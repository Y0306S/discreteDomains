import os
import time

import openai
from openai import AsyncOpenAI

import asyncio
import numpy as np
import pandas as pd
import json
import random
import re
from aiohttp import ClientSession
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from .prompt import MCTSReactPromptBuilder
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import LLMCONFIGS, AVAILABLE_LLMs, LLM_COST_DICT
from mindware.utils.logging_utils import get_logger

from mindware.components.optimizers.mcts import LLMEXPLORATION, LLMEXPLOITATION, WARMUP, ExpandStrategy

import logging

# 配置日志输出为 INFO
logging.basicConfig(level=logging.INFO)

from .utils import (
    CustomEncoder,
    _convert_to_dict,
    _filter_and_validate,
    random_sample_algo_config
)


class ReActMCTSLLM_ACQ:
    def __init__(self, optimizer: 'LLMMCTSOptimizer', **kwargs):
        from mindware.components.optimizers.llmmcts_optimizer import LLMMCTSOptimizer

        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.optimizer = optimizer
        self.task_context = optimizer.task_context
        self.n_candidates = 1

        self.KD = KnowledgeDatabase.instance()
        self.lower_is_better = self.KD.lower_is_better

        self.chat_engine = optimizer.chat_engine
        self.prompt_cost_per_1k = LLM_COST_DICT[LLMCONFIGS.opt_chat_engine]['promt_cost_per_1k']
        self.completion_cost_per_1k = LLM_COST_DICT[LLMCONFIGS.opt_chat_engine]['completion_cost_per_1k']
        self.temperature = AVAILABLE_LLMs[LLMCONFIGS.opt_chat_engine].get("temperature", 0.8)
        print(f"[ReAct MCTS ACQ] Using chat engine: {self.chat_engine} with costs - Prompt: {self.prompt_cost_per_1k}/1k tokens, Completion: {self.completion_cost_per_1k}/1k tokens")

        self.constraints = self.task_context.get("hyperparameter_constraints", {})
        self.algo_choices = self.task_context.get("algorithm_choices", [])
        self.algo2params = self.task_context.get("algorithm_param_names", {})
        self.task_id = self.task_context.get("task_id", {})

        self.warping_transformer = optimizer.warping_transformer
        self.apply_warping = optimizer.warping_transformer is not None

        # 注册绑定这两个历史记录，注意：如果只改历史记录内部的信息，能全局更新，但是如果改了引用，需要重新绑定。todo: 确保没有改引用
        self.algo_selection_history = self.KD.algo_selection_history
        self.param_tuning_history = self.KD.param_tuning_history


        self.total_llm_cost = 0.0

        self.client = AsyncOpenAI(
            api_key=LLMCONFIGS.OPENAI_KEY,
            base_url=LLMCONFIGS.OPENAI_BASE_URL,
            default_headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        )

        self.output_dir = self.optimizer.output_dir
        self.conversation_dir = os.path.join(self.output_dir, "llm_conversations")
        self.debug_dir = os.path.join(self.output_dir, "debug_states") # 新增
        os.makedirs(self.conversation_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    def get_total_llm_cost(self) -> float:
        return self.total_llm_cost

    def _dump_conversation(self, filename_prefix: str, messages: List[Dict]):
        filename = f"{filename_prefix}.txt"
        filepath = os.path.join(self.conversation_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for msg in messages:
                    role, content = msg.get('role', 'unknown'), msg.get('content', '')
                    f.write(f"--- ROLE: {role.upper()} ---\n{content}\n\n")
        except Exception as e:
            print(f"[ERROR] Failed to dump conversation to {filepath}: {e}")

    def _dump_trial_state(self, trial_idx: int, debug_data: Dict):
        """将一轮trial的完整状态保存到JSON文件"""
        filename = f"trial_{trial_idx}_debug_state.json"
        filepath = os.path.join(self.debug_dir, filename) 
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
        except Exception as e:
            print(f"[ERROR] Failed to dump trial state to {filepath}: {e}")

    # --- 辅助方法 ---
    async def _async_generate(self, messages: List[Dict], temperature: float, n: int) -> Tuple[Any, float, float]:
        start_time = time.time()
        request_text = "".join([m.get("content", "") for m in messages])
        # await self.optimizer.rate_limiter.check_and_wait(request_text, start_time)
        try:
            resp = await self.client.chat.completions.create(model=self.chat_engine, messages=messages, temperature=temperature, max_tokens=4096, top_p=0.95, n=n, timeout=180)

            duration = time.time() - start_time
            total_tokens = resp.usage.total_tokens
            # self.optimizer.rate_limiter.add_request_info(total_tokens, start_time)

            cost = (self.prompt_cost_per_1k * (resp.usage.prompt_tokens / 1000) + self.completion_cost_per_1k * (resp.usage.completion_tokens / 1000))
            return resp, cost, duration
        except openai.AuthenticationError as e: print(f"[ERROR] OpenAI Authentication Error: Invalid API Key or token. {e}"); raise e
        except openai.RateLimitError as e: print(f"[ERROR] OpenAI Rate Limit Error: {e}. Check your API plan and usage."); return None, 0.0, time.time() - start_time
        except openai.BadRequestError as e: print(f"[ERROR] OpenAI Invalid Request Error: {e}. Check model name and parameters."); return None, 0.0, time.time() - start_time
        except (openai.APIConnectionError, openai.APITimeoutError) as e: print(f"[ERROR] OpenAI Connection/Timeout Error: {e}"); return None, 0.0, time.time() - start_time
        except openai.APIStatusError as e: print(f"[ERROR] OpenAI API Error: {e.status_code}. Response: {e.response}"); return None, 0.0, time.time() - start_time  
        except Exception as e: print(f"[ERROR] An unexpected error occurred: {type(e).__name__}: {e}"); return None, 0.0, time.time() - start_time


    def _parse_thought_action_summary(self, response: str) -> Dict:
        try:
            # detailed_thought = re.search(r"Thought:(.*?)Thought Summary:", response, re.DOTALL).group(1).strip()
            # summary_thought = re.search(r"Thought Summary:(.*?)Action:", response, re.DOTALL).group(1).strip()
            # 支持 **Thought**: 和 **Action**: 格式
            detailed_thought = re.search(r"\*\*?Thought\*\*?:(.*?)\*\*?Action\*\*?:", response, re.DOTALL)
            if not detailed_thought:
                detailed_thought_match = re.search(r"\*\*?Thought\*\*?:(.*?)```json", response, re.DOTALL)
                if detailed_thought_match:
                    detailed_thought = detailed_thought_match.group(1).strip()
                else:
                    # 尝试不带星号的格式
                    detailed_thought_match = re.search(r"Thought:(.*?)Action:", response, re.DOTALL)
                    if detailed_thought_match:
                        detailed_thought = detailed_thought_match.group(1).strip()
                    else:
                        detailed_thought_match = re.search(r"Thought:(.*?)```json", response, re.DOTALL)
                        if detailed_thought_match:
                            detailed_thought = detailed_thought_match.group(1).strip()
                        else:
                            detailed_thought = ""
            else:
                detailed_thought = detailed_thought.group(1).strip()
            summary_thought = detailed_thought
            #  去掉首尾的*，并用空格替代\n
            detailed_thought = detailed_thought.strip('*').strip().replace('\n', ' ')
            summary_thought = summary_thought.strip('*').strip().replace('\n', ' ')

            pattern = r'```json\n\s*((?:{.*?})|(?:\[.*?\]))\s*,?\n```'
            json_str_match = re.search(pattern, response, re.DOTALL)
            if not json_str_match:
                raise ValueError("No JSON block found in the response.")
            action_str = json_str_match.group(1).strip()
            action_cleaned = json.loads(action_str)

            return {"detailed_thought": detailed_thought, "summary_thought": summary_thought, "action": action_cleaned}
        except Exception:
            # "detailed_thought" 键名保留，但内容是精简后的 "Thought"
            return {"detailed_thought": response, "summary_thought": "Parsing failed.", "action": "PARSING_ERROR"}

    def _parse_reflection_summary(self, response: str) -> Dict:
        try:
            # 支持多种格式：**Reflection**:, *Reflection*:, Reflection:
            # 先尝试匹配 Markdown 加粗格式
            detailed_reflection_match = re.search(r"\*\*Reflection\*\*:(.*?)\*\*Reflection Summary\*\*:", response, re.DOTALL)
            if not detailed_reflection_match:
                # 尝试单星号格式
                detailed_reflection_match = re.search(r"\*Reflection\*:(.*?)\*Reflection Summary\*:", response, re.DOTALL)
            if not detailed_reflection_match:
                # 尝试无星号格式
                detailed_reflection_match = re.search(r"Reflection:(.*?)Reflection Summary:", response, re.DOTALL)
            
            if detailed_reflection_match:
                detailed_reflection = detailed_reflection_match.group(1).strip()
            else:
                # 如果找不到 Reflection: 部分，尝试找到 Reflection Summary: 之前的所有内容
                summary_match = re.search(r"\*\*Reflection Summary\*\*:", response, re.DOTALL)
                if not summary_match:
                    summary_match = re.search(r"\*Reflection Summary\*:", response, re.DOTALL)
                if not summary_match:
                    summary_match = re.search(r"Reflection Summary:", response, re.DOTALL)
                
                if summary_match:
                    # 提取 Reflection Summary: 之前的所有内容作为 detailed_reflection
                    detailed_reflection = response[:summary_match.start()].strip()
                    # 移除开头的 "**Reflection**:" 或类似格式
                    detailed_reflection = re.sub(r'^\*\*?Reflection\*\*?:', '', detailed_reflection, flags=re.MULTILINE).strip()
                else:
                    detailed_reflection = response.strip()
            
            # 解析 Reflection Summary
            # 尝试匹配 **Reflection Summary**: 格式
            summary_reflection_match = re.search(r"\*\*Reflection Summary\*\*:\s*\n(.*)", response, re.DOTALL)
            if not summary_reflection_match:
                # 尝试 *Reflection Summary*: 格式
                summary_reflection_match = re.search(r"\*Reflection Summary\*:\s*\n(.*)", response, re.DOTALL)
            if not summary_reflection_match:
                # 尝试无星号格式
                summary_reflection_match = re.search(r"Reflection Summary:\s*\n(.*)", response, re.DOTALL)
            if not summary_reflection_match:
                # 尝试匹配同一行的内容
                summary_reflection_match = re.search(r"\*\*Reflection Summary\*\*:(.*)", response, re.DOTALL)
            if not summary_reflection_match:
                summary_reflection_match = re.search(r"\*Reflection Summary\*:(.*)", response, re.DOTALL)
            if not summary_reflection_match:
                summary_reflection_match = re.search(r"Reflection Summary:(.*)", response, re.DOTALL)
            
            if summary_reflection_match:
                summary_reflection = summary_reflection_match.group(1).strip()
            else:
                summary_reflection = detailed_reflection  # 如果找不到，使用 detailed_reflection 作为 summary
            
            return {"detailed_reflection": detailed_reflection, "summary_reflection": summary_reflection}
        except Exception as e:
            # 解析失败时，尝试提取整个 response 作为 reflection
            # 但不要返回 "Parsing failed."，而是返回实际的 response 内容
            return {"detailed_reflection": response, "summary_reflection": response[:500] if len(response) > 500 else response}


    def get_candidate_points(self, iter_idx, expand_strategy, node_path):

        return asyncio.run(self._run_react_loop(iter_idx, expand_strategy, node_path))

    async def _run_react_loop(self, iter_idx, expand_strategy, node_path):
        total_cost, total_llm_time = 0.0, 0.0
        evaluated_configs_for_return, eval_times_for_return = [], []
        expand_strategys = []

        from mindware.components.optimizers.llm_optimizer_utils import _evaluate_dict

        # --- 主循环：使用 while 和 max_tries 保证生成足够的候选点 ---
        max_tries = self.n_candidates + 2 # 允许额外5次失败尝试

        # --- 为日志和错误处理预先初始化变量 ---
        selected_algo = None

        dup_num = 0 # 是否避免重复配置，如果第一次出现重复，则后续在promot里面开启避免重复

        while len(evaluated_configs_for_return) < self.n_candidates and max_tries > 0:
            candidate_idx = len(evaluated_configs_for_return)
            max_tries -= 1
            self.logger.info(f"\n[ReAct MCTS ACQ] Trial [{iter_idx}] - Generating Candidate [{candidate_idx + 1}/{self.n_candidates}]... (Tries left: {max_tries})")

            # --- 步骤 B: 根据扩展决策推荐配置 ---
            self.logger.info("[ReAct MCTS ACQ] === Step B: Parameter Recommendation ===")
            param_strategy_name = expand_strategy.strategy
            if expand_strategy.strategy in [LLMEXPLORATION, LLMEXPLOITATION, WARMUP]:
                selected_algo = expand_strategy.selected_algo
                parent_id = expand_strategy.parent_id
                self.logger.info(f"[ReAct MCTS ACQ] MCTS-based strategy selected: {expand_strategy.strategy} for algorithm: {selected_algo}")
                if expand_strategy.strategy in [LLMEXPLORATION, LLMEXPLOITATION]:
                    self.logger.info(f"[ReAct MCTS ACQ] Parameter Tuning for {selected_algo} based on node {parent_id}")
                param_messages = MCTSReactPromptBuilder._get_param_tuning_messages(self, selected_algo, expand_strategy.strategy, node_path, dup_num=dup_num)
                # --- 调用参数选择 ---
                resp, cost, duration = await self._async_generate(param_messages, self.temperature, 1)

                if resp is None:
                    self.logger.debug("[ReAct MCTS ACQ] Step B failed due to API error. Skipping candidate.")
                    continue
                total_cost += cost; total_llm_time += duration

                response_text_param = resp.choices[0].message.content
                self.logger.info(f"[LLM Response Preview]: {response_text_param}")
                parsed_param = self._parse_thought_action_summary(response_text_param)
                detailed_param_thought_param, summary_param_thought_param, param_action_str = parsed_param['detailed_thought'], parsed_param['summary_thought'], parsed_param['action']

                if param_action_str == "PARSING_ERROR":
                    print("[ReAct MCTS ACQ] Warning: Parameter parsing failed. Sampling randomly.")
                                    # 保存本次独立对话
                    self._dump_conversation(
                        f"failed_trial_{iter_idx}_candidate_{candidate_idx}_B_ParamTune",
                        param_messages + [{"role": "assistant", "content": response_text_param}]
                    )
                    continue
                else:
                    config_dict = param_action_str
                
                # for k, v in config_dict.items():
                #     if isinstance(v, str):
                #         try:
                #             # 尝试转 float，能转就转 (处理 "1e-5" -> 1e-5)
                #             config_dict[k] = float(v)
                #         except ValueError:
                #             # 转不了就算了 (比如是 categorical 的字符串 "gini")
                #             pass
                        
                config_dict['algorithm'] = selected_algo # 确保算法被添加

                # 保存本次独立对话
                self._dump_conversation(
                    f"trial_{iter_idx}_candidate_{candidate_idx}_B_ParamTune",
                    param_messages + [{"role": "assistant", "content": response_text_param}]
                )


            else:
                raise ValueError(f"[ReAct MCTS ACQ] Invalid expansion strategy: {expand_strategy.strategy}")


            # --- 步骤 C: 评估 ---
            self.logger.info("[ReAct MCTS ACQ] === Step C: Evaluation ===")
            all_observed_points = [cfg.get_dictionary() for cfg in self.optimizer.configs]
            tmp_conf_dict = self.KD._dict_to_cfg(config_dict).get_dictionary()  # 先填充再验证
            validation_result = _filter_and_validate(self, all_observed_points, [tmp_conf_dict])
            if validation_result.empty:
                print(f"[ReAct MCTS ACQ] Proposed config was invalid or duplicate. Skipping...")
                dup_num += 1 # 开启避免重复
                continue

            obs, eval_time = _evaluate_dict(self.optimizer, config_dict)
            perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)  # obs.objectives[0] 来自于 evaluator，越小越好，所以这里根据方向调整
            self.logger.info(f"[ReAct MCTS ACQ] -> Observed Performance: {perf:.6f}")

            # --- 步骤 D: 反思 (独立调用) ---
            self.logger.info("[ReAct MCTS ACQ] === Step D: Reflection ===")

            reflection_context = { 
                "trail_id": f"{iter_idx}-{candidate_idx}", 
                "action_config": config_dict, 
                "observation_perf": perf,
                "param_thought_summary": summary_param_thought_param # 传入B阶段的思考
            }
            reflect_messages = MCTSReactPromptBuilder._get_reflection_messages(self, reflection_context, node_path)
            resp, cost, duration = await self._async_generate(reflect_messages, self.temperature, 1)

            response_text_reflection = ""
            summary_reflection = "Reflection failed."
            if resp is None: 
                print("[ReAct MCTS ACQ] Warning: Step D (Reflection) failed due to API error.")
                response_text_reflection = "Reflection failed due to API error."

                # 保存本次独立对话
                self._dump_conversation(
                    f"failed_trial_{iter_idx}_candidate_{candidate_idx}_D_Reflect", 
                    reflect_messages + [{"role": "assistant", "content": response_text_reflection}]
                )
            else:
                total_cost += cost; total_llm_time += duration
                response_text_reflection = resp.choices[0].message.content
                parsed_reflection = self._parse_reflection_summary(response_text_reflection)
                summary_reflection = parsed_reflection['summary_reflection']

            # 保存本次独立对话
            self._dump_conversation(
                f"trial_{iter_idx}_candidate_{candidate_idx}_D_Reflect", 
                reflect_messages + [{"role": "assistant", "content": response_text_reflection}]
            )

            # --- 步骤 E: 记录与立即更新 (实现轮内学习的关键) ---
            print("[ReAct MCTS ACQ] === Step E: Logging History ===")
            full_history_entry = { 
                "trail_id": f"{iter_idx}-{candidate_idx}", 
                "detailed_thought": detailed_param_thought_param, 
                "thought_summary": summary_param_thought_param, 
                "action_config": config_dict, 
                "observation_perf": perf, 
                "reflection_summary": summary_reflection, 
                "param_strategy_name": param_strategy_name
            }

            self.optimizer.update_history(obs, expand_strategy, full_history_entry)

            evaluated_configs_for_return.append(config_dict)
            eval_times_for_return.append(eval_time)
            expand_strategys.append(expand_strategy)

        # --- 循环结束后，统一记录本轮 Trial 的产出 ---

        performance_scores_this_trial = []
        if selected_algo and selected_algo in self.param_tuning_history:
             performance_scores_this_trial = [
                 cfg['observation_perf'] for cfg in self.param_tuning_history[selected_algo] 
                 if cfg['trail_id'].startswith(f"{iter_idx}-")
             ]

        trial_debug_data = {
            "trial_index": iter_idx, "total_llm_cost_this_trial": total_cost, "total_llm_time_this_trial": total_llm_time,
            "generated_configs": evaluated_configs_for_return,
            "performance_scores": performance_scores_this_trial,
            "evaluation_times": eval_times_for_return,
            "final_algo_selection_history_stats": self.algo_selection_history,
            "final_param_tuning_history": self.param_tuning_history
        }
        self._dump_trial_state(iter_idx, trial_debug_data)

        self.total_llm_cost += total_cost

        return expand_strategys, evaluated_configs_for_return, total_cost, total_llm_time, eval_times_for_return


