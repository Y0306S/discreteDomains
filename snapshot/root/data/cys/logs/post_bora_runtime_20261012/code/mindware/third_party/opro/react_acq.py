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
from .prompt import OPROReactPromptBuilder
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import LLMCONFIGS, AVAILABLE_LLMs
from mindware.utils.logging_utils import get_logger

import logging

# 配置日志输出为 INFO
logging.basicConfig(level=logging.INFO)

from .utils import (
    CustomEncoder,
    _convert_to_dict,
    _filter_and_validate
)


class ReActLLM_ACQ:
    def __init__(self, optimizer: 'LLMOptimizer', **kwargs):
        from mindware.components.optimizers.llm_optimizer import LLMOptimizer

        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.optimizer = optimizer
        self.task_context = optimizer.task_context
        self.n_candidates = optimizer.n_candidates

        self.KD = KnowledgeDatabase.instance()
        self.lower_is_better = self.KD.lower_is_better

        # # gpt-4.1-mini
        # # self.prompt_cost_per_1k = 0.0004
        # # self.completion_cost_per_1k = 0.0016

        # # deepseek-v3
        # self.prompt_cost_per_1k = 0.000225352
        # self.completion_cost_per_1k = 0.000901408

        self.chat_engine = optimizer.chat_engine
        self.prompt_cost_per_1k = optimizer.prompt_cost_per_1k
        self.completion_cost_per_1k = optimizer.completion_cost_per_1k
        print(f"[ReAct ACQ] Using chat engine: {self.chat_engine} with costs - Prompt: {self.prompt_cost_per_1k}/1k tokens, Completion: {self.completion_cost_per_1k}/1k tokens")

        self.constraints = self.task_context.get("hyperparameter_constraints", {})
        self.algo_choices = self.task_context.get("algorithm_choices", [])
        self.algo2params = self.task_context.get("algorithm_param_names", {})
        self.task_id = self.task_context.get("task_id", {})

        self.warping_transformer = optimizer.warping_transformer
        self.apply_warping = optimizer.warping_transformer is not None

        # 注册绑定这两个历史记录，注意：如果只改历史记录内部的信息，能全局更新，但是如果改了引用，需要重新绑定。todo: 确保没有改引用
        self.algo_selection_history = self.KD.algo_selection_history
        self.param_tuning_history = self.KD.param_tuning_history

        self.reflection_interval = kwargs.get("reflection_interval", 5) # 每 5 轮进行一次元反思
        self.exploration_algos: List[str] = []
        self.exploitation_algos: List[str] = []
        self.meta_reflection_history: List[Dict] = []


        self.total_llm_cost = 0.0

        self.client = AsyncOpenAI(
            api_key=LLMCONFIGS.OPENAI_KEY,
            base_url=LLMCONFIGS.OPENAI_BASE_URL
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
        await self.optimizer.rate_limiter.check_and_wait(request_text, start_time)
        try:
            resp = await self.client.chat.completions.create(model=self.chat_engine, messages=messages, temperature=temperature, max_tokens=2048, top_p=0.95, n=n, timeout=120)
            duration = time.time() - start_time
            total_tokens = resp.usage.total_tokens
            self.optimizer.rate_limiter.add_request_info(total_tokens, start_time)

            cost = (self.prompt_cost_per_1k * (resp.usage.prompt_tokens / 1000) + self.completion_cost_per_1k * (resp.usage.completion_tokens / 1000))
            return resp, cost, duration
        except openai.APIStatusError as e: print(f"[ERROR] OpenAI API Error: {e.status_code}. Response: {e.response}"); return None, 0.0, time.time() - start_time
        except openai.AuthenticationError as e: print(f"[ERROR] OpenAI Authentication Error: Invalid API Key or token. {e}"); raise e
        except openai.RateLimitError as e: print(f"[ERROR] OpenAI Rate Limit Error: {e}. Check your API plan and usage."); return None, 0.0, time.time() - start_time
        except openai.BadRequestError as e: print(f"[ERROR] OpenAI Invalid Request Error: {e}. Check model name and parameters."); return None, 0.0, time.time() - start_time
        except (openai.APIConnectionError, openai.Timeout) as e: print(f"[ERROR] OpenAI Connection/Timeout Error: {e}"); return None, 0.0, time.time() - start_time
        except Exception as e: print(f"[ERROR] An unexpected error occurred: {type(e).__name__}: {e}"); return None, 0.0, time.time() - start_time


    def _parse_thought_action_summary(self, response: str) -> Dict:
        try:
            detailed_thought = re.search(r"Thought:(.*?)Thought Summary:", response, re.DOTALL).group(1).strip()
            summary_thought = re.search(r"Thought Summary:(.*?)Action:", response, re.DOTALL).group(1).strip()
            action_raw = re.search(r"Action:(.*)", response, re.DOTALL).group(1)
            action_cleaned = re.sub(r'^[*\s`]*(json\s*)?', '', action_raw.strip())
            if action_cleaned.endswith('```'):
                action_cleaned = action_cleaned[:-3].strip()
            action_cleaned = action_cleaned.rstrip('`').strip()
            return {"detailed_thought": detailed_thought, "summary_thought": summary_thought, "action": action_cleaned}
        except Exception:
            # "detailed_thought" 键名保留，但内容是精简后的 "Thought"
            return {"detailed_thought": response, "summary_thought": "Parsing failed.", "action": "PARSING_ERROR"}

    def _parse_reflection_summary(self, response: str) -> Dict:
        try:
            # --- 修改：匹配你精简后的 Prompt ---
            detailed_reflection = re.search(r"Reflection:(.*?)Reflection Summary:", response, re.DOTALL).group(1).strip()
            summary_reflection = re.search(r"Reflection Summary:(.*)", response, re.DOTALL).group(1).strip()
            # "detailed_reflection" 键名保留，但内容是精简后的 "Reflection"
            return {"detailed_reflection": detailed_reflection, "summary_reflection": summary_reflection}
        except Exception:
            return {"detailed_reflection": response, "summary_reflection": "Parsing failed."}

    def _parse_meta_reflection_response(self, response: str) -> Tuple[List[str], List[str]]:
        try:
            json_str_match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
            if not json_str_match:
                return [], []

            data = json.loads(json_str_match.group(1))

            exploration = data.get("exploration_algorithms", [])
            exploitation = data.get("exploitation_algorithms", [])

            valid_exploration = [str(a) for a in exploration if isinstance(a, str)]
            valid_exploitation = [str(a) for a in exploitation if isinstance(a, str)]

            return valid_exploration, valid_exploitation
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[ERROR] Failed to parse meta-reflection JSON response: {e}")
            return [], []

    async def _run_meta_reflection(self, iter_idx: int):
        print(f"\n--- [ReAct ACQ] Running Meta-Reflection at trial {iter_idx} ---")
        messages = OPROReactPromptBuilder._get_meta_reflection_prompt(self)
        resp, cost, duration = await self._async_generate(messages, 0.5, 1)
        if cost is not None: 
            # self.optimizer.llm_query_cost[-1] += cost
            self.total_llm_cost += cost
        # if duration is not None: self.optimizer.llm_query_time[-1] += duration
        if resp is None: print("[ReAct ACQ] Meta-Reflection failed due to API error."); return
        response_text = resp.choices[0].message.content
        self.meta_reflection_history.append({"trial_id": iter_idx, "response": response_text})
        exploration_algos, exploitation_algos = self._parse_meta_reflection_response(response_text)
        if exploration_algos or exploitation_algos:
            self.exploration_algos, self.exploitation_algos = exploration_algos, exploitation_algos
            print(f"[ReAct ACQ] Meta-Reflection complete. Updated strategic lists:\n  - Exploration: {self.exploration_algos}\n  - Exploitation: {self.exploitation_algos}")
        final_messages = messages + [{"role": "assistant", "content": response_text}]
        self._dump_conversation(f"trial_{iter_idx}_meta_reflection", final_messages)

    def get_candidate_points(self, observed_configs, observed_fvals, alpha, iter_idx):

        return asyncio.run(self._run_react_loop(observed_configs, observed_fvals, iter_idx))

    async def _run_react_loop(self, observed_configs, observed_fvals, iter_idx):
        total_cost, total_llm_time = 0.0, 0.0
        evaluated_configs_for_return, eval_times_for_return = [], []

        from mindware.components.optimizers.llm_optimizer_utils import _evaluate_dict

        # --- 主循环：使用 while 和 max_tries 保证生成足够的候选点 ---
        max_tries = self.n_candidates + 5 # 允许额外5次失败尝试

        # --- 为日志和错误处理预先初始化变量 ---
        selected_algo = None

        while len(evaluated_configs_for_return) < self.n_candidates and max_tries > 0:
            candidate_idx = len(evaluated_configs_for_return)
            max_tries -= 1
            self.logger.info(f"\n[ReAct ACQ] Trial [{iter_idx}] - Generating Candidate [{candidate_idx + 1}/{self.n_candidates}]... (Tries left: {max_tries})")

            # --- 重构：不再有共享的 `messages` 列表 ---

            # --- 步骤 A: 算法选择 (独立调用) ---
            self.logger.info("[ReAct ACQ] === Step A: Algorithm Selection ===")
            algo_strategy_name, algo_messages = OPROReactPromptBuilder._get_algo_selection_messages(self)
            resp, cost, duration = await self._async_generate(algo_messages, 0.7, 1)

            if resp is None:
                self.logger.debug("[ReAct ACQ] Step A failed due to API error. Skipping candidate.")
                continue
            total_cost += cost; total_llm_time += duration

            response_text_algo = resp.choices[0].message.content
            parsed_algo = self._parse_thought_action_summary(response_text_algo)
            detailed_algo_thought, summary_algo_thought, selected_algo_raw = parsed_algo['detailed_thought'], parsed_algo['summary_thought'], parsed_algo['action']
            selected_algo = selected_algo_raw.strip().strip(' *`_')

            if selected_algo == "PARSING_ERROR" or selected_algo not in self.algo_choices:
                self.logger.debug(f"[ReAct ACQ] Warning: Parsed algorithm '{selected_algo}' is invalid. Choosing randomly.")
                selected_algo = random.choice(self.algo_choices)

            # # 保存本次独立对话
            # self._dump_conversation(
            #     f"trial_{iter_idx}_candidate_{candidate_idx}_A_AlgoSelect",
            #     algo_messages + [{"role": "assistant", "content": response_text_algo}]
            # )

            # --- 步骤 B: 参数选择 (独立调用) ---
            self.logger.info(f"[ReAct ACQ] === Step B: Parameter Tuning for {selected_algo} ===")
            param_strategy_name, param_messages = OPROReactPromptBuilder._get_param_tuning_messages(self, selected_algo)
            resp, cost, duration = await self._async_generate(param_messages, 0.8, 1)

            if resp is None:
                self.logger.debug("[ReAct ACQ] Step B failed due to API error. Skipping candidate.")
                continue
            total_cost += cost; total_llm_time += duration

            response_text_param = resp.choices[0].message.content
            parsed_param = self._parse_thought_action_summary(response_text_param)
            detailed_param_thought_param, summary_param_thought_param, param_action_str = parsed_param['detailed_thought'], parsed_param['summary_thought'], parsed_param['action']

            if param_action_str == "PARSING_ERROR":
                print("[ReAct ACQ] Warning: Parameter parsing failed. Sampling randomly.")
                config_dict = self.optimizer.config_space.sample_configuration().get_dictionary()
            else:
                config_dict = _convert_to_dict(param_action_str)

            config_dict['algorithm'] = selected_algo # 确保算法被添加

            # # 保存本次独立对话
            # self._dump_conversation(
            #     f"trial_{iter_idx}_candidate_{candidate_idx}_B_ParamTune",
            #     param_messages + [{"role": "assistant", "content": response_text_param}]
            # )

            # --- 步骤 C: 评估 ---
            self.logger.info("[ReAct ACQ] === Step C: Evaluation ===")
            all_observed_points = [cfg.get_dictionary() for cfg in self.optimizer.configs]
            validation_result = _filter_and_validate(self, all_observed_points, [config_dict])
            if validation_result.empty:
                print(f"[ReAct ACQ] Proposed config was invalid or duplicate. Skipping...")
                continue

            obs, eval_time = _evaluate_dict(self.optimizer, config_dict)
            perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)  # obs.objectives[0] 来自于 evaluator，越小越好，所以这里根据方向调整
            self.logger.info(f"[ReAct ACQ] -> Observed Performance: {perf:.6f}")

            # --- 步骤 D: 反思 (独立调用) ---
            self.logger.info("[ReAct ACQ] === Step D: Reflection ===")
            reflection_context = { 
                "trail_id": f"{iter_idx}-{candidate_idx}", 
                "action_config": config_dict, 
                "observation_perf": perf,
                "param_thought_summary": summary_param_thought_param # 传入B阶段的思考
            }
            reflect_messages = OPROReactPromptBuilder._get_reflection_messages(self, reflection_context)
            resp, cost, duration = await self._async_generate(reflect_messages, 0.5, 1)

            response_text_reflection = ""
            summary_reflection = "Reflection failed."
            if resp is None: 
                print("[ReAct ACQ] Warning: Step D (Reflection) failed due to API error.")
                response_text_reflection = "Reflection failed due to API error."
            else:
                total_cost += cost; total_llm_time += duration
                response_text_reflection = resp.choices[0].message.content
                parsed_reflection = self._parse_reflection_summary(response_text_reflection)
                summary_reflection = parsed_reflection['summary_reflection']

            # # 保存本次独立对话
            # self._dump_conversation(
            #     f"trial_{iter_idx}_candidate_{candidate_idx}_D_Reflect", 
            #     reflect_messages + [{"role": "assistant", "content": response_text_reflection}]
            # )

            # --- 步骤 E: 记录与立即更新 (实现轮内学习的关键) ---
            print("[ReAct ACQ] === Step E: Logging History ===")
            full_history_entry = { 
                "trail_id": f"{iter_idx}-{candidate_idx}", 
                "detailed_thought": detailed_param_thought_param, 
                "thought_summary": summary_param_thought_param, 
                "action_config": config_dict, 
                "observation_perf": perf, 
                "reflection_summary": summary_reflection, 
                "algo_strategy_name": algo_strategy_name,
                "param_strategy_name": param_strategy_name
            }

            self.optimizer.update_history(obs, full_history_entry)

            evaluated_configs_for_return.append(config_dict)
            eval_times_for_return.append(eval_time)

        # --- 循环结束后，统一记录本轮 Trial 的产出 ---

        # (我们之前讨论的 UnboundLocalError 修复)
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
            "final_param_tuning_history": self.param_tuning_history,
            "meta_reflection_history": self.meta_reflection_history
        }
        self._dump_trial_state(iter_idx, trial_debug_data)

        self.total_llm_cost += total_cost

        # 检查是否触发元反思
        if iter_idx  % self.reflection_interval == 0 and self.algo_selection_history:
            await self._run_meta_reflection(iter_idx)

        return pd.DataFrame(evaluated_configs_for_return), total_cost, total_llm_time, eval_times_for_return


