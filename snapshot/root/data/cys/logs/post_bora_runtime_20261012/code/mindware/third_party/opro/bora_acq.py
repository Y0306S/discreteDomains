import os
import time
import json
import re
import asyncio
import pandas as pd
from typing import Any, Dict, List, Tuple
from openai import AsyncOpenAI

from mindware.utils.logging_utils import get_logger
from mindware.modules.configs import LLMCONFIGS, AVAILABLE_LLMs, LLM_COST_DICT
from .prompt import BORAPromptBuilder

class BORALLM_ACQ:
    """
    BORA Acquisition Function.
    Handles the interaction with LLM for BORA's specific actions (a2, a3).
    
    Responsibilities:
    1. Construct Prompts using BORAPromptBuilder.
    2. Call LLM API (Async).
    3. Parse strict JSON response (Insight + Hypotheses).
    """
    def __init__(self, optimizer, **kwargs):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.optimizer = optimizer
        self.task_context = optimizer.task_context
        
        self.chat_engine = optimizer.chat_engine
        self.prompt_cost_per_1k = LLM_COST_DICT[LLMCONFIGS.opt_chat_engine]['promt_cost_per_1k']
        self.completion_cost_per_1k = LLM_COST_DICT[LLMCONFIGS.opt_chat_engine]['completion_cost_per_1k']

        self.client = AsyncOpenAI(
            api_key=LLMCONFIGS.OPENAI_KEY,
            base_url=LLMCONFIGS.OPENAI_BASE_URL
        )
        
        # BORA 温度设置：通常需要一定的创造性但又要保证格式
        # 相比 BOPRO 的 0.7，这里可以保持一致或根据实验微调
        self.temperature = kwargs.get("temperature", 0.7)
        self.total_llm_cost = 0.0

        self.output_dir = optimizer.output_dir
        self.conversation_dir = os.path.join(self.output_dir, "bora_conversations")
        os.makedirs(self.conversation_dir, exist_ok=True)

    def get_total_llm_cost(self) -> float:
        return self.total_llm_cost

    def _dump_conversation(self, filename: str, messages: List[Dict]):
        """Dump the conversation log for debugging."""
        try:
            with open(os.path.join(self.conversation_dir, filename), 'w', encoding='utf-8') as f:
                for msg in messages:
                    f.write(f"--- {msg['role'].upper()} ---\n{msg['content']}\n\n")
        except Exception as e:
            self.logger.warning(f"Failed to dump conversation: {e}")

    def _parse_response(self, response: str) -> Dict:
        """
        BORA 专用解析器。
        目标：提取 JSON 中的 'insight' 和 'hypotheses' 列表中的配置。
        
        Expected Format:
        {
            "insight": "...",
            "hypotheses": [
                {"name": "...", "config": {...}}, ...
            ]
        }
        """
        result = {"valid": False, "configs": [], "insight": "No insight parsed."}
        
        try:
            # 1. 尝试提取 Markdown 代码块中的 JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            
            # 2. 如果没有代码块，尝试提取最外层大括号
            if not json_match:
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                # 常见 JSON 错误修复
                json_str = re.sub(r',\s*\}', '}', json_str)
                json_str = re.sub(r',\s*\]', ']', json_str)
                
                data = json.loads(json_str)
                
                # 提取 Insight (兼容 comment/insight 键名)
                result["insight"] = data.get("insight", data.get("comment", ""))
                
                # 提取配置列表
                configs = []
                raw_hypotheses = data.get("hypotheses", [])
                
                if isinstance(raw_hypotheses, list):
                    for h in raw_hypotheses:
                        if isinstance(h, dict):
                            # Prompt 中定义配置在 "config" 字段下
                            if "config" in h:
                                configs.append(h["config"])
                            # 兼容性处理
                            elif "points" in h and isinstance(h["points"], list) and len(h["points"]) > 0:
                                configs.append(h["points"][0])
                
                result["configs"] = configs
                if configs:
                    result["valid"] = True
            else:
                self.logger.warning("[BORA ACQ] No JSON structure found in response.")

        except json.JSONDecodeError as e:
            self.logger.warning(f"[BORA ACQ] JSON Decode Error: {e}. Raw snippet: {response[:100]}...")
        except Exception as e:
            self.logger.warning(f"[BORA ACQ] Parsing Error: {e}")

        return result

    async def _async_generate(self, messages: List[Dict]) -> Tuple[str, float, float]:
        t0 = time.time()
        try:
            resp = await self.client.chat.completions.create(
                model=self.chat_engine,
                messages=messages,
                temperature=self.temperature,
                # top_p=0.9, # 可选，根据 BOPRO 代码规范保留或移除
                max_tokens=2048,
                n=1 
            )
            content = resp.choices[0].message.content
            
            p_tokens = resp.usage.prompt_tokens
            c_tokens = resp.usage.completion_tokens
            cost = (p_tokens / 1000 * self.prompt_cost_per_1k) + (c_tokens / 1000 * self.completion_cost_per_1k)
            
            return content, cost, time.time() - t0
        except Exception as e:
            self.logger.error(f"OpenAI API Error: {e}")
            return None, 0.0, time.time() - t0

    def get_action_a2_suggestions(self, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, comments_history: List[str], n_generate: int = 1):
        """Action a2: LLM Suggests (High Exploration)"""
        return asyncio.run(self._run_bora_loop(
            action_type="a2",
            obs_cfg_df=obs_cfg_df,
            obs_acc_df=obs_acc_df,
            comments_history=comments_history,
            n_generate=n_generate
        ))

    def get_action_a3_selection(self, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, comments_history: List[str], bo_candidates: List[Any], n_select: int = 1):
        """Action a3: LLM Selects (Guided Exploration)"""
        return asyncio.run(self._run_bora_loop(
            action_type="a3",
            obs_cfg_df=obs_cfg_df,
            obs_acc_df=obs_acc_df,
            comments_history=comments_history,
            bo_candidates=bo_candidates,
            n_select=n_select
        ))

    async def _run_bora_loop(self, action_type, **kwargs):
        # 1. 根据动作类型构建 Prompt
        if action_type == "a2":
            messages = BORAPromptBuilder.get_action_a2_messages(
                self.task_context, 
                obs_cfg_df=kwargs['obs_cfg_df'],
                obs_acc_df=kwargs['obs_acc_df'],
                comments_history=kwargs['comments_history'],
                n_generate=kwargs['n_generate']
            )
        elif action_type == "a3":
            messages = BORAPromptBuilder.get_action_a3_messages(
                self.task_context,
                obs_cfg_df=kwargs['obs_cfg_df'],
                obs_acc_df=kwargs['obs_acc_df'],
                comments_history=kwargs['comments_history'],
                bo_candidates=kwargs['bo_candidates'],
                n_select=kwargs['n_select']
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # 2. 调用 LLM
        response_text, cost, duration = await self._async_generate(messages)
        self.total_llm_cost += cost

        # 3. 解析结果
        configs = []
        insight = ""
        
        if response_text:
            parsed = self._parse_response(response_text)
            
            # 保存日志 (附加 LLM 回复)
            log_messages = messages + [{"role": "assistant", "content": response_text}]
            timestamp = int(time.time())
            self._dump_conversation(f"bora_{action_type}_{timestamp}.txt", log_messages)
            
            if parsed["valid"]:
                configs = parsed["configs"]
                insight = parsed["insight"]
            else:
                self.logger.warning(f"[BORA ACQ] Parsing failed or no valid configs for action {action_type}")
        
        return configs, insight, cost, duration