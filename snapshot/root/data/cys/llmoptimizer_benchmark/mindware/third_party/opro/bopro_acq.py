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
from .prompt import BOPROPromptBuilder 

class BOPROLLM_ACQ:
    """
    BOPRO Acquisition Function.
    Implements the logic: Latent Proposal -> Similarity Search -> Direct Prompting -> Generation.
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
        
        self.temperature = kwargs.get("temperature", 0.7)
        
        
        self.total_llm_cost = 0.0

        self.output_dir = optimizer.output_dir
        self.conversation_dir = os.path.join(self.output_dir, "bopro_conversations")
        os.makedirs(self.conversation_dir, exist_ok=True)

    def get_total_llm_cost(self) -> float:
        return self.total_llm_cost

    def _dump_conversation(self, filename: str, messages: List[Dict]):
        try:
            with open(os.path.join(self.conversation_dir, filename), 'w', encoding='utf-8') as f:
                for msg in messages:
                    f.write(f"--- {msg['role'].upper()} ---\n{msg['content']}\n\n")
        except Exception as e:
            self.logger.warning(f"Failed to dump conversation: {e}")

    def _parse_response(self, response: str) -> Dict:
        """
        BOPRO 专用解析器。
        目标：尽可能从响应中提取出合法的 JSON 配置。
        """
        thought = "BOPRO direct generation (No explicit thought captured)"
        config = None
        valid = False

        try:
            # 1. 尝试提取 Markdown JSON 块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            
            # 2. 如果没有 markdown，尝试提取大括号包围的内容
            if not json_match:
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                config_str = json_match.group(1).strip()
                # 尝试修复常见的 JSON 错误（比如末尾逗号）
                config_str = re.sub(r',\s*\}', '}', config_str)
                config = json.loads(config_str)
                valid = True
                
                # 如果 LLM 在 JSON 之前说了话，把它作为 Thought 存起来（方便调试）
                # 找到 JSON 之前的所有文本
                pre_text_match = response[:json_match.start()].strip()
                if pre_text_match:
                    thought = pre_text_match
            else:
                self.logger.warning("[BOPRO ACQ] No JSON structure found in response.")

        except json.JSONDecodeError as e:
            self.logger.warning(f"[BOPRO ACQ] JSON Decode Error: {e}")
            thought = f"JSON Decode Error. Raw: {response}"
        except Exception as e:
            self.logger.warning(f"[BOPRO ACQ] Parsing Error: {e}")
            thought = f"Parsing Error. Raw: {response}"

        return {"thought": thought, "config": config, "valid": valid}

    async def _async_generate(self, messages: List[Dict]) -> Tuple[str, float, float]:
        t0 = time.time()
        try:
            resp = await self.client.chat.completions.create(
                model=self.chat_engine,
                messages=messages,
                temperature=1.0,
                top_p=0.9,
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

    def get_candidate_points(self, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, alpha=0.0, iter_idx=0):
        return asyncio.run(self._run_bopro_loop(obs_cfg_df, obs_acc_df, iter_idx))

    async def _run_bopro_loop(self, obs_cfg_df, obs_acc_df, iter_idx):
        # 1. 构建 Prompt (使用新的 BOPROPromptBuilder)
        messages = BOPROPromptBuilder.get_generation_messages(
            self.task_context, obs_cfg_df, obs_acc_df
        )

        # 2. 调用 LLM
        response_text, cost, duration = await self._async_generate(messages)
        self.total_llm_cost += cost

        # 3. 解析
        cand_df = None
        if response_text:
            parsed = self._parse_response(response_text)
            
            # 保存 Debug 日志
            self._dump_conversation(f"trial_{iter_idx}_bopro.txt", messages + [{"role": "assistant", "content": response_text}])

            if parsed['valid']:
                # 转换为 DataFrame 格式返回
                cand_df = pd.DataFrame([parsed['config']])
            else:
                self.logger.warning(f"[BOPRO ACQ] Parsing failed for trial {iter_idx}")
        
        return cand_df, cost, duration