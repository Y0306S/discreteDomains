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
# 引入我们刚才写好的 OPROPromptBuilder
from .prompt import OPROPromptBuilder

class OPROLLM_ACQ:
    """
    OPRO Acquisition Function.
    Based on Google DeepMind's "Large Language Models as Optimizers".
    
    Mechanism:
    1. Construct Meta-Prompt with Optimization Trajectory (sorted history).
    2. Ask LLM to generate 'n' new configurations.
    3. Parse the list of configurations.
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
        
        # OPRO Paper suggests temperature 1.0 for better exploration 
        # BOPRO default was 0.7. We stick to OPRO paper defaults here.
        self.temperature = kwargs.get("temperature", 1.0)
        self.n_candidates = kwargs.get("n_candidates", 1) # Paper uses 8 for stability [cite: 158, 291]
        
        self.total_llm_cost = 0.0

        self.output_dir = optimizer.output_dir
        self.conversation_dir = os.path.join(self.output_dir, "opro_conversations")
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
        OPRO 专用解析器。
        目标：提取 JSON 中的 'configs' 列表。
        
        Expected Format from Prompt:
        {
            "configs": [
                {"algorithm": "...", ...},
                {"algorithm": "...", ...}
            ]
        }
        """
        valid = False
        configs = []
        thought = "OPRO Direct Generation"

        try:
            # 1. 尝试提取 Markdown JSON 块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            
            # 2. 如果没有 markdown，尝试提取大括号包围的内容
            if not json_match:
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                # 常见 JSON 错误修复
                json_str = re.sub(r',\s*\}', '}', json_str)
                json_str = re.sub(r',\s*\]', ']', json_str)
                
                data = json.loads(json_str)
                
                # 提取配置列表
                if "configs" in data and isinstance(data["configs"], list):
                    configs = data["configs"]
                    valid = True
                # 兼容性处理：如果 LLM 只返回了一个单对象 (没放在列表里)
                elif "algorithm" in data:
                    configs = [data]
                    valid = True
            else:
                self.logger.warning("[OPRO ACQ] No JSON structure found in response.")

        except json.JSONDecodeError as e:
            self.logger.warning(f"[OPRO ACQ] JSON Decode Error: {e}. Raw: {response[:100]}...")
        except Exception as e:
            self.logger.warning(f"[OPRO ACQ] Parsing Error: {e}")

        return {"configs": configs, "valid": valid, "thought": thought}

    async def _async_generate(self, messages: List[Dict]) -> Tuple[str, float, float]:
        t0 = time.time()
        try:
            resp = await self.client.chat.completions.create(
                model=self.chat_engine,
                messages=messages,
                temperature=self.temperature, 
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

    def get_candidate_points(self, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, iter_idx=0, n_generate=None):
        """
        外部调用的主接口。
        """
        # 允许 Optimizer 覆盖生成的数量
        n = n_generate if n_generate is not None else self.n_candidates
        return asyncio.run(self._run_opro_loop(obs_cfg_df, obs_acc_df, iter_idx, n))

    async def _run_opro_loop(self, obs_cfg_df, obs_acc_df, iter_idx, n_generate):
        # 1. 构建 Prompt (使用 OPROPromptBuilder)
        # 注意：不再需要 BOPRO 的 latent_proposal 等参数
        messages = OPROPromptBuilder.get_generation_messages(
            self.task_context, 
            context_cfg_df=obs_cfg_df, 
            context_acc_df=obs_acc_df,
            n_generate=n_generate
        )

        # 2. 调用 LLM
        response_text, cost, duration = await self._async_generate(messages)
        self.total_llm_cost += cost

        # 3. 解析
        cand_df = None
        
        if response_text:
            parsed = self._parse_response(response_text)
            
            # 保存日志
            log_messages = messages + [{"role": "assistant", "content": response_text}]
            self._dump_conversation(f"trial_{iter_idx}_opro.txt", log_messages)

            if parsed['valid'] and parsed['configs']:
                # 将配置列表转换为 DataFrame
                # 这样 Optimizer 可以一次性获得多个候选点
                cand_df = pd.DataFrame(parsed['configs'])
            else:
                self.logger.warning(f"[OPRO ACQ] Parsing failed or no configs for trial {iter_idx}")
        
        return cand_df, cost, duration