import os
import time
import json
import re
import asyncio
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Any, Dict, List, Tuple
from openai import AsyncOpenAI

from mindware.utils.logging_utils import get_logger
from mindware.modules.configs import LLMCONFIGS, AVAILABLE_LLMs, LLM_COST_DICT
from .prompt import LLAMBOPromptBuilder

class LLAMBO_ACQ:
    """
    LLAMBO Acquisition Function.
    Implements the complete LLAMBO loop components:
    1. Candidate Sampling (Inverse: Score -> Configs)
    2. Discriminative Surrogate (Forward: Configs -> Scores, with Uncertainty via Shuffle)
    3. Acquisition Selection (Expected Improvement)
    
    Adheres to the "Low-Resource" strategy:
    - M = 8 candidates per step.
    - K = 3 shuffle rounds for surrogate scoring.
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
        
        # Hyperparameters (Reference: LLAMBO Appendix D.4)
        # alpha = -0.1 (Conservative exploration)
        self.alpha = kwargs.get("alpha", -0.1)
        # M candidates per sampling step (Low resource: 8, Paper: 20)
        self.n_candidates = kwargs.get("n_candidates", 8)
        # K Monte Carlo evaluations for uncertainty (Low resource: 3, Paper: 10)
        self.n_surrogate_rounds = kwargs.get("n_surrogate_rounds", 3)
        self.temperature = kwargs.get("temperature", 0.7)
        
        self.total_llm_cost = 0.0
        self.output_dir = optimizer.output_dir
        self.conversation_dir = os.path.join(self.output_dir, "llambo_conversations")
        os.makedirs(self.conversation_dir, exist_ok=True)

    def get_total_llm_cost(self) -> float:
        return self.total_llm_cost

    def _dump_conversation(self, filename: str, messages: List[Dict], append: bool = False):
        try:
            mode = 'a' if append else 'w'
            with open(os.path.join(self.conversation_dir, filename), mode, encoding='utf-8') as f:
                if append:
                    f.write("\n\n" + "="*40 + "\n\n")
                for msg in messages:
                    f.write(f"--- {msg['role'].upper()} ---\n{msg['content']}\n\n")
        except Exception as e:
            self.logger.warning(f"Failed to dump conversation: {e}")

    # =========================================================
    # Part 1: Candidate Sampling (Inverse Generation)
    # =========================================================

    def _calculate_target_score(self, scores: List[float], lower_is_better: bool) -> float:
        """
        Calculate target score s' based on historical best/worst and alpha.
        Paper Formula (Minimization): s' = s_min - alpha * (s_max - s_min)
        Adapted for Maximization: s' = s_max + alpha * (s_max - s_min)
        (Assuming alpha is negative for conservative, positive for aggressive)
        """
        if not scores:
            return 0.5 # Default fallback
            
        s_min, s_max = min(scores), max(scores)
        r = s_max - s_min
        if r == 0: r = 0.1 * abs(s_max) if s_max != 0 else 0.1 # Avoid zero range

        if lower_is_better:
            # Minimization: Best is s_min. 
            # If alpha=-0.1: s' = s_min - (-0.1)*r = s_min + 0.1*r (Worse than best)
            target = s_min - self.alpha * r
        else:
            # Maximization: Best is s_max.
            # If alpha=-0.1: s' = s_max + (-0.1)*r = s_max - 0.1*r (Worse than best)
            target = s_max + self.alpha * r
            
        return target

    def _parse_sampler_response(self, response: str) -> List[Dict]:
        """Extract configs list from Sampler response."""
        configs = []
        try:
            # Try finding JSON block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                # Fix common JSON trailing comma errors
                json_str = re.sub(r',\s*\}', '}', json_str)
                json_str = re.sub(r',\s*\]', ']', json_str)
                data = json.loads(json_str)
                
                if "configs" in data and isinstance(data["configs"], list):
                    configs = data["configs"]
                elif "config" in data: # Fallback single
                    configs = [data["config"]]
        except Exception as e:
            self.logger.warning(f"[LLAMBO Sampler] Parsing Error: {e}")
        
        return configs

    # =========================================================
    # Part 2: Discriminative Surrogate (Forward Prediction)
    # =========================================================

    def _parse_surrogate_response(self, response: str, n_expected: int) -> List[float]:
        """Extract predictions list from Surrogate response."""
        scores = []
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                json_str = re.sub(r',\s*\}', '}', json_str)
                json_str = re.sub(r',\s*\]', ']', json_str)
                data = json.loads(json_str)
                
                if "predictions" in data and isinstance(data["predictions"], list):
                    # Ensure we get floats
                    scores = [float(s) if s is not None else float('nan') for s in data["predictions"]]
        except Exception as e:
            self.logger.warning(f"[LLAMBO Surrogate] Parsing Error: {e}")
        
        # Pad with NaNs if count doesn't match
        if len(scores) < n_expected:
            scores.extend([float('nan')] * (n_expected - len(scores)))
        return scores[:n_expected]

    # =========================================================
    # Core LLM Interface
    # =========================================================

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

    # =========================================================
    # Main Workflow
    # =========================================================

    def get_candidate(self, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, iter_idx: int) -> Dict:
        """
        Executes the full LLAMBO loop:
        1. Sample Candidates (Batch)
        2. Evaluate Candidates (Batch + Shuffle)
        3. Select Best via EI
        
        Returns: A single configuration dictionary.
        """
        return asyncio.run(self._run_llambo_loop(obs_cfg_df, obs_acc_df, iter_idx))

    async def _run_llambo_loop(self, obs_cfg_df, obs_acc_df, iter_idx):
        # 0. Prep Data
        lower_is_better = self.task_context.get("lower_is_better", False)
        scores_history = obs_acc_df['score'].tolist()
        
        # -----------------------------------------------------
        # Step 1: Candidate Sampling (Inverse)
        # -----------------------------------------------------
        target_score = self._calculate_target_score(scores_history, lower_is_better)
        
        sampler_msgs = LLAMBOPromptBuilder.get_sampler_messages(
            self.task_context, obs_cfg_df, obs_acc_df, 
            target_score=target_score, n_generate=self.n_candidates
        )
        
        samp_text, samp_cost, _ = await self._async_generate(sampler_msgs)
        self.total_llm_cost += samp_cost
        
        # Log Sampler
        if samp_text:
            self._dump_conversation(f"iter_{iter_idx}_1_sampler.txt", sampler_msgs + [{"role": "assistant", "content": samp_text}])
        
        candidates_configs = self._parse_sampler_response(samp_text or "")
        
        if not candidates_configs:
            self.logger.warning(f"[LLAMBO] Sampler failed to generate valid configs at iter {iter_idx}")
            return None # Optimizer will fallback to random

        # -----------------------------------------------------
        # Step 2: Surrogate Evaluation (Forward, Shuffle x K)
        # -----------------------------------------------------
        # We need to evaluate ALL candidates K times
        # Shape of predictions: [K_rounds, M_candidates]
        prediction_matrix = [] 

        surrogate_tasks = []
        all_round_messages = []

        for k in range(self.n_surrogate_rounds):
            # Shuffle history for each round
            msgs = LLAMBOPromptBuilder.get_surrogate_messages(
                self.task_context, obs_cfg_df, obs_acc_df, 
                candidate_configs=candidates_configs, shuffle=True
            )
            all_round_messages.append(msgs)
            surrogate_tasks.append(self._async_generate(msgs))
        
        # Execute K rounds in parallel
        results = await asyncio.gather(*surrogate_tasks)
        
        for k, (text, cost, _) in enumerate(results):
            self.total_llm_cost += cost
            current_msgs = all_round_messages[k]
            if text:
                full_conversation = current_msgs + [{"role": "assistant", "content": text}]
                self._dump_conversation(f"iter_{iter_idx}_2_surrogate_round_{k}.txt", full_conversation)
                
                preds = self._parse_surrogate_response(text, len(candidates_configs))
                prediction_matrix.append(preds)
            else:
                self.logger.warning(f"[LLAMBO] Surrogate round {k} failed (empty response).")
                prediction_matrix.append([float('nan')] * len(candidates_configs))

        # -----------------------------------------------------
        # Step 3: Acquisition (EI Calculation)
        # -----------------------------------------------------
        # Convert to numpy for easier calc: [K, M] -> [M, K]
        preds_arr = np.array(prediction_matrix).T # Rows = Candidates, Cols = Rounds
        
        # Calculate Mean and Std for each candidate
        means = np.nanmean(preds_arr, axis=1)
        stds = np.nanstd(preds_arr, axis=1)
        # Handle case where std is 0 or NaN
        stds = np.nan_to_num(stds, nan=1e-6) 
        stds[stds < 1e-9] = 1e-9

        # Best observed so far
        if lower_is_better:
            best_f = min(scores_history)
            # EI for minimization: (best_f - mu) ...
            # Normalize directions: We transform minimization to maximization of improvement
            # Imp = best_f - means
            imp = best_f - means
        else:
            best_f = max(scores_history)
            # EI for maximization: (mu - best_f) ...
            imp = means - best_f

        # EI Formula
        with np.errstate(divide='ignore'):
            z = imp / stds
            ei = imp * norm.cdf(z) + stds * norm.pdf(z)
        
        # Handle NaNs (failed predictions)
        ei = np.nan_to_num(ei, nan=-1.0)
        
        # Select Best
        best_idx = np.argmax(ei)
        selected_config = candidates_configs[best_idx]
        
        self.logger.info(f"[LLAMBO] Iter {iter_idx}: Selected candidate {best_idx} with EI={ei[best_idx]:.4f} (Mean={means[best_idx]:.4f}, Std={stds[best_idx]:.4f})")
        
        return selected_config