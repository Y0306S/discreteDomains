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

from .utils import (
    _constraints_text_for_algo,
    _build_dataset_summary_text,
    _convert_to_dict, 
    _filter_and_validate
)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
        
class ReActLLM_ACQ:
    def __init__(self, optimizer: 'LLMOptimizer', **kwargs):
        from mindware.components.optimizers.llm_optimizer import LLMOptimizer

        self.optimizer = optimizer
        self.task_context = optimizer.task_context
        self.n_candidates = optimizer.n_candidates
        self.lower_is_better = self.task_context.get('lower_is_better', False)
        self.chat_engine = self.task_context.get('chat_engine', "gpt-4o")
        self.constraints = self.task_context.get("hyperparameter_constraints", {})
        self.algo_choices = self.task_context.get("algorithm_choices", [])
        self.algo2params = self.task_context.get("algorithm_param_names", {})
        self.task_id = self.task_context.get("task_id", {})
        
        self.warping_transformer = optimizer.warping_transformer
        self.apply_warping = optimizer.warping_transformer is not None

        def history_entry_factory():
            best_init = np.inf if self.lower_is_better else -np.inf
            worst_init = -np.inf if self.lower_is_better else np.inf
            return {
                'count': 0,
                'scores': [],
                'best_score': best_init,
                'worst_score': worst_init,
                'avg_score': 0.0,
                'thoughts': [] # 记录与该算法相关的思考
            }
        self.algo_selection_history: Dict[str, Dict] = defaultdict(history_entry_factory)
        
        self.param_tuning_history: Dict[str, List[Dict]] = defaultdict(list)

        self.reflection_interval = kwargs.get("reflection_interval", 5) # 每 5 轮进行一次元反思
        self.exploration_algos: List[str] = []
        self.exploitation_algos: List[str] = []
        self.meta_reflection_history: List[Dict] = []


        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", "")
        )

        self.output_dir = self.optimizer.output_dir
        self.conversation_dir = os.path.join(self.output_dir, "llm_conversations")
        self.debug_dir = os.path.join(self.output_dir, "debug_states") # 新增
        os.makedirs(self.conversation_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

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
        filepath = os.path.join(self.debug_dir, filename) # 修改了保存目录
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
            prompt_cost_per_1k, completion_cost_per_1k = 0.005, 0.015
            cost = (prompt_cost_per_1k * (resp.usage.prompt_tokens / 1000) + completion_cost_per_1k * (resp.usage.completion_tokens / 1000))
            return resp, cost, duration
        except openai.APIStatusError as e: print(f"[ERROR] OpenAI API Error: {e.status_code}. Response: {e.response}"); return None, 0.0, time.time() - start_time
        except openai.AuthenticationError as e: print(f"[ERROR] OpenAI Authentication Error: Invalid API Key or token. {e}"); raise e
        except openai.RateLimitError as e: print(f"[ERROR] OpenAI Rate Limit Error: {e}. Check your API plan and usage."); return None, 0.0, time.time() - start_time
        except openai.BadRequestError as e: print(f"[ERROR] OpenAI Invalid Request Error: {e}. Check model name and parameters."); return None, 0.0, time.time() - start_time
        except (openai.APIConnectionError, openai.Timeout) as e: print(f"[ERROR] OpenAI Connection/Timeout Error: {e}"); return None, 0.0, time.time() - start_time
        except Exception as e: print(f"[ERROR] An unexpected error occurred: {type(e).__name__}: {e}"); return None, 0.0, time.time() - start_time


    def _parse_thought_action_summary(self, response: str) -> Dict:
        try:
            detailed_thought = re.search(r"Detailed Thought:(.*?)Thought Summary:", response, re.DOTALL).group(1).strip()
            summary_thought = re.search(r"Thought Summary:(.*?)Action:", response, re.DOTALL).group(1).strip()           
            action_raw = re.search(r"Action:(.*)", response, re.DOTALL).group(1)          
            action_cleaned = re.sub(r'^[*\s`]*(json\s*)?', '', action_raw.strip())         
            if action_cleaned.endswith('```'):
                action_cleaned = action_cleaned[:-3].strip()
            return {"detailed_thought": detailed_thought, "summary_thought": summary_thought, "action": action_cleaned}
        except Exception:
            return {"detailed_thought": response, "summary_thought": "Parsing failed.", "action": "PARSING_ERROR"}

    def _parse_reflection_summary(self, response: str) -> Dict:
        try:
            detailed_reflection = re.search(r"Detailed Reflection:(.*?)Reflection Summary:", response, re.DOTALL).group(1).strip()
            summary_reflection = re.search(r"Reflection Summary:(.*)", response, re.DOTALL).group(1).strip()
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

    def _format_history_text(self, history: List[Dict], max_entries=5, include_reflection: bool = True):
        if not history: return "No tuning history for this algorithm yet. This is the first attempt."
        if not isinstance(history, list): return "Invalid history format for display."
        text_parts = [f"Here are the most recent {len(history[-max_entries:])} historical trials:\n"]
        for entry in history[-max_entries:]:
            text_parts.append(f"--- Start of Trial {entry.get('trail_id', 'N/A')} ---")
            text_parts.append(f"Action: {json.dumps(entry['action_config'])}")
            text_parts.append(f"Observation (Performance): {entry['observation_perf']:.6f}")
            # if include_reflection and 'reflection_summary' in entry:
            #     text_parts.append(f"Reflection Summary: {entry['reflection_summary']}")
            text_parts.append(f"--- End of Trial {entry.get('trail_id', 'N/A')} ---\n")
        return "\n".join(text_parts)

    def _format_statistical_history(self) -> str:
        if not self.algo_selection_history: return "No historical trials have been completed yet. Please make your decision based on the dataset summary and your general knowledge."
        summary_parts = ["Here is a summary of the performance of algorithms tried so far:\n"]
        tried_algos = set()
        sorted_history = sorted(self.algo_selection_history.items(), key=lambda item: item[1]['avg_score'], reverse=not self.lower_is_better)
        for algo, stats in sorted_history:
            tried_algos.add(algo)
            summary_parts.append(f"- **{algo}**: Tried {stats['count']} time(s). Best Score: {stats['best_score']:.4f}, Avg Score: {stats['avg_score']:.4f}")
        all_algos = set(self.algo_choices)
        untried_algos = list(all_algos - tried_algos)
        if untried_algos: summary_parts.append(f"- **Untried Algorithms**: {untried_algos}")
        return "\n".join(summary_parts)

    def _get_strategy_for_algo_selection(self) -> Tuple[str, str]:
        """为算法选择阶段生成策略指令（根据元反思状态动态调整）。"""
        is_exploration = random.random() < 0.5
        # 检查元反思是否已经运行并生成了推荐列表
        has_meta_reflection = bool(self.exploration_algos or self.exploitation_algos)
        if is_exploration:
            strategy_name = "EXPLORATION"
            if has_meta_reflection:
                strategy_instruction = (
                    "Your goal is to explore different algorithms. "
                    "**Choose an algorithm from the 'Algorithms Recommended for EXPLORATION' list.**"
                )
            else:
                strategy_instruction = (
                    "Your goal is to explore different algorithms. "
                    "Based on the dataset summary and 'Current Performance Statistics', "
                    "**choose an algorithm that is UNTRIED or has shown varied results and needs more investigation.**"
                )
        else:
            strategy_name = "EXPLOITATION"
            if has_meta_reflection:
                strategy_instruction = (
                    "Your goal is to exploit the currently best-performing algorithm. "
                    "**Choose an algorithm from the 'Algorithms Recommended for EXPLOITATION' list.**"
                )
            else:
                # 状态 2: 尚无元反思，依赖统计数据
                strategy_instruction = (
                    "Your goal is to exploit the currently best-performing algorithm. "
                    "Based on the 'Current Performance Statistics', "
                    "**choose the algorithm that currently shows the best Average Score or Best Score.**"
                )
                
        return strategy_name, strategy_instruction

    def _get_strategy_for_param_tuning(self) -> Tuple[str, str]:
        """为参数优化阶段生成策略指令"""
        is_exploration = random.random() < 0.5
        if is_exploration:
            strategy_name = "EXPLORATION"
            strategy_instruction = (
                "You have selected the algorithm; the goal now is to explore within its parameter space. "
            )
        else:
            strategy_name = "EXPLOITATION"
            strategy_instruction = (
                "You have selected the algorithm; the goal now is to exploit its most promising region. "
            )
        return strategy_name, strategy_instruction

    def _build_system_prompt(self) -> str:
        return """You are an advanced AutoML agent. Your primary mission is to solve the Combined Algorithm Selection and Hyperparameter optimization (CASH) problem by finding the best-performing configuration.
        Your operational workflow is a cycle of Thought -> Action -> Observation -> Reflection:
        1.  **Thought**: Based on the history and the current strategy (Exploration or Exploitation), you will first reason step-by-step about your next move. You should analyze past trials, compare results, and form a hypothesis about what to try next.
        2.  **Action**: Based on your thought process, you will output a single, specific action (e.g., choosing an algorithm or proposing a hyperparameter configuration).
        3.  **Observation**: Your action will be executed by an external tool, which will return a performance score. This is your observation.
        4.  **Reflection**: You will be shown your thought, your action, and the resulting observation. You must then reflect on the outcome to draw conclusions and learn lessons for future trials.
        You will be guided to switch between Exploration and Exploitation. Pay close attention to the strategy provided in each user prompt. Your success depends on your ability to learn from the full history of `{Thought, Action, Observation, Reflection}` to identify patterns and continuously refine your strategy."""



    def _get_algo_selection_messages(self) -> List[Dict]:
        strategy_name, strategy_instruction = self._get_strategy_for_algo_selection()
        task_type_str, metric_name, goal_direction = self.task_context.get("dataset_profile", {}).get("task_type", "unknown"), self.task_context.get("metric", "score"), 'Higher is better' if not self.lower_is_better else 'Lower is better'
        
        history_or_insights_display = ""
        if self.exploration_algos or self.exploitation_algos:
            shuffled_exploration, shuffled_exploitation = self.exploration_algos.copy(), self.exploitation_algos.copy()
            random.shuffle(shuffled_exploration); random.shuffle(shuffled_exploitation)
            strategic_insights = f"""Based on a periodic review of the entire history, here are the current strategic insights:
- **Algorithms Recommended for EXPLORATION**: {shuffled_exploration}
- **Algorithms Recommended for EXPLOITATION**: {shuffled_exploitation}"""
            history_or_insights_display = f"**3. Strategic Insights (from Meta-Reflection):**\n{strategic_insights}"
        else:
            stats_summary = self._format_statistical_history()
            history_or_insights_display = f"**3. Current Algorithm Performance Statistics:**\n{stats_summary}"
 
        user_prompt = f"""
### **MISSION BRIEFING**

**1. Mission Objective:**
Your mission is to find an optimal `configuration` (consisting of an algorithm and its hyperparameters). This `configuration` will be used to complete the following machine learning pipeline:
  a. An ML model will be instantiated using your proposed algorithm and hyperparameters.
  b. The model will be **trained** on a training subset of the **'{self.task_id}'** dataset.
  c. The trained model will then make **predictions** on an unseen test subset of the dataset.
  d. Finally, these predictions will be compared to the true labels to calculate the **{metric_name}** score.

Your ultimate goal is to find the `configuration` that results in the best possible score at the end of this entire process.
**2. Core Task:**
- **Dataset:** `{self.task_id}`
- **Task Type:** You are solving a **{task_type_str}** problem.
- **Performance Metric:** Your success will be measured by the **{metric_name}** score ({goal_direction}).

**3. Workflow Protocol:**
Remember to follow the **Thought -> Action -> Observation -> Reflection** cycle. For this turn, you are at the beginning of the cycle. You must first provide your reasoning (`Thought`) and then your decision (`Action`).

### **AVAILABLE INTELLIGENCE**

**1. Dataset Summary:**
{_build_dataset_summary_text(self.task_context)}

**2. Available Algorithms (Your Toolbox):**
{self.algo_choices}

{history_or_insights_display}


### **CURRENT DIRECTIVE**

**1. Current Stage:** Stage 1 - Algorithm Selection.
**2. Strategy for this turn:** {strategy_name}
**3. Instruction:** {strategy_instruction}
**4. Required Output Format:**
   - **Detailed Thought:** (Your step-by-step reasoning for choosing the algorithm).
   - **Thought Summary:** (A one-sentence summary of your core reasoning).
   - **Action:** (ONLY the name of the chosen algorithm).

Begin.
"""
        return [{"role": "system", "content": self._build_system_prompt()}, {"role": "user", "content": user_prompt}]

    def _get_param_tuning_messages(self, current_messages: List[Dict], selected_algo: str) -> List[Dict]:
        strategy_name, strategy_instruction = self._get_strategy_for_param_tuning()
        
        history_display = self._format_history_text(self.param_tuning_history[selected_algo], max_entries=5, include_reflection=True)
        
        user_prompt = f"""Excellent, you have chosen '{selected_algo}'. Now, let's determine its hyperparameters.
### Hyperparameter Space for '{selected_algo}'
{_constraints_text_for_algo(selected_algo, self.algo2params, self.constraints, self.algo_choices)}
### 2. Recent Tuning History for '{selected_algo}'
{history_display}
### Your Task for This Turn (Stage 2: Parameter Tuning)
- **Strategy**: {strategy_name}
- **Instruction**: {strategy_instruction}
- **Key Analysis**: **You MUST analyze the 'Recent Tuning History' above.** Pay close attention to the `Reflection Summary` from previous trials to learn what worked, what didn't, and why.
- **Required Output Format**: Provide a detailed thought, a one-sentence summary, and the action.

**Detailed Thought:**
(Explain your rationale for choosing hyperparameter values.)
**Thought Summary:**
(Summarize your thought into a single, actionable sentence.)
**Action:**
(Output the configuration in the format: `param1: value1, param2: value2, ...`)
"""
        return current_messages + [{"role": "user", "content": user_prompt}]

    def _get_reflection_messages(self, current_messages: List[Dict], context: Dict) -> List[Dict]:
        
        # 从上下文中提取核心信息
        current_perf = context['observation_perf']
        current_config = context['action_config']
        selected_algo = current_config['algorithm']
        
        # 获取当前算法的历史记录
        history_for_algo = self.param_tuning_history[selected_algo]
        
        # 准备历史记录文本
        history_text = self._format_history_text(history_for_algo, max_entries=5, include_reflection=True)

        user_prompt = f"""
### **REFLECTION TASK (Parameter Tuning)**

Your last action has been executed. Now, you must perform a deep reflection to extract knowledge.

**1. CONTEXT: YOUR PREVIOUS TURN**
(The full context of your thought process and action is included in the conversation history above.)
- **Your Chosen Action**: `{json.dumps(current_config)}`
- **Observed Performance**: `{current_perf:.6f}`

**2. AVAILABLE DATA: RECENT HISTORY FOR '{selected_algo}'**
{history_text}

**3. YOUR TASK: EXTRACT SPECIFIC TUNING INSIGHTS**
Based on all the information above, provide your analysis. **Avoid vague statements. Be specific and use numerical values** from the context and history to support your conclusions.

   **A. Comparison & Attribution (Why did this work or not work?)**
   - **Compare this trial with the recent history to determine *why* it worked or didn't work.** - Pinpoint the specific parameter changes (e.g., `'min_samples_leaf': 2 -> 4`) you believe are responsible for the performance change (e.g., score improved by `+0.005` or decreased by `-0.002`). Be specific and use numbers to support your analysis.
   
   **B. Knowledge Synthesis (What did we learn?)**
   - **Parameter Importance:** Based on this trial, update your ranking of which parameters seem most influential.
   - **Promising Values/Ranges:** For the most important parameters, what specific values or **numerical ranges** appear to be effective? **Be precise.** (e.g., `'learning_rate': 'around 0.05', 'max_depth': '> 8', 'min_samples_leaf': '[4, 10]'`)
   - **Parameter Correlations:** Have you observed any potential relationships? (e.g., "A lower `subsample` seems to require a higher `n_estimators`.")

**4. REQUIRED OUTPUT FORMAT**
Provide a detailed reflection by answering the questions above, then summarize your key takeaway.

**Detailed Reflection:**
(Your structured, quantitative analysis based on the sections A and B above.)

**Reflection Summary:**
(Provide a **concise summary of the key, quantitative lesson learned from *this* trial.** This summary must be an *analytical insight*, NOT a *plan for the next step*.)
"""
        return current_messages + [{"role": "user", "content": user_prompt}]


    def _get_meta_reflection_prompt(self) -> List[Dict]:
        history_stats_for_prompt = {algo: {
            'count': data['count'],
            'best_score': data['best_score'],
            'worst_score': data['worst_score'],
            'avg_score': data['avg_score'],
        } for algo, data in self.algo_selection_history.items()}
        history_json_str = json.dumps(history_stats_for_prompt, indent=2, cls=CustomEncoder)
        goal_direction = 'Higher is better' if not self.lower_is_better else 'Lower is better'
        history_block = f"```json\n{history_json_str}\n```"
        output_format_block = """```json
{
  "reasoning": "(Your brief summary of why you are updating the lists)",
  "exploration_algorithms": ["list", "of", "updated", "algorithm", "names"],
  "exploitation_algorithms": ["list", "of", "updated", "algorithm", "names"]
}
```"""
        user_prompt = f"""
### **META-REFLECTION TASK**
**1. Objective:**
Your task is to act as a senior AutoML strategist. Your goal is to **update** the strategic lists (`Exploration` and `Exploitation`) based on the latest performance statistics.
**2. Performance Metric:**
The performance score is **{self.task_context.get("metric", "score")}** ({goal_direction}).
**3. Current Strategic Lists:**
- **Current Exploration List**: {self.exploration_algos}
- **Current Exploitation List**: {self.exploitation_algos}
**4. Latest Performance Statistics (Your Knowledge Base):**
This JSON provides a summary of all algorithms tried so far, including how many times each was chosen (`count`) and their performance statistics.
{history_block}
**5. Your Analysis and Update Task:**
- Review the **Latest Performance Statistics**.
- Compare them with the **Current Strategic Lists**.
- Decide if any algorithms should be moved...
- Provide the **complete, updated lists** in the required JSON format.
**6. Required Output Format:**
You must provide your final categorization in a single JSON block. Do not add any text before or after the JSON block.
{output_format_block}

Begin.
"""
        return [{"role": "system", "content": self._build_system_prompt()}, {"role": "user", "content": user_prompt}]


    async def _run_meta_reflection(self, iter_idx: int):
        print(f"\n--- [ReAct ACQ] Running Meta-Reflection at trial {iter_idx} ---")
        messages = self._get_meta_reflection_prompt()
        resp, cost, duration = await self._async_generate(messages, 0.5, 1)
        if cost is not None: self.optimizer.llm_query_cost[-1] += cost
        if duration is not None: self.optimizer.llm_query_time[-1] += duration
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
        while len(evaluated_configs_for_return) < self.n_candidates and max_tries > 0:
            candidate_idx = len(evaluated_configs_for_return)
            max_tries -= 1
            print(f"\n[ReAct ACQ] Trial [{iter_idx}] - Generating Candidate [{candidate_idx + 1}/{self.n_candidates}]... (Tries left: {max_tries})")
            
            # --- 为每个候选点创建独立的、连续的对话上下文 ---
            messages = self._get_algo_selection_messages()
            
            # --- 步骤 A: 算法选择 ---
            resp, cost, duration = await self._async_generate(messages, 0.7, 1)
            if resp is None: continue 
            total_cost += cost; total_llm_time += duration
            
            response_text_algo = resp.choices[0].message.content
            parsed_algo = self._parse_thought_action_summary(response_text_algo)
            detailed_algo_thought, summary_algo_thought, selected_algo_raw = parsed_algo['detailed_thought'], parsed_algo['summary_thought'], parsed_algo['action']
            selected_algo = selected_algo_raw.strip().strip(' *`_')
            if selected_algo == "PARSING_ERROR" or selected_algo not in self.algo_choices:
                print(f"[ReAct ACQ] Warning: Parsed algorithm '{selected_algo}' is invalid. Choosing randomly.")
                selected_algo = random.choice(self.algo_choices)
            
            assistant_response_A = f"Thought Summary: {summary_algo_thought}\n Action: {selected_algo_raw}"
            messages.append({"role": "assistant", "content": assistant_response_A})

            # --- 步骤 B: 参数选择 ---
            messages = self._get_param_tuning_messages(messages, selected_algo)
            resp, cost, duration = await self._async_generate(messages, 0.8, 1)
            if resp is None: continue
            total_cost += cost; total_llm_time += duration

            response_text_param = resp.choices[0].message.content
            parsed_param = self._parse_thought_action_summary(response_text_param)
            detailed_param_thought_param, summary_param_thought_param, param_action_str = parsed_param['detailed_thought'], parsed_param['summary_thought'], parsed_param['action']
            
            if param_action_str == "PARSING_ERROR": config_dict = self.optimizer.config_space.sample_configuration().get_dictionary()
            else: config_dict = _convert_to_dict(param_action_str); config_dict['algorithm'] = selected_algo
            
            assistant_response_B = f"Thought Summary: {summary_param_thought_param}\nAction: {param_action_str}"
            messages.append({"role": "assistant", "content": assistant_response_B})
        
            # --- 步骤 C: 评估 ---
            all_observed_points = [cfg.get_dictionary() for cfg in self.optimizer.configs]
            validation_result = _filter_and_validate(self, all_observed_points, [config_dict])
            if validation_result.empty:
                print(f"[ReAct ACQ] Proposed config was invalid or duplicate. Skipping...")
                continue
            
            perf, eval_time = _evaluate_dict(self.optimizer, config_dict)

            # --- 步骤 D: 反思 ---
            reflection_context = { "trail_id": f"{iter_idx}-{candidate_idx}", "action_config": config_dict, "observation_perf": perf }
            messages = self._get_reflection_messages(messages, reflection_context)
            resp, cost, duration = await self._async_generate(messages, 0.5, 1)
            if resp is None: response_text_reflection = "Reflection failed due to API error."
            else:
                total_cost += cost; total_llm_time += duration
                response_text_reflection = resp.choices[0].message.content
            
            parsed_reflection = self._parse_reflection_summary(response_text_reflection)
            summary_reflection = parsed_reflection['summary_reflection']
            messages.append({"role": "assistant", "content": response_text_reflection})

            # --- 步骤 E: 记录与立即更新 (实现轮内学习的关键) ---
            self._dump_conversation(f"trial_{iter_idx}_candidate_{candidate_idx}", messages)

            full_history_entry = { 
                "trail_id": f"{iter_idx}-{candidate_idx}", "detailed_thought": detailed_param_thought_param, 
                "thought_summary": summary_param_thought_param, "action_config": config_dict, 
                "observation_perf": perf, "reflection_summary": summary_reflection
            }
            self.param_tuning_history[selected_algo].append(full_history_entry)
            
            cfg_obj = self.optimizer._dict_to_cfg(config_dict)
            self.optimizer.configs.append(cfg_obj)
            self.optimizer.perfs.append(perf)

            algo_stats = self.algo_selection_history[selected_algo]
            algo_stats['count'] += 1
            algo_stats['scores'].append(perf)
            algo_stats['avg_score'] = np.mean(algo_stats['scores'])
            if self.lower_is_better:
                algo_stats['best_score'] = min(algo_stats['best_score'], perf)
                algo_stats['worst_score'] = max(algo_stats['worst_score'], perf)
            else:
                algo_stats['best_score'] = max(algo_stats['best_score'], perf)
                algo_stats['worst_score'] = min(algo_stats['worst_score'], perf)
            algo_stats['thoughts'].append({
                'trial_id': f"{iter_idx}-{candidate_idx}",
                'detailed_thought': detailed_algo_thought, 'summary_thought': summary_algo_thought
            })
            
            evaluated_configs_for_return.append(config_dict)
            eval_times_for_return.append(eval_time)

        # --- 循环结束后，统一记录本轮 Trial 的产出 ---
        trial_debug_data = {
            "trial_index": iter_idx, "total_llm_cost_this_trial": total_cost, "total_llm_time_this_trial": total_llm_time,
            "generated_configs": evaluated_configs_for_return, 
            "performance_scores": [cfg['observation_perf'] for cfg in self.param_tuning_history[selected_algo] if cfg['trail_id'].startswith(f"{iter_idx}-")],
            "evaluation_times": eval_times_for_return,
            "final_algo_selection_history_stats": self.algo_selection_history,
            "final_param_tuning_history": self.param_tuning_history,
            "meta_reflection_history": self.meta_reflection_history
        }
        self._dump_trial_state(iter_idx, trial_debug_data)

        # 检查是否触发元反思
        if (iter_idx + 1) % self.reflection_interval == 0 and self.algo_selection_history:
            await self._run_meta_reflection(iter_idx)

        return pd.DataFrame(evaluated_configs_for_return), total_cost, total_llm_time, eval_times_for_return