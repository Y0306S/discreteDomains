import json
import random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


from .utils import (
    CustomEncoder,
    _constraints_text_for_algo,
    _build_algorithm_analysis_text,
    _build_dataset_summary_text,
    _build_dataset_summary_text_onehot,
    _constraints_text_for_selected_algo,
    _format_number_scientific,
)


from mindware.components.optimizers.mcts import LLMEXPLORATION, LLMEXPLOITATION, WARMUP, Node
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase


class PromptBuilder:

    @classmethod
    def _build_reflector_system_prompt(cls) -> str:
        """为反思阶段构建独立的System Prompt"""
        return """You are an expert AutoML strategist. Your current role is **Performance Analyst** in a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** task.
A trial has just been completed. Your goal is to analyze the result to extract valuable, quantitative insights that will improve future decisions in this CASH campaign.
"""




class OPROReactPromptBuilder(PromptBuilder):
    """OPRO Prompt Builder with REACT-specific methods."""

    @classmethod
    def _build_algo_selector_system_prompt(cls) -> str:
        """为算法选择阶段构建独立的System Prompt"""
        return """You are an expert AutoML strategist.Your current role is **Algorithm Selector** in a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** task.
Your goal is to choose the most promising algorithm to evaluate next, contributing to the overall mission of finding the best model for a given dataset. Analyze all provided data to make a strategic choice.
"""

    @classmethod
    def _build_param_tuner_system_prompt(cls) -> str:

        return """You are an expert AutoML strategist. Your current role is **Hyperparameter Tuner** in a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** task.
An algorithm has been selected. Your goal is to find the best hyperparameter configuration for it. Your proposal should be informed by historical data and the overall mission of achieving peak performance.
"""

    @classmethod
    def _build_meta_reflector_system_prompt(cls) -> str:
        """为算法选择反思阶段构建独立的System Prompt"""
        return """You are an expert AutoML strategist. Your current role is **Chief Strategist** in a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** campaign.
Your task is to conduct a periodic review of **algorithm performance** across the campaign and update the high-level **strategic algorithm lists** (Exploration and Exploitation) to guide the next phase of the search.
"""

    # 算法选择层面，如果还没有分为Exploration/Exploitation，则展示统计数据
    @classmethod
    def _format_statistical_history(cls, REACTACQ) -> str:
        if not REACTACQ.algo_selection_history:
             return "No historical trials have been completed yet. Please make your decision based on the dataset summary and your general knowledge."
        summary_parts = ["Here is a summary of the performance of algorithms tried so far:\n"]
        tried_algos = set()
        sorted_history = sorted(REACTACQ.algo_selection_history.items(), key=lambda item: item[1]['avg_score'], reverse=not REACTACQ.lower_is_better)
        for algo, stats in sorted_history:
            tried_algos.add(algo)
            summary_parts.append(f"- **{algo}**: Tried {stats['count']} time(s). Best Score: {stats['best_score']:.4f}, Avg Score: {stats['avg_score']:.4f}")
        all_algos = set(REACTACQ.algo_choices)
        untried_algos = list(all_algos - tried_algos)
        if untried_algos: summary_parts.append(f"- **Untried Algorithms**: {untried_algos}")
        return "\n".join(summary_parts)

    # 算法选择阶段的智能策略生成
    @classmethod
    def _get_strategy_for_algo_selection(cls, REACTACQ) -> Tuple[str, str]:
        """为算法选择阶段生成策略指令（根据元反思状态动态调整）。"""
        is_exploration = random.random() < REACTACQ.optimizer.exploration_exploitation_balancer.get_algo_exploration_prob()
        # 检查元反思是否已经运行并生成了推荐列表
        has_meta_reflection = bool(REACTACQ.exploration_algos or REACTACQ.exploitation_algos)
        if is_exploration:
            strategy_name = LLMEXPLORATION
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
            strategy_name = LLMEXPLOITATION
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

    # 算法选择阶段的Prompt生成
    @classmethod
    def _get_algo_selection_messages(cls, REACTACQ) -> List[Dict]:
        """构建算法选择的独立 Prompt（重构以提高连贯性）"""
        strategy_name, strategy_instruction = cls._get_strategy_for_algo_selection(REACTACQ)
        
        # --- 重新获取 task_type_str 以便在指令中使用 ---
        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        history_or_insights_display = ""
        if REACTACQ.exploration_algos or REACTACQ.exploitation_algos:
            shuffled_exploration, shuffled_exploitation = REACTACQ.exploration_algos.copy(), REACTACQ.exploitation_algos.copy()
            random.shuffle(shuffled_exploration); random.shuffle(shuffled_exploitation)
            strategic_insights = f"""Based on a periodic review of the entire history, here are the current strategic insights:
- **Algorithms Recommended for EXPLORATION**: {shuffled_exploration}
- **Algorithms Recommended for EXPLOITATION**: {shuffled_exploitation}"""
            history_or_insights_display = f"**Performance History (Strategic Insights):**\n{strategic_insights}"
        else:
            stats_summary = cls._format_statistical_history(REACTACQ)
            history_or_insights_display = f"**Performance History (Current Statistics):**\n{stats_summary}"
 
        # --- 重构 User Prompt ---
        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Select Algorithm ({strategy_name})**

**1. Directive:**
Your immediate objective is to select the most promising algorithm for a **{task_type_str}** task.
You must follow the **{strategy_name}** strategy: {strategy_instruction}

**2. Available Intelligence:**
- **Dataset & Metric Summary:**
{_build_dataset_summary_text(REACTACQ.task_context)}
- **Available Algorithms (Your Toolbox):**
{REACTACQ.algo_choices}
- {history_or_insights_display}

**3. Required Output Format:**
You must provide your reasoning and your final choice in this format:

**Thought:**
(Your reasoning for choosing the algorithm based on the directive and all available intelligence.)
**Thought Summary:**
(A short summary of your core reasoning.)
**Action:**
(ONLY the name of the chosen algorithm.)

"""
        return strategy_name, [{"role": "system", "content": cls._build_algo_selector_system_prompt()}, {"role": "user", "content": user_prompt}]

    # 算法HPO的智能策略生成，直接指导探索空间
    @classmethod
    def _get_strategy_for_param_tuning(cls, REACTACQ, selected_algo) -> Tuple[str, str]:
        """为参数优化阶段生成策略指令"""
        is_exploration = random.random() < REACTACQ.optimizer.exploration_exploitation_balancer.get_param_exploration_prob(selected_algo)
        if is_exploration:
            strategy_name = LLMEXPLORATION
            strategy_instruction = (
                "You have selected the algorithm; the goal now is to explore within its parameter space. The exploration strategy should focus on trying diverse and less-tested configurations to uncover potentially high-performing areas."
            )
        else:
            strategy_name = LLMEXPLOITATION
            strategy_instruction = (
                "You have selected the algorithm; the goal now is to exploit its most promising region. The exploitation strategy should focus on refining and optimizing configurations around the best-known settings to maximize performance."
            )
        return strategy_name, strategy_instruction


    # 算法HPO的prompt中的examples，初始版本，展示最好的，最差的，最近的3个
    @classmethod
    def _build_bwr_param_examples(cls, REACTACQ, selected_algo: str, history: List[Dict], max_k: int = 3, include_reflection: bool = True) -> str:
        """
        [MODIFIED] 
        构建参数优化阶段的BWR历史样本，包含智能切换逻辑。
        """
        if not history: 
            return "No tuning history for this algorithm yet. This is the first attempt."
        if not isinstance(history, list): 
            return "Invalid history format for display."

        # --- 1. 定义智能切换的阈值 ---
        # 我们定义“历史充足”为至少需要6条记录，这样B/W/R才开始有意义
        MIN_HISTORY_FOR_BWR = 6

        df = pd.DataFrame(history).drop_duplicates(subset='trail_id', keep='last')
        ascending_order = REACTACQ.lower_is_better 
        sorted_df_perf = df.sort_values(by='observation_perf', ascending=ascending_order)

        output_parts = []

        # --- 2. 内部辅助函数，用于格式化单个条目 ---
        def format_entry(row, show_reflection: bool) -> str:
            entry = [
                f"  --- Trial {row.get('trail_id', 'N/A')} ---",
                f"  Configuration: {json.dumps(row['action_config'])}",
                f"  Observation (Performance): {row['observation_perf']:.6f}"
            ]
            if show_reflection and 'reflection_summary' in row and not pd.isna(row['reflection_summary']):
                entry.append(f"  Reflection Summary: {row['reflection_summary']}")
            return "\n".join(entry)

        if len(df) < MIN_HISTORY_FOR_BWR:
            output_parts.append(
                f"Here is the **complete (but limited) tuning history** for '{selected_algo}', sorted by performance:"
            )
            for _, row in sorted_df_perf.iterrows():
                output_parts.append(format_entry(row, include_reflection))
        else:
            output_parts.append(
                f"Here is a **strategic summary of the {len(df)} trials** for '{selected_algo}':"
            )

            top_df = sorted_df_perf.head(max_k)
            least_df = sorted_df_perf.tail(max_k).sort_values(by='observation_perf', ascending=not ascending_order)
            recent_df = df.tail(max_k).iloc[::-1]

            displayed_trail_ids = set()

            def add_block(title: str, dataframe: pd.DataFrame):
                nonlocal output_parts, displayed_trail_ids
                entries_to_show = []
                for _, row in dataframe.iterrows():
                    trail_id = row.get('trail_id', 'N/A')
                    if trail_id not in displayed_trail_ids:
                        entries_to_show.append(format_entry(row, include_reflection))
                        displayed_trail_ids.add(trail_id)

                if entries_to_show:
                    output_parts.append(f"\n**{title}:**")
                    output_parts.extend(entries_to_show)

            add_block(f"Top-Performing {len(top_df)} Configurations (by Performance)", top_df)
            add_block(f"Least-Performing {len(least_df)} Configurations (by Performance)", least_df)
            add_block(f"Most Recent {len(recent_df)} Configurations (Chronological)", recent_df)

        return "\n".join(output_parts)

    # 算法HPO的Prompt生成
    @classmethod
    def _get_param_tuning_messages(cls, REACTACQ, selected_algo: str) -> List[Dict]:
        """构建参数调优的独立 Prompt"""
        strategy_name, strategy_instruction = cls._get_strategy_for_param_tuning(REACTACQ, selected_algo)
        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'

        add_shap = REACTACQ.optimizer.add_shap
        historical_tuning_experience = ""
        if add_shap:
            historical_tuning_experience = f"""
- **Historical Tuning Experience that can be Leveraged:**
    {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}
            """

        history_display = cls._build_bwr_param_examples(
            REACTACQ,
            selected_algo, # 传入算法名称
            REACTACQ.param_tuning_history[selected_algo], 
            max_k=3,
            include_reflection=True
        )

        dataset_summary_text = _build_dataset_summary_text(REACTACQ.task_context)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
Your strategic goal is **{strategy_name}**: {strategy_instruction}
Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}'.

**2. Performance Metric:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

**3. Available Intelligence:**
- **Dataset & Metric Summary:**
    {dataset_summary_text}{historical_tuning_experience}
- **Hyperparameter Space for '{selected_algo}':**
    {_constraints_text_for_algo(selected_algo, REACTACQ.algo2params, REACTACQ.constraints, REACTACQ.algo_choices)}
- **Strategic Trial Summary for '{selected_algo}':**
    {history_display}

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Strategic Trial Summary' (Top/Least/Recent) and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Thought Summary:**
(A short summary of your core reasoning.)
**Action:**
(Output the configuration in the format: `param1: value1, param2: value2, ...`)
"""

        return strategy_name, [{"role": "system", "content": cls._build_param_tuner_system_prompt()}, {"role": "user", "content": user_prompt}]


# - **Historical Tuning Experience that can be Leveraged:**
#     {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}

    @classmethod
    def _get_reflection_messages(cls, REACTACQ, reflection_context: Dict) -> List[Dict]:
        """构建反思的独立 Prompt"""

        selected_algo = reflection_context['action_config']['algorithm']

        history_for_algo = REACTACQ.param_tuning_history[selected_algo]

        history_text = cls._build_bwr_param_examples(
            REACTACQ,
            selected_algo, # 传入算法名称
            REACTACQ.param_tuning_history[selected_algo], 
            max_k=3,
            include_reflection=False # 在反思阶段，不看过去的反思
        )

        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the dataset '{REACTACQ.task_id}'.

### **CURRENT TASK: Reflect on Trial Performance**

**1. Context: The Trial to Analyze**
- **Original Rationale (Thought Summary)**: {reflection_context['param_thought_summary']}
- **Your Chosen Config**: `{json.dumps(reflection_context['action_config'])}`
- **Observed Performance**: `{reflection_context['observation_perf']:.6f}`

**2. Available Data: Strategic Trial Summary for '{selected_algo}'**
{history_text}

### 3. YOUR TASK: ANALYTICAL REFLECTION
Your goal is to synthesize the key lessons learned from this trial. Your analysis must be quantitative and structured.

**1. Triage & Compare:**
First, where does this trial's performance ({reflection_context['observation_perf']:.6f}) rank? Is it closer to the Top-Performing, Least-Performing, or somewhere in between?

**2. Attribute & Synthesize:**
Based on its ranking, pinpoint the parameter changes relative to the historical examples that likely caused this outcome. What does this trial teach you about parameter importance, promising value ranges, or interactions? 

**4. REQUIRED OUTPUT FORMAT**

**Reflection:**
(Your structured analysis addressing the two points above.)

**Reflection Summary:**
(A concise, analytical summary of the key quantitative lesson learned from this trial. Do not suggest next steps.)
"""
        return [{"role": "system", "content": cls._build_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]


    @classmethod
    def _get_meta_reflection_prompt(cls, REACTACQ) -> List[Dict]:
        # --- PROMPT DATA PREPARATION ---
        history_stats_for_prompt = {
            algo: {
                'count': data['count'],
                'best_score': data['best_score'],
                'worst_score': data['worst_score'],
                'avg_score': data['avg_score'],
            } for algo, data in REACTACQ.algo_selection_history.items()
        }
        history_json_str = json.dumps(history_stats_for_prompt, indent=2, cls=CustomEncoder)
        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'
        history_block = f"```json\n{history_json_str}\n```"
        output_format_block = """```json
    {
    "reasoning": "(Your brief summary of why you are updating the algorithm lists based on the guiding principles and latest data.)",
    "exploration_algorithms": ["list", "of", "updated", "algorithm", "names"],
    "exploitation_algorithms": ["list", "of", "updated", "algorithm", "names"]
    }
    ```"""

        user_prompt = f"""
    ### **OVERALL MISSION: Automated Machine Learning (AutoML)**
    Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible **algorithm** and its optimal hyperparameters for the current dataset.

    ### **CURRENT TASK: High-Level Strategic Review of Algorithm Performance**

    **1. Objective:**
    As the Chief Strategist, your goal is to intelligently update the strategic **algorithm lists** (`Exploration` and `Exploitation`) using the latest **algorithm performance data**.

    **2. Guiding Principles for Algorithm Categorization:**
    Use these principles to strategically categorize **each algorithm**. Your goal is not just to find the single best, but to identify groups of promising candidates.
    - **Exploitation Candidates**: Identify **a small, elite group of algorithms** to focus on. A good candidate demonstrates a strong combination of the following:
        - **Peak Performance**: Has one of the top `best_score` values, showing high potential.
        - **High Consistency**: The `avg_score` is close to its `best_score`, indicating reliability and less risk.
        - **Sufficient Evidence**: The `count` of trials is high enough to trust the performance metrics.
    - **Exploration Candidates**:
        - **Algorithms** that are **untried** (present in the toolbox but not in the performance statistics).
        - **Algorithms** with **high performance variance** (a large gap between `best_score` and `worst_score`).
        - **Algorithms** with few trials (`count` is low).

    **3. Performance Metric:**
    The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

    **4. Available Intelligence:**
    - **Complete Algorithm Toolbox (All available options):**
    {REACTACQ.algo_choices}
    - **Current Strategic Algorithm Lists (Before Update):**
        - **Current Exploration List**: {REACTACQ.exploration_algos}
        - **Current Exploitation List**: {REACTACQ.exploitation_algos}
    - **Latest Algorithm Performance Statistics (Your Knowledge Base):**
    {history_block}

    **5. Your Analysis and Update Task:**
    - Review the **Latest Algorithm Performance Statistics** and the **Complete Algorithm Toolbox** through the lens of the **Guiding Principles**.
    - Compare your analysis with the **Current Strategic Algorithm Lists**.
    - Decide which **algorithms** belong in the updated lists. An algorithm can be in both lists if it's a top performer that still warrants further exploration.
    - Ensure every algorithm from the 'Complete Algorithm Toolbox' is placed in at least one of the updated lists.
    - Provide the **complete, updated algorithm lists** in the required JSON format.

    **6. Required Output Format:**
    You must provide your final categorization of algorithms in a single JSON block. Do not add any text before or after it.
    {output_format_block}
    """
        return [{"role": "system", "content": cls._build_meta_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]


class MCTSReactPromptBuilder(PromptBuilder):
    """Prompt Builder with MCTS REACT-specific methods."""

    @classmethod
    def _format_config_for_llm(cls, REACTACQ, config_dict: Dict) -> str:
        """
        将配置字典格式化为JSON字符串。
        对于Log分布的参数，将其值转换为科学计数法字符串 (e.g. "1e-5")。
        """
        if not config_dict:
            return "{}"
            
        cs = REACTACQ.optimizer.config_space
        formatted_dict = {}
        
        for k, v in config_dict.items():
            # 默认直接用原值
            formatted_dict[k] = v
            
            # 尝试获取对应的 HP 对象，判断是否是 log
            try:
                hp_name = k
                # 处理算法前缀问题 (CASH 场景通常是 algorithm:param)
                if ':' not in k and 'algorithm' in config_dict:
                    potential_name = f"{config_dict['algorithm']}:{k}"
                    # 检查拼凑的名字是否在 config space 里
                    if potential_name in cs.get_hyperparameter_names():
                        hp_name = potential_name
                
                hp = cs.get_hyperparameter(hp_name)
                is_log = getattr(hp, 'log', False)
                
                # [核心] 调用 utils 中的科学计数法格式化函数
                formatted_dict[k] = _format_number_scientific(v, is_log)
                
            except Exception:
                # 如果找不到参数定义（比如 algorithm 字段），保持原样
                pass
                
        return json.dumps(formatted_dict)

    @classmethod
    def _build_param_tuner_system_prompt(cls) -> str:

        return """You are an expert AutoML strategist. Your current role is **Hyperparameter Tuner** in a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** task.
An algorithm has been selected. Your goal is to adjust a basic hyperparameter configuration on a monte carlo tree search node to find the best hyperparameter configuration for it. Your proposal should be informed by historical data and the overall mission of achieving peak performance.
"""

    # 算法HPO的prompt中的examples，初始版本，展示最好的，最差的，最近的3个
    @classmethod
    def _build_bwr_param_examples(cls, REACTACQ, selected_algo: str, history: List[Dict], max_k: int = 3, include_reflection: bool = True) -> str:
        """
        [MODIFIED] 
        构建参数优化阶段的BWR历史样本，包含智能切换逻辑。
        """
        if not history: 
            return "No tuning history for this algorithm yet. This is the first attempt."
        if not isinstance(history, list): 
            return "Invalid history format for display."

        # --- 1. 定义智能切换的阈值 ---
        # 我们定义“历史充足”为至少需要6条记录，这样B/W/R才开始有意义
        MIN_HISTORY_FOR_BWR = 6

        df = pd.DataFrame(history).drop_duplicates(subset='trail_id', keep='last')
        ascending_order = REACTACQ.lower_is_better 
        sorted_df_perf = df.sort_values(by='observation_perf', ascending=ascending_order)

        output_parts = []

        # --- 2. 内部辅助函数，用于格式化单个条目 ---
        def format_entry(row, show_reflection: bool) -> str:
            config_str = cls._format_config_for_llm(REACTACQ, row['action_config'])
            
            entry = [
                f"  --- Trial {row.get('trail_id', 'N/A')} ---",
                f"  Configuration: {config_str}", 
                f"  Observation (Performance): {row['observation_perf']:.6f}"
            ]
            if show_reflection and 'reflection_summary' in row and not pd.isna(row['reflection_summary']):
                entry.append(f"  Reflection Summary: {row['reflection_summary']}")
            return "\n".join(entry)

        if len(df) <= MIN_HISTORY_FOR_BWR:
            output_parts.append(
                f"Here is the **complete (but limited) tuning history** for '{selected_algo}', sorted by performance:"
            )
            for _, row in sorted_df_perf.iterrows():
                output_parts.append(format_entry(row, include_reflection))
        elif len(df) > MIN_HISTORY_FOR_BWR:
            output_parts.append(
                f"Here is a summary of the {len(df)} trials for '{selected_algo}', highlighting the best and worst attempts:"
            )

            top_df = sorted_df_perf.head(max_k)
            least_df = sorted_df_perf.tail(max_k).sort_values(by='observation_perf', ascending=not ascending_order)

            displayed_trail_ids = set()

            def add_block(title: str, dataframe: pd.DataFrame):
                nonlocal output_parts, displayed_trail_ids
                entries_to_show = []
                for _, row in dataframe.iterrows():
                    trail_id = row.get('trail_id', 'N/A')
                    if trail_id not in displayed_trail_ids:
                        entries_to_show.append(format_entry(row, include_reflection))
                        displayed_trail_ids.add(trail_id)

                if entries_to_show:
                    output_parts.append(f"\n**{title}:**")
                    output_parts.extend(entries_to_show)

            add_block(f"Least-Performing {len(least_df)} Configurations (by Performance)", least_df)
            top_df_display = top_df.sort_values(by='observation_perf', ascending= not ascending_order)
            add_block(f"Top-Performing {len(top_df_display)} Configurations (by Performance)", top_df_display)
            

        
        # 加入蒙特卡洛树搜索的最近尝试

        return "\n".join(output_parts)



    @classmethod
    def _build_mctsbwr_param_examples(cls, REACTACQ, selected_algo: str, node_path: Dict, include_reflection: bool = True, shuffle = False, current_perf: float = None) -> str:
        """
            构建参数优化阶段的MCTS历史样本，基于和当前配置的相似度以及提升和降低幅度，分别选取相似的提升最多的，降低最多的
        """

        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        show_reflection = include_reflection and (len(algo_history)>50)

        def _get_relative_comparison_text(other_perf: float) -> str:
            """返回纯文本的比较描述，不包含 performance 数值本身"""
            if current_perf is None:
                return ""

            # 处理 Inf/NaN
            if not np.isfinite(current_perf) or not np.isfinite(other_perf):
                if not np.isfinite(current_perf) and np.isfinite(other_perf):
                    return "better than current (current FAILED)"
                if np.isfinite(current_perf) and not np.isfinite(other_perf):
                    return "FAILED configuration"
                return "same (both FAILED)"

            delta = other_perf - current_perf
            denom = max(abs(current_perf), 1e-12)
            change_pct = abs(delta) / denom * 100.0

            if REACTACQ.lower_is_better:
                if other_perf < current_perf:
                    relation = "better"
                elif other_perf > current_perf:
                    relation = "worse"
                else:
                    relation = "same"
            else:
                if other_perf > current_perf:
                    relation = "better"
                elif other_perf < current_perf:
                    relation = "worse"
                else:
                    relation = "same"

            if relation == "same":
                return "same as current"
            else:
                return f"≈ {change_pct:.2f}% {relation} than current"
            
        def format_mcts_node_json(node, role_label: str) -> str:
            # 1. 属性提取
            if hasattr(node, 'current_config'): 
                raw_config = node.current_config
                raw_perf = node.current_perf
                node_id = node.current_id
                reflection = getattr(node, 'reflection', None)
            else: 
                raw_config = node.config
                raw_perf = node.perf
                node_id = node.id
                reflection = getattr(node, 'reflection', None)

            # 2. Config 处理
            config_dict = KnowledgeDatabase.instance()._cfg_to_dict(raw_config)
            config_json_str = cls._format_config_for_llm(REACTACQ, config_dict)
            config_obj = json.loads(config_json_str)

            # 3. Perf 处理
            perf = raw_perf
            if REACTACQ.lower_is_better:
                perf = -perf
            
            # 4. 构建结构化对象
            node_data = {
                "role": role_label,  # <--- [新增] 明确身份
                "node_id": node_id,
                "performance_score": float(f"{perf:.8g}"),
                "configuration": config_obj
            }

            # 增加比较信息
            comparison = _get_relative_comparison_text(perf)
            if comparison:
                node_data["comparison_to_current"] = comparison

            # 5. Reflection 处理
            if show_reflection and reflection:
                if not reflection.startswith("Bayesian Local search") and not reflection.startswith("Warmup"):
                    node_data["reflection_summary"] = reflection
                
            return json.dumps(node_data, ensure_ascii=False)
        
        output_parts = []
        if len(node_path['parent']) == 0:
            # Warmup
            all_paths = REACTACQ.optimizer.mcts_tree.cpp_dict[selected_algo].all_paths[:8]
            output_parts = [f"\n** Summary of Historical Configurations (JSON format, sorted by Performance): **"]
            for node in all_paths:
                # [修改点 2] 传入 Warmup 标签
                output_parts.append(format_mcts_node_json(node, "Historical Warmup Trial"))
        else:
            current_node = node_path['parent'][-1]
            current_config = current_node.config
            cpp = REACTACQ.optimizer.mcts_tree.cpp_dict[selected_algo]
            exclude_ids = []
            cpp.update_similarities(current_config)
            cpp.update_pareto(exclude_ids=exclude_ids)

            output_parts = []

            if len(cpp.decline_pareto) > 0:
                sorted_decline = sorted(cpp.decline_pareto, key=lambda n: n.current_perf)
                output_parts.append(f"\n** Least-Performing {len(sorted_decline)} Configurations: **")
                for node in sorted_decline:
                    output_parts.append(format_mcts_node_json(node, "Least-Performing Configuration"))

            if len(cpp.improve_pareto) > 0:
                nodes_to_show = cpp.improve_pareto
                if shuffle:
                    all_paths = REACTACQ.optimizer.mcts_tree.cpp_dict[selected_algo].all_paths[:10]
                    sel_idxs = np.random.choice(len(all_paths), size=min(6, len(all_paths)), replace=False)
                    sel_idxs = sorted(sel_idxs)
                    nodes_to_show = [all_paths[idx] for idx in sel_idxs]

                sorted_improve = sorted(nodes_to_show, key=lambda n: n.current_perf)
                output_parts.append(f"\n** Top-Performing {len(sorted_improve)} Configurations (worst to BEST): **")
                for node in sorted_improve:
                    # [修改点 4] 传入正面教材标签
                    output_parts.append(format_mcts_node_json(node, "Top-Performing Configuration"))


            # if len(cpp.improve_pareto) > 0:
            #     # mcts树上与当前配置相似且提升最多的expand
            #     output_parts.append(f"\n** Top-Performing {len(cpp.improve_pareto)} Configurations (sorted by Performance): **")
                
            #     nodes_to_show = cpp.improve_pareto
            #     if shuffle:
            #         all_paths = REACTACQ.optimizer.mcts_tree.cpp_dict[selected_algo].all_paths[:10]
            #         sel_idxs = np.random.choice(len(all_paths), size=min(6, len(all_paths)), replace=False)
            #         sel_idxs = sorted(sel_idxs)
            #         nodes_to_show = [all_paths[idx] for idx in sel_idxs]

            #     for node in nodes_to_show:
            #         output_parts.append(format_mcts_node_str(node))

            # if len(cpp.decline_pareto) > 0:
            #     # mcts树上与当前配置相似且下降最多的expand
            #     output_parts.append(f"\n** Least-Performing {len(cpp.decline_pareto)} Configurations (sorted by worst performance first): **")
            #     for node in cpp.decline_pareto:
            #         output_parts.append(format_mcts_node_str(node))

        return "\n".join(output_parts)

    @classmethod
    def _build_top10_param_examples(cls, REACTACQ, selected_algo: str, node_path: Dict, include_reflection: bool = True, current_perf: float = None) -> str:
        """
        消融实验版本：仅显示 top 10 performance 的配置（基于 performance 排序，不使用 pareto 和相似度）
        """
        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        show_reflection = include_reflection and (len(algo_history) > 50)

        def _get_relative_comparison_text(other_perf: float) -> str:
            """返回纯文本的比较描述，不包含 performance 数值本身"""
            if current_perf is None:
                return ""

            # 处理 Inf/NaN
            if not np.isfinite(current_perf) or not np.isfinite(other_perf):
                if not np.isfinite(current_perf) and np.isfinite(other_perf):
                    return "better than current (current FAILED)"
                if np.isfinite(current_perf) and not np.isfinite(other_perf):
                    return "FAILED configuration"
                return "same (both FAILED)"

            delta = other_perf - current_perf
            denom = max(abs(current_perf), 1e-12)
            change_pct = abs(delta) / denom * 100.0

            if REACTACQ.lower_is_better:
                if other_perf < current_perf:
                    relation = "better"
                elif other_perf > current_perf:
                    relation = "worse"
                else:
                    relation = "same"
            else:
                if other_perf > current_perf:
                    relation = "better"
                elif other_perf < current_perf:
                    relation = "worse"
                else:
                    relation = "same"

            if relation == "same":
                return "same as current"
            else:
                return f"≈ {change_pct:.2f}% {relation} than current"
            
        def format_mcts_node_json(node, role_label: str) -> str:
            # 1. 属性提取
            if hasattr(node, 'current_config'): 
                raw_config = node.current_config
                raw_perf = node.current_perf
                node_id = node.current_id
                reflection = getattr(node, 'reflection', None)
            else: 
                raw_config = node.config
                raw_perf = node.perf
                node_id = node.id
                reflection = getattr(node, 'reflection', None)

            # 2. Config 处理
            config_dict = KnowledgeDatabase.instance()._cfg_to_dict(raw_config)
            config_json_str = cls._format_config_for_llm(REACTACQ, config_dict)
            config_obj = json.loads(config_json_str)

            # 3. Perf 处理
            # raw_perf 是内部值（"越大越好"），需要转换为任务指标值用于显示和比较
            perf = raw_perf
            if REACTACQ.lower_is_better:
                perf = -perf  # 转换为任务指标值（如果是loss任务，取反）
            
            # 4. 构建结构化对象
            node_data = {
                "role": role_label,
                "node_id": node_id,
                "performance_score": float(f"{perf:.8g}"),
                "configuration": config_obj
            }

            # 增加比较信息（传入任务指标值，_get_relative_comparison_text内部会根据lower_is_better处理）
            comparison = _get_relative_comparison_text(perf)
            if comparison:
                node_data["comparison_to_current"] = comparison

            # 5. Reflection 处理
            if show_reflection and reflection:
                if not reflection.startswith("Bayesian Local search") and not reflection.startswith("Warmup"):
                    node_data["reflection_summary"] = reflection
                
            return json.dumps(node_data, ensure_ascii=False)
        
        output_parts = []
        cpp = REACTACQ.optimizer.mcts_tree.cpp_dict[selected_algo]
        
        # 获取所有路径并按 performance 排序（all_paths 已经按 current_perf 降序排序，最好的在前面）
        all_paths = cpp.all_paths
        
        if len(all_paths) == 0:
            return "No historical configurations available yet."
        
        # 取 top 10（all_paths 已经是降序排序，最好的在前面）
        top10_paths = all_paths[:10]
        
        # 按 performance 升序排列（worst to BEST），让最好的在最后，符合 ICL 的 recency bias
        # 参考 _build_mctsbwr_param_examples 的逻辑
        sorted_top10 = sorted(top10_paths, key=lambda n: n.current_perf, reverse=False)  # 升序：worst to best
        
        output_parts.append(f"\n** Top 10 Performing Configurations (sorted by Performance, worst to BEST): **")
        for node in sorted_top10:
            output_parts.append(format_mcts_node_json(node, "Top-Performing Configuration"))
        
        return "\n".join(output_parts)

    @classmethod
    def _build_random50_param_examples(cls, REACTACQ, selected_algo: str, node_path: Dict, include_reflection: bool = True, current_perf: float = None) -> str:
        """
        消融实验版本：随机选出50个配置（如果不足50个就全部展示），顺序随机
        """
        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        show_reflection = include_reflection and (len(algo_history) > 50)

        def _get_relative_comparison_text(other_perf: float) -> str:
            """返回纯文本的比较描述，不包含 performance 数值本身"""
            if current_perf is None:
                return ""

            # 处理 Inf/NaN
            if not np.isfinite(current_perf) or not np.isfinite(other_perf):
                if not np.isfinite(current_perf) and np.isfinite(other_perf):
                    return "better than current (current FAILED)"
                if np.isfinite(current_perf) and not np.isfinite(other_perf):
                    return "FAILED configuration"
                return "same (both FAILED)"

            delta = other_perf - current_perf
            denom = max(abs(current_perf), 1e-12)
            change_pct = abs(delta) / denom * 100.0

            if REACTACQ.lower_is_better:
                if other_perf < current_perf:
                    relation = "better"
                elif other_perf > current_perf:
                    relation = "worse"
                else:
                    relation = "same"
            else:
                if other_perf > current_perf:
                    relation = "better"
                elif other_perf < current_perf:
                    relation = "worse"
                else:
                    relation = "same"

            if relation == "same":
                return "same as current"
            else:
                return f"≈ {change_pct:.2f}% {relation} than current"
            
        def format_mcts_node_json(node, role_label: str) -> str:
            # 1. 属性提取
            if hasattr(node, 'current_config'): 
                raw_config = node.current_config
                raw_perf = node.current_perf
                node_id = node.current_id
                reflection = getattr(node, 'reflection', None)
            else: 
                raw_config = node.config
                raw_perf = node.perf
                node_id = node.id
                reflection = getattr(node, 'reflection', None)

            # 2. Config 处理
            config_dict = KnowledgeDatabase.instance()._cfg_to_dict(raw_config)
            config_json_str = cls._format_config_for_llm(REACTACQ, config_dict)
            config_obj = json.loads(config_json_str)

            # 3. Perf 处理
            # raw_perf 是内部值（"越大越好"），需要转换为任务指标值用于显示和比较
            perf = raw_perf
            if REACTACQ.lower_is_better:
                perf = -perf  # 转换为任务指标值（如果是loss任务，取反）
            
            # 4. 构建结构化对象
            node_data = {
                "role": role_label,
                "node_id": node_id,
                "performance_score": float(f"{perf:.8g}"),
                "configuration": config_obj
            }

            # 增加比较信息（传入任务指标值，_get_relative_comparison_text内部会根据lower_is_better处理）
            comparison = _get_relative_comparison_text(perf)
            if comparison:
                node_data["comparison_to_current"] = comparison

            # 5. Reflection 处理
            if show_reflection and reflection:
                if not reflection.startswith("Bayesian Local search") and not reflection.startswith("Warmup"):
                    node_data["reflection_summary"] = reflection
                
            return json.dumps(node_data, ensure_ascii=False)
        
        output_parts = []
        cpp = REACTACQ.optimizer.mcts_tree.cpp_dict[selected_algo]
        
        # 获取所有路径
        all_paths = cpp.all_paths
        
        if len(all_paths) == 0:
            return "No historical configurations available yet."
        
        # 随机选择最多50个配置
        num_to_select = min(50, len(all_paths))
        if len(all_paths) <= num_to_select:
            # 如果不足50个，全部展示
            selected_paths = all_paths.copy()
        else:
            # 随机选择50个
            selected_indices = np.random.choice(len(all_paths), size=num_to_select, replace=False)
            selected_paths = [all_paths[i] for i in selected_indices]
        
        # 随机打乱顺序（不需要按performance排序）
        random.shuffle(selected_paths)
        
        output_parts.append(f"\n** Random {len(selected_paths)} Historical Configurations (randomly selected and shuffled): **")
        for node in selected_paths:
            output_parts.append(format_mcts_node_json(node, "Historical Configuration"))
        
        return "\n".join(output_parts)

    @classmethod
    def _build_mcts_param_examples(cls, REACTACQ, selected_algo: str, node_path: Dict, include_reflection: bool = True, current_perf: float=None) -> str:
        """
        构建参数优化阶段的MCTS历史样本，基于节点路径。
        """
        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        show_reflection = include_reflection and (len(algo_history) > 50)


        def _get_relative_comparison_text(other_perf: float) -> str:
            """返回纯文本的比较描述，不包含 performance 数值本身"""
            if current_perf is None:
                return ""

            # 处理 Inf/NaN
            if not np.isfinite(current_perf) or not np.isfinite(other_perf):
                if not np.isfinite(current_perf) and np.isfinite(other_perf):
                    return "better than current (current FAILED)"
                if np.isfinite(current_perf) and not np.isfinite(other_perf):
                    return "FAILED configuration"
                return "same (both FAILED)"

            delta = other_perf - current_perf
            denom = max(abs(current_perf), 1e-12)
            change_pct = abs(delta) / denom * 100.0

            if REACTACQ.lower_is_better:
                if other_perf < current_perf:
                    relation = "better"
                elif other_perf > current_perf:
                    relation = "worse"
                else:
                    relation = "same"
            else:
                if other_perf > current_perf:
                    relation = "better"
                elif other_perf < current_perf:
                    relation = "worse"
                else:
                    relation = "same"

            if relation == "same":
                return "same as current"
            else:
                return f"≈ {change_pct:.2f}% {relation} than current"
            
            
        output_parts = [f"\n** Here is the recent tuning path towards the current basic configuration of '{selected_algo}': **"]

        # if len(node_path) <= 1:
        #     output_parts.append("No prior tuning steps available; this is the initial configuration.")
        #     return "\n".join(output_parts)


        for idx, node in enumerate(node_path['parent']):
            if node.config.get('algorithm') != selected_algo:
                raise ValueError(f"Node {node.id} has mismatched algorithm: expected '{selected_algo}', got '{node.config.get('algorithm')}'")
            # entry = [
            #     f"  --- Trial {node.id} ---",
            #     f"  Action: {json.dumps(node.action_config)}",
            #     f"  Observation (Performance): {node.observation_perf:.6f}"
            # ]
            # if include_reflection and node.reflection_summary:
            #     entry.append(f"  Reflection Summary: {node.reflection_summary}")
            # output_parts.append("\n".join(entry))
            # 修改上面的prompt，要能表现出MCTS的路径特性，第一个节点是基础配置，然后每个节点是基于上一个节点的调整，第一个node不用写action。最后一个节点是当前节点要调优的节点
            # 1. Title / Role
            if len(node_path['parent']) > 1 and idx == 0:
                role = "Initial Configuration"
            elif idx == len(node_path['parent']) - 1:
                role = "Current Basic Node (To be tuned)"
            else:
                role = "Intermediate Node"

            # 2. Config
            config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node.config)
            config_json_str = cls._format_config_for_llm(REACTACQ, config_dict)
            config_obj = json.loads(config_json_str)

            # 3. Perf
            perf = node.perf
            if REACTACQ.lower_is_better: 
                perf = -perf

            # 4. Build Object
            node_data = {
                "role_in_path": role,
                "node_id": node.id,
                "performance_score": float(f"{perf:.8g}"),
                "configuration": config_obj
            }
            
            # Comparison
            comparison = _get_relative_comparison_text(perf)
            if comparison:
                node_data["comparison_to_current"] = comparison

            # Reflection
            if show_reflection and node.reflection:
                if not node.reflection.startswith("Bayesian Local search") and not node.reflection.startswith("Warmup"):
                    node_data["reflection_summary"] = node.reflection

            output_parts.append(json.dumps(node_data, ensure_ascii=False))


        # 加入兄弟节点
        if len(node_path['sibling']) > 0:
            output_parts.append(f"\n** Here is {len(node_path['sibling'])} child nodes of the same basic configuration: **")
            for sibling in node_path['sibling']:
                if sibling.config.get('algorithm') != selected_algo:
                    raise ValueError(f"Sibling Node {sibling.id} has mismatched algorithm: expected '{selected_algo}', got '{sibling.config.get('algorithm')}'")
                config_dict = KnowledgeDatabase.instance()._cfg_to_dict(sibling.config)
                config_json_str = cls._format_config_for_llm(REACTACQ, config_dict)
                config_obj = json.loads(config_json_str)
                
                perf = sibling.perf
                if REACTACQ.lower_is_better: perf = -perf
                
                sibling_data = {
                    "role_in_path": "Sibling Node",
                    "node_id": sibling.id,
                    "performance_score": float(f"{perf:.8g}"),
                    "configuration": config_obj
                }
                
                comparison = _get_relative_comparison_text(perf)
                if comparison:
                    sibling_data["comparison_to_current"] = comparison

                if show_reflection and sibling.reflection:
                    if not sibling.reflection.startswith("Bayesian Local search") and not sibling.reflection.startswith("Warmup"):
                        sibling_data["reflection_summary"] = sibling.reflection
                
                output_parts.append(json.dumps(sibling_data, ensure_ascii=False))

        return "\n".join(output_parts)


    # 算法HPO的Prompt生成
    # ===== 主算法版本（使用_build_mctsbwr_param_examples + _build_mcts_param_examples）=====
    @classmethod
    def _get_param_tuning_messages_original(cls, REACTACQ, selected_algo: str, strategy_name: str, node_path: Dict, dup_num=0) -> List[Dict]:
        """
        主算法版本：同时使用Strategic Tuning History Summary (BWR)和MCTS Tuning Path
        """
        constraints_str = _constraints_text_for_selected_algo(
            REACTACQ.optimizer.config_space, 
            selected_algo=selected_algo
        )

        basic_config_str = "{}"
        if len(node_path['parent']) > 0:
             basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
             basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)
             
        if strategy_name == LLMEXPLORATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}."
                " Your goal now is to modify this configuration to explore new areas of the hyperparameter space."
                " The exploration strategy should focus on making changes that introduce diversity and help uncover potentially high-performing configurations."
                " Analyze unexplored areas and make bold adjustments to discover new possibilities."
            )
        elif strategy_name == LLMEXPLOITATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}." 
                " Your goal now is to refine and optimize this configuration to achieve peak performance."
                " The exploitation strategy should focus on making minor, incremental adjustments to the basic configuration, guiding it towards a more optimal setup."
                " Leverage historical configurations that have shown great performance to make informed and precise adjustments."
            )
        elif strategy_name == WARMUP:
            strategy_instruction = (
                f"You have selected the algorithm '{selected_algo}' and have access to historical tuning data for it."
                " Your goal is to propose a promising hyperparameter configuration from scratch."
                " For this warm-up phase, focus on identifying configurations that are distinct from previous attempts while still demonstrating strong performance."
                " Analyze the historical tuning data to find configurations that balance novelty and effectiveness."
            )
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'

        add_shap = REACTACQ.optimizer.add_shap
        historical_tuning_experience = ""
        if add_shap:
            historical_tuning_experience = f"""
- **Historical Tuning Experience that can be Leveraged:**
    {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}
            """

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        output_format_block = """```json
    {
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }
```"""
        dup_str = "" if dup_num == 0 else "(3) Avoid proposing a configuration that is the same as any previous configurations in the historical data."

        if strategy_name in [LLMEXPLORATION, LLMEXPLOITATION]:
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                shuffle=dup_num > 1,
                current_perf=None
            )

            mcts_history_display = cls._build_mcts_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. The Basic Configuration to Build Upon:**
- This configuration is derived from previous trials and represents a promising starting point: 
    {basic_config_str}


**3. Dataset & Metric Summary:**
    {dataset_summary_text}


**4. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}

- **Recent MCTS Tuning Path for the Basic Configuration:**
    {mcts_history_display}

**5. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Strategic Tuning History Summary' and the 'MCTS Tuning Path' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""

        elif strategy_name == WARMUP:
            assert len(node_path['parent']) == 0, "In WARMUP strategy, node_path['parent'] should be empty."
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. Dataset & Metric Summary:**
    {dataset_summary_text}


**3. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Strategic Tuning History Summary' and the 'MCTS Tuning Path' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        return [{"role": "system", "content": cls._build_param_tuner_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（No Reflection：不展示reflection，但保留主算法的memory机制）=====
    @classmethod
    def _get_param_tuning_messages(cls, REACTACQ, selected_algo: str, strategy_name: str, node_path: Dict, dup_num=0) -> List[Dict]:
        """
        消融实验版本：No Reflection - 不展示reflection_summary，但保留主算法的memory机制（同时使用Strategic Tuning History Summary和MCTS Tuning Path）
        """
        constraints_str = _constraints_text_for_selected_algo(
            REACTACQ.optimizer.config_space, 
            selected_algo=selected_algo
        )

        basic_config_str = "{}"
        if len(node_path['parent']) > 0:
             basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
             basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)
             
        if strategy_name == LLMEXPLORATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}."
                " Your goal now is to modify this configuration to explore new areas of the hyperparameter space."
                " The exploration strategy should focus on making changes that introduce diversity and help uncover potentially high-performing configurations."
                " Analyze unexplored areas and make bold adjustments to discover new possibilities."
            )
        elif strategy_name == LLMEXPLOITATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}." 
                " Your goal now is to refine and optimize this configuration to achieve peak performance."
                " The exploitation strategy should focus on making minor, incremental adjustments to the basic configuration, guiding it towards a more optimal setup."
                " Leverage historical configurations that have shown great performance to make informed and precise adjustments."
            )
        elif strategy_name == WARMUP:
            strategy_instruction = (
                f"You have selected the algorithm '{selected_algo}' and have access to historical tuning data for it."
                " Your goal is to propose a promising hyperparameter configuration from scratch."
                " For this warm-up phase, focus on identifying configurations that are distinct from previous attempts while still demonstrating strong performance."
                " Analyze the historical tuning data to find configurations that balance novelty and effectiveness."
            )
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'

        add_shap = REACTACQ.optimizer.add_shap
        historical_tuning_experience = ""
        if add_shap:
            historical_tuning_experience = f"""
- **Historical Tuning Experience that can be Leveraged:**
    {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}
            """

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        output_format_block = """```json
    {
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }
```"""
        dup_str = "" if dup_num == 0 else "(3) Avoid proposing a configuration that is the same as any previous configurations in the historical data."

        if strategy_name in [LLMEXPLORATION, LLMEXPLOITATION]:
            # 消融实验：不展示reflection_summary，设置 include_reflection=False
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=False,  # No Reflection：不展示reflection
                shuffle=dup_num > 1,
                current_perf=None
            )

            mcts_history_display = cls._build_mcts_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=False,  # No Reflection：不展示reflection
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. The Basic Configuration to Build Upon:**
- This configuration is derived from previous trials and represents a promising starting point: 
    {basic_config_str}


**3. Dataset & Metric Summary:**
    {dataset_summary_text}{historical_tuning_experience}


**4. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}

- **Recent MCTS Tuning Path for the Basic Configuration:**
    {mcts_history_display}

**5. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Strategic Tuning History Summary' and the 'MCTS Tuning Path' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""

        elif strategy_name == WARMUP:
            assert len(node_path['parent']) == 0, "In WARMUP strategy, node_path['parent'] should be empty."
            # 消融实验：不展示reflection_summary，设置 include_reflection=False
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=False,  # No Reflection：不展示reflection
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. Dataset & Metric Summary:**
    {dataset_summary_text}


**3. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Strategic Tuning History Summary' and the 'MCTS Tuning Path' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        return [{"role": "system", "content": cls._build_param_tuner_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（仅使用top 10 performance配置）=====
    @classmethod
    def _get_param_tuning_messages_ablation_top10(cls, REACTACQ, selected_algo: str, strategy_name: str, node_path: Dict, dup_num=0) -> List[Dict]:
        """
        消融实验版本：仅使用Top 10 Performing Configurations（按performance排序），不使用pareto和相似度筛选
        """
        constraints_str = _constraints_text_for_selected_algo(
            REACTACQ.optimizer.config_space, 
            selected_algo=selected_algo
        )

        basic_config_str = "{}"
        if len(node_path['parent']) > 0:
             basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
             basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)
             
        if strategy_name == LLMEXPLORATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}."
                " Your goal now is to modify this configuration to explore new areas of the hyperparameter space."
                " The exploration strategy should focus on making changes that introduce diversity and help uncover potentially high-performing configurations."
                " Analyze unexplored areas and make bold adjustments to discover new possibilities."
            )
        elif strategy_name == LLMEXPLOITATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}." 
                " Your goal now is to refine and optimize this configuration to achieve peak performance."
                " The exploitation strategy should focus on making minor, incremental adjustments to the basic configuration, guiding it towards a more optimal setup."
                " Leverage historical configurations that have shown great performance to make informed and precise adjustments."
            )
        elif strategy_name == WARMUP:
            strategy_instruction = (
                f"You have selected the algorithm '{selected_algo}' and have access to historical tuning data for it."
                " Your goal is to propose a promising hyperparameter configuration from scratch."
                " For this warm-up phase, focus on identifying configurations that are distinct from previous attempts while still demonstrating strong performance."
                " Analyze the historical tuning data to find configurations that balance novelty and effectiveness."
            )
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'

        add_shap = REACTACQ.optimizer.add_shap
        historical_tuning_experience = ""
        if add_shap:
            historical_tuning_experience = f"""
- **Historical Tuning Experience that can be Leveraged:**
    {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}
            """

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        output_format_block = """```json
    {
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }
```"""
        dup_str = "" if dup_num == 0 else "(3) Avoid proposing a configuration that is the same as any previous configurations in the historical data."

        if strategy_name in [LLMEXPLORATION, LLMEXPLOITATION]:
            # 消融实验：只使用Top 10 Performing Configurations
            history_display = cls._build_top10_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. The Basic Configuration to Build Upon:**
- This configuration is derived from previous trials and represents a promising starting point: 
    {basic_config_str}


**3. Dataset & Metric Summary:**
    {dataset_summary_text}


**4. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Top 10 Performing Configurations for '{selected_algo}':**
    {history_display}

**5. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Top 10 Performing Configurations' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""

        elif strategy_name == WARMUP:
            assert len(node_path['parent']) == 0, "In WARMUP strategy, node_path['parent'] should be empty."
            # WARMUP阶段：使用Top 10 Performing Configurations
            history_display = cls._build_top10_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. Dataset & Metric Summary:**
    {dataset_summary_text}


**3. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Top 10 Performing Configurations for '{selected_algo}':**
    {history_display}

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Top 10 Performing Configurations' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        return [{"role": "system", "content": cls._build_param_tuner_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（随机选出50个配置，顺序随机）=====
    @classmethod
    def _get_param_tuning_messages_ablation_random50(cls, REACTACQ, selected_algo: str, strategy_name: str, node_path: Dict, dup_num=0) -> List[Dict]:
        """
        消融实验版本：随机选出50个配置（如果不足50个就全部展示），顺序随机
        """
        constraints_str = _constraints_text_for_selected_algo(
            REACTACQ.optimizer.config_space, 
            selected_algo=selected_algo
        )

        basic_config_str = "{}"
        if len(node_path['parent']) > 0:
             basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
             basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)
             
        if strategy_name == LLMEXPLORATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}."
                " Your goal now is to modify this configuration to explore new areas of the hyperparameter space."
                " The exploration strategy should focus on making changes that introduce diversity and help uncover potentially high-performing configurations."
                " Analyze unexplored areas and make bold adjustments to discover new possibilities."
            )
        elif strategy_name == LLMEXPLOITATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}." 
                " Your goal now is to refine and optimize this configuration to achieve peak performance."
                " The exploitation strategy should focus on making minor, incremental adjustments to the basic configuration, guiding it towards a more optimal setup."
                " Leverage historical configurations that have shown great performance to make informed and precise adjustments."
            )
        elif strategy_name == WARMUP:
            strategy_instruction = (
                f"You have selected the algorithm '{selected_algo}' and have access to historical tuning data for it."
                " Your goal is to propose a promising hyperparameter configuration from scratch."
                " For this warm-up phase, focus on identifying configurations that are distinct from previous attempts while still demonstrating strong performance."
                " Analyze the historical tuning data to find configurations that balance novelty and effectiveness."
            )
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'

        add_shap = REACTACQ.optimizer.add_shap
        historical_tuning_experience = ""
        if add_shap:
            historical_tuning_experience = f"""
- **Historical Tuning Experience that can be Leveraged:**
    {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}
            """

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        output_format_block = """```json
    {
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }
```"""
        dup_str = "" if dup_num == 0 else "(3) Avoid proposing a configuration that is the same as any previous configurations in the historical data."

        if strategy_name in [LLMEXPLORATION, LLMEXPLOITATION]:
            # 消融实验：随机选出50个配置（如果不足50个就全部展示），顺序随机
            history_display = cls._build_random50_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. The Basic Configuration to Build Upon:**
- This configuration is derived from previous trials and represents a promising starting point: 
    {basic_config_str}


**3. Dataset & Metric Summary:**
    {dataset_summary_text}


**4. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Random Historical Configurations for '{selected_algo}':**
    {history_display}

**5. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Random Historical Configurations' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""

        elif strategy_name == WARMUP:
            assert len(node_path['parent']) == 0, "In WARMUP strategy, node_path['parent'] should be empty."
            # WARMUP阶段：随机选出50个配置
            history_display = cls._build_random50_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. Dataset & Metric Summary:**
    {dataset_summary_text}


**3. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Random Historical Configurations for '{selected_algo}':**
    {history_display}

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Random Historical Configurations' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        return [{"role": "system", "content": cls._build_param_tuner_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（不展示任何历史信息）=====
    @classmethod
    def _get_param_tuning_messages_ablation_no_history(cls, REACTACQ, selected_algo: str, strategy_name: str, node_path: Dict, dup_num=0) -> List[Dict]:
        """
        消融实验版本：不展示任何历史信息（既不展示local，也不展示global）
        """
        constraints_str = _constraints_text_for_selected_algo(
            REACTACQ.optimizer.config_space, 
            selected_algo=selected_algo
        )

        basic_config_str = "{}"
        if len(node_path['parent']) > 0:
             basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
             basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)
             
        if strategy_name == LLMEXPLORATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials."
                " Your goal now is to modify this configuration to explore new areas of the hyperparameter space."
                " The exploration strategy should focus on making changes that introduce diversity and help uncover potentially high-performing configurations."
                " Analyze unexplored areas and make bold adjustments to discover new possibilities."
            )
        elif strategy_name == LLMEXPLOITATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials." 
                " Your goal now is to refine and optimize this configuration to achieve peak performance."
                " The exploitation strategy should focus on making minor, incremental adjustments to the basic configuration, guiding it towards a more optimal setup."
                " Use your understanding of hyperparameter optimization principles to make informed and precise adjustments."
            )
        elif strategy_name == WARMUP:
            strategy_instruction = (
                f"You have selected the algorithm '{selected_algo}'."
                " Your goal is to propose a promising hyperparameter configuration from scratch."
                " For this warm-up phase, focus on identifying configurations that demonstrate strong performance based on general optimization principles."
                " Consider typical effective parameter ranges for this algorithm type."
            )
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'

        add_shap = REACTACQ.optimizer.add_shap
        historical_tuning_experience = ""
        if add_shap:
            historical_tuning_experience = f"""
- **Historical Tuning Experience that can be Leveraged:**
    {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}
            """

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        output_format_block = """```json
    {
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }
```"""
        dup_str = "" if dup_num == 0 else "(3) Avoid proposing a configuration that is the same as any previous configurations."

        if strategy_name in [LLMEXPLORATION, LLMEXPLOITATION]:
            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. The Basic Configuration to Build Upon:**
- This configuration is derived from previous trials and represents a promising starting point: 
    {basic_config_str}


**3. Dataset & Metric Summary:**
    {dataset_summary_text}{historical_tuning_experience}

**4. Performance Metric:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

**5. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""

        elif strategy_name == WARMUP:
            assert len(node_path['parent']) == 0, "In WARMUP strategy, node_path['parent'] should be empty."

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. Dataset & Metric Summary:**
    {dataset_summary_text}{historical_tuning_experience}

**3. Performance Metric:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        return [{"role": "system", "content": cls._build_param_tuner_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（仅使用_build_mctsbwr_param_examples，不使用_build_mcts_param_examples）=====
    @classmethod
    def _get_param_tuning_messages_ablation_global_only(cls, REACTACQ, selected_algo: str, strategy_name: str, node_path: Dict, dup_num=0) -> List[Dict]:
        """
        消融实验版本：仅使用Strategic Tuning History Summary (BWR/Global)，不使用MCTS Tuning Path (Local)
        """
        constraints_str = _constraints_text_for_selected_algo(
            REACTACQ.optimizer.config_space, 
            selected_algo=selected_algo
        )

        basic_config_str = "{}"
        if len(node_path['parent']) > 0:
             basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
             basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)
             
        if strategy_name == LLMEXPLORATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}."
                " Your goal now is to modify this configuration to explore new areas of the hyperparameter space."
                " The exploration strategy should focus on making changes that introduce diversity and help uncover potentially high-performing configurations."
                " Analyze unexplored areas and make bold adjustments to discover new possibilities."
            )
        elif strategy_name == LLMEXPLOITATION:
            strategy_instruction = (
                f"You have a basic configuration for algorithm '{selected_algo}' from previous trials, along with some historical tuning data for {selected_algo}." 
                " Your goal now is to refine and optimize this configuration to achieve peak performance."
                " The exploitation strategy should focus on making minor, incremental adjustments to the basic configuration, guiding it towards a more optimal setup."
                " Leverage historical configurations that have shown great performance to make informed and precise adjustments."
            )
        elif strategy_name == WARMUP:
            strategy_instruction = (
                f"You have selected the algorithm '{selected_algo}' and have access to historical tuning data for it."
                " Your goal is to propose a promising hyperparameter configuration from scratch."
                " For this warm-up phase, focus on identifying configurations that are distinct from previous attempts while still demonstrating strong performance."
                " Analyze the historical tuning data to find configurations that balance novelty and effectiveness."
            )
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'

        add_shap = REACTACQ.optimizer.add_shap
        historical_tuning_experience = ""
        if add_shap:
            historical_tuning_experience = f"""
- **Historical Tuning Experience that can be Leveraged:**
    {_build_algorithm_analysis_text(REACTACQ.optimizer.algorithm_analysis_knowledge[selected_algo], strategy_name)}
            """

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        output_format_block = """```json
    {
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }
```"""
        dup_str = "" if dup_num == 0 else "(3) Avoid proposing a configuration that is the same as any previous configurations in the historical data."

        if strategy_name in [LLMEXPLORATION, LLMEXPLOITATION]:
            # 消融实验：只使用Strategic Tuning History Summary (BWR)，不使用MCTS Tuning Path (Local)
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                shuffle=dup_num > 1, # 两次以上的重复，打乱顺序
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. The Basic Configuration to Build Upon:**
- This configuration is derived from previous trials and represents a promising starting point: 
    {basic_config_str}


**3. Dataset & Metric Summary:**
    {dataset_summary_text}


**4. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}

**5. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Strategic Tuning History Summary' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""

        elif strategy_name == WARMUP:
            assert len(node_path['parent']) == 0, "In WARMUP strategy, node_path['parent'] should be empty."
            # WARMUP阶段：使用Strategic Tuning History Summary
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=None
            )

            user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset **'{REACTACQ.task_id}'**.

### **CURRENT TASK: Tune Hyperparameters ({strategy_name})**

**1. Directive:**
An algorithm, **'{selected_algo}'**, has been selected. Your immediate objective is to propose a hyperparameter configuration for it.
(1) Your strategic goal is **{strategy_name}**: {strategy_instruction}
(2) Make sure that the configuration you propose is within the hyperparameter space defined for '{selected_algo}':
  {constraints_str}
{dup_str}

**2. Dataset & Metric Summary:**
    {dataset_summary_text}


**3. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).

- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. Crucially, analyze the 'Strategic Tuning History Summary' and connect your insights with the Dataset & Metric Summary and your {strategy_name} directive to justify your new parameter choices.)
**Action:**
Output the configuration in a single JSON block as shown below. Do not add any text before or after it.
{output_format_block}
"""
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

        return [{"role": "system", "content": cls._build_param_tuner_system_prompt()}, {"role": "user", "content": user_prompt}]

 
#     @classmethod
#     def _get_reflection_messages(cls, REACTACQ, reflection_context: Dict, node_path: List[Node]) -> List[Dict]:
#         """构建反思的独立 Prompt"""

#         selected_algo = reflection_context['action_config']['algorithm']

#         history_display = cls._build_bwr_param_examples(
#             REACTACQ,
#             selected_algo, # 传入算法名称
#             REACTACQ.param_tuning_history[selected_algo],
#             max_k=3,
#             include_reflection=True
#         )
#         if history_display:
#             history_display = f"\n- **Strategic Trial Summary for '{selected_algo}':**\n    {history_display}"

#         mcts_history_display = cls._build_mcts_param_examples(
#             REACTACQ,
#             selected_algo,
#             node_path,
#             include_reflection=True
#         )

#         user_prompt = f"""
# ### **OVERALL MISSION: Automated Machine Learning (AutoML)**
# Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the dataset '{REACTACQ.task_id}'.

# ### **CURRENT TASK: Reflect on Trial Performance**

# **1. Context: The Trial to Analyze**
# - **Basic Configuration to Build Upon:** {json.dumps(KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config))}
# - **Original Rationale (Thought Summary)**: {reflection_context['param_thought_summary']}
# - **Your Chosen Config**: `{json.dumps(reflection_context['action_config'])}`
# - **Observed Performance**: `{reflection_context['observation_perf']:.6f}`

# **2. Available Data for '{selected_algo}'**
# {history_display}
# - **Recent MCTS Tuning Path for the basic configuration:**
#     {mcts_history_display}

# ### 3. YOUR TASK: ANALYTICAL REFLECTION
# Your goal is to synthesize the key lessons learned from this trial. Your analysis must be quantitative and structured.

# **1. Triage & Compare:**
# First, where does this trial's performance ({reflection_context['observation_perf']:.6f}) rank? Is it closer to the Top-Performing, Least-Performing, or somewhere in the MCTS tuning path?

# **2. Attribute & Synthesize:**
# Based on its ranking, pinpoint the parameter changes relative to the basic configuration that likely caused this outcome. What does this trial teach you about parameter importance, promising value ranges, or interactions? 

# **4. REQUIRED OUTPUT FORMAT**

# **Reflection:**
# (Your structured analysis addressing the two points above.)

# **Reflection Summary:**
# (A concise, analytical summary of the key quantitative lesson learned from this trial. Do not suggest next steps.)
# """
#         return [{"role": "system", "content": cls._build_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]


    # ===== 主算法版本（使用_build_mctsbwr_param_examples + _build_mcts_param_examples）=====
    @classmethod
    def _get_reflection_messages_original(cls, REACTACQ, reflection_context: Dict, node_path: Dict) -> List[Dict]:
        """
        主算法版本：同时使用Strategic Tuning History Summary (BWR)和MCTS Tuning Path
        """
        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'
        selected_algo = reflection_context['action_config']['algorithm']

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        result_config_str = cls._format_config_for_llm(REACTACQ, reflection_context['action_config'])

        # 当前perf真实值
        current_perf = reflection_context['observation_perf']
        perf_str = f"{current_perf:.8g}"

        #真实指标
        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        history_perfs = [h["observation_perf"] for h in algo_history]

        # 默认不写排名句子
        trial_rank_text = ""

        if history_perfs:
            # 把当前 trial 一起放进列表
            perfs_with_current = history_perfs + [current_perf]

            # 根据任务方向排序：
            # - lower_is_better=True: 越小越好 => 升序
            # - lower_is_better=False: 越大越好 => 降序
            sorted_perfs = sorted(perfs_with_current, reverse=not REACTACQ.lower_is_better)

            # 找到当前 trial 的索引。为避免浮点精度问题，按"最接近值"的方式匹配
            def _find_best_index(values, target):
                best_idx = 0
                best_diff = float("inf")
                for i, v in enumerate(values):
                    d = abs(v - target)
                    if d < best_diff:
                        best_diff = d
                        best_idx = i
                return best_idx

            rank_index = _find_best_index(sorted_perfs, current_perf)
            rank = rank_index + 1
            total = len(sorted_perfs)
            best_desc = "lowest" if REACTACQ.lower_is_better else "highest"
            trial_rank_text = (
                f"{rank} out of {total} trials "
                f"(1 = {best_desc} value of the task metric among all trials of '{selected_algo}')."
            )

        if len(node_path['parent']) > 0:
            
            # 与 current_perf 同源的任务指标，真实值
            parent_raw_perf = node_path['parent'][-1].perf
            if REACTACQ.lower_is_better:
                # 内部统一"越大越好"时，loss 任务需要取反还原
                parent_task_perf = -parent_raw_perf
            else:
                parent_task_perf = parent_raw_perf

            parent_perf_str = f"{parent_task_perf:.8g}"

            basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
            basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)

            delta = current_perf - parent_task_perf
            if parent_task_perf == 0:
                relative_change_text = (
                    "undefined (basic configuration performance is 0, cannot compute relative percentage change safely)"
                )
            else:
                change_pct = (delta / abs(parent_task_perf)) * 100.0

                # 根据任务方向判断 improved / degraded
                if REACTACQ.lower_is_better:
                    # 越小越好：current < parent => improved
                    if current_perf < parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf > parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"
                else:
                    # 越大越好：current > parent => improved
                    if current_perf > parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf < parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"

                relative_change_text = (
                    f"{better_or_worse} by {abs(delta):.6g} in the task metric "
                    f"(≈ {abs(change_pct):.2f}% change relative to the basic configuration)."
                )


            analysis_str = f"""
- **Basic Configuration to Change From:** {basic_config_str}
- **Basic Configuration Performance:** `{parent_perf_str}`
- **Original Thought to Adjust Basic Configuration**: {reflection_context['param_thought_summary']}
- **Resulting Configuration**: `{result_config_str}`
- **Resulting Performance**: `{perf_str}`
- **Change vs. Basic Configuration:** {relative_change_text}
"""
            if trial_rank_text:
                analysis_str += f"- **Current Trial's Ranking among all historical trials for '{selected_algo}':** {trial_rank_text}\n"

            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            mcts_history_display = cls._build_mcts_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}

- **Recent MCTS Tuning Path for the Basic Configuration:**
    {mcts_history_display}
"""

        else:
            # warmup阶段，没有basic configuration，直接根据历史数据进行调优，要找到和现有不太一样的，并且尽量效果好的

            analysis_str = f"""
- **Original Thought to Recommend Configuration**: {reflection_context['param_thought_summary']}
- **Configuration**: `{result_config_str}`
- **Performance**: `{perf_str}`
"""

            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}
"""

        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset '{REACTACQ.task_id}'.

### **CURRENT TASK: Reflect on Trial Performance**

**1. Dataset & Metric Summary:**
    {dataset_summary_text}


**2. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).
{historical_data_str}


**3. Context: The Trial to Analyze**
Use the context below together with the historical data above when performing your analytical reflection:
{analysis_str}


**4. YOUR TASK: ANALYTICAL REFLECTION
Your goal is to synthesize **reliable and reusable** lessons learned from this trial. Your analysis must be "quantitative, structured, and globally useful" for future trials.

- In **Reflection**, you should:
  1. **Triage & Compare**: Position this trial among the historical data, using the numerical performance comparisons (absolute differences and percentages) from the context.
  2. **Attribute & Synthesize**: Explain which parameter deviation from the historical configurations most likely led to the observed outcome.
     - Highlight only those hyperparameters that appear **clearly important**, and the **value ranges** that look consistently promising or risky, and any **interactions** you can infer from the numbers.
     - Make your reasoning as reusable as possible (not just describing this single trial), so that it can guide **future tuning decisions**.
     - Only include conclusions that you are **confident about** based on the provided quantitative evidence.

- In **Reflection Summary**, distill your analysis into a reliable and high-confidence summary of global lessons :
  - Include **only** those insights that are:
    - Clearly supported by multiple numerical observations in the context, and
    - Likely to remain valid for future trials of this algorithm on this dataset.
  - Do **not** invent rules for hyperparameters where the data is noisy or contradictory — in those cases, **do not mention them** in the Reflection Summary.
  - Do **not** suggest explicit "next steps" or new configurations.

**5. REQUIRED OUTPUT FORMAT**

**Reflection:**
(Your structured analysis addressing the two points above, focused on quantitative comparison and attribution of the outcome.)

**Reflection Summary:**
(A compact, single paragraph summary of global, reusable and high-confidence conclusions derived from this analysis.)
"""
        return [{"role": "system", "content": cls._build_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（No Reflection：这个方法在no reflection实验中不会被调用）=====
    @classmethod
    def _get_reflection_messages(cls, REACTACQ, reflection_context: Dict, node_path: Dict) -> List[Dict]:
        """
        消融实验版本：No Reflection - 这个方法在no reflection实验中不会被调用
        （因为reactmcts_acq.py中会跳过reflection步骤）
        """
        # 这个方法不应该被调用，但为了代码完整性，提供一个占位实现
        raise RuntimeError("Reflection is disabled in the no-reflection ablation experiment. This method should not be called.")

    # ===== 消融实验版本（仅使用_build_mctsbwr_param_examples，不使用_build_mcts_param_examples）=====
    @classmethod
    def _get_reflection_messages_ablation_global_only(cls, REACTACQ, reflection_context: Dict, node_path: Dict) -> List[Dict]:
        """
        消融实验版本：仅使用Strategic Tuning History Summary (BWR/Global)，不使用MCTS Tuning Path (Local)
        """
        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'
        selected_algo = reflection_context['action_config']['algorithm']

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        result_config_str = cls._format_config_for_llm(REACTACQ, reflection_context['action_config'])

        # 当前perf真实值
        current_perf = reflection_context['observation_perf']
        perf_str = f"{current_perf:.8g}"

        #真实指标
        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        history_perfs = [h["observation_perf"] for h in algo_history]

        # 默认不写排名句子
        trial_rank_text = ""

        if history_perfs:
            # 把当前 trial 一起放进列表
            perfs_with_current = history_perfs + [current_perf]

            # 根据任务方向排序：
            # - lower_is_better=True: 越小越好 => 升序
            # - lower_is_better=False: 越大越好 => 降序
            sorted_perfs = sorted(perfs_with_current, reverse=not REACTACQ.lower_is_better)

            # 找到当前 trial 的索引。为避免浮点精度问题，按"最接近值"的方式匹配
            def _find_best_index(values, target):
                best_idx = 0
                best_diff = float("inf")
                for i, v in enumerate(values):
                    d = abs(v - target)
                    if d < best_diff:
                        best_diff = d
                        best_idx = i
                return best_idx

            rank_index = _find_best_index(sorted_perfs, current_perf)
            rank = rank_index + 1
            total = len(sorted_perfs)
            best_desc = "lowest" if REACTACQ.lower_is_better else "highest"
            trial_rank_text = (
                f"{rank} out of {total} trials "
                f"(1 = {best_desc} value of the task metric among all trials of '{selected_algo}')."
            )

        if len(node_path['parent']) > 0:
            
            # 与 current_perf 同源的任务指标，真实值
            parent_raw_perf = node_path['parent'][-1].perf
            if REACTACQ.lower_is_better:
                # 内部统一"越大越好"时，loss 任务需要取反还原
                parent_task_perf = -parent_raw_perf
            else:
                parent_task_perf = parent_raw_perf

            parent_perf_str = f"{parent_task_perf:.8g}"

            basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
            basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)

            delta = current_perf - parent_task_perf
            if parent_task_perf == 0:
                relative_change_text = (
                    "undefined (basic configuration performance is 0, cannot compute relative percentage change safely)"
                )
            else:
                change_pct = (delta / abs(parent_task_perf)) * 100.0

                # 根据任务方向判断 improved / degraded
                if REACTACQ.lower_is_better:
                    # 越小越好：current < parent => improved
                    if current_perf < parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf > parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"
                else:
                    # 越大越好：current > parent => improved
                    if current_perf > parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf < parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"

                relative_change_text = (
                    f"{better_or_worse} by {abs(delta):.6g} in the task metric "
                    f"(≈ {abs(change_pct):.2f}% change relative to the basic configuration)."
                )


            analysis_str = f"""
- **Basic Configuration to Change From:** {basic_config_str}
- **Basic Configuration Performance:** `{parent_perf_str}`
- **Original Thought to Adjust Basic Configuration**: {reflection_context['param_thought_summary']}
- **Resulting Configuration**: `{result_config_str}`
- **Resulting Performance**: `{perf_str}`
- **Change vs. Basic Configuration:** {relative_change_text}
"""
            if trial_rank_text:
                analysis_str += f"- **Current Trial's Ranking among all historical trials for '{selected_algo}':** {trial_rank_text}\n"

            # 消融实验：只使用Strategic Tuning History Summary (BWR)，不使用MCTS Tuning Path (Local)
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}
"""

        else:
            # warmup阶段，没有basic configuration，直接根据历史数据进行调优，要找到和现有不太一样的，并且尽量效果好的

            analysis_str = f"""
- **Original Thought to Recommend Configuration**: {reflection_context['param_thought_summary']}
- **Configuration**: `{result_config_str}`
- **Performance**: `{perf_str}`
"""

            # WARMUP阶段：使用Strategic Tuning History Summary
            history_display = cls._build_mctsbwr_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Strategic Tuning History Summary for '{selected_algo}':**
    {history_display}
"""

        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset '{REACTACQ.task_id}'.

### **CURRENT TASK: Reflect on Trial Performance**

**1. Dataset & Metric Summary:**
    {dataset_summary_text}


**2. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).
{historical_data_str}


**3. Context: The Trial to Analyze**
Use the context below together with the historical data above when performing your analytical reflection:
{analysis_str}


**4. YOUR TASK: ANALYTICAL REFLECTION
Your goal is to synthesize **reliable and reusable** lessons learned from this trial. Your analysis must be "quantitative, structured, and globally useful" for future trials.

- In **Reflection**, you should:
  1. **Triage & Compare**: Position this trial among the historical data, using the numerical performance comparisons (absolute differences and percentages) from the context.
  2. **Attribute & Synthesize**: Explain which parameter deviation from the historical configurations most likely led to the observed outcome.
     - Highlight only those hyperparameters that appear **clearly important**, and the **value ranges** that look consistently promising or risky, and any **interactions** you can infer from the numbers.
     - Make your reasoning as reusable as possible (not just describing this single trial), so that it can guide **future tuning decisions**.
     - Only include conclusions that you are **confident about** based on the provided quantitative evidence.

- In **Reflection Summary**, distill your analysis into a reliable and high-confidence summary of global lessons :
  - Include **only** those insights that are:
    - Clearly supported by multiple numerical observations in the context, and
    - Likely to remain valid for future trials of this algorithm on this dataset.
  - Do **not** invent rules for hyperparameters where the data is noisy or contradictory — in those cases, **do not mention them** in the Reflection Summary.
  - Do **not** suggest explicit "next steps" or new configurations.

**5. REQUIRED OUTPUT FORMAT**

**Reflection:**
(Your structured analysis addressing the two points above, focused on quantitative comparison and attribution of the outcome.)

**Reflection Summary:**
(A compact, single paragraph summary of global, reusable and high-confidence conclusions derived from this analysis.)
"""
        return [{"role": "system", "content": cls._build_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（随机选出50个配置，顺序随机）=====
    @classmethod
    def _get_reflection_messages_ablation_random50(cls, REACTACQ, reflection_context: Dict, node_path: Dict) -> List[Dict]:
        """
        消融实验版本：随机选出50个配置（如果不足50个就全部展示），顺序随机
        """
        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'
        selected_algo = reflection_context['action_config']['algorithm']

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        result_config_str = cls._format_config_for_llm(REACTACQ, reflection_context['action_config'])

        # 当前perf真实值
        current_perf = reflection_context['observation_perf']
        perf_str = f"{current_perf:.8g}"

        #真实指标
        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        history_perfs = [h["observation_perf"] for h in algo_history]

        # 默认不写排名句子
        trial_rank_text = ""

        if history_perfs:
            # 把当前 trial 一起放进列表
            perfs_with_current = history_perfs + [current_perf]

            # 根据任务方向排序：
            # - lower_is_better=True: 越小越好 => 升序
            # - lower_is_better=False: 越大越好 => 降序
            sorted_perfs = sorted(perfs_with_current, reverse=not REACTACQ.lower_is_better)

            # 找到当前 trial 的索引。为避免浮点精度问题，按"最接近值"的方式匹配
            def _find_best_index(values, target):
                best_idx = 0
                best_diff = float("inf")
                for i, v in enumerate(values):
                    d = abs(v - target)
                    if d < best_diff:
                        best_diff = d
                        best_idx = i
                return best_idx

            rank_index = _find_best_index(sorted_perfs, current_perf)
            rank = rank_index + 1
            total = len(sorted_perfs)
            best_desc = "lowest" if REACTACQ.lower_is_better else "highest"
            trial_rank_text = (
                f"{rank} out of {total} trials "
                f"(1 = {best_desc} value of the task metric among all trials of '{selected_algo}')."
            )

        if len(node_path['parent']) > 0:
            
            # 与 current_perf 同源的任务指标，真实值
            parent_raw_perf = node_path['parent'][-1].perf
            if REACTACQ.lower_is_better:
                # 内部统一"越大越好"时，loss 任务需要取反还原
                parent_task_perf = -parent_raw_perf
            else:
                parent_task_perf = parent_raw_perf

            parent_perf_str = f"{parent_task_perf:.8g}"

            basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
            basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)

            delta = current_perf - parent_task_perf
            if parent_task_perf == 0:
                relative_change_text = (
                    "undefined (basic configuration performance is 0, cannot compute relative percentage change safely)"
                )
            else:
                change_pct = (delta / abs(parent_task_perf)) * 100.0

                # 根据任务方向判断 improved / degraded
                if REACTACQ.lower_is_better:
                    # 越小越好：current < parent => improved
                    if current_perf < parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf > parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"
                else:
                    # 越大越好：current > parent => improved
                    if current_perf > parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf < parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"

                relative_change_text = (
                    f"{better_or_worse} by {abs(delta):.6g} in the task metric "
                    f"(≈ {abs(change_pct):.2f}% change relative to the basic configuration)."
                )


            analysis_str = f"""
- **Basic Configuration to Change From:** {basic_config_str}
- **Basic Configuration Performance:** `{parent_perf_str}`
- **Original Thought to Adjust Basic Configuration**: {reflection_context['param_thought_summary']}
- **Resulting Configuration**: `{result_config_str}`
- **Resulting Performance**: `{perf_str}`
- **Change vs. Basic Configuration:** {relative_change_text}
"""
            if trial_rank_text:
                analysis_str += f"- **Current Trial's Ranking among all historical trials for '{selected_algo}':** {trial_rank_text}\n"

            # 消融实验：随机选出50个配置（如果不足50个就全部展示），顺序随机
            history_display = cls._build_random50_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Random Historical Configurations for '{selected_algo}':**
    {history_display}
"""

        else:
            # warmup阶段，没有basic configuration，直接根据历史数据进行调优，要找到和现有不太一样的，并且尽量效果好的

            analysis_str = f"""
- **Original Thought to Recommend Configuration**: {reflection_context['param_thought_summary']}
- **Configuration**: `{result_config_str}`
- **Performance**: `{perf_str}`
"""

            # WARMUP阶段：随机选出50个配置
            history_display = cls._build_random50_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Random Historical Configurations for '{selected_algo}':**
    {history_display}
"""

        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset '{REACTACQ.task_id}'.

### **CURRENT TASK: Reflect on Trial Performance**

**1. Dataset & Metric Summary:**
    {dataset_summary_text}


**2. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).
{historical_data_str}


**3. Context: The Trial to Analyze**
Use the context below together with the historical data above when performing your analytical reflection:
{analysis_str}


**4. YOUR TASK: ANALYTICAL REFLECTION
Your goal is to synthesize **reliable and reusable** lessons learned from this trial. Your analysis must be "quantitative, structured, and globally useful" for future trials.

- In **Reflection**, you should:
  1. **Triage & Compare**: Position this trial among the historical data, using the numerical performance comparisons (absolute differences and percentages) from the context.
  2. **Attribute & Synthesize**: Explain which parameter deviation from the historical configurations most likely led to the observed outcome.
     - Highlight only those hyperparameters that appear **clearly important**, and the **value ranges** that look consistently promising or risky, and any **interactions** you can infer from the numbers.
     - Make your reasoning as reusable as possible (not just describing this single trial), so that it can guide **future tuning decisions**.
     - Only include conclusions that you are **confident about** based on the provided quantitative evidence.

- In **Reflection Summary**, distill your analysis into a reliable and high-confidence summary of global lessons :
  - Include **only** those insights that are:
    - Clearly supported by multiple numerical observations in the context, and
    - Likely to remain valid for future trials of this algorithm on this dataset.
  - Do **not** invent rules for hyperparameters where the data is noisy or contradictory — in those cases, **do not mention them** in the Reflection Summary.
  - Do **not** suggest explicit "next steps" or new configurations.

**5. REQUIRED OUTPUT FORMAT**

**Reflection:**
(Your structured analysis addressing the two points above, focused on quantitative comparison and attribution of the outcome.)

**Reflection Summary:**
(A compact, single paragraph summary of global, reusable and high-confidence conclusions derived from this analysis.)
"""
        return [{"role": "system", "content": cls._build_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（不展示任何历史信息）=====
    @classmethod
    def _get_reflection_messages_ablation_no_history(cls, REACTACQ, reflection_context: Dict, node_path: Dict) -> List[Dict]:
        """
        消融实验版本：不展示任何历史信息（既不展示local，也不展示global）
        """
        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'
        selected_algo = reflection_context['action_config']['algorithm']

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        result_config_str = cls._format_config_for_llm(REACTACQ, reflection_context['action_config'])

        # 当前perf真实值
        current_perf = reflection_context['observation_perf']
        perf_str = f"{current_perf:.8g}"

        if len(node_path['parent']) > 0:
            # 与 current_perf 同源的任务指标，真实值
            parent_raw_perf = node_path['parent'][-1].perf
            if REACTACQ.lower_is_better:
                # 内部统一"越大越好"时，loss 任务需要取反还原
                parent_task_perf = -parent_raw_perf
            else:
                parent_task_perf = parent_raw_perf

            parent_perf_str = f"{parent_task_perf:.8g}"

            basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
            basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)

            delta = current_perf - parent_task_perf
            if parent_task_perf == 0:
                relative_change_text = (
                    "undefined (basic configuration performance is 0, cannot compute relative percentage change safely)"
                )
            else:
                change_pct = (delta / abs(parent_task_perf)) * 100.0

                # 根据任务方向判断 improved / degraded
                if REACTACQ.lower_is_better:
                    # 越小越好：current < parent => improved
                    if current_perf < parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf > parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"
                else:
                    # 越大越好：current > parent => improved
                    if current_perf > parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf < parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"

                relative_change_text = (
                    f"{better_or_worse} by {abs(delta):.6g} in the task metric "
                    f"(≈ {abs(change_pct):.2f}% change relative to the basic configuration)."
                )

            analysis_str = f"""
- **Basic Configuration to Change From:** {basic_config_str}
- **Basic Configuration Performance:** `{parent_perf_str}`
- **Original Thought to Adjust Basic Configuration**: {reflection_context['param_thought_summary']}
- **Resulting Configuration**: `{result_config_str}`
- **Resulting Performance**: `{perf_str}`
- **Change vs. Basic Configuration:** {relative_change_text}
"""

        else:
            # warmup阶段，没有basic configuration
            analysis_str = f"""
- **Original Thought to Recommend Configuration**: {reflection_context['param_thought_summary']}
- **Configuration**: `{result_config_str}`
- **Performance**: `{perf_str}`
"""

        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset '{REACTACQ.task_id}'.

### **CURRENT TASK: Reflect on Trial Performance**

**1. Dataset & Metric Summary:**
    {dataset_summary_text}


**2. Performance Metric:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).


**3. Context: The Trial to Analyze**
Use the context below when performing your analytical reflection:
{analysis_str}


**4. YOUR TASK: ANALYTICAL REFLECTION
Your goal is to synthesize **reliable and reusable** lessons learned from this trial. Your analysis must be "quantitative, structured, and globally useful" for future trials.

- In **Reflection**, you should:
  1. **Analyze the Outcome**: Evaluate the performance of this trial relative to the basic configuration (if applicable) using the numerical performance comparisons (absolute differences and percentages) from the context.
  2. **Attribute & Synthesize**: Explain which parameter choices most likely led to the observed outcome.
     - Highlight only those hyperparameters that appear **clearly important**, and the **value ranges** that look promising or risky, and any **interactions** you can infer.
     - Make your reasoning as reusable as possible (not just describing this single trial), so that it can guide **future tuning decisions**.
     - Only include conclusions that you are **confident about** based on the provided quantitative evidence and general optimization principles.

- In **Reflection Summary**, distill your analysis into a reliable and high-confidence summary of global lessons :
  - Include **only** those insights that are:
    - Clearly supported by the numerical observations in the context, and
    - Likely to remain valid for future trials of this algorithm on this dataset.
  - Do **not** invent rules for hyperparameters where the evidence is insufficient — in those cases, **do not mention them** in the Reflection Summary.
  - Do **not** suggest explicit "next steps" or new configurations.

**5. REQUIRED OUTPUT FORMAT**

**Reflection:**
(Your structured analysis addressing the two points above, focused on quantitative comparison and attribution of the outcome.)

**Reflection Summary:**
(A compact, single paragraph summary of global, reusable and high-confidence conclusions derived from this analysis.)
"""
        return [{"role": "system", "content": cls._build_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]

    # ===== 消融实验版本（仅使用top 10 performance配置）=====
    @classmethod
    def _get_reflection_messages_ablation_top10(cls, REACTACQ, reflection_context: Dict, node_path: Dict) -> List[Dict]:
        """
        消融实验版本：仅使用Top 10 Performing Configurations（按performance排序），不使用pareto和相似度筛选
        """
        goal_direction = 'Higher is better' if not REACTACQ.lower_is_better else 'Lower is better'
        selected_algo = reflection_context['action_config']['algorithm']

        dataset_summary_text = _build_dataset_summary_text_onehot(REACTACQ.task_context, selected_algo=selected_algo)

        task_type_str = REACTACQ.task_context.get("dataset_profile", {}).get("task_type", "unknown")

        result_config_str = cls._format_config_for_llm(REACTACQ, reflection_context['action_config'])

        # 当前perf真实值
        current_perf = reflection_context['observation_perf']
        perf_str = f"{current_perf:.8g}"

        #真实指标
        algo_history = REACTACQ.param_tuning_history.get(selected_algo, [])
        history_perfs = [h["observation_perf"] for h in algo_history]

        # 默认不写排名句子
        trial_rank_text = ""

        if history_perfs:
            # 把当前 trial 一起放进列表
            perfs_with_current = history_perfs + [current_perf]

            # 根据任务方向排序：
            # - lower_is_better=True: 越小越好 => 升序
            # - lower_is_better=False: 越大越好 => 降序
            sorted_perfs = sorted(perfs_with_current, reverse=not REACTACQ.lower_is_better)

            # 找到当前 trial 的索引。为避免浮点精度问题，按"最接近值"的方式匹配
            def _find_best_index(values, target):
                best_idx = 0
                best_diff = float("inf")
                for i, v in enumerate(values):
                    d = abs(v - target)
                    if d < best_diff:
                        best_diff = d
                        best_idx = i
                return best_idx

            rank_index = _find_best_index(sorted_perfs, current_perf)
            rank = rank_index + 1
            total = len(sorted_perfs)
            best_desc = "lowest" if REACTACQ.lower_is_better else "highest"
            trial_rank_text = (
                f"{rank} out of {total} trials "
                f"(1 = {best_desc} value of the task metric among all trials of '{selected_algo}')."
            )

        if len(node_path['parent']) > 0:
            
            # 与 current_perf 同源的任务指标，真实值
            parent_raw_perf = node_path['parent'][-1].perf
            if REACTACQ.lower_is_better:
                # 内部统一"越大越好"时，loss 任务需要取反还原
                parent_task_perf = -parent_raw_perf
            else:
                parent_task_perf = parent_raw_perf

            parent_perf_str = f"{parent_task_perf:.8g}"

            basic_config_dict = KnowledgeDatabase.instance()._cfg_to_dict(node_path['parent'][-1].config)
            basic_config_str = cls._format_config_for_llm(REACTACQ, basic_config_dict)

            delta = current_perf - parent_task_perf
            if parent_task_perf == 0:
                relative_change_text = (
                    "undefined (basic configuration performance is 0, cannot compute relative percentage change safely)"
                )
            else:
                change_pct = (delta / abs(parent_task_perf)) * 100.0

                # 根据任务方向判断 improved / degraded
                if REACTACQ.lower_is_better:
                    # 越小越好：current < parent => improved
                    if current_perf < parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf > parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"
                else:
                    # 越大越好：current > parent => improved
                    if current_perf > parent_task_perf:
                        better_or_worse = "improved"
                    elif current_perf < parent_task_perf:
                        better_or_worse = "degraded"
                    else:
                        better_or_worse = "no change"

                relative_change_text = (
                    f"{better_or_worse} by {abs(delta):.6g} in the task metric "
                    f"(≈ {abs(change_pct):.2f}% change relative to the basic configuration)."
                )


            analysis_str = f"""
- **Basic Configuration to Change From:** {basic_config_str}
- **Basic Configuration Performance:** `{parent_perf_str}`
- **Original Thought to Adjust Basic Configuration**: {reflection_context['param_thought_summary']}
- **Resulting Configuration**: `{result_config_str}`
- **Resulting Performance**: `{perf_str}`
- **Change vs. Basic Configuration:** {relative_change_text}
"""
            if trial_rank_text:
                analysis_str += f"- **Current Trial's Ranking among all historical trials for '{selected_algo}':** {trial_rank_text}\n"

            # 消融实验：只使用Top 10 Performing Configurations
            history_display = cls._build_top10_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Top 10 Performing Configurations for '{selected_algo}':**
    {history_display}
"""

        else:
            # warmup阶段，没有basic configuration，直接根据历史数据进行调优，要找到和现有不太一样的，并且尽量效果好的

            analysis_str = f"""
- **Original Thought to Recommend Configuration**: {reflection_context['param_thought_summary']}
- **Configuration**: `{result_config_str}`
- **Performance**: `{perf_str}`
"""

            # WARMUP阶段：使用Top 10 Performing Configurations
            history_display = cls._build_top10_param_examples(
                REACTACQ,
                selected_algo,
                node_path,
                include_reflection=True,
                current_perf=current_perf
            )

            historical_data_str = f"""
- **Top 10 Performing Configurations for '{selected_algo}':**
    {history_display}
"""

        user_prompt = f"""
### **OVERALL MISSION: Automated Machine Learning (AutoML)**
Your ultimate goal is to solve a **Combined Algorithm Selection and Hyperparameter optimization (CASH)** problem: finding the best possible algorithm and its optimal hyperparameters for the **{task_type_str}** task on dataset '{REACTACQ.task_id}'.

### **CURRENT TASK: Reflect on Trial Performance**

**1. Dataset & Metric Summary:**
    {dataset_summary_text}


**2. Available Historical Data:**
The performance score is **{REACTACQ.task_context.get("metric", "score")}** ({goal_direction}).
{historical_data_str}


**3. Context: The Trial to Analyze**
Use the context below together with the historical data above when performing your analytical reflection:
{analysis_str}


**4. YOUR TASK: ANALYTICAL REFLECTION
Your goal is to synthesize **reliable and reusable** lessons learned from this trial. Your analysis must be "quantitative, structured, and globally useful" for future trials.

- In **Reflection**, you should:
  1. **Triage & Compare**: Position this trial among the historical data, using the numerical performance comparisons (absolute differences and percentages) from the context.
  2. **Attribute & Synthesize**: Explain which parameter deviation from the historical configurations most likely led to the observed outcome.
     - Highlight only those hyperparameters that appear **clearly important**, and the **value ranges** that look consistently promising or risky, and any **interactions** you can infer from the numbers.
     - Make your reasoning as reusable as possible (not just describing this single trial), so that it can guide **future tuning decisions**.
     - Only include conclusions that you are **confident about** based on the provided quantitative evidence.

- In **Reflection Summary**, distill your analysis into a reliable and high-confidence summary of global lessons :
  - Include **only** those insights that are:
    - Clearly supported by multiple numerical observations in the context, and
    - Likely to remain valid for future trials of this algorithm on this dataset.
  - Do **not** invent rules for hyperparameters where the data is noisy or contradictory — in those cases, **do not mention them** in the Reflection Summary.
  - Do **not** suggest explicit "next steps" or new configurations.

**5. REQUIRED OUTPUT FORMAT**

**Reflection:**
(Your structured analysis addressing the two points above, focused on quantitative comparison and attribution of the outcome.)

**Reflection Summary:**
(A compact, single paragraph summary of global, reusable and high-confidence conclusions derived from this analysis.)
"""
        return [{"role": "system", "content": cls._build_reflector_system_prompt()}, {"role": "user", "content": user_prompt}]








class BOPROPromptBuilder(PromptBuilder):


    """
    BOPRO 专用的 Prompt 构建器。
    风格：Direct Generation with Contextual Guidance (模仿 ICLR 2025 BOPRO 原文风格)
    """

    @classmethod
    def _build_system_prompt(cls) -> str:
        return """You are an expert AutoML strategist. Your goal is to solve a **Combined Algorithm Selection and Hyperparameter Optimization (CASH)** problem.
You work in collaboration with a Bayesian Optimization (BO) engine. The BO engine identifies promising regions in the search space, and your job is to propose concrete hyperparameter configurations within those regions based on provided similar historical examples."""
    
    @classmethod
    def _build_bopro_history(cls, context_cfg_df: pd.DataFrame, context_acc_df: pd.DataFrame, lower_is_better: bool) -> str:
        """
        构建符合 BOPRO 原文风格的历史展示：
        GUESS (score: ...):
        {config}
        """
        if context_cfg_df is None or context_cfg_df.empty:
            return "No similar historical configurations available."

        output_parts = [
            "Here are your top previous guesses and scores (retrieved based on similarity to the current search region):"
        ]

        records = []
        for idx in range(len(context_cfg_df)):
            cfg = context_cfg_df.iloc[idx].to_dict()
            clean_cfg = {k: v for k, v in cfg.items() if v is not None and str(v) != 'nan'}
            score = context_acc_df.iloc[idx]['score']
            records.append((clean_cfg, score))

        # BOPRO 原文通常按分数升序排列 (ascending order of accuracy)
        # 这样最好的结果在最后面，符合 LLM 的 Recency Bias
        ascending = True 
        if lower_is_better:
            # 如果越小越好，原本分数小的就是好的。
            # 我们希望“好的”排在后面。如果按数值升序，小的（好的）在前面。
            # 所以如果是 Loss，我们要降序排列（大的坏的在前，小的好在后）。
            ascending = False
        
        # 排序
        records.sort(key=lambda x: x[1], reverse=not ascending)

        for cfg, score in records:
            output_parts.append(f"GUESS (score: {score:.6f}):")
            output_parts.append(f"{json.dumps(cfg)}")
        
        return "\n".join(output_parts)

    @classmethod
    def get_generation_messages(cls, task_context: Dict, context_cfg_df: pd.DataFrame, context_acc_df: pd.DataFrame) -> List[Dict]:
        
        task_id = task_context.get("task_id", "Unknown")
        metric = task_context.get("metric", "score")
        lower_is_better = task_context.get("lower_is_better", False)
        
        # 1. 计算 Target Score (BOPRO 的 Trick)
        # 告诉 LLM：“我要你达到这个分数”。通常设为比当前历史最好还要好一点。
        best_so_far = context_acc_df['score'].max() if not context_acc_df.empty else 0.0
        if not context_acc_df.empty:
            if lower_is_better:
                best_so_far = context_acc_df['score'].min()
                # 目标是更低：减少 1% 或者 0.01
                target_score = best_so_far * 0.99 if best_so_far > 0 else best_so_far - 0.01
                goal_desc = f"{target_score:.6f}"
            else:
                best_so_far = context_acc_df['score'].max()
                # 目标是更高
                target_score = best_so_far * 1.01 if best_so_far > 0 else best_so_far + 0.01
                if best_so_far >= 1.0: target_score = 1.0
                goal_desc = f"{target_score:.6f}"
        else:
            target_score = "optimal"
            goal_desc = "the best possible value"

        # 2. 数据集与搜索空间
        dataset_summary = _build_dataset_summary_text(task_context)
        
        algo_choices = task_context.get("algorithm_choices", [])
        constraints_text = ""
        for algo in algo_choices:
            constraints_text += f"\n--- Hyperparameter Space for '{algo}' ---\n"
            constraints_text += _constraints_text_for_algo(
                algo, 
                task_context.get("algorithm_param_names", {}), 
                task_context.get("hyperparameter_constraints", {}),
                algo_choices
            )

        # 3. 历史上下文 (BOPRO Style)
        history_text = cls._build_bopro_history(context_cfg_df, context_acc_df, lower_is_better)

        # 4. 构建符合 BOPRO 逻辑的 User Prompt
        user_prompt = f"""
Given the following task definition and search space, your task is to determine the optimal hyperparameter configuration.

**1. Task Definition:**
- Dataset: {task_id}
- Metric: {metric} ({'Lower is better' if lower_is_better else 'Higher is better'})
- Dataset Summary:
{dataset_summary}

**2. Search Space:**
You must select one algorithm and valid hyperparameters from the following options:
{constraints_text}

**3. Optimization History:**
{history_text}

**4. Instruction:**
Analyze the previous guesses above to make **exactly 1 new guess** likely to give a target score of **{goal_desc}**.
- The provided guesses are semantically similar to a promising region identified by Bayesian Optimization.
- **Strictly adhere to the Search Space.** Do not invent parameters not listed above. Make sure that the configuration you propose is within the hyperparameter space defined for the selected algorithm.


**5. Output Format:**
(Note: directly start your response using the specified JSON format. Do not output any other text.)
```json
{{
    "algorithm": "...",
    "parameter": value,
    ...
}}""" 
        return [ {"role": "system", "content": cls._build_system_prompt()}, {"role": "user", "content": user_prompt} ]
    

class BORAPromptBuilder(PromptBuilder):
    """Prompt Builder with BORA-specific methods."""


    @classmethod
    def _get_bora_system_prompt(cls) -> str:
        """
        LLM Comments and Hypotheses.
        定义 BORA 特有的 JSON 输出格式，包含 Icomment 和 Hypotheses。
        """
        return """You are an expert Research Assistant (BORA) helping to optimize a machine learning pipeline (CASH task).
Your goal is to analyze the optimization history and propose new configurations (hypotheses) to improve performance.

**CRITICAL: RESPONSE FORMAT**
You must strictly output a single valid JSON object. No other text.
The JSON must follow this structure:

```json
{
    "comment": "Briefly comment on the optimization progress and highlight the most important findings.",
    "hypotheses": [
        {
            "name": "short meaningful name here",
            "rationale": "Your short summary of ratioale here...",
            "confidence": "high"/"medium"/"low",
            "config": {
                "algorithm": "...",
                "param_1": "value",
                ...
            }
        },
        ...
    ]
}"""

    @staticmethod
    def _format_comments(comments_history: List[str]) -> str:
        """
        格式化历史评论 C。
        为了防止 Context 过长，只保留最近的几条。
        """
        if not comments_history:
            return "No previous research insights recorded."
        
        # 只保留最近 3 条，避免干扰
        recent = comments_history[-3:]
        return "\n".join([f"- {c}" for c in recent])
    
    @staticmethod
    def _format_bo_candidates(bo_candidates: List[Any]) -> str:
        """
        格式化 BO 推荐的候选点 {x_BO}，供 Action a3 使用。
        格式参考 Figure 25: [suggestions]
        """
        output = []
        for i, cand in enumerate(bo_candidates):
            # 将 Configuration 对象转为字典字符串
            try:
                # 过滤掉 None (Inactive 参数) 使其更清晰
                cfg_dict = {k: v for k, v in cand.get_dictionary().items() if v is not None}
                cfg_str = json.dumps(cfg_dict)
            except:
                cfg_str = str(cand)
            output.append(f"Suggestion {i+1}: {cfg_str}")
        return "\n".join(output)
    
    @classmethod
    def _build_search_space_section(cls, task_context) -> str:
        """
        构建搜索空间描述（即 Experiment Card 的一部分）。
        复用现有的 helper 函数。
        """
        algo_choices = task_context.get("algorithm_choices", [])
        constraints_text = ""
        for algo in algo_choices:
            constraints_text += f"\n--- Hyperparameter Space for '{algo}' ---\n"
            constraints_text += _constraints_text_for_algo(
                algo, 
                task_context.get("algorithm_param_names", {}), 
                task_context.get("hyperparameter_constraints", {}),
                algo_choices
            )
        return constraints_text
    
    @classmethod
    def _build_bora_history(cls, context_cfg_df: pd.DataFrame, context_acc_df: pd.DataFrame, lower_is_better: bool) -> str:
        """
        构建符合 BORA 风格的历史展示。
        排序逻辑:Worst -> Best (让最好的排在最后，利于 ICL)。
        """
        if context_cfg_df is None or context_cfg_df.empty:
            return "No similar historical configurations available."

        output_parts = [
            "Here are your top previous guesses and scores:"
        ]

        records = []
        for idx in range(len(context_cfg_df)):
            cfg = context_cfg_df.iloc[idx].to_dict()
            # 清理 config，去除 None/NaN
            clean_cfg = {k: v for k, v in cfg.items() if v is not None and str(v) != 'nan'}
            score = context_acc_df.iloc[idx]['score']
            records.append((clean_cfg, score))

        # 排序逻辑验证无误：确保好的结果在列表底部
        ascending = True 
        if lower_is_better:
            ascending = False
        
        records.sort(key=lambda x: x[1], reverse=not ascending)
        top_records = records[:8]
        display_ascending = not lower_is_better


        final_reverse = False if not lower_is_better else True
        top_records.sort(key=lambda x: x[1], reverse=final_reverse)

        output_parts = [
            f"Here are your top {len(top_records)} performing previous configs and scores:"
        ]

        for cfg, score in top_records:
            output_parts.append(f"CONFIG (score: {score:.6f}):")
            output_parts.append(f"{json.dumps(cfg, default=str)}")
        
        return "\n".join(output_parts)
    
    @classmethod
    def get_action_a3_messages(cls, task_context, obs_cfg_df, obs_acc_df, comments_history, bo_candidates,n_select) -> List[Dict]:
        """
        Action a3 Prompt: LLM Comments and Selects n_LBO Points.
        """
        # 1. 准备基础信息 (Experiment Card 内容)
        task_id = task_context.get("task_id", "Unknown")
        metric = task_context.get("metric", "score")
        lower_is_better = task_context.get("lower_is_better", False)
        target_obj = "minimizing" if lower_is_better else "maximizing"  

        iteration_count = len(obs_acc_df)
        dataset_count = len(obs_acc_df)

        dataset_summary = _build_dataset_summary_text(task_context)
        search_space_text = cls._build_search_space_section(task_context)

        # 2. 准备历史信息 (Data D) - 复用 BOPRO 的格式化
        history_text = cls._build_bora_history(obs_cfg_df, obs_acc_df, lower_is_better)
        
        # 3. 准备评论历史 (Comments C)
        comments_text = cls._format_comments(comments_history)
        bo_suggestions_text = cls._format_bo_candidates(bo_candidates)


        user_prompt = f"""OVERALL MISSION
You are solving a Combined Algorithm Selection and Hyperparameter Optimization (CASH) problem for dataset '{task_id}'. 
Current Status: The optimization is in progress at Iteration {iteration_count}

1. EXPERIMENT CARD (Context)
Dataset Summary: {dataset_summary}

Search Space (Constraints): {search_space_text}

2. AVAILABLE INTELLIGENCE
Optimization History (Data D - {dataset_count} samples):
Top performing configurations so far: {history_text}

Previous Research Insights (Comments C): {comments_text}

3. CURRENT TASK: Select 

Using the points suggested by the optimizer below, give **{n_select} hypotheses** {target_obj} '{metric}'.
**CRITICAL CONSTRAINT:** Each hypothesis must only contain **1 unique point from the optimizer suggestions** listed below. Do not invent new points.

The Bayesian Optimizer suggestions these points to try next:
{bo_suggestions_text}

**Output: Strictly follow the JSON format defined in the system prompt and Provide a JSON object  with exactly {n_select} entries (corresponding to your selected/refined candidates).Remember the hypotheses points must be chosen from the ones recommended by the optimizer.**
""" 
        
        return [ {"role": "system", "content": cls._get_bora_system_prompt()}, {"role": "user", "content": user_prompt} ]


        
    @classmethod
    def get_action_a2_messages(cls, task_context, obs_cfg_df, obs_acc_df, comments_history,  n_generate) -> List[Dict]:
        """
        Action a3 Prompt: LLM Suggests.
        """
        # 1. 准备基础信息
        task_id = task_context.get("task_id", "Unknown")
        metric = task_context.get("metric", "score")
        lower_is_better = task_context.get("lower_is_better", False)
        target_obj = "Minimize" if task_context.get("lower_is_better") else "Maximize"
        dataset_summary = _build_dataset_summary_text(task_context)
        search_space_text = cls._build_search_space_section(task_context)

        # 2. 准备历史和评论
        history_text = cls._build_bora_history(obs_cfg_df, obs_acc_df, lower_is_better)
        comments_text = cls._format_comments(comments_history)

        iteration_count = len(obs_acc_df)
        dataset_count = len(obs_acc_df)
        

        user_prompt = f"""OVERALL MISSION
You are solving a Combined Algorithm Selection and Hyperparameter Optimization (CASH) problem for dataset '{task_id}'. 
Current Status: The optimization is in progress at Iteration {iteration_count}.
1. EXPERIMENT CARD (Context)
Dataset Summary: {dataset_summary}

Search Space (Constraints): You must select one algorithm and its parameters from: {search_space_text}

2. AVAILABLE INTELLIGENCE
Optimization History (Data D - {dataset_count} samples): 
The following are the top performing configurations found so far: 
{history_text}

Previous Research Insights (Comments C): 
Here are your previous thoughts on the optimization: {comments_text}

3. CURRENT TASK: Suggest Hypotheses (Action a2)
The optimization process is currently experiencing High Uncertainty or stagnation. I need you to take over and propose {n_generate} new, diverse hypotheses (configurations) to explore the search space.

Instructions:

- Reflect on your previous hypotheses and the entire data.
- Analyze Subspaces: 
    - What are the subspaces/parameters that seem to **maximize** performance?
    - What are the subspaces that seem detrimental (poor performance)?
- Improve & Discard: Improve promising previous hypotheses or discard those that consistently performed poorly.
- Generate: Propose {n_generate} valid configurations based on this reasoning.

**Output: Strictly follow the JSON format defined in the system prompt.Provide a JSON object  with exactly {n_generate} entries.""" 

        
        return [ {"role": "system", "content": cls._get_bora_system_prompt()}, {"role": "user", "content": user_prompt} ]

# class OPROPromptBuilder(BOPROPromptBuilder):
#     """
#     OPRO (Optimization by PROmpting) 专用 Prompt 构建器。
#     Based on Google DeepMind's "Large Language Models as Optimizers".
    
#     Difference from BOPRO:
#     - No BO guidance context.
#     - History is explicitly presented as an "Optimization Trajectory" sorted by score.
#     """

#     @classmethod
#     def _build_system_prompt(cls) -> str:
#         """
#         [cite: 31, 108] LLM as Optimizer.
#         """
#         return """You are an expert AutoML strategist acting as an **Optimization Algorithm**.
# Your goal is to iteratively generate new hyperparameter configurations that maximize the performance metric on a given dataset.
# You should analyze the provided **Optimization Trajectory** (past solutions and their scores) to identify patterns and propose better solutions."""

#     @classmethod
#     def _build_opro_history(cls, context_cfg_df: pd.DataFrame, context_acc_df: pd.DataFrame, lower_is_better: bool) -> str:
#         """
#         构建 Optimization Trajectory。
#         [修正] 仅保留历史中最好的 Top-K 个配置，并按“质量由低到高”排序展示（最好的在最后）。
#         Reference: OPRO Paper Section 3.1 & 4.2[cite: 155, 268].
#         """
#         if context_cfg_df is None or context_cfg_df.empty:
#             return "No previous trials available. This is the first step."

#         records = []
#         for idx in range(len(context_cfg_df)):
#             cfg = context_cfg_df.iloc[idx].to_dict()
#             clean_cfg = {k: v for k, v in cfg.items() if v is not None and str(v) != 'nan'}
#             score = context_acc_df.iloc[idx]['score']
#             records.append((clean_cfg, score))


#         if lower_is_better:
#             records.sort(key=lambda x: x[1], reverse=False)
#         else:
#             records.sort(key=lambda x: x[1], reverse=True)
            
#         top_records = records[:20]

        
#         if lower_is_better:
#             top_records.sort(key=lambda x: x[1], reverse=True)
#         else:

#             top_records.sort(key=lambda x: x[1], reverse=False)

#         output_parts = [
#             f"Here are the top {len(top_records)} past configurations and their scores, arranged in ascending order of quality (Best solutions are at the bottom):"
#         ]

#         for cfg, score in top_records:
#             output_parts.append(f"CONFIG (score: {score:.6f}):")
#             output_parts.append(f"{json.dumps(cfg)}")
        
#         return "\n".join(output_parts)

#     @classmethod
#     def get_generation_messages(cls, task_context: Dict, context_cfg_df: pd.DataFrame, context_acc_df: pd.DataFrame, n_generate: int = 1) -> List[Dict]:
#         """
#         构建 OPRO 的 Generation Prompt。
#         [cite: 127] Meta-Prompt Design: Problem Description + Optimization Trajectory.
#         """
#         task_id = task_context.get("task_id", "Unknown")
#         metric = task_context.get("metric", "score")
#         lower_is_better = task_context.get("lower_is_better", False)
        
#         # 1. 目标描述
#         goal_direction = "minimize" if lower_is_better else "maximize"
        
#         # 2. 上下文构建
#         dataset_summary = _build_dataset_summary_text(task_context)
        
#         algo_choices = task_context.get("algorithm_choices", [])
#         constraints_text = ""
#         for algo in algo_choices:
#             constraints_text += f"\n--- Hyperparameter Space for '{algo}' ---\n"
#             constraints_text += _constraints_text_for_algo(
#                 algo, 
#                 task_context.get("algorithm_param_names", {}), 
#                 task_context.get("hyperparameter_constraints", {}),
#                 algo_choices
#             )

#         # 3. 历史轨迹 (Optimization Trajectory)
#         history_text = cls._build_opro_history(context_cfg_df, context_acc_df, lower_is_better)

#         # 4. User Prompt [cite: 243, 1055]
#         # 参考论文话术: "Give me a new (w, b) pair that is different from all pairs above, and has a function value lower than any of the above."
        
#         user_prompt = f"""
# You are solving a **Combined Algorithm Selection and Hyperparameter Optimization (CASH)** problem.

# **1. Task Definition:**
# - Dataset: {task_id}
# - Metric: {metric} (Goal: **{goal_direction}**)
# - Dataset Summary:
# {dataset_summary}

# **2. Search Space (Constraints):**
# You must select one algorithm and valid hyperparameters from:
# {constraints_text}

# **3. Optimization Trajectory:**
# {history_text}

# **4. Your Task:**
# Based on the patterns in the optimization trajectory above, generate **{n_generate} new configuration(s)** that are different from the previous ones and push the performance metric further towards the goal ({goal_direction}).
# - **Identify patterns**: Look at which algorithms and parameters lead to better scores at the bottom of the list.
# - **Exploration & Exploitation**: You can refine the best configurations or explore promising but untried regions.
# - **Constraints**: Strictly adhere to the parameter ranges defined in the Search Space.

# **5. Output Format:**
# Strictly output a JSON object containing a list of configurations under the key "configs".
# ```json
# {{
#     "configs": [
#         {{
#             "algorithm": "...",
#             "param_1": value,
#             ...
#         }},
#         ...
#     ]
# }}
# """ 
#         return [ {"role": "system", "content": cls._build_system_prompt()}, {"role": "user", "content": user_prompt} ]



class OPROPromptBuilder(PromptBuilder):
    """
    OPRO (Optimization by PROmpting) Prompt Builder.
    Strictly follows the "Simple and Effective" philosophy of the Google DeepMind paper.
    
    Structure based on OPRO Figure 2 & Figure 19:
    1. Problem Description (Context + Search Space)
    2. Optimization Trajectory (History sorted worst -> best)
    3. Meta-Instruction (Generate new, valid config)
    """

    @classmethod
    def _build_system_prompt(cls) -> str:
        """
        OPRO 风格：直接、功能性。
        """
        return "You are an optimization assistant. Your task is to find the best hyperparameter configuration for a machine learning pipeline given the optimization history."

    @classmethod
    def _build_opro_history(cls, context_cfg_df: pd.DataFrame, context_acc_df: pd.DataFrame, lower_is_better: bool, top_k: int = 8) -> str:
        """
        构建 Optimization Trajectory。
        核心逻辑：Sort in ascending order of quality (Worst -> Best).
        [cite: 136] "sorted in the ascending order... encouraging the LLM to build upon existing good solutions"
        """
        if context_cfg_df is None or context_cfg_df.empty:
            return ""

        # 1. 收集
        records = []
        for idx in range(len(context_cfg_df)):
            cfg = context_cfg_df.iloc[idx].to_dict()
            clean_cfg = {k: v for k, v in cfg.items() if v is not None and str(v) != 'nan'}
            score = context_acc_df.iloc[idx]['score']
            records.append((clean_cfg, score))

        # 2. 筛选 Top-K (先选出最好的 K 个)
        # lower_is_better=True (Loss): 小的好。按升序排，取头部。
        # lower_is_better=False (Acc): 大的好。按降序排，取头部。
        if lower_is_better:
            records.sort(key=lambda x: x[1], reverse=False)
        else:
            records.sort(key=lambda x: x[1], reverse=True)
        
        top_records = records[:top_k]

        # 3. 展示排序 (Worst -> Best)
        # [cite: 1047] "The pairs are arranged in descending order based on their function values, where lower values are better."
        # 这句话有点绕，但核心目的是：让最好的结果出现在 Prompt 的最末尾。
        
        if lower_is_better:
            # Loss: 我们希望最后一行是 0.01 (最小)。所以是降序排列 (0.5 -> 0.1)。
            top_records.sort(key=lambda x: x[1], reverse=True)
        else:
            # Acc: 我们希望最后一行是 0.99 (最大)。所以是升序排列 (0.6 -> 0.9)。
            top_records.sort(key=lambda x: x[1], reverse=False)

        # 4. 格式化
        output_parts = [
            "Below are some previous configurations and their scores, arranged so that the best configurations are at the bottom:"
        ]
        for cfg, score in top_records:
            # 保持 JSON 格式以便 LLM 模仿
            output_parts.append(f"score: {score:.6f}\nconfig: {json.dumps(cfg)}")
        
        return "\n\n".join(output_parts)

    @classmethod
    def get_generation_messages(cls, task_context: Dict, context_cfg_df: pd.DataFrame, context_acc_df: pd.DataFrame, n_generate: int = 1) -> List[Dict]:
        """
        构建 OPRO Meta-Prompt。
        """
        task_id = task_context.get("task_id", "Unknown")
        metric = task_context.get("metric", "score")
        lower_is_better = task_context.get("lower_is_better", False)
        
        goal_verb = "minimize" if lower_is_better else "maximize"
        quality_desc = "lower" if lower_is_better else "higher"
        
        # 1. 搜索空间 (The Search Space)
        algo_choices = task_context.get("algorithm_choices", [])
        constraints_text = ""
        for algo in algo_choices:
            constraints_text += f"\n[Algorithm: {algo}]\n"
            constraints_text += _constraints_text_for_algo(
                algo, 
                task_context.get("algorithm_param_names", {}), 
                task_context.get("hyperparameter_constraints", {}),
                algo_choices
            )

        # 2. 历史轨迹 (Optimization Trajectory)
        history_text = cls._build_opro_history(context_cfg_df, context_acc_df, lower_is_better)
        
        # 3. 数据集描述 (Context)
        dataset_summary = _build_dataset_summary_text(task_context)

        # 4. User Prompt (模仿 OPRO 论文 Figure 19 和 Figure 22 的朴素风格)
        user_prompt = f"""
I need your help to **{goal_verb}** the metric '{metric}' for a machine learning task on dataset '{task_id}'.

**Task Context:**
{dataset_summary}

**Search Space (Constraints):**
You must choose exactly one algorithm from the list below and set its hyperparameters.
{constraints_text}

**Optimization Trajectory:**
{history_text}

**Your Task:**
Generate **{n_generate} new configuration(s)** that are different from all configurations above, and have a score **{quality_desc}** than any of the above.

**CRITICAL CONSTRAINTS:**
1.  **Format**: Output strictly a valid JSON object.
2.  **NO EXPLANATION**: **DO NOT** output any reasoning, "Key rationale", "Based on...", introductory text, or concluding remarks. I only want the JSON.
3.  **Start Immediately**: Start your response directly with ```json.
4.  **Validity**: The configuration MUST consist of:
    -   Key "algorithm": One string chosen from the headers above (e.g., "lightgbm").
    -   Hyperparameters: Only parameters belonging to that specific algorithm.
    -   Values: Must strictly respect the types (int/float/str) and ranges defined in the Search Space.

**Output Format:**
```json
{{
    "configs": [
        {{
            "algorithm": "...",
            "param_name": value,
            ...
        }}
    ]
}}"""
        return [ {"role": "system", "content": cls._build_system_prompt()}, {"role": "user", "content": user_prompt} ]
    

class LLAMBOPromptBuilder(PromptBuilder):
    """
    LLAMBO Prompt Builder.
    Implements the specific Data Card, Model Card, and dual-direction history formatting 
    (Inverse for Sampling, Forward for Surrogate) as described in the LLAMBO paper.
    """

    @classmethod
    def _build_system_prompt(cls) -> str:
        return "You are an expert machine learning researcher and optimizer."

    # @classmethod
    # def _build_data_card(cls, task_context: Dict) -> str:
    #     """
    #     [cite: 2178-2180] Data Card Construction.
    #     Format: "The model is evaluated on a tabular {task} task containing {classes} classes..."
    #     """
    #     profile = task_context.get("dataset_profile", {})
    #     task_type = profile.get("task_type", "classification")
    #     n_samples = profile.get("n_samples", "unknown")
    #     n_features = profile.get("n_features", "unknown")
    #     n_cat = profile.get("n_categorical", 0)
    #     n_num = profile.get("n_numeric", 0)
    #     # 尝试获取类别数，回归任务可能没有
    #     classes_info = f" containing {profile.get('n_classes', 'unknown')} classes" if task_type == 'classification' else ""

    #     return (
    #         f"The model is evaluated on a tabular {task_type} task{classes_info}. "
    #         f"The tabular dataset contains {n_samples} samples and {n_features} features "
    #         f"({n_cat} categorical, {n_num} numerical)."
    #     )

    @classmethod
    def _build_model_card(cls, task_context: Dict) -> str:
        """
        [cite: 2200] Model Card Construction.
        Format: "The allowable ranges for the hyperparameters are: ..."
        """
        algo_choices = task_context.get("algorithm_choices", [])
        constraints_text = ""
        for algo in algo_choices:
            constraints_text += f"\n[Algorithm: {algo}]\n"
            constraints_text += _constraints_text_for_algo(
                algo, 
                task_context.get("algorithm_param_names", {}), 
                task_context.get("hyperparameter_constraints", {}),
                algo_choices
            )
        return f"The allowable ranges for the hyperparameters are:\n{constraints_text}"

    @classmethod
    def _format_history_records(cls, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, n_bins: int = 8, shuffle: bool = False, metric_lower_is_better: bool = False):
        """
        [核心修改] 分层采样 (Stratified Sampling)。
        将历史按分数排序后切分为 n_bins 层，从每层随机选取 1 个，以展示从差到好的完整分布。
        """
        if obs_cfg_df is None or obs_cfg_df.empty:
            return []

        # 1. 收集所有历史
        records = []
        for idx in range(len(obs_cfg_df)):
            cfg = obs_cfg_df.iloc[idx].to_dict()
            clean_cfg = {k: v for k, v in cfg.items() if v is not None and str(v) != 'nan'}
            score = obs_acc_df.iloc[idx]['score']
            records.append((clean_cfg, score))

        # 2. 统一按数值升序排序 (方便分层)
        records.sort(key=lambda x: x[1])

        selected_records = []
        total_records = len(records)

        # 3. 分层抽取
        if total_records <= n_bins:
            selected_records = records
        else:
            k, m = divmod(total_records, n_bins)
            current_idx = 0
            for i in range(n_bins):
                chunk_size = k + 1 if i < m else k
                chunk = records[current_idx : current_idx + chunk_size]
                if chunk:
                    selected_records.append(random.choice(chunk))
                current_idx += chunk_size

        # 4. 最终展示顺序调整
        if shuffle:
            # Surrogate 需要 Shuffle 来模拟不确定性
            random.shuffle(selected_records)
        else:
            # Sampler 需要有规律的顺序 (Worst -> Best) 以利用 ICL
            if metric_lower_is_better:
                # Loss: 降序 (0.9 -> 0.1)
                selected_records.sort(key=lambda x: x[1], reverse=True)
            else:
                # Accuracy: 升序 (0.1 -> 0.9)
                selected_records.sort(key=lambda x: x[1], reverse=False)
        
        return selected_records
    
    @classmethod
    def get_sampler_messages(cls, task_context: Dict, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, target_score: float, n_generate: int = 8) -> List[Dict]:
        """
        Candidate Sampler (Inverse: Perf -> Config).
        展示 8 层分层历史，要求生成达到 target_score 的配置。
        """
        dataset_summary = _build_dataset_summary_text(task_context)
        metric = task_context.get("metric", "score")

        algo_choices = task_context.get("algorithm_choices", [])
        constraints_parts = []
        for algo in algo_choices:
            constraints_parts.append(f"[Algorithm: {algo}]")
            constraints_parts.append(_constraints_text_for_algo(
                algo, 
                task_context.get("algorithm_param_names", {}), 
                task_context.get("hyperparameter_constraints", {}),
                algo_choices
            ))
        constraints_text = "\n".join(constraints_parts)

        # 2. History (Stratified 8 bins, No Shuffle, Worst -> Best)
        records = cls._format_history_records(
            obs_cfg_df, obs_acc_df, n_bins=8, shuffle=False, 
            metric_lower_is_better=task_context.get("lower_is_better", False)
        )
        
        # 优化：构建 History (Sampler 逻辑: 先 Performance, 后 Config)
        history_parts = []
        for cfg, score in records:
            history_parts.append(f"Performance: {score:.6f}\nHyperparameter configuration: {json.dumps(cfg)}")
        # 使用双换行分隔每条记录，保持清晰
        history_text = "\n\n".join(history_parts)

        # 3. User Prompt
        # 修正：移除 f-string 内部的缩进，防止多余的空格 Token
        user_prompt = f"""1. Data Card: {dataset_summary}

2. Model Card: {constraints_text}

3. Observations (Performance -> Configuration): 
The following are examples of the performance of the model measured in '{metric}' and the corresponding configurations.
{history_text}

4. Task: Recommend {n_generate} configurations that can achieve the target performance of {target_score:.6f}.
Use the insights from the Observations to infer which algorithms and parameters lead to high performance.
Strictly adhere to the parameter ranges in the Model Card.
Focus on the most promising algorithms found in the Observations. Do NOT distribute guesses evenly across all algorithms; allocate more guesses to algorithms that have performed well.

Output Format:
Strictly output a single JSON object containing the configurations.

**CRITICAL CONSTRAINTS:**
1. Output **ONLY** the JSON block. 
2. **DO NOT** output any reasoning, "Key patterns", summary, or text before or after the JSON.
3. Start your response directly with ```json.

```json
{{
    "configs": [
        {{
            "algorithm": "...",
            ...
        }}
    ]
}}
```""" 
        return [ {"role": "system", "content": cls._build_system_prompt()}, {"role": "user", "content": user_prompt} ]
    @classmethod
    def get_surrogate_messages(cls, task_context: Dict, obs_cfg_df: pd.DataFrame, obs_acc_df: pd.DataFrame, candidate_configs: List[Dict], shuffle: bool = True) -> List[Dict]:
        """
        [cite_start][cite: 2182-2185] Discriminative Surrogate (Forward: Config -> Perf).
        展示 8 层分层历史，支持 Shuffle 以进行不确定性估计。
        """

        dataset_summary = _build_dataset_summary_text(task_context)
        metric = task_context.get("metric", "score")

        records = cls._format_history_records(
            obs_cfg_df, obs_acc_df, n_bins=8, shuffle=shuffle, 
            metric_lower_is_better=task_context.get("lower_is_better", False)
        )

        history_parts = []
        for cfg, score in records:
            history_parts.append(f"Hyperparameter configuration: {json.dumps(cfg)}\nPerformance: {score:.6f}")
        history_text = "\n\n".join(history_parts)

        query_parts = []
        for i, cfg in enumerate(candidate_configs):
            query_parts.append(f"Hyperparameter configuration {i+1}: {json.dumps(cfg)}")
        query_text = "\n".join(query_parts)


        user_prompt = f"""Data Card: {dataset_summary}

Observations (Configuration -> Performance): 
The following are examples of the performance of the model measured in '{metric}'. 
{history_text}

Task: 
Based on the observations above, predict the performance ('{metric}') for the following {len(candidate_configs)} candidate configurations.
{query_text}

Output Format: 
Strictly output a JSON object with a list of float scores corresponding to the order above.
**CRITICAL:** 1. Do NOT output any reasoning, thinking process, or introductory text.
2. Start your response directly with ```json.
3. Output ONLY the JSON block.

```json
{{
    "predictions": [0.123, 0.456, ...]
}}
```"""
        return [ {"role": "system", "content": cls._build_system_prompt()}, {"role": "user", "content": user_prompt} ]