import os
import time
import openai
import asyncio
import numpy as np
import pandas as pd
import json
from aiohttp import ClientSession
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from mindware.third_party.opro.rate_limiter import RateLimiter
from typing import Any, Dict, List, Tuple

from .utils import (
    _serialize_row,
    _prepare_configurations_acquisition,
    _constraints_text,
    _dataset_summary_block,
    _convert_to_dict,
    _filter_and_validate,
)

# Set up OpenAI API credentials from environment variables.
openai.api_type = os.environ.get("OPENAI_API_TYPE", "")
openai.api_version = os.environ.get("OPENAI_API_VERSION", "")
openai.api_base = os.environ.get("OPENAI_API_BASE", "")
openai.api_key = os.environ.get("OPENAI_API_KEY", "")


class BaselineLLM_ACQ:
    """
    Handles all prompt engineering and interaction with the LLM API.

    This class is the "brain" of the OPRO process. It constructs sophisticated
    prompts based on the optimization history, dataset profile, and search space
    constraints. It then queries the LLM to get new candidate configurations.
    """
    def __init__(
        self,
        task_context,
        n_candidates,
        n_templates,
        lower_is_better,
        jitter=False,
        rate_limiter=None,
        warping_transformer=None,
        chat_engine=None,
        prompt_setting=None,
        shuffle_features=False,
        # Parameters for controlling the few-shot example selection strategy.
        history_budget: int = 30,
        top_k: int = 15,
        worst_k: int = 3,
        recent_k: int = 8,
        per_algo_cap: int = 10,
        interleave_order: tuple = ("best", "recent", "worst"),
    ):
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = max(1, int(n_candidates / max(1, n_templates)))
        self.lower_is_better = lower_is_better
        self.apply_jitter = jitter
        self.rate_limiter = rate_limiter or RateLimiter(max_tokens=800000000, time_frame=300)
        self.warping_transformer = warping_transformer
        self.apply_warping = warping_transformer is not None
        self.chat_engine = chat_engine
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        # Unpack schema information from the task context.
        self.constraints: Dict[str, Dict[str, Any]] = task_context.get("hyperparameter_constraints", {})
        self.algo_choices: List[str] = task_context.get("algorithm_choices", [])
        self.algo2params: Dict[str, List[str]] = task_context.get("algorithm_param_names", {})

        # --- Few-shot example selection strategy ---
        self.history_budget = history_budget
        self.top_k = top_k
        self.worst_k = worst_k
        self.recent_k = recent_k
        self.per_algo_cap = per_algo_cap
        self.interleave_order = interleave_order

        # Internal state for tracking optimization progress.
        self.observed_best = None
        self.observed_worst = None
        self.alpha = 0.2
        
        # Create a dedicated directory for saving LLM conversation logs.
        self.output_dir = self.task_context.get("output_dir", "./")
        print("[LLM_ACQ] Conversation logs will be saved to:", self.output_dir)
        self.conversation_log_dir = os.path.join(self.output_dir, "llm_conversations")
        os.makedirs(self.conversation_log_dir, exist_ok=True)

    # =============== Prompt Construction Methods ===============

    def _build_system_prompt(self) -> str:
        """Builds the base system prompt that defines the LLM's role and goal."""
        metric = self.task_context.get("metric", "score")
        direction = "minimize" if self.lower_is_better else "maximize"
        algo_choices_txt = (
            f" Choose one algorithm from {self.algo_choices} and set ONLY its parameters."
            if self.algo_choices else
            " Choose a valid algorithm and set ONLY its parameters."
        )
        return (
            "You are a model selection and hyperparameter optimization assistant. "
            f"Your job is to {direction} the evaluation metric '{metric}' by proposing a valid algorithm "
            f"and its hyperparameters. {algo_choices_txt} "
            "Always follow the search-space constraints and output exactly in the requested format."
        )

    def _build_exploration_prompt(self) -> str:
        """Builds a system prompt that encourages exploration of diverse configurations."""
        metric = self.task_context.get("metric", "score")
        direction = "minimize" if self.lower_is_better else "maximize"
        algo_choices_txt = (
            f" Choose one algorithm from {self.algo_choices} and set ONLY its parameters."
            if self.algo_choices else
            " Choose a valid algorithm and set ONLY its parameters."
        )
        return (
            "You are a model selection and hyperparameter optimization assistant. "
            f"Your job is to {direction} the evaluation metric '{metric}' by proposing a valid algorithm "
            f"and its hyperparameters. {algo_choices_txt} "
            "During this early stage of optimization, we encourage a wider exploration of the search space."
            " Propose configurations that are diverse and innovative, rather than trying to refine the current solutions."
            " Always follow the search-space constraints and output exactly in the requested format."
        )
    
    def _build_exploitation_prompt(self) -> str:
        """Builds a system prompt that encourages exploitation of high-performing regions."""
        metric = self.task_context.get("metric", "score")
        direction = "minimize" if self.lower_is_better else "maximize"
        algo_choices_txt = (
            f" Choose one algorithm from {self.algo_choices} and set ONLY its parameters."
            if self.algo_choices else
            " Choose a valid algorithm and set ONLY its parameters."
        )
        return (
            "You are a model selection and hyperparameter optimization assistant. "
            f"Your job is to {direction} the evaluation metric '{metric}' by proposing a valid algorithm "
            f"and its hyperparameters. {algo_choices_txt} "
            "Based on the history of previous trials, please focus on configurations that have been successful so far."
            " Consider refining these solutions and propose configurations that are likely to yield better results."
            " Always follow the search-space constraints and output exactly in the requested format."
        )
    
    def _build_dataset_summary_sentences(self) -> str:
        """
        Creates a natural language description of the dataset and task from profile data.
        This makes the LLM data-aware.
        """
        prof = self.task_context.get("dataset_profile", {}) or {}
        metric = self.task_context.get("metric", "score")

        task_type = prof.get("task_type", "unknown task")
        parts = [f"This is a {task_type} problem. The optimization metric is '{metric}'."]
        
        n_samples = prof.get("n_samples")
        n_features = prof.get("n_features")
        n_num = prof.get("n_numeric")
        n_cat = prof.get("n_categorical")
        if n_samples is not None and n_features is not None and n_num is not None and n_cat is not None:
            parts.append(
                f"The dataset contains {n_samples} samples and {n_features} features "
                f"({n_num} numeric and {n_cat} categorical)."
            )
        elif n_samples is not None and n_features is not None:
            parts.append(f"The dataset contains {n_samples} samples and {n_features} features.")

        mr = prof.get("missing_rate", 0.0)
        if isinstance(mr, (int, float)) and mr > 0:
            parts.append(f"There is a non-zero missing rate of {mr:.4f}.")

        if prof.get("task_type") == "classification":
            cd = prof.get("class_distribution", {})
            dist = cd.get("distribution", {})
            if dist:
                dist_str = ", ".join([f"{str(k)}: {float(v):.3f}" for k, v in dist.items()])
                parts.append(f"Class distribution is: {dist_str}.")

        if prof.get("task_type") == "regression":
            ts = prof.get("target_stats", {})
            stats_frag = []
            for k in ["mean", "std", "min", "median", "max"]:
                if k in ts:
                    stats_frag.append(f"{k}={float(ts[k]):.4f}")
            if stats_frag:
                parts.append("Target statistics: " + ", ".join(stats_frag) + ".")

        return " ".join(parts)

    def _select_labeled_examples(self, observed_configs: pd.DataFrame, observed_fvals: pd.DataFrame):
        """
        Selects, labels, and formats historical configurations for few-shot prompting.

        This method implements the core logic for the few-shot selection strategy,
        picking the best, worst, and most recent examples based on `__init__` parameters.
        """
        if observed_configs is None or observed_fvals is None or len(observed_configs) == 0:
            return []
        df = observed_configs.copy()
        df["_score"] = observed_fvals.values.astype(float).ravel()
        ascending = True if self.lower_is_better else False
        df_sorted = df.sort_values("_score", ascending=ascending)
        
        # Select indices for each category (best, worst, recent).
        best_idx = list(df_sorted.index[: self.top_k])
        worst_idx = list(df_sorted.index[-self.worst_k:]) if self.worst_k > 0 else []
        recent_idx = list(df.index[-self.recent_k:]) if self.recent_k > 0 else []
        
        # Merge and deduplicate indices based on the specified order.
        buckets = {"best": best_idx, "recent": recent_idx, "worst": worst_idx}
        merged = []
        for key in self.interleave_order:
            for i in buckets.get(key, []):
                if i not in merged:
                    merged.append(i)
        
        # Apply per-algorithm caps to ensure diversity.
        algo_col = "algorithm"
        algo_counts: Dict[str, int] = {}
        selected = []
        for i in merged:
            algo = df.loc[i, algo_col] if algo_col in df.columns else None
            if algo is None:
                for c in df.columns:
                    if ":" in c and not pd.isna(df.loc[i, c]):
                        algo = c.split(":", 1)[0]
                        break
            if algo is None:
                selected.append(i)
                continue
            cnt = algo_counts.get(algo, 0)
            if cnt < self.per_algo_cap:
                selected.append(i)
                algo_counts[algo] = cnt + 1

        # Enforce the total history budget.
        if len(selected) > self.history_budget:
            selected = selected[: self.history_budget]

        # Format the selected examples with labels (BEST, RECENT, WORST).
        examples = []
        best_set, recent_set, worst_set = set(best_idx), set(recent_idx), set(worst_idx)
        for i in selected:
            row = observed_configs.loc[i]
            A = float(observed_fvals.loc[i].values[0])
            if i in best_set:
                label = "BEST"
            elif i in recent_set:
                label = "RECENT"
            elif i in worst_set:
                label = "WORST"
            else:
                label = "EXAMPLE"
            examples.append({
                "Label": label,
                "Q": _serialize_row(self, row),
                "A": f"{A:.6f}",
            })
        return examples

    def _gen_prompt_tempates_acquisitions(
        self, observed_configs, observed_fvals, desired_fval, n_prompts=1, use_context="full_context", iter_idx=None
    ):
        """
        (Deprecated) An older method for generating a complete LangChain prompt template.
        The current implementation uses a more direct two-stage approach.
        """
        all_prompt_templates, all_query_templates = [], []

        # Switch between exploration and exploitation prompts based on iteration index.
        if iter_idx < 25:
            system_goal = self._build_exploration_prompt()
        else:
            system_goal = self._build_exploitation_prompt()

        for i in range(n_prompts):
            few_shot_examples = self._select_labeled_examples(observed_configs, observed_fvals)

            example_template = (
                "{Label}\n"
                "Performance: {A}\n"
                "Hyperparameter configuration: {Q}"
            )
            example_prompt = PromptTemplate(
                input_variables=["Q", "A", "Label"],
                template=example_template,
            )

            data_summary = self._build_dataset_summary_sentences() if self.task_context.get(
                "add_dataset_summary_to_prompt", True
            ) else ""
            constraints_text = _constraints_text(self)

            history_intro = (
                "The following are historical examples collected on this same dataset and task. "
                "Each block shows a past trial's evaluation Performance and the corresponding model hyperparameter configuration."
            )
            label_guidance = (
                "Examples are labeled as follows:\n"
                "- BEST: top-performing configurations; explore around these regions.\n"
                "- RECENT: most recent trials; they indicate the current search trend.\n"
                "- WORST: poor configurations; generally avoid these directions.\n"
            )

            metric = self.task_context.get("metric", "score")
            direction = "lower is better" if self.lower_is_better else "higher is better"
            prefix_parts = [
                f"{system_goal}\n",
                (f"Dataset & Task:\n{data_summary}\n" if data_summary else ""),
                "Search space constraints:\n" + constraints_text + "\n",
                history_intro + "\n",
                label_guidance,
                f"Target: propose one configuration expected to achieve {desired_fval:.6f} {metric} "
                f"({direction}).\n"
                "Do not use boundary values. Use the highest precision allowed by the ranges.\n"
                "Do not copy any example verbatim; propose an improved configuration.\n"
                "Your response must ONLY contain the predicted configuration in the exact format:\n"
                "## algorithm: <name>, <algo>:<param>: <value>, <algo>:<param>: <value>, ... ##",
            ]
            prefix = "\n".join([p for p in prefix_parts if p])
            suffix = "\nPerformance: {A}\nHyperparameter configuration:"

            few_shot_prompt = FewShotPromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["A"],
                example_separator="\n\n",
            )
            all_prompt_templates.append(few_shot_prompt)
            all_query_templates.append([{"A": f"{desired_fval:.6f}"}])

        return all_prompt_templates, all_query_templates

    # =============== LLM API Interaction ===============
    async def _async_generate_with_context(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.8,
        max_tokens: int = 500,
        n: int = 1
    ):
        """
        Asynchronously calls the OpenAI ChatCompletion API with rate limiting and retries.
        """
        MAX_RETRIES = 3
        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)
            resp = None
            for retry in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                    self.rate_limiter.add_request(request_text=last_user_msg, current_time=start_time)
                    
                    resp = await openai.ChatCompletion.acreate(
                        model=self.chat_engine,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=0.95,
                        n=n,
                        request_timeout=30,
                    )
                    
                    self.rate_limiter.add_request(
                        request_token_count=resp["usage"]["total_tokens"],
                        current_time=start_time
                    )
                    break
                except Exception as e:
                    print(f"[AF] RETRYING {retry+1}/{MAX_RETRIES}...", e)
                    await openai.aiosession.get().close()
            
            if resp is None:
                return None, 0.0, 0
            
            # Simplified cost calculation for GPT-4 series.
            tot_cost = 0.0015 * (resp["usage"]["prompt_tokens"] / 1000) + \
                       0.002 * (resp["usage"]["completion_tokens"] / 1000)
            return resp, tot_cost, resp["usage"]["total_tokens"]

    def _build_algo_selection_examples(self, observed_configs, observed_fvals):
        """Formats the selected few-shot examples for the algorithm selection stage."""
        examples = self._select_labeled_examples(observed_configs, observed_fvals)
        formatted = []
        for ex in examples:
            cfg_text = ex["Q"]
            algo_match = cfg_text.split("algorithm: ")[1].split(",")[0].strip() if "algorithm: " in cfg_text else "unknown"
            formatted.append({
                "Label": ex["Label"],
                "Algorithm": algo_match,
                "Performance": ex["A"],
                "Config": cfg_text
            })
        return formatted

    def _build_ranked_param_optimization_examples(self, observed_configs, observed_fvals, selected_algo):
        """
        Builds few-shot examples for the parameter optimization stage.

        It intelligently shows either separate "good" and "poor" examples or a single
        ranked list, depending on the diversity of the available history for the chosen algorithm.
        """
        if observed_configs is None or observed_fvals is None or len(observed_configs) < 1:
            return ""

        full_history = observed_configs.copy()
        full_history["_score"] = observed_fvals.values.astype(float).ravel()
        ascending = self.lower_is_better
        full_history_sorted = full_history.sort_values("_score", ascending=ascending)

        # Define thresholds for "good" (top 20%) and "poor" (bottom 30%) performance.
        total_trials = len(full_history_sorted)
        top_20_percent_index = int(total_trials * 0.2)
        bottom_30_percent_index = int(total_trials * 0.7)
        top_threshold = full_history_sorted.iloc[min(top_20_percent_index, total_trials - 1)]["_score"]
        bottom_threshold = full_history_sorted.iloc[min(bottom_30_percent_index, total_trials - 1)]["_score"]

        algo_history = full_history[full_history['algorithm'] == selected_algo]
        if algo_history.empty:
            return ""

        if self.lower_is_better:
            good_examples_df = algo_history[algo_history["_score"] <= top_threshold]
            bad_examples_df = algo_history[algo_history["_score"] >= bottom_threshold]
        else: # Higher is better
            good_examples_df = algo_history[algo_history["_score"] >= top_threshold]
            bad_examples_df = algo_history[algo_history["_score"] <= bottom_threshold]

        output_parts = []
        
        # Conditionally format the examples based on data availability.
        if len(good_examples_df) >= 2 and len(bad_examples_df) >= 2:
            output_parts.append("The following are good historical configurations for this algorithm (from the top 20% of all trials):")
            for _, row in good_examples_df.iterrows():
                config_row = row.drop(['_score', 'algorithm']).dropna()
                serialized_config = ", ".join([f"{k}: {v}" for k, v in config_row.items()])
                output_parts.append(f"Performance: {row['_score']:.6f}\nConfig: ## {serialized_config} ##")

            output_parts.append("\nThe following are poor historical configurations for this algorithm (from the bottom 30% of all trials):")
            for _, row in bad_examples_df.iterrows():
                config_row = row.drop(['_score', 'algorithm']).dropna()
                serialized_config = ", ".join([f"{k}: {v}" for k, v in config_row.items()])
                output_parts.append(f"Performance: {row['_score']:.6f}\nConfig: ## {serialized_config} ##")
        else:
            output_parts.append("The following are all available historical configurations for this algorithm, sorted by performance:")
            algo_history_sorted = algo_history.sort_values("_score", ascending=ascending)
            for _, row in algo_history_sorted.iterrows():
                config_row = row.drop(['_score', 'algorithm']).dropna()
                serialized_config = ", ".join([f"{k}: {v}" for k, v in config_row.items()])
                output_parts.append(f"Performance: {row['_score']:.6f}\nConfig: ## {serialized_config} ##")

        return "\n".join(output_parts)

    
    def get_candidate_points(
        self,
        observed_configs,
        observed_fvals,
        alpha=0.2,
        iter_idx=None
    ):
        """
        Public entry point to get new candidate configurations from the LLM.

        This method orchestrates the two-stage optimization process synchronously.
        """
        t0 = time.time()
        try:
            # Wrap the core async logic to be called from a synchronous context.
            proposed, total_cost = asyncio.run(
                self._run_two_stage_optimization(observed_configs, observed_fvals, alpha, iter_idx)
            )
            time_taken = time.time() - t0
            return proposed, total_cost, time_taken
        except Exception as e:
            print(f"[LLM_ACQ] Two-stage generation failed: {e}")
            return pd.DataFrame(), 0.0, time.time() - t0

    async def _run_two_stage_optimization(
        self,
        observed_configs,
        observed_fvals,
        alpha=0.2,
        iter_idx=None
    ):
        """
        The core async logic for the two-stage (Algorithm -> Parameters) optimization.
        """
        
        # --- Stage 1: Algorithm Selection ---
        print("[LLM_ACQ] Stage 1: Algorithm Selection")
        algo_examples = self._build_algo_selection_examples(observed_configs, observed_fvals)
        data_summary = self._build_dataset_summary_sentences()
        algo_selection_prompt = self._format_algo_selection_prompt(
            algo_examples, data_summary, iter_idx
        )
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": algo_selection_prompt}
        ]
        
        resp1, cost1, tokens1 = await self._async_generate_with_context(messages, temperature=0.7, n=1)
        if resp1 is None:
            return pd.DataFrame(), 0.0
        
        assistant_msg = resp1["choices"][0]["message"]["content"]
        selected_algo = self._parse_selected_algorithm(assistant_msg)
        print(f"[LLM_ACQ] Selected algorithm: {selected_algo}")
        
        messages.append({"role": "assistant", "content": assistant_msg})
        self._save_conversation_log(messages, iter_idx, "stage1_algo_selection")

        # --- Stage 2: Parameter Optimization ---
        print("[LLM_ACQ] Stage 2: Parameter Optimization")
        
        # This loop retries the generation if the LLM fails to produce valid configurations.
        MAX_TRIES = 5
        all_validated_points = pd.DataFrame()
        total_cost = cost1
        num_tries = 0

        while len(all_validated_points) < self.n_candidates and num_tries < MAX_TRIES:
            num_tries += 1
            print(f"[LLM_ACQ] Attempt {num_tries}/{MAX_TRIES} to generate {self.n_candidates - len(all_validated_points)} more candidates...")

            param_examples = self._build_ranked_param_optimization_examples(
                observed_configs, observed_fvals, selected_algo
            )
            
            # Set a target performance value to guide the LLM.
            vals = observed_fvals.values.astype(float).ravel()
            rng = float(np.abs(np.max(vals) - np.min(vals)))
            if rng == 0: rng = 0.1 * float(np.abs(np.max(vals)))
            if self.lower_is_better:
                desired_fval = float(np.min(vals)) - self.alpha * rng
            else:
                desired_fval = float(np.max(vals)) + self.alpha * rng
            
            param_optimization_prompt = self._format_param_optimization_prompt(
                observed_configs,
                observed_fvals,
                selected_algo,
                desired_fval
            )
            
            messages_stage2 = messages + [{"role": "user", "content": param_optimization_prompt}]
            
            resp2, cost2, tokens2 = await self._async_generate_with_context(
                messages_stage2, 
                temperature=0.8 + num_tries * 0.05, # Slightly increase temp on retries.
                max_tokens=500,
                n=self.n_candidates # Request enough candidates in one go.
            )
            
            total_cost += cost2
            if resp2 is None:
                continue
            
            # Parse and validate the configurations from the LLM's response.
            candidate_points = []
            assistant_responses = []
            for ch in resp2.get("choices", []):
                content = ch.get("message", {}).get("content", "")
                assistant_responses.append(content)
                cand = self._parse_config_from_response(content, selected_algo)
                if cand:
                    candidate_points.append(cand)
            
            final_messages = messages_stage2 + [{"role": "assistant", "content": "\n---\n".join(assistant_responses)}]
            self._save_conversation_log(final_messages, iter_idx, f"stage2_attempt_{num_tries}")

            if not candidate_points:
                continue

            # Filter out duplicate or invalid points.
            current_observed_configs = pd.concat([
                observed_configs,
                all_validated_points
            ]).reset_index(drop=True)

            proposed = _filter_and_validate(
                self, 
                current_observed_configs.to_dict(orient="records"), 
                candidate_points
            )

            if not proposed.empty:
                all_validated_points = pd.concat([all_validated_points, proposed]).drop_duplicates().reset_index(drop=True)
            
            print(f"[LLM_ACQ] After attempt {num_tries}, got {len(proposed)} new valid points. Total valid points: {len(all_validated_points)}.")
        
        if all_validated_points.empty:
            print("[LLM_ACQ] Failed to generate any valid candidates after all tries.")
        
        # Return up to n_candidates valid points.
        return all_validated_points.head(self.n_candidates), total_cost

    def _format_algo_selection_prompt(self, examples, data_summary, iter_idx):
        """
        Formats the prompt for the algorithm selection stage, switching between
        exploration and exploitation instructions based on the iteration number.
        """
        metric = self.task_context.get("metric", "score")
        direction = "minimize" if self.lower_is_better else "maximize"
        
        EXPLORATION_ITER_THRESHOLD = 25
        if iter_idx is None or iter_idx < EXPLORATION_ITER_THRESHOLD:
            instruction_header = "You are in the EXPLORATION PHASE."
            instruction_body = (
                "Your goal is to explore diverse algorithms to understand the search space.\n"
                "Do NOT solely focus on the configurations labeled 'BEST'. "
                "Consider algorithms that are under-explored or look promising based on the dataset characteristics."
            )
        else:
            instruction_header = "You are in the EXPLOITATION PHASE."
            instruction_body = (
                "Sufficient data has been collected. The focus should now shift from broad exploration to exploiting promising regions.\n"
                "Analyze the high-performing configurations in the history to identify the algorithm with the most potential for achieving a new best score."
            )
        examples_text = "\n\n".join([
            f"{ex['Label']}\n"
            f"Algorithm: {ex['Algorithm']}\n"
            f"Performance: {ex['Performance']}\n"
            f"Full Config: {ex['Config']}"
            for ex in examples
        ])
            
        return f"""{instruction_header}
    {instruction_body}

    Dataset & Task:
    {data_summary}

    Historical trials across all algorithms:
    {examples_text}

    Based on the instructions and the history, which algorithm should we use next to {direction} {metric}?
    Output ONLY the algorithm name, e.g.: "random_forest" or "xgboost"
    """

    def _format_param_optimization_prompt(self, observed_configs, observed_fvals, selected_algo, desired_fval):
        """
        Formats the prompt for the parameter optimization stage, including
        algorithm-specific constraints and ranked historical examples.
        """
        metric = self.task_context.get("metric", "score")
        direction = "lower is better" if self.lower_is_better else "higher is better"
        
        constraints_text = self._constraints_text_for_algo(selected_algo)
        examples_text = self._build_ranked_param_optimization_examples(
            observed_configs, 
            observed_fvals, 
            selected_algo
        )
        
        # Dynamically choose the final instruction based on available history.
        if "good historical configurations" in examples_text:
            instruction = "Based on the good and bad examples, propose a new configuration"
        elif "all available historical configurations" in examples_text:
            instruction = "Based on the historical data provided, propose a new configuration"
        else:
            instruction = "Propose a good starting configuration"
            examples_text = "There is no prior history for this specific algorithm. Please propose a good starting configuration based on the search space constraints."
        
        return f"""Now, optimize the hyperparameters for the algorithm: '{selected_algo}'

Search space constraints for '{selected_algo}':
{constraints_text}

Historical trials for this algorithm:
{examples_text}

{instruction} expected to achieve a '{metric}' of {desired_fval:.6f} ({direction}).
IMPORTANT: Pay close attention to the ranges in the constraints. For integer ranges like [min, max], the value must be between min and max inclusive. For float ranges, avoid using the exact boundary values.

Your response must be in the following format:
## algorithm: {selected_algo}, {selected_algo}:<param>: <value>, ... ##
"""

    def _parse_selected_algorithm(self, response_text: str) -> str:
        """Parses the algorithm name from the LLM's response."""
        text = response_text.strip().lower()
        for algo in self.algo_choices:
            if algo.lower() in text:
                return algo
        # Fallback to a random choice if parsing fails.
        print(f"Warning: Could not parse algorithm '{response_text}'. Falling back to a random choice.")
        return np.random.choice(self.algo_choices) if self.algo_choices else "unknown"

    def _parse_config_from_response(self, response_text: str, expected_algo: str) -> Dict[str, Any]:
        """Parses a hyperparameter configuration from the LLM's response string."""
        try:
            first = response_text.find("##")
            last = response_text.rfind("##")
            if first != -1 and last != -1 and last > first:
                segment = response_text[first + 2: last].strip()
            else:
                segment = response_text.split("##")[1].strip()
            
            cand = _convert_to_dict(segment)
            # Ensure the parsed algorithm matches the expected one.
            if cand and cand.get("algorithm") == expected_algo:
                return cand
        except Exception:
            pass
        return None

    def _save_conversation_log(self, messages: List[Dict[str, str]], iter_idx: int, stage_name: str):
        """Saves the full conversation history to a human-readable .txt file for debugging."""
        if iter_idx is None:
            return
        
        log_filename = f"iter_{iter_idx:03d}_{stage_name}.txt"
        log_path = os.path.join(self.conversation_log_dir, log_filename)
        
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                for message in messages:
                    role = message.get("role", "unknown").upper()
                    content = message.get("content", "")
                    
                    f.write("="*20 + f" ROLE: {role} " + "="*20 + "\n\n")
                    f.write(content)
                    f.write("\n\n")
        except Exception as e:
            print(f"[LLM_ACQ] Error saving conversation log to {log_path}: {e}")
    
    def _constraints_text_for_algo(self, selected_algo: str) -> str:
        """Generates the constraints text block for ONLY the selected algorithm."""
        
        algo_params = self.algo2params.get(selected_algo, [])
        relevant_params = ['algorithm'] + algo_params
        algo_constraints = {k: v for k, v in self.constraints.items() if k in relevant_params}
        
        lines = []
        for name, properties in algo_constraints.items():
            if name == 'algorithm':
                lines.append(f"- {name}: Categorical, Choices: {self.algo_choices}")
                continue
            
            p_type = properties.get("type")
            if p_type == "categorical":
                lines.append(f"- {name}: Categorical, Choices: {properties.get('choices')}")
            elif p_type in ("int", "float"):
                log_scale = " (log scale)" if properties.get("log") else ""
                lines.append(f"- {name}: {p_type.capitalize()}, Range: [{properties.get('lower')}, {properties.get('upper')}]" + log_scale)
        return "\n".join(lines)