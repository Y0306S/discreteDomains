import time
import os
import numpy as np
import pandas as pd
from functools import partial
from typing import Any, Dict, List, Tuple

from ConfigSpace import Configuration, ConfigurationSpace
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT

from mindware.third_party.opro.baseline_acq import BaselineLLM_ACQ
from mindware.third_party.opro.react_acq import ReActLLM_ACQ
from mindware.third_party.opro.rate_limiter import RateLimiter
from mindware.third_party.opro.warping import NumericalTransformer
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS


# Utility functions for the optimizer
from mindware.components.optimizers.llm_optimizer_utils import (
    _record_event,
    _evaluate_dict,
    _cfgs_df,
    _init_f,
    _update_incumbent_from_current,
    _time_is_up,
    _quota_is_up,
    ExplorationExploitationBalancer,
    wrapper_evaluater
)
from mindware.third_party.opro.utils import _build_hparam_constraints






class LLMOptimizer(BaseOptimizer):
    """
    An optimizer that leverages a Large Language Model (LLM) to solve CASH problems.

    This optimizer orchestrates the Optimization by Prompting (OPRO) workflow. It manages
    the optimization loop, evaluates configurations suggested by the LLM, and keeps
    track of the search history and performance.
    """
    def __init__(
        self,
        evaluator,
        config_space: ConfigurationSpace,
        name: str,
        eval_type: str,
        task_id: str = "test",
        time_limit: int = None,
        evaluation_limit: int = None,
        per_run_time_limit: int = 300,
        per_run_mem_limit: int = 1024,
        inner_iter_num_per_iter: int = 1,
        timestamp: float = None,
        output_dir: str = "./",
        seed: int = 1,
        n_jobs: int = 1,
        topk: int = 50,
        **kwargs,
    ):
        super().__init__(
            evaluator=evaluator,
            config_space=config_space,
            name=name,
            eval_type=eval_type,
            task_id=task_id,
            time_limit=time_limit,
            evaluation_limit=evaluation_limit,
            per_run_time_limit=per_run_time_limit,
            per_run_mem_limit=per_run_mem_limit,
            inner_iter_num_per_iter=inner_iter_num_per_iter,
            timestamp=timestamp,
            output_dir=output_dir,
            seed=seed,
            topk=topk,
        )


        self.general_transfer = kwargs.get('general_transfer', True)
        self.add_shap = kwargs.get('add_shap', True)


        # 将evaluator包装为超时控制的形式
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)


        self.prompt_strategy = kwargs.get("prompt_strategy", "baseline")

        # --- LLM and OPRO specific parameters ---

        # Default: deepseek-v3
        self.chat_engine = AVAILABLE_LLMs[kwargs.get("chat_engine", LLMCONFIGS.opt_chat_engine)]['model']
        self.prompt_cost_per_1k = kwargs.get("prompt_cost_per_1k", 0.000225352)
        self.completion_cost_per_1k = kwargs.get("completion_cost_per_1k", 0.000901408)

        self.n_candidates = kwargs.get("n_candidates", 1)
        self.n_templates = kwargs.get("n_templates", 3)
        self.alpha = kwargs.get("alpha", 0.2)
        self.n_initial_samples = kwargs.get("n_initial_samples", 4)
        self.n_trials = kwargs.get("n_trials", 10)
        self.use_input_warping = kwargs.get("use_input_warping", False)
        self.prompt_setting = kwargs.get("prompt_setting", "full_context")
        self.shuffle_features = kwargs.get("shuffle_features", False)
        self.task_name = kwargs.get("task_name", "CASH")

        self.KD = KnowledgeDatabase.instance()
        self.configs = self.KD.configs
        self.perfs = self.KD.perfs
        self.lower_is_better = self.KD.lower_is_better

        # --- Context for the LLM Acquisition Function ---
        # This context bundles all necessary information for the LLM to generate prompts.
        self.task_context = dict(
            task=self.task_name,
            task_id=self.task_id,
            lower_is_better=self.lower_is_better,
            hyperparameter_constraints=_build_hparam_constraints(self.config_space),
        )

        # Extract algorithm choices and their specific parameters if available.
        # if 'algorithm' in [hp.name for hp in self.config_space.get_hyperparameters()]:
        algo_hp = self.config_space.get_hyperparameter('algorithm')
        algo_choices = list(algo_hp.choices)
        self.task_context['algorithm_choices'] = algo_choices
        self.task_context['algorithm_param_names'] = {
            a: [hp.name for hp in self.config_space.get_hyperparameters() if hp.name.startswith(a + ':')]
            for a in algo_choices
        }

        self.exploration_exploitation_balancer = ExplorationExploitationBalancer(alpha=0.6, algo_choices=algo_choices)

        # --- Internal State and Utilities ---
        # self.rate_limiter = RateLimiter(max_tokens=100000, time_frame=60, max_requests=1000)
        self.rate_limiter = RateLimiter(max_tokens=50000, time_frame=60, max_requests=1000)
        self.warping_transformer = (
            NumericalTransformer(self.task_context["hyperparameter_constraints"])
            if self.use_input_warping else None
        )
        self.llm_query_cost: List[float] = []
        self.llm_query_time: List[float] = []
        self.exp_output = dict()
        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_LLM_opt_traj.csv")
        self.start_wall_time = None

        # Estimate the number of unique configurations to set an early stopping threshold.
        _uniq = set()
        for _ in range(5000):
            cfg = self.config_space.sample_configuration()
            _uniq.add(tuple(sorted(cfg.get_dictionary().items())))
        self.config_num_threshold = max(1, len(_uniq))
        self.maximum_config_num = min(1500, self.config_num_threshold)
        self.logger.debug(f"[LLMOptimizer] estimated unique configs: {self.config_num_threshold}")

        os.makedirs(self.output_dir, exist_ok=True)

        # --- Dataset Profiling ---
        # Build and add dataset profile to the task context to make the LLM data-aware.
        self.add_dataset_summary_to_prompt = kwargs.get("add_dataset_summary_to_prompt", True)
        self.task_context.update({
            "add_dataset_summary_to_prompt": self.add_dataset_summary_to_prompt,
        })

        profile = kwargs.get("dataset_profile")
        # tpi = kwargs.get("task_profile_input")
        # if tpi and isinstance(tpi, dict) and 'df' in tpi and 'label_col' in tpi:
        #     try:
        #         profile = _build_dataset_profile_from_df(
        #             self, tpi['df'], tpi['label_col'], tpi.get('task_type_str', 'CLS'), tpi.get('metric_name', '')
        #         )
        #     except Exception as e:
        #         self.logger.warning(f"[LLMOptimizer] build profile from df failed, fallback to provided profile. Err={e}")

        # if profile is not None:
        self.task_context["dataset_profile"] = profile
        self.task_context["metric"] = profile["metric"]

        # --- Initialize the Acquisition Function ---
        # The LLM_ACQ class handles all prompt engineering and interaction with the LLM API.
        self.task_context.update({
            "output_dir": self.output_dir,
            "run_id": self.timestamp,
            "dump_llm_templates": kwargs.get("dump_llm_templates", True),
            "chat_engine": self.chat_engine
        })

        if self.prompt_strategy == "react":
            self.logger.debug(f"[LLMOptimizer] : Using ReAct/Reflexion Strategy")
            self.acq_func = ReActLLM_ACQ(self, **kwargs)
        else:
            self.logger.debug(f"[LLMOptimizer] : Using Baseline Strategy")
            self.acq_func = BaselineLLM_ACQ(
                self.task_context,
                self.n_candidates,
                self.n_templates,
                lower_is_better=self.lower_is_better,
                rate_limiter=self.rate_limiter,
                warping_transformer=self.warping_transformer,
                chat_engine=self.chat_engine,
                prompt_setting=self.prompt_setting,
                shuffle_features=self.shuffle_features,
            )


        self.algorithm_analysis_knowledge = None
        if self.general_transfer:
            algo_cate = self.KD.fetch_algo_cate_knowledge()  # {"exploration_algorithms": [...], "exploitation_algorithms": [...]}
            self.acq_func.exploration_algos = algo_cate["exploration_algorithms"]
            self.acq_func.exploitation_algos = algo_cate["exploitation_algorithms"]
            self.logger.info(f"[ReAct ACQ] Initialize lists:\n  - Exploration: {algo_cate['exploration_algorithms']}\n  - Exploitation: {algo_cate['exploitation_algorithms']}")

            """
            {
                algorithm: {
                    "dataset_similarity": [...],
                    "SHAP Analysis": [...],
                    "SHAP Overall Importance": [...]
                }
            }
            """

            if self.add_shap:
                self.algorithm_analysis_knowledge = self.KD.fetch_historical_analysis_knowledge()


    def iterate(self, budget=MAX_INT):
        """
        A single iteration step, designed to be called by an external scheduler.

        In one call, this method generates and evaluates all candidates for a single trial.
        """
        t0 = time.time()
        iter_idx = len(self.KD.bo_history) + 1

        # Cold-start with initial samples if history is empty.
        if iter_idx <= self.n_initial_samples:
            cfg_d = _init_f(self, 1)[0]
            t0_iter = time.time()
            obs, eval_time = _evaluate_dict(self, cfg_d)
            cfg_obj = obs.config
            max_perf = - float(obs.objectives[0])

            self.exploration_exploitation_balancer.update_best_perf(max_perf, cfg_d['algorithm'])
            self.update_saver([cfg_obj], [-max_perf])
            self.KD.update_history(obs, source='random', trail_id = f'{iter_idx}_warmup')

            iter_time = time.time() - t0_iter
            _record_event(
                self, stage="iterate-init", cfg=cfg_obj, perf=max_perf,
                eval_time=eval_time, iter_time=time.time() - t0_iter,
                note="warmup in iterate()", iter_idx=iter_idx,
            )

            row_data = {
                "perf": max_perf,
                "config": cfg_obj,
                "eval_time": eval_time,
                "llm_time": 0,
                "iter_time": iter_time,
            }

            _update_incumbent_from_current(self)

            iter_cost = time.time() - t0
            return self.incumbent_perf, iter_cost, self.incumbent_config, [row_data]

        # Check budget and quotas.
        if _time_is_up(self, self.timestamp) or _quota_is_up(self):
            self.timeout_flag = True

        obs_cfg_df = _cfgs_df(self)
        obs_acc_df = pd.DataFrame({'score': self.perfs})

        t0_iter = time.time()
        
        if self.prompt_strategy == 'react': 
            cand_df, cost, tcost, eval_times = self.acq_func.get_candidate_points(
                obs_cfg_df, obs_acc_df[["score"]], alpha=self.alpha, iter_idx=iter_idx
            )
        else: 
            cand_df, cost, tcost = self.acq_func.get_candidate_points(
                obs_cfg_df, obs_acc_df[["score"]], alpha=self.alpha, iter_idx=iter_idx
            )
            eval_times = [] 

        if cand_df is None or len(cand_df) == 0:
            self.early_stopped_flag = True
            return self.incumbent_perf, time.time() - t0, self.incumbent_config, []

        # Record LLM cost for this trial.
        self.llm_query_cost.append(cost)
        self.llm_query_time.append(tcost)

        newly_observed_rows = []
        for i in range(len(cand_df)):

            if self.prompt_strategy == 'baseline':
                acc, eval_time = _evaluate_dict(self, cfg_d)
                self.configs.append(self.KD._dict_to_cfg(cfg_d))
                self.perfs.append(acc)
            else:
                cfg_obj = self.configs[-len(cand_df) + i]
                max_perf = self.perfs[-len(cand_df) + i]
                eval_time = eval_times[i]

            self.update_saver([cfg_obj], [-max_perf])
            # Update the incumbent after the trial.
            _update_incumbent_from_current(self)

            iter_time = time.time() - t0_iter if i == 0 else 0.0
            note = f"candidate {i+1}/{len(cand_df)}"

            row_data = {
                "perf": max_perf,
                "config": cfg_obj,
                "eval_time": eval_time,
                "llm_time": self.llm_query_time[-1],
                "iter_time": iter_time,
            }
            newly_observed_rows.append(row_data)

            _record_event(
                self, stage="iterate", cfg=cfg_obj, perf=max_perf,
                llm_cost=(cost if i == 0 else 0.0),
                llm_time=(tcost if i == 0 else 0.0),
                eval_time=eval_time, 
                iter_time=iter_time, 
                iter_idx=iter_idx,
                note=note,
                algo_exploration_prob=self.exploration_exploitation_balancer.algo_prob_record[-1],
                param_exploration_prob=self.exploration_exploitation_balancer.param_prob_record[-1],
            )


        iter_cost = time.time() - t0
        return self.incumbent_perf, iter_cost, self.incumbent_config, newly_observed_rows


    def update_history(self, obs, full_history_entry):
        algo_strategy_name = full_history_entry["algo_strategy_name"]
        param_strategy_name = full_history_entry["param_strategy_name"]
        selected_algo = obs.config.get("algorithm")
        max_perf = - float(obs.objectives[0])
        self.exploration_exploitation_balancer.update_cumulative_rewards(max_perf, algo_strategy_name, param_strategy_name, selected_algo) # 更新一个越大越好的分数
        self.KD.update_history(obs, source='llm', **full_history_entry)


    def get_opt_trajectory(self):
        """Returns a summary of the optimization process."""
        best = float(max(self.perfs)) if self.perfs else float("-inf")

        total_llm_cost = 0.0
        
        if self.prompt_strategy == 'react' and hasattr(self.acq_func, 'get_total_llm_cost'):
            total_llm_cost = self.acq_func.get_total_llm_cost()
        else:
            total_llm_cost = float(sum(self.llm_query_cost) if self.llm_query_cost else 0.0)

        return {
            "n_evals": len(self.perfs),
            "final_best": best,
            "llm_cost_sum": total_llm_cost,
            "llm_calls": len(self.llm_query_cost),
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join([str(p) for p in self.perfs]),
            "history": [
                {
                    "iter_idx": r["iter_idx"],
                    "perf": r["perf"],
                    "best_so_far": r["best_so_far"],
                    "algorithm": r["algorithm"],
                    "llm_cost": r["llm_cost"],
                    "llm_time": r["llm_time"],
                    "eval_time": r.get("eval_time", 0.0),
                    "iter_time": r.get("iter_time", 0.0),
                    "wall_elapsed_sec": r["wall_elapsed_sec"],
                } for r in self.traj_rows
            ][:500]
        }

