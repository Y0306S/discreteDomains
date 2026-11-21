import os
import time
import numpy as np
import pandas as pd
from functools import partial
from typing import List, Dict, Any

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter, 
    UniformIntegerHyperparameter, 
    UniformFloatHyperparameter, 
    Constant,
    UnParametrizedHyperparameter
)

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS

from mindware.third_party.opro.llambo_acq import LLAMBO_ACQ

from openbox.utils.history import Observation
from mindware.components.optimizers.llm_optimizer_utils import (
    _update_incumbent_from_current,
    _time_is_up,
    wrapper_evaluater
)
from mindware.components.optimizers.bopro_optimizers_utils import _evaluate_dict
from mindware.third_party.opro.utils import _build_hparam_constraints

class LLAMBOOptimizer(BaseOptimizer):
    """
    LLAMBO Optimizer (Large Language Models to Enhance Bayesian Optimization).
    
    Mechanism:
    1. Warm Start: Using Knowledge Database (KD) or Random Sampling.
    2. Candidate Sampling: LLM acts as an Inverse Sampler (Score -> Configs).
    3. Surrogate Modeling: LLM acts as a Discriminative Model (Configs -> Scores) with Shuffle for uncertainty.
    4. Acquisition: Expected Improvement (EI) maximization.
    """

    def __init__(
        self,
        evaluator,
        config_space: ConfigurationSpace,
        name: str = "LLAMBO",
        eval_type: str = "holdout",
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

        self.max_llm_cost = 2.0

        # 1. 基础设置
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)
        print("[LLAMBOOptimizer] Using chat_engine:", AVAILABLE_LLMs[LLMCONFIGS.opt_chat_engine]['model'])
        self.chat_engine = AVAILABLE_LLMs[LLMCONFIGS.opt_chat_engine]['model']
        self.task_name = kwargs.get("task_name", "CASH")
        self.lower_is_better = kwargs.get("lower_is_better", False)
        
        # [配置] LLAMBO 采样参数 (低资源版默认值)
        self.n_candidates = kwargs.get("n_candidates", 8)
        self.alpha = kwargs.get("alpha", -0.1)
        
        # [配置] 随机初始化数量 (如果没有 KD 数据)
        self.init_num = 10 

        # 2. 数据容器
        self.configs = [] 
        self.perfs = []   # 存储原始 metric
        
        # 3. 构建任务上下文
        algo_hp = self.config_space.get_hyperparameter('algorithm')
        algo_choices = list(algo_hp.choices)
        
        self.task_context = dict(
            task=self.task_name,
            task_id=self.task_id,
            lower_is_better=self.lower_is_better,
            hyperparameter_constraints=_build_hparam_constraints(self.config_space),
            algorithm_choices=algo_choices
        )
        self.task_context['algorithm_param_names'] = {
            a: [hp.name for hp in self.config_space.get_hyperparameters() if hp.name.startswith(a + ':')]
            for a in algo_choices
        }
        self.task_context["add_dataset_summary_to_prompt"] = kwargs.get("add_dataset_summary_to_prompt", True)
        profile = kwargs.get("dataset_profile")
        self.task_context["dataset_profile"] = profile
        self.task_context["metric"] = profile["metric"] if profile else "score"
        self.task_context.update({
            "output_dir": self.output_dir,
            "run_id": self.timestamp,
        })

        # 4. 初始化 ACQ (Controller)
        self.acq_func = LLAMBO_ACQ(
           optimizer=self,
           temperature=0.7, 
           n_candidates=8,
           n_surrogate_rounds=3, # Shuffle 次数
           alpha=self.alpha
        )

        # 5. 准备热启动 (Warm Start) 队列
        self.initial_configs_queue = []
        self._init_warm_start(seed, algo_choices)

        # 日志记录
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm_query_cost: List[float] = []
        self.llm_query_time: List[float] = []
        self.eval_time : List[float] = []
        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_LLAMBO_opt_traj.csv")

        self.early_stopped_flag = False

    def _init_warm_start(self, seed, algo_choices):
        """加载 Knowledge Database 中的配置 (KD Warmup)"""
        rng = np.random.RandomState(seed)
        self.KD = KnowledgeDatabase.instance()
        bestconf_knowledge = self.KD.fetch_best_conf_knowledge()
        
        algo_queues = {}
        for alg in algo_choices:
            if alg in bestconf_knowledge and "Best Configurations" in bestconf_knowledge[alg]:
                confs = list(bestconf_knowledge[alg]["Best Configurations"])
                if len(confs) > 0:
                    # 随机丢弃一个增加方差
                    idx_to_remove = rng.randint(0, len(confs))
                    confs.pop(idx_to_remove)
                algo_queues[alg] = confs
            else:
                algo_queues[alg] = []
        
        # Round-Robin 提取
        while any(len(q) > 0 for q in algo_queues.values()):
            for alg in algo_choices:
                if len(algo_queues[alg]) > 0:
                    conf_dict = algo_queues[alg].pop(0) 
                    try:
                        if 'algorithm' not in conf_dict:
                            conf_dict['algorithm'] = alg
                        config_obj = Configuration(self.config_space, values=conf_dict)
                        self.initial_configs_queue.append(config_obj)
                    except Exception as e:
                        self.logger.warning(f"[LLAMBO] Failed to load KD config: {e}")

        if len(self.initial_configs_queue) > 0:
            print(f"[LLAMBO] Warm start loaded with {len(self.initial_configs_queue)} configurations.")
        else:
            print(f"[LLAMBO] KD empty. Will use random initialization for first {self.init_num} steps.")

    def _safe_dict_to_config(self, d: Dict[str, Any]) -> Configuration:
        """
        Robustly converts a dictionary (from LLM) into a valid ConfigSpace.Configuration.
        """
        # 复用 BOPRO/OPRO 的逻辑，确保类型安全
        import math
        def _is_nan(x): return isinstance(x, float) and math.isnan(x)
        def _none_like(x): return (x is None) or (isinstance(x, str) and x.strip().lower() in ("none", "null"))

        hp_list = self.config_space.get_hyperparameters()
        hp_map = {hp.name: hp for hp in hp_list}

        algo_hp = hp_map.get('algorithm', None)
        algo = d.get('algorithm')
        if algo is None: 
            for k in d.keys():
                if k.strip() == "algorithm": algo = d[k]; break
        
        if algo is None:
            prefixes = {k.split(':', 1)[0] for k in d.keys() if ':' in k}
            if prefixes: algo = max(prefixes, key=lambda p: len([k for k in d.keys() if k.startswith(p)]))
            elif algo_hp: algo = algo_hp.default_value
            else: raise ValueError("No 'algorithm' provided.")
        
        if isinstance(algo, str): algo = algo.strip()
        
        values: Dict[str, Any] = {'algorithm': algo}
        for hp in hp_list:
            if hp.name == 'algorithm': continue
            if hp.name.startswith(algo + ':'):
                if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                    val = getattr(hp, 'value', getattr(hp, 'default_value', None))
                    values[hp.name] = val
                elif hasattr(hp, 'default_value'):
                    values[hp.name] = hp.default_value

        for k_in, v_in in d.items():
            k_clean = k_in.strip()
            if k_clean == 'algorithm' or _is_nan(v_in) or _none_like(v_in): continue
            k = k_clean if k_clean.startswith(algo + ':') else f"{algo}:{k_clean}"
            hp = hp_map.get(k)
            if hp is None or isinstance(hp, (Constant, UnParametrizedHyperparameter)): continue

            try:
                if isinstance(hp, UniformIntegerHyperparameter):
                    v = int(round(float(v_in)))
                    v = max(hp.lower, min(hp.upper, v))
                elif isinstance(hp, UniformFloatHyperparameter):
                    v = float(v_in)
                    v = max(hp.lower, min(hp.upper, v))
                elif isinstance(hp, CategoricalHyperparameter):
                    choices = list(hp.choices)
                    v_str = str(v_in).strip().strip("'").strip('"')
                    if v_str in choices: v = v_str
                    else:
                        matched = next((c for c in choices if str(c).lower() == v_str.lower()), None)
                        if matched: v = matched
                        else: continue
                else:
                    v = v_in
                values[k] = v
            except: continue

        return Configuration(self.config_space, values=values)

    def iterate(self, budget=MAX_INT):
        
        if self.early_stopped_flag:
            return self.incumbent_perf, 0, self.incumbent_config
        
        current_total_cost = self.acq_func.get_total_llm_cost()
        if current_total_cost > self.max_llm_cost:
            self.logger.info(f"[LLAMBO] STOPPING: Cost limit exceeded (${current_total_cost:.4f} > ${self.max_llm_cost})")
            self.early_stopped_flag = True 
            return self.incumbent_perf, 0, self.incumbent_config
        
        if _time_is_up(self, self.timestamp):
            self.timeout_flag = True
            return self.incumbent_perf, 0, self.incumbent_config

        t0 = time.time()

        # =====================================================
        # 1. 确定配置来源 (Warmup vs LLAMBO Loop)
        # =====================================================
        candidate_to_eval = None
        note = ""
        llm_cost = 0.0
        llm_time = 0.0

        # A. KD Warmup
        if len(self.initial_configs_queue) > 0:
            candidate_to_eval = self.initial_configs_queue.pop(0)
            note = "KD Warmup"
        
        # B. Random Warmup (如果历史太少)
        elif len(self.configs) < self.init_num:
            candidate_to_eval = self.config_space.sample_configuration()
            note = f"Random Warmup ({len(self.configs)+1}/{self.init_num})"
        
        # C. LLAMBO Main Loop
        else:
            obs_cfg_df = pd.DataFrame([c.get_dictionary() for c in self.configs])
            if self.lower_is_better:
                raw_scores = [-p for p in self.perfs]
            else:
                raw_scores = self.perfs
            obs_acc_df = pd.DataFrame({'score': raw_scores})
            
            try:
                t0_llm = time.time()
                cost_before = self.acq_func.get_total_llm_cost()
                # 调用 ACQ: Sample -> Evaluate -> Select Best
                best_config_dict = self.acq_func.get_candidate(
                    obs_cfg_df, obs_acc_df, iter_idx=len(self.perfs)
                )

                cost_after = self.acq_func.get_total_llm_cost()

                llm_time = time.time() - t0_llm
                llm_cost = cost_after - cost_before

                if best_config_dict:
                    try:
                        candidate_to_eval = self._safe_dict_to_config(best_config_dict)
                        note = "LLAMBO Selected"
                    except Exception as e:
                        self.logger.warning(f"Failed to convert LLAMBO config: {e}")
                        candidate_to_eval = self.config_space.sample_configuration()
                        note = "Fallback (Conversion Error)"
                else:
                    candidate_to_eval = self.config_space.sample_configuration()
                    note = "Fallback (LLAMBO Failed)"

            except Exception as e:
                self.logger.error(f"LLAMBO ACQ Error: {e}")
                candidate_to_eval = self.config_space.sample_configuration()
                note = f"Fallback (Error: {e})"

        # =====================================================
        # 2. 评估 (Evaluation)
        # =====================================================
        
        # 评估
        obs, eval_time = _evaluate_dict(self, candidate_to_eval.get_dictionary(), self.config_space)
        #perf是原始metric；max_perf是越大越好的metric
        #self.perfs存的是越大越好的max_perf
        perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)  # obs.objectives[0] 来自于 evaluator，越小越好，所以这里根据方向调整
        max_perf = - float(obs.objectives[0])
        cfg_obj = obs.config
        
        self.update_saver([cfg_obj], [-max_perf])
        self.configs.append(cfg_obj)
        self.perfs.append(max_perf)
        _update_incumbent_from_current(self)

        iter_time = time.time() - t0
        opt_time = iter_time - eval_time - llm_time
        
        self.llm_query_cost.append(llm_cost)
        self.llm_query_time.append(llm_time)
        self.eval_time.append(eval_time)
        self._record_event(
            opt="LLAMBO",
            iter_idx=len(self.perfs),
            cfg=candidate_to_eval,
            perf=max_perf,
            eval_time=eval_time,
            opt_time=opt_time,
            iter_time=iter_time,
            llm_cost=llm_cost,
            llm_time=llm_time,
            note=note
        )

        return self.incumbent_perf, iter_time, self.incumbent_config

    def _record_event(self, opt, iter_idx, cfg, perf, eval_time=0.0, opt_time=0.0, iter_time=0.0, llm_cost=0.0, llm_time=0.0, note="", **kwargs):
        wall_elapsed_sec = time.time() - self.timestamp
        algo = cfg.get('algorithm', 'default')
        
        row = {
            "ts": pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "optimizer": opt,
            "iter_idx": iter_idx,
            "perf": perf,
            # "best_so_far": min(self.perfs) if self.lower_is_better else max(self.perfs),
            "best_so_far": max(self.perfs) if self.perfs else float("-inf"),
            "algorithm": algo,
            "opt_time": opt_time,
            "eval_time": eval_time,
            "iter_time": iter_time,
            "llm_cost": llm_cost,   
            "llm_time": llm_time,   
            "wall_elapsed_sec": wall_elapsed_sec,
            "note": note
        }
        self.traj_rows.append(row)
        
        try:
            df = pd.DataFrame(self.traj_rows)
            df.to_csv(self.traj_csv_path, index=False)
        except:
            pass

    def _save_global_summary(self):
        """计算全局统计信息并追加到 CSV 末尾 (无锁版)"""
        try:
            total_llm_cost = self.acq_func.get_total_llm_cost()
            total_wall_time = time.time() - self.timestamp
            total_llm_time = sum(self.llm_query_time) if self.llm_query_time else 0.0
            total_eval_time = sum(self.eval_time) if self.eval_time else 0.0
            total_opt_time = total_wall_time - total_llm_time - total_eval_time
            if total_opt_time < 0: total_opt_time = 0.0
            if self.perfs:
                best_perf = -max(self.perfs) if self.lower_is_better else max(self.perfs)
            else:
                best_perf = 0

            summary_row = {
                "ts": "SUMMARY",
                "optimizer": "TOTAL_STATS",
                "iter_idx": len(self.perfs),
                "perf": best_perf,
                "best_so_far": best_perf,
                "algorithm": "ALL",
                "opt_time": total_opt_time,
                "eval_time": total_eval_time,
                "llm_time": total_llm_time,
                "wall_elapsed_sec": total_wall_time,
                "iter_time": 0,
                "llm_cost": total_llm_cost,
                "note": f"Limit: ${self.max_llm_cost} | EarlyStop: {self.early_stopped_flag}"
            }
            
            self.traj_rows.append(summary_row)
            pd.DataFrame(self.traj_rows).to_csv(self.traj_csv_path, index=False)
            print(f"[LLAMBO] Global summary saved. Cost: ${total_llm_cost:.4f}, TotalTime: {total_wall_time:.1f}s (LLM: {total_llm_time:.1f}s, Eval: {total_eval_time:.1f}s)")            
        except Exception as e:
            self.logger.warning(f"[LLAMBO] Failed to save summary: {e}")

    def get_opt_trajectory(self):
        best = min(self.perfs) if self.lower_is_better else max(self.perfs)
        total_llm_cost = sum(self.llm_query_cost) if self.llm_query_cost else 0.0
        
        return {
            "n_evals": len(self.perfs),
            "final_best": best,
            "llm_cost_sum": total_llm_cost,
            "csv_path": self.traj_csv_path
        }

    def run(self):
        while not self.timeout_flag:
            self.iterate()
        return self.incumbent_perf