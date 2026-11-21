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

# [OPRO 特有组件]
from mindware.third_party.opro.opro_acq import OPROLLM_ACQ

from openbox.utils.history import Observation
from mindware.components.optimizers.llm_optimizer_utils import (
    _update_incumbent_from_current,
    _time_is_up,
    wrapper_evaluater
)
from mindware.components.optimizers.bopro_optimizers_utils import _evaluate_dict
from mindware.third_party.opro.utils import _build_hparam_constraints

class OPROOptimizer(BaseOptimizer):
    """
    OPRO Optimizer (Optimization by PROmpting).
    Re-implementation of Google DeepMind's paper "Large Language Models as Optimizers".
    
    Mechanism:
    1. Warm Start: Use KD or Random Sampling to build an initial trajectory.
    2. Optimization Trajectory: Sort history by score (handled by PromptBuilder).
    3. Generation: LLM predicts the next best configuration based on the trajectory.
    """

    def __init__(
        self,
        evaluator,
        config_space: ConfigurationSpace,
        name: str = "OPRO",
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
        self.max_llm_cost = 1.0
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)
        print("[LLAMBOOptimizer] Using chat_engine:", AVAILABLE_LLMs[LLMCONFIGS.opt_chat_engine]['model'])
        self.chat_engine = AVAILABLE_LLMs[LLMCONFIGS.opt_chat_engine]['model']
        self.task_name = kwargs.get("task_name", "CASH")
        self.lower_is_better = kwargs.get("lower_is_better", False)
        
        self.n_candidates = kwargs.get("n_candidates", 1)
        self.configs = [] 
        self.perfs = []   
        

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

        # 4. 初始化 ACQ
        self.acq_func = OPROLLM_ACQ(
           optimizer=self,
           temperature=1.0, # OPRO 论文建议 1.0 以增加多样性
           n_candidates=self.n_candidates
        )

        self.init_num = 12
        self.initial_configs_queue = []
        self._init_warm_start(seed, algo_choices)

        os.makedirs(self.output_dir, exist_ok=True)
        self.llm_query_cost: List[float] = []
        self.llm_query_time: List[float] = []
        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_OPRO_opt_traj.csv")
        self.exp_output = dict()
        self.start_wall_time = None

    def _init_warm_start(self, seed, algo_choices):
        """加载 Knowledge Database 中的配置"""
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
                        self.logger.warning(f"[OPRO] Failed to load KD config: {e}")

        if len(self.initial_configs_queue) > 0:
            print(f"[OPRO] Warm start loaded with {len(self.initial_configs_queue)} configurations.")
        else:
            print(f"[OPRO] KD empty. Will use random initialization for first {self.init_num} steps.")

    def _safe_dict_to_config(self, d: Dict[str, Any]) -> Configuration:
        """
        [复用 BOPRO 逻辑] 将 LLM 的字典转换为 Configuration 对象。
        处理类型转换、范围截断、默认值填充。
        """

        import math
        def _is_nan(x): return isinstance(x, float) and math.isnan(x)
        def _none_like(x): return (x is None) or (isinstance(x, str) and x.strip().lower() in ("none", "null"))

        hp_list = self.config_space.get_hyperparameters()
        hp_map = {hp.name: hp for hp in hp_list}

        algo_hp = hp_map.get('algorithm', None)
        algo = d.get('algorithm')
        if algo is None: # 尝试修复 key
            for k in d.keys():
                if k.strip() == "algorithm":
                    algo = d[k]; break
        
        if algo is None: # 推断
            prefixes = {k.split(':', 1)[0] for k in d.keys() if ':' in k}
            if prefixes: algo = max(prefixes, key=lambda p: len([k for k in d.keys() if k.startswith(p)]))
            elif algo_hp: algo = algo_hp.default_value
            else: raise ValueError("No 'algorithm' provided.")
        
        if isinstance(algo, str): algo = algo.strip()
        
        # 预填充默认值
        values: Dict[str, Any] = {'algorithm': algo}
        for hp in hp_list:
            if hp.name == 'algorithm': continue
            if hp.name.startswith(algo + ':'):
                if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                    val = getattr(hp, 'value', getattr(hp, 'default_value', None))
                    values[hp.name] = val
                elif hasattr(hp, 'default_value'):
                    values[hp.name] = hp.default_value

        # 覆盖 LLM 值
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
                        # 尝试大小写匹配
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
        candidates_to_eval = []
        action_type = ""
        note = ""
        llm_cost = 0.0
        llm_time = 0.0

        # 情况 A: 消耗 KD 热启动队列
        if len(self.initial_configs_queue) > 0:
            action_type = "Warmup (KD)"
            candidates_to_eval.append(self.initial_configs_queue.pop(0))
            note = "KD Warmup"
        
        # 情况 B: 历史数据太少，且 KD 也没了 -> 随机采样补充
        elif len(self.configs) < self.init_num:
            action_type = "Warmup (Random)"
            candidates_to_eval.append(self.config_space.sample_configuration())
            note = f"Random Warmup ({len(self.configs)+1}/{self.init_num})"
        
        # 情况 C: 历史足够 -> OPRO LLM Generation
        else:
            action_type = "OPRO"
            
            obs_cfg_df = pd.DataFrame([c.get_dictionary() for c in self.configs])
            if self.lower_is_better:
                raw_scores = [-p for p in self.perfs]
            else:
                raw_scores = self.perfs
            obs_acc_df = pd.DataFrame({'score': raw_scores})
            
            try:
                # 调用 ACQ
                cand_df, cost, tcost = self.acq_func.get_candidate_points(
                    obs_cfg_df, obs_acc_df, iter_idx=len(self.perfs), n_generate=self.n_candidates
                )
                llm_cost = cost
                llm_time = tcost
                
                if cand_df is not None and not cand_df.empty:
                    # 转换所有候选点 (支持 n_candidates > 1)
                    for _, row in cand_df.iterrows():
                        try:
                            cfg = self._safe_dict_to_config(row.to_dict())
                            candidates_to_eval.append(cfg)
                        except Exception as e:
                            self.logger.warning(f"Failed to convert LLM output to config: {e}")
                    
                    if candidates_to_eval:
                        note = "OPRO Generation"
                    else:
                        note = "OPRO Gen Failed (Conversion)"
                        # Fallback
                        candidates_to_eval.append(self.config_space.sample_configuration())
                else:
                    note = "OPRO Gen Failed (No Return)"
                    candidates_to_eval.append(self.config_space.sample_configuration())

            except Exception as e:
                self.logger.error(f"OPRO ACQ Error: {e}")
                note = f"Error: {e}"
                candidates_to_eval.append(self.config_space.sample_configuration())

        # =====================================================
        # 2. 评估 (Evaluation)
        # =====================================================
        eval_time_total = 0.0
        
        for cfg in candidates_to_eval:
            if _time_is_up(self, self.timestamp): break
            
            # 评估
            obs, eval_time = _evaluate_dict(self, cfg.get_dictionary(), self.config_space)
            eval_time_total += eval_time
            cfg_obj = obs.config
           
            perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)
            max_perf = -float(obs.objectives[0]) 
            
            self.update_saver([cfg_obj], [-max_perf])
            self.configs.append(cfg_obj)
            self.perfs.append(max_perf)
            _update_incumbent_from_current(self)

        # =====================================================
        # 3. 记录日志 (Logging)
        # =====================================================
        iter_time = time.time() - t0
        opt_time = iter_time - eval_time_total
        
        # 记录本轮最后一个点的结果
        if candidates_to_eval:
            self._record_event(
                opt="OPRO",
                iter_idx=len(self.perfs),
                cfg=candidates_to_eval[-1],
                perf=max_perf,
                eval_time=eval_time_total,
                opt_time=opt_time,
                iter_time=iter_time,
                llm_cost=llm_cost,
                llm_time=llm_time,
                note=f"{note} (Batch:{len(candidates_to_eval)})"
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

    def get_opt_trajectory(self):
        """返回优化轨迹概览，供 benchmark 脚本使用"""
        best = float(max(self.perfs)) if self.perfs else float("-inf")   
        total_llm_cost = sum(self.llm_query_cost) if self.llm_query_cost else 0.0
        
        return {
            "n_evals": len(self.perfs),
            "final_best": best,
            "llm_cost_sum": total_llm_cost,
            "llm_calls": len([c for c in self.llm_query_cost if c > 0]), # 统计实际产生开销的调用次数
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join([str(p) for p in self.perfs]),
            "history": [
                {
                    "iter_idx": r["iter_idx"],
                    "perf": r["perf"],
                    "best_so_far": r["best_so_far"],
                    "algorithm": r["algorithm"],
                    "llm_cost": r.get("llm_cost", 0.0),
                    "llm_time": r.get("llm_time", 0.0),
                    "eval_time": r.get("eval_time", 0.0),
                    "iter_time": r.get("iter_time", 0.0),
                    "wall_elapsed_sec": r["wall_elapsed_sec"],
                } for r in self.traj_rows
            ][:500]
        }

    def run(self):
        while not self.timeout_flag:
            self.iterate()
        return self.incumbent_perf