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
from openbox.utils.history import Observation, History
from openbox.utils.constants import SUCCESS

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.advisor.generic_advisor import MyAdvisor
from mindware.third_party.opro.bopro_acq import BOPROLLM_ACQ
from mindware.third_party.opro.rate_limiter import RateLimiter

from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS
from mindware.components.utils.bopro_embedding import BOPROEmbedding 
from openbox.utils.history import Observation
from openbox.utils.limit import run_obj_func
from openbox.utils.util_funcs import parse_result
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from mindware.components.optimizers.llm_optimizer_utils import (
    _update_incumbent_from_current,
    _record_event,
    _time_is_up,
    wrapper_evaluater
)

from mindware.components.optimizers.bopro_optimizers_utils import _evaluate_dict
from mindware.third_party.opro.utils import _build_hparam_constraints

class BOPROOptimizer(BaseOptimizer):
    """
    BOPRO Optimizer for CASH (Joint Algorithm & Hyperparameter Optimization).
    
    Mechanism:
    1. Use a GLOBAL BO Advisor (SMAC) to propose a 'latent' candidate in the full CASH space.
    2. Retrieve top-k historical configurations (across all algos) that are semantically similar 
       to this Proposal (using SimCSE text embedding + Cosine Similarity).
    3. Use these similar configurations as in-context examples to prompt the LLM.
    4. Evaluate the LLM-generated candidate.
    """

    def __init__(
        self,
        evaluator,
        config_space: ConfigurationSpace,
        name: str = "BOPRO",
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
        n_jobs: int=1,
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
        # 将evaluator包装为超时控制的形式
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)

        print("[BOPROOptimizer] Using chat_engine:", kwargs.get("chat_engine", LLMCONFIGS.opt_chat_engine))
         # --- LLM Parameters ---
        self.chat_engine = AVAILABLE_LLMs[LLMCONFIGS.opt_chat_engine]['model']
        # self.use_input_warping = kwargs.get("use_input_warping", False)
        self.n_candidates = kwargs.get("n_candidates", 1)
        self.n_templates = kwargs.get("n_templates", 1) 
        self.task_name = kwargs.get("task_name", "CASH")
        


        # --- 初始化 Knowledge Database ---
        # self.KD = KnowledgeDatabase.instance()
        # self.configs = self.KD.configs
        # self.perfs = self.KD.perfs
        # self.lower_is_better = self.KD.lower_is_better
        self.configs = [] 
        self.perfs = []   # 存储 score (越大越好)
        
        # 2. 用于 OpenBox BO Advisor 的标准 History 对象
        self.history = History(
            task_id=task_id,
            config_space=config_space,
            ref_point=None,
            meta_info=None,
        )

        self.lower_is_better = kwargs.get("lower_is_better", False)

        # --- 初始化 BOPRO Embedding 组件 ---
        self.embedding_model = BOPROEmbedding.instance()
        self.embedding_model.reset() 
        

        rng = np.random.RandomState(seed)

        # 1. 获取算法列表
        algo_hp = self.config_space.get_hyperparameter('algorithm')
        algo_choices = list(algo_hp.choices)
        
        # 2. 从 KD 获取每个算法的最佳配置
        self.KD = KnowledgeDatabase.instance()
        bestconf_knowledge = self.KD.fetch_best_conf_knowledge()
        
        initial_configs_objs = [] 
        algo_queues = {}

        for alg in algo_choices:
            if alg in bestconf_knowledge and "Best Configurations" in bestconf_knowledge[alg]:
                confs = list(bestconf_knowledge[alg]["Best Configurations"])
                if len(confs) > 0:
                    idx_to_remove = rng.randint(0, len(confs))
                    removed_conf = confs.pop(idx_to_remove)                
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
                        initial_configs_objs.append(config_obj)
                    except Exception as e:
                        print(f"[Warning] Failed to convert KD config for {alg}: {e}")
                        continue

        if len(initial_configs_objs) > 0:
            final_init_trials = len(initial_configs_objs)
            print(f"[BOPRO] Warm start with {final_init_trials} configurations (randomly dropped 1 per algo).")
        else:
            final_init_trials = 18 
            print(f"[BOPRO] KD empty. Fallback to {final_init_trials} random initial trials.")



        self.advisor = MyAdvisor(
            config_space=self.config_space,
            num_objectives=1,
            num_constraints=0,
            initial_strategy='random',
            initial_trials=final_init_trials,
            initial_configurations=initial_configs_objs if len(initial_configs_objs) > 0 else None, 
            rand_prob=0.1,   
            acq_optimizer_rand_prob=0.0, 
            surrogate_type='gp', 
            acq_type='ei',
            task_id=f"{task_id}_global_bo",
            output_dir=output_dir,
            random_state=seed
        )
        self.advisor.history = self.history 
    
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
        self.task_context["metric"] = profile["metric"]
        self.task_context.update({
            "output_dir": self.output_dir,
            "run_id": self.timestamp,
            "dump_llm_templates": kwargs.get("dump_llm_templates", True),
            "chat_engine": self.chat_engine
        })

        self.rate_limiter = RateLimiter(max_tokens=100000, time_frame=60, max_requests=2000)
        self.acq_func = BOPROLLM_ACQ(
           optimizer=self,
           chat_engine=self.chat_engine,
           temperature=0.7,
           n_candidates=self.n_candidates
        )

        _uniq = set()
        for _ in range(5000):
            cfg = self.config_space.sample_configuration()
            _uniq.add(tuple(sorted(cfg.get_dictionary().items())))
        self.config_num_threshold = max(1, len(_uniq))
        self.maximum_config_num = min(1500, self.config_num_threshold)
        self.logger.debug(f"[LLMOptimizer] estimated unique configs: {self.config_num_threshold}")

        os.makedirs(self.output_dir, exist_ok=True)
        
        self.llm_query_cost: List[float] = []
        self.llm_query_time: List[float] = []
        self.exp_output = dict()
        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_LLM_opt_traj.csv")
        self.start_wall_time = None


    def _safe_dict_to_config(self, d: Dict[str, Any]) -> Configuration:
        """
        Robustly converts a dictionary (from LLM) into a valid ConfigSpace.Configuration.
        Handles:
        - Type casting (float -> int, str -> categorical)
        - Constant/UnParametrized parameters (ignores LLM output, uses default)
        - Range clipping
        - Active/Inactive logic (CASH)
        """
        import math
        
        # 内部帮助函数
        def _is_nan(x):
            return isinstance(x, float) and math.isnan(x)

        def _none_like(x):
            return (x is None) or (isinstance(x, str) and x.strip().lower() in ("none", "null"))

        hp_list = self.config_space.get_hyperparameters()
        hp_map = {hp.name: hp for hp in hp_list}

        # 1. 确定当前选择的算法 (Algorithm)
        algo_hp = hp_map.get('algorithm', None)
        algo = d.get('algorithm')
        
        # 尝试处理 Key 中可能存在的空格 (LLM 常见问题)
        if algo is None:
            for k in d.keys():
                if k.strip() == "algorithm":
                    algo = d[k]
                    break
        
        # 如果还是没找到，尝试从前缀推断 (e.g., xgboost:n_estimators -> xgboost)
        if algo is None:
            prefixes = {k.split(':', 1)[0] for k in d.keys() if ':' in k}
            if prefixes:
                # 统计出现最多的前缀作为算法
                algo = max(prefixes, key=lambda p: len([k for k in d.keys() if k.startswith(p)]))
            elif algo_hp is not None:
                algo = algo_hp.default_value
            else:
                # 实在没招了，随机选一个或者报错，这里选择 Fallback
                raise ValueError("No 'algorithm' provided and cannot infer from keys.")
        
        # 清洗 algo 字符串
        if isinstance(algo, str):
            algo = algo.strip()

        if algo_hp is not None and algo not in list(algo_hp.choices):
            raise ValueError(f"Invalid algorithm '{algo}'. Choices: {algo_hp.choices}")

        # 2. 预填充该算法分支的默认值 (Pre-fill defaults)
        values: Dict[str, Any] = {'algorithm': algo}
        for hp in hp_list:
            name = hp.name
            if name == 'algorithm':
                continue
            
            # 只处理属于当前算法的参数 (CASH 逻辑: algo:param)
            if name.startswith(algo + ':'):
                if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                    # [关键修复] 对于 Constant，永远使用定义的值，忽略 LLM 的输入
                    val = getattr(hp, 'value', getattr(hp, 'default_value', None))
                    values[name] = val
                else:
                    if hasattr(hp, 'default_value'):
                        values[name] = hp.default_value

        # 3. 使用 LLM 的字典覆盖默认值 (Override with LLM inputs)
        for k_in, v_in in d.items():
            k_clean = k_in.strip() # 去除键名空格
            
            if k_clean == 'algorithm':
                continue
            if _is_nan(v_in):
                continue
            if _none_like(v_in):
                v_in = None

            # 处理参数名映射 (LLM 可能只输出了 'learning_rate' 而不是 'xgb:learning_rate')
            if not k_clean.startswith(algo + ':'):
                target = f"{algo}:{k_clean}"
                if target in hp_map:
                    k = target
                else:
                    continue # 既不是全名也不是短名，可能是幻觉参数，跳过
            else:
                k = k_clean

            hp = hp_map.get(k)
            
            # [关键] 如果是 Constant，直接跳过覆盖步骤，保留上面的默认值
            if hp is None or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                continue

            # 类型转换与范围检查
            try:
                if isinstance(hp, UniformIntegerHyperparameter):
                    v = int(round(float(v_in)))
                    if v < hp.lower: v = hp.lower
                    if v > hp.upper: v = hp.upper
                
                elif isinstance(hp, UniformFloatHyperparameter):
                    if v_in is None:
                        v = None
                    else:
                        v = float(v_in)
                        if v < hp.lower: v = hp.lower
                        if v > hp.upper: v = hp.upper
                
                elif isinstance(hp, CategoricalHyperparameter):
                    choices = list(hp.choices)
                    # 尝试清理字符串值 (比如 " 'gini' " -> "gini")
                    v_str = str(v_in).strip().strip("'").strip('"')
                    
                    if v_str not in choices:
                        # 尝试做一下类型转换匹配 (比如 LLM 输出 "true" 但选项是 True)
                        matched = False
                        for c in choices:
                            if str(c).lower() == v_str.lower():
                                v = c
                                matched = True
                                break
                        if not matched:
                            continue # 值不合法，使用默认值
                    else:
                        v = v_str # 这里可能需要根据 choices 的真实类型再转一次，但通常 ConfigSpace 能处理匹配的 str
                        # 为了保险，找到原始对象
                        for c in choices:
                            if str(c) == v_str:
                                v = c
                                break
                else:
                    v = v_in
            except Exception:
                continue # 转换失败，保留默认值

            values[k] = v

        # 返回构建好的 Configuration 对象
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

        try:
            proposal_config = self.advisor.get_suggestion()
            bo_suggestion_time = time.time() - t0
        except Exception as e:
            self.logger.warning(f"[BOPRO] Advisor failed to suggest: {e}. Fallback to random.")
            import traceback
            traceback.print_exc()
            proposal_config = self.config_space.sample_configuration()
            bo_suggestion_time = time.time() - t0


        
        history_configs = self.configs
        history_perfs = self.perfs # 这些通常是评估指标 (越大越好)
        
        embedding_time = 0.0
        similraity_time = 0.0
        # 如果历史太少，退化为 BO Proposal 直接评估
        if len(history_configs) < self.advisor.init_num:
            final_config = proposal_config
            note = "Warmup to get enough history"
        else:
            t0_similarity = time.time()
            sim_scores = self.embedding_model.get_similarity_scores(proposal_config)
            similraity_time = time.time() - t0_similarity

            k = min(len(history_configs), 5)
            top_k_indices = np.argsort(sim_scores)[-k:][::-1]

            context_dicts = []
            context_scores = []
            
            for idx in top_k_indices:
                cfg = history_configs[idx]
                score = history_perfs[idx] # 假设已经是越大越好的 score
                context_dicts.append(cfg.get_dictionary())
                context_scores.append(score)
            
            obs_cfg_df = pd.DataFrame(context_dicts)

            if self.lower_is_better:                
                raw_context_scores = [-s for s in context_scores]
            else:
                raw_context_scores = context_scores

            obs_acc_df = pd.DataFrame({'score': raw_context_scores})
            
            
            # -----------------------------------------------------------------
            # [Step 3] LLM Generation: 基于相似样本 Prompting
            # -----------------------------------------------------------------
            try:
                cand_df, cost, tcost = self.acq_func.get_candidate_points(
                    obs_cfg_df, obs_acc_df, alpha=0.0, iter_idx=len(self.perfs)
                )
                
                self.llm_query_cost.append(cost)
                self.llm_query_time.append(tcost)

                if cand_df is not None and len(cand_df) > 0:
                    llm_config_dict = cand_df.iloc[0].to_dict()
                    

                    # 尝试转换为 Configuration 对象 (包含合法性检查)
                    try:
                        final_config = self._safe_dict_to_config(llm_config_dict)
                        note = "BOPRO Generation"
                    except Exception as e:
                        # 如果 LLM 生成的配置不合法 (比如参数越界，或者缺少条件参数)
                        # 我们回退到 BO 的 Proposal，保证流程不中断
                        self.logger.warning(f"LLM generated invalid config: {e}. Fallback to BO Proposal.")
                        final_config = proposal_config
                        note = "Fallback (Invalid LLM Config)"
                else:
                    final_config = proposal_config
                    note = "Fallback (LLM No Return)"

            except Exception as e:
                self.logger.error(f"LLM ACQ Error: {e}")
                final_config = proposal_config
                note = f"Fallback (Error: {str(e)})"

        # ---------------------------------------------------------------------
        # [Step 4] Evaluation: 评估最终配置
        # ---------------------------------------------------------------------
        
        
        # 将 Configuration 放入 Observation 包装 (OpenBox 标准)
        # 注意：这里我们还没有 Observation 对象，先评估
        cfg_dict = final_config.get_dictionary()
        
        # 评估
        obs, eval_time = _evaluate_dict(self, cfg_dict, self.config_space)
        cfg_obj = obs.config
        perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)
        max_perf = -float(obs.objectives[0]) # 转换为越大越好 (KD和Saver通常需要)
        
        # ---------------------------------------------------------------------
        # [Step 5] Update: 更新数据库和 Advisor
        # ---------------------------------------------------------------------
        
        # 1. 更新 KD (KnowledgeDatabase)
        self.update_saver([cfg_obj], [-max_perf])
        self.configs.append(cfg_obj)
        self.perfs.append(max_perf)
        self.history.update_observation(obs)

        t0_embedding = time.time()
        self.embedding_model.add_observation(cfg_obj)
        embedding_time = time.time() - t0_embedding
        
        _update_incumbent_from_current(self)
        
        # try:
        #     self.advisor.update_observation(obs)
        # except Exception as e:
        #     self.logger.warning(f"Failed to update advisor: {e}")

        iter_time = time.time() - t0
        opt_time = iter_time - eval_time
        if opt_time < 0:
            opt_time = 0

        # 5. 记录详细日志
        self._record_event(
            opt="BOPRO",
            iter_idx=len(self.perfs),
            cfg=obs.config,
            perf=max_perf,
            eval_time=eval_time,
            opt_time=opt_time, 
            bo_suggestion_time = bo_suggestion_time,
            embedding_time=embedding_time,
            similarity_time=similraity_time,
            iter_time=iter_time,
            llm_cost=cost if 'cost' in locals() else 0.0,
            llm_time=tcost if 'tcost' in locals() else 0.0,
            note=note
        )


        return self.incumbent_perf, iter_time , self.incumbent_config

    def _record_event(self, opt, iter_idx, cfg, perf,iter_time=0.0, eval_time=0.0, opt_time=0.0,bo_suggestion_time=0.0,embedding_time=0.0,similarity_time=0.0,llm_cost=0.0,llm_time=0.0,note="", **kwargs):
        wall_elapsed_sec = time.time() - self.timestamp
        algo = cfg.get('algorithm', 'default')
        
        row = {
            "ts": pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "optimizer": opt,
            "iter_idx": iter_idx,
            "perf": perf,
            "best_so_far": max(self.perfs) if self.perfs else float("-inf"),
            "algorithm": algo,
            "iter_time": iter_time,
            "eval_time": eval_time,
            "opt_time": opt_time,
            "bo_suggestion_time":bo_suggestion_time,
            "embedding_time": embedding_time,
            "similarity_time": similarity_time,
            "llm_time": llm_time,   
            "llm_cost": llm_cost,   
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
        best = float(max(self.perfs)) if self.perfs else float("-inf")   
            
        total_llm_cost = float(sum(self.llm_query_cost) if self.llm_query_cost else 0.0)


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
        # return {
        #     "final_best": self.incumbent_perf,
        #     "csv_path": self.traj_csv_path
        # }

    def run(self):
        while not self.timeout_flag:
            self.iterate()
        return self.incumbent_perf