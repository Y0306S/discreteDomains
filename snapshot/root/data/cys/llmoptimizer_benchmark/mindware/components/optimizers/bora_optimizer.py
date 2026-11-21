import os
import time
import math
import numpy as np
import pandas as pd
from functools import partial
from typing import List, Dict, Any
from collections import deque  # 用于维护信任历史的滑动窗口

from typing import Tuple
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter, 
    UniformIntegerHyperparameter, 
    UniformFloatHyperparameter, 
    Constant,
    UnParametrizedHyperparameter
)

from openbox.utils.history import History
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.advisor.generic_advisor import MyAdvisor
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS

# 引入 BORA 组件 (稍后我们会编写 ACQ)
from mindware.third_party.opro.bora_acq import BORALLM_ACQ
from mindware.third_party.opro.rate_limiter import RateLimiter

from mindware.components.optimizers.llm_optimizer_utils import (
    _update_incumbent_from_current,
    _record_event,
    _time_is_up,
    wrapper_evaluater
)
from mindware.components.optimizers.bopro_optimizers_utils import _evaluate_dict
from mindware.third_party.opro.utils import _build_hparam_constraints

class BORAOptimizer(BaseOptimizer):
    """
    BORA: Bayesian Optimization with Research Assistant.
    
    Implements a state-machine based optimizer that dynamically switches between:
    - a1: Vanilla BO (Exploitation)
    - a2: LLM Suggests (High Exploration)
    - a3: LLM Selects (Guided Exploration)
    
    Based on GP uncertainty and performance plateau detection.
    """

    def __init__(
        self,
        evaluator,
        config_space: ConfigurationSpace,
        name: str = "BORA",
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
        # 1. 基础设置
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)
        print("[BORAOptimizer] Using chat_engine:", kwargs.get("chat_engine", LLMCONFIGS.opt_chat_engine))
        self.chat_engine = AVAILABLE_LLMs[LLMCONFIGS.opt_chat_engine]['model']
        self.task_name = kwargs.get("task_name", "CASH")
        self.lower_is_better = kwargs.get("lower_is_better", False)

        # 2. 初始化数据容器
        self.configs = [] 
        self.perfs = []
        self.history = History(
            task_id=task_id,
            config_space=config_space,
            ref_point=None,
            meta_info=None,
        )
        
        # BORA 特有：记录 LLM 的历史评论 (Context C)
        self.comments_history: List[str] = []

        # =========================================================
        # [BORA 核心参数初始化] 严格遵循论文配置
        # =========================================================
        
        # --- A. 停滞检测 (Plateau Detection) ---
        # "A performance plateau is detected... w.r.t a set percentage gamma"
        self.gamma = 0.05  # 5% 阈值

        # --- B. 自适应窗口 (Adaptive Window m) ---
        # "m is initialized at m_init = ceil(2 * sqrt(d))"
        # 计算 CASH 空间的近似维度 d
        # 这里我们取所有超参数的数量，虽然对于 conditional space 来说这是上界，但比较稳健
        d_approx = len(self.config_space.get_hyperparameters())
        self.m_init = int(math.ceil(2 * math.sqrt(d_approx)))
        # 动态变化的当前窗口 m
        self.m = self.m_init
        
        # 窗口边界 "vary between m_min = 0 and m_max = 3 * m_init"
        self.m_min = 0
        self.m_max = 3 * self.m_init
        
        # 调整步长 "Delta_max = 15"
        self.delta_max = 15

        # --- C. 信任机制 (Trust Mechanism) ---
        # "H is initialized as {0.9}"
        self.trust_history = deque([0.9], maxlen=3) # 滑动窗口 W=3
        self.current_trust = 0.9 # T_l
        self.epsilon = 1e-6 # 防止除零

        # --- D. 不确定性监控 (Uncertainty) ---
        # "set of fixed q monitoring points x_mon"
        # 我们随机采样 5000 个点作为监控集，永久固定
        rng_mon = np.random.RandomState(seed)
        self.monitoring_configs = self.config_space.sample_configuration(500)
        self.monitoring_arrays = np.array([cfg.get_array() for cfg in self.monitoring_configs])
        
        # 记录历史最大的不确定性，用于归一化
        self.sigma_max_global = 1e-9 
        # 阈值比例 "0.5 * sigma_max and 0.3 * sigma_max"
        self.sigma_upper_ratio = 0.5
        self.sigma_lower_ratio = 0.3
        
        # --- E. 动作采样参数 (Action Parameters) ---
        self.n_bo_candidates = 5 # a3: n_BO
        self.n_lbo_selected = 1  # a3: n_LBO
        self.n_llm_suggest = 1   # a2: n_LLM (建议值5)

        # =========================================================
        # [Warm Start] 复用 BOPRO 的 KD 热启动逻辑
        # =========================================================
        
        rng = np.random.RandomState(seed)
        algo_hp = self.config_space.get_hyperparameter('algorithm')
        algo_choices = list(algo_hp.choices)
        
        self.KD = KnowledgeDatabase.instance()
        bestconf_knowledge = self.KD.fetch_best_conf_knowledge()
        
        initial_configs_objs = [] 
        algo_queues = {}

        # 构建队列 & 随机丢弃 (BOPRO Logic)
        for alg in algo_choices:
            if alg in bestconf_knowledge and "Best Configurations" in bestconf_knowledge[alg]:
                confs = list(bestconf_knowledge[alg]["Best Configurations"])
                if len(confs) > 0:
                    idx_to_remove = rng.randint(0, len(confs))
                    removed_conf = confs.pop(idx_to_remove)
                algo_queues[alg] = confs
            else:
                algo_queues[alg] = []
        
        # 轮询提取
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

        # 确定 initial_trials
        if len(initial_configs_objs) > 0:
            final_init_trials = len(initial_configs_objs)
            print(f"[BORA] Warm start with {final_init_trials} configurations (randomly dropped 1 per algo).")
        else:
            final_init_trials = 5 
            print(f"[BORA] KD empty. Fallback to {final_init_trials} random initial trials.")

 
        self.advisor = MyAdvisor(
            config_space=self.config_space,
            num_objectives=1,
            num_constraints=0,
            initial_strategy='random',
            # initial_trials=final_init_trials,
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
         # Context setup for Prompt Builder
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

        # --- BORA ACQ Initialization ---
        self.acq_func = BORALLM_ACQ(optimizer=self, temperature=0.7)
        
       
        
        _uniq = set()
        for _ in range(5000):
            cfg = self.config_space.sample_configuration()
            _uniq.add(tuple(sorted(cfg.get_dictionary().items())))
        self.config_num_threshold = max(1, len(_uniq))
        self.maximum_config_num = min(1500, self.config_num_threshold)
        self.logger.debug(f"[LLMOptimizer] estimated unique configs: {self.config_num_threshold}")

        # Logging Setup
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm_query_cost: List[float] = []
        self.llm_query_time: List[float] = []
        self.exp_output = dict()
        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_BORA_opt_traj.csv")
        self.start_wall_time = None
        
        # 性能优化：a2训练计数器（每5轮训练一次）
        self._a2_train_counter = 0

    def _calculate_uncertainty(self, skip_training: bool = False) -> Tuple[float, float]:
        """
        计算代理模型 (Common GP/PRF) 在固定监控点上的平均不确定性。
        对应论文公式 (2): sigma_mean = 1/q * sum(sigma(x_mon))
        
        Args:
            skip_training: 如果为True，跳过GP训练（用于a1和a3，因为它们会通过get_suggestion训练）
        """
        if self.advisor.surrogate_model is None:
            return 0.0, 0.0

        try:
            t1 = time.time()
            X = self.history.get_config_array(transform='scale')
            Y = self.history.get_objectives(transform='infeasible')
            t_get_data = time.time() - t1

            if len(X) == 0:
                return 0.0, 0.0

            # 训练模型（如果需要）
            t_train = 0.0
            # 即使skip_training=True，如果模型还没有被训练过，也需要先训练一次
            need_training = not skip_training or (not hasattr(self.advisor.surrogate_model, 'is_trained') or not self.advisor.surrogate_model.is_trained)
            
            if need_training:
                if len(X) % 10 == 0 or len(X) <= 20:
                    self.logger.info(f"[BORA GP Training Data] n_samples={len(X)}, n_features={X.shape[1]}, Y_shape={Y.shape}, monitoring_points={len(self.monitoring_arrays)}")
                
                t2 = time.time()
                self.advisor.surrogate_model.train(X, Y[:, 0])
                t_train = time.time() - t2

            # 进行预测
            t3 = time.time()
            _mean, var = self.advisor.surrogate_model.predict(self.monitoring_arrays)
            t_predict = time.time() - t3
            
            # 记录耗时（仅在训练或耗时较长时）
            total_time = t_get_data + t_train + t_predict
            if need_training and total_time > 0.5:
                self.logger.info(f"[BORA Uncertainty Timing] Data: {t_get_data:.3f}s, Train: {t_train:.3f}s, Predict: {t_predict:.3f}s, Total: {total_time:.3f}s (n_samples={len(X)})")
            elif not need_training and total_time > 0.5:
                self.logger.info(f"[BORA Uncertainty Timing] Data: {t_get_data:.3f}s, Train: skipped, Predict: {t_predict:.3f}s, Total: {total_time:.3f}s (n_samples={len(X)})")
            
            var = np.clip(var, 1e-9, np.inf) 
            std = np.sqrt(var)
            
            sigma_mean = np.mean(std)
            sigma_current_max = np.max(std)
            
            return sigma_mean, sigma_current_max
        except Exception as e:
            self.logger.warning(f"[BORA] _calculate_uncertainty failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0.0, 0.0
    
    def _check_plateau(self) -> bool:
        """
        检测是否停滞 (Plateau Detection)。
        逻辑：检查过去 m 步中，最佳值是否有显著提升 ( > gamma)。
        对应论文公式 (5)。
       
        """
        # 如果历史数据不足 m+1 步，无法判断停滞
        if len(self.perfs) < self.m + 1:
            return False 
        
        # 1. 获取 "Best-so-far" 轨迹
        # self.perfs 存储的是每一步的性能。我们需要计算截止到每一步的历史最佳值。
        # np.maximum.accumulate([1, 0, 2, 1]) -> [1, 1, 2, 2]
        best_trace = np.maximum.accumulate(self.perfs)
        
        # 2. 确定检查窗口
        # 我们需要检查从 t-m+1 到 t 的这一段区间
        # 这里的 t 对应 len(best_trace) - 1
        t = len(best_trace) - 1
        start_idx = t - self.m + 1
        
        # 3. 遍历检查显著提升
        # 只要在这个窗口内，存在任意一步 j，满足 y_j_max > y_{j-1}_max * (1 + gamma)
        # 就说明"有提升"，即"非停滞"
        
        is_plateau = True # 默认假设是停滞的，除非找到反例
        
        for j in range(start_idx, t + 1):
            if j <= 0: continue # 保护索引
            
            y_j_max = best_trace[j]
            y_prev_max = best_trace[j-1]
            
            # 计算提升阈值
            # 注意处理负数 metric 的情况 (sign)
            # 公式: y_prev * (1 + sign(y_prev) * gamma)
            threshold = y_prev_max * (1 + np.sign(y_prev_max) * self.gamma)
            
            # 如果这一步的最佳值超过了阈值，说明有显著提升
            if y_j_max > threshold:
                is_plateau = False 
                break
        
        # self.logger.debug(f"[BORA Plateau] Window m={self.m}, Detected={is_plateau}")
        return is_plateau

    def _update_trust_mechanism(self, llm_perfs: List[float]):
        """
        更新信任度 T_l 和停滞窗口 m。
        在 Action a2 或 a3 执行后调用。
        对应论文 Trust Mechanism 章节及公式 (6-10)。
       
        
        Args:
            llm_perfs: LLM 刚刚生成的这批配置的实际评估分数
        """
        if not llm_perfs:
            return

        # 1. 计算 Reward r_l (Eq 8)
        # LLM 这批点里的最大值
        current_batch_max = max(llm_perfs)
        
        # 调用 LLM 之前的历史最大值
        # 注意：此时 self.perfs 已经包含了 llm_perfs (因为我们在 iterate 里先评估再更新)
        # 所以我们需要倒回去看
        idx_before_batch = len(self.perfs) - len(llm_perfs)
        if idx_before_batch > 0:
            prev_history_perfs = self.perfs[:idx_before_batch]
            y_prev_max = max(prev_history_perfs)
        else:
            y_prev_max = -np.inf # 极少见情况，全是 LLM 生成的
        
        r_l = current_batch_max - y_prev_max
        
        # 2. 计算 Score (Eq 9)
        if r_l > 0:
            score = 1.0 # 成功突破
        else:
            val_y = abs(y_prev_max) if np.isfinite(y_prev_max) else 1.0
            denominator_term = val_y + self.epsilon
            exponent = - (r_l / denominator_term)
            score = 1.0 / (1.0 + np.exp(exponent))
            
        # 3. 更新信任历史 H
        self.trust_history.append(score)
        
        # 4. 计算当前平均信任度 T_l (Eq 10)
        T_l = np.mean(self.trust_history)
        
        # 5. 计算调整量 (Eq 6)
        # T_l_prev 是上一次更新后的信任度
        T_l_prev = self.current_trust
        diff = T_l - T_l_prev

        # 更新当前信任度供下次使用
        self.current_trust = T_l 
        
        # 公式: floor((T_l - T_l-1) * Delta_max)
        m_adjustment = math.floor(diff * self.delta_max)
        
        # 6. 更新窗口 m (Eq 7)
        # 逻辑：信任度增加 -> adjustment > 0 -> m 减小 -> 更频繁调用 LLM
        old_m = self.m
        new_m = self.m - m_adjustment
        
        # Clip 限制范围
        self.m = max(self.m_min, min(self.m_max, new_m))
        
        self.logger.info(f"[BORA Trust] BatchMax: {current_batch_max:.4f}, PrevMax: {y_prev_max:.4f}, r_l: {r_l:.4f}")
        self.logger.info(f"[BORA Trust] Score: {score:.4f} (Sigmoid exp: {-r_l/(abs(y_prev_max)+1e-6):.2f})")
        self.logger.info(f"[BORA Trust] T_prev: {T_l_prev:.3f} -> T_curr: {T_l:.3f}")
        self.logger.info(f"[BORA Trust] Adj: {m_adjustment} (floor({diff:.3f}*15)), m: {old_m} -> {self.m}")

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
        
    
    def _perform_action_a1(self) -> Tuple[List[Configuration], str]:
        """
        Action a1: Vanilla BO (Exploitation).
       
        """
        note = "a1: Vanilla BO"
        try:
            # BO 推荐 1 个点
            t_start = time.time()
            conf = self.advisor.get_suggestion()
            t_elapsed = time.time() - t_start
            if t_elapsed > 1.0:
                self.logger.info(f"[BORA Action a1 Timing] get_suggestion took {t_elapsed:.3f}s")
            return [conf], note, 0.0, 0.0
        except Exception as e:
            self.logger.warning(f"[BORA] a1 failed: {e}. Fallback to random.")
            return [self.config_space.sample_configuration()], "a1 (Fallback)", 0.0, 0.0
    
    def _perform_action_a2(self) -> Tuple[List[Configuration], str]:
        """
        Action a2: LLM Suggests (High Exploration).
       
        LLM takes over and suggests n_LLM points.
        """
        note = "a2: LLM Suggest"
        
        # 1. 准备上下文数据 (Data D and Comments C)
        obs_cfg_df = pd.DataFrame([c.get_dictionary() for c in self.configs])
        if self.lower_is_better:
            raw_scores = [-p for p in self.perfs]
        else:
            raw_scores = self.perfs
        obs_acc_df = pd.DataFrame({'score': raw_scores})
        
        # 2. 调用 ACQ (需等待异步结果)
        # 注意: n_generate 对应论文中的 n_LLM
        llm_dicts, insight, cost, tcost = self.acq_func.get_action_a2_suggestions(
            obs_cfg_df, obs_acc_df, self.comments_history, self.n_llm_suggest
        )
        
        # 3. 记录开销与 Insight
        if insight: 
            self.comments_history.append(insight)
        self.llm_query_cost.append(cost)
        self.llm_query_time.append(tcost)
        
        # 4. 转换结果
        candidates = []
        for d in llm_dicts:
            try:
                candidates.append(self._safe_dict_to_config(d))
            except Exception:
                pass
        
        if not candidates:
            self.logger.warning("[BORA] a2 returned no valid configs. Fallback to BO.")
            return self._perform_action_a1() # Fallback

        return candidates, note, cost, tcost

    def _perform_action_a3(self) -> Tuple[List[Configuration], str]:
        """
        Action a3: LLM Selects (Guided Exploration).
       
        BO proposes n_BO points, LLM selects n_LBO points.
        """
        note = "a3: LLM Select"
        
        # 1. BO 生成候选集 (n_BO)
        try:
            # get_suggestion(return_list=True) 可以一次拿多个点
            # 这是一个循环调用，如果 OpenBox 版本不支持 batch，需要循环调用
            t_bo_start = time.time()
            bo_candidates = []
            for i in range(self.n_bo_candidates):
                t_call_start = time.time()
                bo_candidates.append(self.advisor.get_suggestion())
                t_call_elapsed = time.time() - t_call_start
                if t_call_elapsed > 1.0:
                    self.logger.info(f"[BORA Action a3 Timing] get_suggestion[{i+1}/{self.n_bo_candidates}] took {t_call_elapsed:.3f}s")
            t_bo_total = time.time() - t_bo_start
            if t_bo_total > 2.0:
                self.logger.info(f"[BORA Action a3 Timing] Total BO generation ({self.n_bo_candidates} calls) took {t_bo_total:.3f}s")
        except Exception as e:
            self.logger.warning(f"[BORA] a3 BO generation failed: {e}")
            return self._perform_action_a1()

        # 2. 准备上下文
        obs_cfg_df = pd.DataFrame([c.get_dictionary() for c in self.configs])
        if self.lower_is_better:
            raw_scores = [-p for p in self.perfs]
        else:
            raw_scores = self.perfs
        obs_acc_df = pd.DataFrame({'score': raw_scores})
        
        # 3. 调用 ACQ 进行筛选
        llm_dicts, insight, cost, tcost = self.acq_func.get_action_a3_selection(
            obs_cfg_df, obs_acc_df, self.comments_history, bo_candidates, n_select=self.n_lbo_selected
        )
        
        # 4. 记录
        if insight: 
            self.comments_history.append(insight)
        self.llm_query_cost.append(cost)
        self.llm_query_time.append(tcost)
        
        # 5. 转换结果
        candidates = []
        for d in llm_dicts:
            try:
                candidates.append(self._safe_dict_to_config(d))
            except Exception:
                pass
        
        if not candidates:
            self.logger.warning("[BORA] a3 returned no valid selection. Fallback to BO candidates.")
            # 如果 LLM 选不出来，为了不浪费，可以评估 BO 提出来的最好的 1-2 个
            return bo_candidates[:self.n_lbo_selected], "a3 (Fallback Select)"

        return candidates, note, cost, tcost
    
    def iterate(self, budget=MAX_INT):
        
        
        if self.early_stopped_flag:
            return self.incumbent_perf, 0, self.incumbent_config
        
        current_total_cost = self.acq_func.get_total_llm_cost()
        # if current_total_cost > self.max_llm_cost:
        #     self.logger.info(f"[LLAMBO] STOPPING: Cost limit exceeded (${current_total_cost:.4f} > ${self.max_llm_cost})")
        #     self.early_stopped_flag = True 
        #     return self.incumbent_perf, 0, self.incumbent_config
        
        if _time_is_up(self, self.timestamp):
            self.timeout_flag = True
            return self.incumbent_perf, 0, self.incumbent_config
        t0 = time.time()
        timing = {}
        
        # 判断当前阶段
        is_warmup = len(self.configs) < self.advisor.init_num

        # -----------------------------------------------------
        # 1. Update Policy Parameters (Sigma) [Algorithm 1 Line 3, 15]
        # -----------------------------------------------------
        # 性能优化：先计算uncertainty（不训练）用于决策，然后根据action_type决定是否需要额外训练
        # - a1/a3: 跳过训练（get_suggestion会训练）
        # - a2: 需要训练，但每5轮训练一次
        t_sigma_start = time.time()
        if not is_warmup:
            # 先计算uncertainty（不训练）用于决策
            sigma_mean, sigma_current_max = self._calculate_uncertainty(skip_training=True)
            
            # 更新历史最大值 (水位线)
            if sigma_current_max > self.sigma_max_global:
                self.sigma_max_global = sigma_current_max
            
            # 动态更新阈值
            sigma_upper = self.sigma_upper_ratio * self.sigma_max_global
            sigma_lower = self.sigma_lower_ratio * self.sigma_max_global
        else:
            sigma_mean = 0.0
            sigma_upper = 0.0
            sigma_lower = 0.0
        timing['sigma_calculation'] = time.time() - t_sigma_start

        # Logging
        iter_idx = len(self.perfs) + 1
        if not is_warmup:
            self.logger.debug(f"[BORA Uncertainty] ({iter_idx}) sigma_mean: {sigma_mean:.4f}, sigma_max: {self.sigma_max_global:.4f}, Thresholds: [{sigma_lower:.4f}, {sigma_upper:.4f}]")
        
        # -----------------------------------------------------
        # 2. Check Plateau [Algorithm 1 Line 6]
        # -----------------------------------------------------
        t_plateau_start = time.time()
        is_plateau = self._check_plateau() if not is_warmup else False
        timing['plateau_check'] = time.time() - t_plateau_start

        # -----------------------------------------------------
        # 3. Decision & Execution [Algorithm 1 Line 6-12]
        # -----------------------------------------------------
        candidates_to_eval = []
        action_type = ""
        note = ""
        llm_cost = 0.0
        llm_time = 0.0
        # A. Warmup Phase (Follow Advisor / KD)
        if len(self.configs) < self.advisor.init_num:
            action_type = "Warmup"
            t_action_start = time.time()
            try:
                t_warmup_start = time.time()
                conf = self.advisor.get_suggestion()
                t_warmup_elapsed = time.time() - t_warmup_start
                if t_warmup_elapsed > 1.0:
                    self.logger.info(f"[BORA Warmup Timing] get_suggestion took {t_warmup_elapsed:.3f}s")
                candidates_to_eval.append(conf)
            except:
                candidates_to_eval.append(self.config_space.sample_configuration())
            note = f"Warmup ({len(self.configs)+1}/{self.advisor.init_num})"
            timing['action_perform'] = time.time() - t_action_start

        # B. BORA Policy Phase
        else:
            current_total_cost = self.acq_func.get_total_llm_cost()

            t_action_start = time.time()
            if current_total_cost > self.max_llm_cost:
                candidates_to_eval, note, llm_cost , llm_time = self._perform_action_a1()
                action_type = "a1"
                note += " (Cost Limit Exceeded)"

            elif sigma_mean < sigma_lower or (not is_plateau):
                candidates_to_eval, note, llm_cost , llm_time = self._perform_action_a1()
                action_type = "a1"
                
            elif sigma_mean > sigma_upper:
                action_type = "a2"
                
                # a2需要训练，但每5轮训练一次
                self._a2_train_counter += 1
                if self._a2_train_counter % 5 == 0:
                    X = self.history.get_config_array(transform='scale')
                    Y = self.history.get_objectives(transform='infeasible')
                    if len(X) > 0:
                        t_train_start = time.time()
                        self.advisor.surrogate_model.train(X, Y[:, 0])
                        t_train = time.time() - t_train_start
                        self.logger.info(f"[BORA a2 GP Training] n_samples={len(X)}, Train: {t_train:.3f}s (每5轮训练一次)")
                
                candidates_to_eval, note,llm_cost , llm_time = self._perform_action_a2()

            else:
                candidates_to_eval, note,llm_cost , llm_time = self._perform_action_a3()
                action_type = "a3"
            timing['action_perform'] = time.time() - t_action_start

        # -----------------------------------------------------
        # 4. Evaluation & Data Update [Algorithm 1 Line 13-14]
        # -----------------------------------------------------
        # BORA 的动作可能一次产生多个点 (a2, a3)，我们需要全部评估
        
        current_batch_perfs = [] # 记录这一批次的分数，用于 Trust Update
        
        eval_time = 0.0
        for cfg in candidates_to_eval:
            if _time_is_up(self, self.timestamp): break
            
            # Evaluate
            obs, eval_time_once = _evaluate_dict(self, cfg.get_dictionary(), self.config_space)
            eval_time += eval_time_once 
            max_perf = -float(obs.objectives[0]) # Score (Higher is better)
            perf = float(obs.objectives[0]) * (1 if self.lower_is_better else -1)

            # Update History / KD
            self.update_saver([obs.config], [-max_perf])
            self.configs.append(obs.config)
            self.perfs.append(max_perf)
            self.history.update_observation(obs)
            
            current_batch_perfs.append(max_perf)
            
            # Update Incumbent
            _update_incumbent_from_current(self)


        # 只有在 LLM 真正介入 (a2, a3) 且处于非 Warmup 阶段时才更新信任
        t_trust_start = time.time()
        if action_type in ["a2", "a3"] and current_batch_perfs:
            self._update_trust_mechanism(current_batch_perfs)
        timing['trust_update'] = time.time() - t_trust_start

        iter_time = time.time() - t0
        opt_time = iter_time - eval_time
        
        if iter_idx % 10 == 0 or opt_time > 5.0:
            timing_str = ", ".join([f"{k}: {v:.3f}s" for k, v in sorted(timing.items(), key=lambda x: x[1], reverse=True)])
            self.logger.info(f"[BORA Timing] Iter {iter_idx}: {timing_str} | Total opt_time: {opt_time:.3f}s, iter_time: {iter_time:.3f}s")
        
        if candidates_to_eval:
             self._record_event(
                opt="BORA",
                iter_idx=len(self.perfs),
                cfg=candidates_to_eval[-1],
                perf=current_batch_perfs[-1] if current_batch_perfs else 0.0,
                eval_time=eval_time, 
                opt_time=opt_time, 
                iter_time=iter_time,
                llm_time = llm_time,
                llm_cost= llm_cost,
                note=f"{note} (Batch:{len(candidates_to_eval)})"
            )

        return self.incumbent_perf, iter_time, self.incumbent_config

    def _record_event(self, opt, iter_idx, cfg, perf,iter_time=0.0, eval_time=0.0, opt_time=0.0,llm_cost=0.0,llm_time=0.0,note="", **kwargs):
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