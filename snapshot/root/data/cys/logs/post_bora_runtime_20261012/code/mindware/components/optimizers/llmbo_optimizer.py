from mindware.components.optimizers.llm_optimizer import LLMOptimizer
from mindware.components.optimizers.llmmcts_optimizer import LLMMCTSOptimizer
from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from openbox.core.base import build_surrogate

import os
import time
import numpy as np
import pandas as pd
from functools import partial
from collections import OrderedDict
from typing import List, Dict
from sklearn.model_selection import KFold

from mindware.components.optimizers.llm_optimizer_utils import _evaluate_dict, wrapper_evaluater
from mindware.components.optimizers.mcts import BAYESIANLOCAL, BAYESIANRANDOM, ExpandStrategy, Node


class LLMBOOptimizer(BaseOptimizer):
    def __init__(
        self,
        evaluator,
        config_space,
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
        **kwargs
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

        self.llm_mode = kwargs.get('llm_mode', 'mcts')  # mcts, opro
        assert self.llm_mode in ['mcts', 'opro'], "Invalid llm_mode. Supported modes are 'mcts' and 'opro'."
        # bo和llm切换的模式
        self.alternative_mode = kwargs.get('alternative_mode', 'adaptive')  # 'fixed' or 'adaptive'
        assert self.alternative_mode in ['fixed_0.5', 'pure_bo', 'pure_llm', 'adaptive']

        # 不可以llm_mode为 opro时使用pure_bo
        assert not (self.llm_mode == 'opro' and self.alternative_mode == 'pure_bo'), "Cannot use 'pure_bo' alternative_mode with 'opro' llm_mode."

        self.KD = KnowledgeDatabase.instance()

        LLMCLASS = LLMMCTSOptimizer if self.llm_mode == 'mcts' else LLMOptimizer
        # Initialize LLM-based optimizer here
        self.llm_optimizer = LLMCLASS(
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
            n_jobs=n_jobs,
            topk=topk,
            **kwargs
        )

        if self.alternative_mode == 'pure_llm':
            self.bo_optimizer = None
        else:
            self.bo_optimizer = SMACOptimizer(
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
                n_jobs=n_jobs,
                topk=topk
            )

        # 将evaluator包装为超时控制的形式
        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)

        self.general_transfer = kwargs.get('general_transfer', True)

        self.algorithm_order = None  # 迁移学习的算法顺序
        self.initial_configs = []  # 迁移学习的热启动配置
        if self.general_transfer:

            # 获取算法优先级
            # model_scores = self.KD.recommand_models() # [("random_forest", 0.95), ("lightgbm", 0.90), ...],
            model_scores = self.KD.fetch_algo_cate_knowledge()  # [("random_forest", 0.95), ("lightgbm", 0.90), ...],
            self.algorithm_order = [item[0] for item in model_scores]

            # 为每个算法获取最佳配置作为热启动
            bestconf_knowledge = self.KD.fetch_best_conf_knowledge()  # algorithm: [...]

            # 每轮每个算法放一个，直到放完
            while len(bestconf_knowledge) > 0:
                for alg in self.algorithm_order:
                    best_confs = bestconf_knowledge[alg]["Best Configurations"]
                    if len(best_confs) == 0:
                        continue
                    self.initial_configs.append(best_confs.pop(0))
                    if len(best_confs) == 0:
                        bestconf_knowledge.pop(alg)

        self.configs = self.KD.configs
        self.perfs = self.KD.perfs

        if self.alternative_mode == 'fixed_0.5':
            self.bo_prob = 0.5 # 初始时探索与利用各占一半
        elif self.alternative_mode == 'pure_bo':
            self.bo_prob = 1.0 # 始终使用BO优化器
        elif self.alternative_mode == 'pure_llm':
            self.bo_prob = 0.0 # 始终使用LLM优化器
        elif self.alternative_mode == 'adaptive':
            self.bo_prob = 1.0 # 初始化为0，表示初始时完全依赖LLM优化器
        else:
            raise ValueError("Invalid alternative_mode: %s" % self.alternative_mode)
        self.alternative_interval = kwargs.get('alternative_interval', 10)  # 每隔多少次迭代调整一次概率

        self.exp_output = dict()

        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_LLMBO_opt_traj.csv")

    def build_single_surrogate(self, X: np.ndarray, y: np.array):
        model = build_surrogate(func_str='prf', config_space=self.config_space, rng=np.random.RandomState(self.seed))

        if np.all(y == y[0]):
            y[0] += 1e-4

        model.train(X, y)
        return model

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if not ((y_true[idx] > y_true[inner_idx]) ^ (y_pred[idx] > y_pred[inner_idx])):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def adjust_bo_probability(self):
        """
        根据历史表现动态调整BO优化器的使用概率
        """
        X = self.KD.bo_history.get_config_array()
        y = self.KD.bo_history.get_objectives()

        _mu, _var = list(), list()

        # Conduct K-fold cross validation.
        k_fold_num = min(self.alternative_interval, X.shape[0])
        kf = KFold(n_splits=k_fold_num, shuffle=True, random_state=self.seed)
        idxs = list()
        for train_idx, val_idx in kf.split(X):
            idxs.extend(list(val_idx))
            X_train, X_val, y_train, y_val = X[train_idx, :], X[val_idx, :], y[train_idx], y[val_idx]
            model = self.build_single_surrogate(X_train, y_train)
            mu, var = model.predict(X_val)
            mu, var = mu.flatten(), var.flatten()
            _mu.extend(list(mu))
            _var.extend(list(var))
        assert (np.array(sorted(idxs)) == np.arange(X.shape[0])).all()

        _mu, _var = np.asarray(_mu), np.asarray(_var)

        # 采样计算权重
        num_sample = 50
        preorder_ratio_list = []
        # for _ in range(num_sample):
        #     sampled_y = np.random.normal(_mu, np.sqrt(_var))
        #     preorder_num, pair_num = self.calculate_preserving_order_num(sampled_y, y)
        #     preorder_ratio_list.append(preorder_num / pair_num)

        preorder_num, pair_num = self.calculate_preserving_order_num(_mu, y)
        preorder_ratio_list.append(preorder_num / pair_num)

        avg_order_ratio = np.mean(preorder_ratio_list)

        # 0.5 - 1 对应 0 - 1

        # self.bo_prob = max(0, avg_order_ratio - 0.5) * 2
        # 做一些限制，防止波动过大： 最多一次提升0.3，最多一次下降0.1

        new_bo_prob = avg_order_ratio
        new_bo_prob = min(new_bo_prob, 1.0)
        if new_bo_prob > self.bo_prob:
            self.bo_prob = min(self.bo_prob + 0.3, new_bo_prob)
        else:
            self.bo_prob = max(self.bo_prob / 2, new_bo_prob)


    def iterate(self, budget=MAX_INT):
        """
        A single iteration step, designed to be called by an external scheduler.

        In one call, this method generates and evaluates all candidates for a single trial.
        """

        t0 = time.time()
        iter_idx = len(self.KD.bo_history) + 1

        # Cold-start with initial samples if history is empty.
        if len(self.initial_configs) > 0:
            # first_key = next(iter(self.initial_configs))
            # cfg_d = self.initial_configs[first_key].pop(0)
            # if len(self.initial_configs[first_key]) == 0:
            #     self.initial_configs.pop(first_key)
            cfg_d = self.initial_configs.pop(0)

            t0_iter = time.time()
            obs, eval_time = _evaluate_dict(self, cfg_d)
            cfg_obj = obs.config
            max_perf = - float(obs.objectives[0])

            iter_time = time.time() - t0_iter

            expand_strategy = {}
            if self.llm_optimizer is not None:
                if self.llm_mode == 'opro':
                    self.llm_optimizer.exploration_exploitation_balancer.update_best_perf(max_perf, cfg_d['algorithm'])
                elif self.llm_mode == 'mcts':
                    candidate_node = self.llm_optimizer.mcts_tree.warmup_selection(cfg_d['algorithm']) # 找到算法对应的HPORootNode
                    expand_strategy = self.llm_optimizer.mcts_tree.expansion(candidate_node)[0]
                    self.llm_optimizer.mcts_tree.update_results(
                        config=cfg_obj,
                        perf=max_perf,
                        action="Warmup from historical knowledge; no thought process.",
                        reflection="Warmup strategy.",
                        expand_strategy=expand_strategy
                    )

                    expand_strategy = expand_strategy.to_dict()
                else:
                    raise ValueError(f"Unknown llm_mode: {self.llm_mode}")

            self.update_saver([cfg_obj], [-max_perf])
            self.KD.update_history(obs, source='warmup', trail_id = f'{iter_idx}_warmup')

            self._record_event(
                opt="historical-warmup",
                iter_idx=iter_idx,
                cfg=cfg_obj,
                perf=max_perf,
                eval_time=eval_time,
                opt_time=0,
                iter_time=0,
                **expand_strategy
            )

            run_history = self.KD.bo_history
            if len(run_history.get_incumbents()) > 0:
                incumbent_obs = run_history.get_incumbents()[0]
                self.incumbent_config = incumbent_obs.config.get_dictionary().copy()
                self.incumbent_perf = -incumbent_obs.objectives[0]


            iter_cost = time.time() - t0

            iter_idx = len(self.KD.bo_history) + 1
            if self.alternative_mode == 'adaptive' and iter_idx % self.alternative_interval == 0 and iter_idx > 0:
                ori_bo_prob = self.bo_prob
                self.adjust_bo_probability()
                self.logger.info(f"Adjusted BO optimizer probability from {ori_bo_prob:.2f} to {self.bo_prob:.2f} at iteration {iter_idx}.")


            return self.incumbent_perf, iter_cost, self.incumbent_config



        # self.logger.info(f"Iteration {iter_idx}: Using BO optimizer (BO probability {self.bo_prob:.2f})")
        # optimizer_to_use = self.bo_optimizer
        # opt = 'bo'
        # Decide which optimizer to use based on the current probability


        # --- 步骤 A: 根据MCTS生成扩展决策 ---
        self.logger.info("[LLMMAB] === Step A: MCTS Expansion ===")
        candidate_node = self.llm_optimizer.mcts_tree.selection()  # 选择待扩展节点
        expand_strategy, node_path = self.llm_optimizer.mcts_tree.expansion(candidate_node)  # 扩展节点，返回扩展策略和节点回溯路径
        selected_algo = expand_strategy.selected_algo
        self.logger.info(f"[LLMMAB] Selected algorithm: {selected_algo} for iteration {iter_idx}. Strategy: {expand_strategy.strategy}, Parent Node ID: {expand_strategy.parent_id}")

        if self.alternative_mode=='pure_bo' or np.random.rand() < self.bo_prob:
            self.logger.info(f"Iteration {iter_idx}: Using BO optimizer (BO probability {self.bo_prob:.2f})")
            optimizer_to_use = self.bo_optimizer
            opt = 'bo'
            _, _, _, newly_observed_rows = optimizer_to_use.iterate(budget=budget)
        else:
            # 更新change path pareto中的gp kernel
            gp_kernel = self.bo_optimizer.get_gp_kernel(selected_algo)
            self.llm_optimizer.mcts_tree.cpp_dict[selected_algo].update_gp_kernel(gp_kernel)
            self.logger.info(f"Iteration {iter_idx}: Using LLM optimizer (BO probability {self.bo_prob:.2f})")
            optimizer_to_use = self.llm_optimizer
            opt = 'llm'
            _, _, _, newly_observed_rows = optimizer_to_use.iterate(budget=budget, expand_strategy=expand_strategy, node_path=node_path)

        # llm推荐失败，则用bo补上
        if opt == 'llm' and len(newly_observed_rows) == 0:
            self.logger.info(f"LLM optimizer failed to propose valid configurations. Falling back to BO optimizer.")
            optimizer_to_use = self.bo_optimizer
            opt = 'bo'
            _, _, _, newly_observed_rows = optimizer_to_use.iterate(budget=budget)

        expand_strategys = []
        # 如果是BO优化器，则更新llm优化器的探索与利用平衡器
        if self.llm_optimizer is not None:
            if opt == 'bo':
                for row in newly_observed_rows:
                    if self.llm_mode == 'opro':
                        self.llm_optimizer.exploration_exploitation_balancer.update_best_perf(
                            row["perf"],
                            row["config"]['algorithm']
                        )
                    else:

                        config = row["config"]
                        origin = config.origin
                        # 解析origin，有几种情况：Random Search开头，或者Local Search (mcts_node_id: {start_point.extra_info['mcts_node_id']})
                        if origin.startswith("Random Search"):
                            action = "Warmup random sampling; no thought process."
                            reflection = "Warmup strategy; no reflection performed."
                            candidate_node = self.llm_optimizer.mcts_tree.warmup_selection(config['algorithm'])
                            expand_strategy = ExpandStrategy(
                                selected_algo=config['algorithm'],
                                strategy=BAYESIANRANDOM,
                                parent_id=candidate_node.id
                            )
                        elif origin.startswith("Local Search"):
                            # 提取mcts_node_id
                            mcts_node_id_str = origin.split("mcts_node_id:")[1].strip().rstrip(")")
                            mcts_node_id = int(mcts_node_id_str)
                            candidate_node = Node.node_dict[mcts_node_id]
                            # 分析一下config的变化，写进action
                            changes = ""
                            origin_config = candidate_node.config
                            changed_params = []
                            for param in config.get_dictionary().keys():
                                if param in origin_config and config.get_dictionary()[param] != origin_config[param]:
                                    changed_params.append(param)
                            if changed_params:
                                # 写出哪些参数被修改了，如何修改的，用->表示
                                changes = ", ".join([f"{param}: {origin_config[param]} -> {config.get_dictionary()[param]}" for param in changed_params])
                            action = "Bayesian local search." + (f" Changed parameters: {changes}." if changes else "")
                            reflection = "Bayesian Local search performed; no additional reflection."
                            # 将BO Local Search出的节点挂载到对应的父节点下
                            expand_strategy = ExpandStrategy(
                                selected_algo=config['algorithm'],
                                strategy=BAYESIANLOCAL,
                                parent_id=candidate_node.id
                            )

                        else:
                            raise ValueError(f"Unknown origin format: {origin}")

                        self.llm_optimizer.mcts_tree.update_results(
                            config=config,
                            perf=row["perf"],
                            action=action,
                            reflection=reflection,
                            expand_strategy=expand_strategy
                        )
                        expand_strategys.append(expand_strategy.to_dict())

            else:
                if self.llm_mode == 'mcts':
                    for row in newly_observed_rows:
                        if 'expand_strategy' in row:
                            expand_strategys.append(row['expand_strategy'].to_dict())

            # 每10次迭代可视化一次树结构
            if self.llm_mode == 'mcts' and iter_idx % 10 == 0:
                self.llm_optimizer.mcts_tree.visualize()

        run_history = self.KD.bo_history
        if len(run_history.get_incumbents()) > 0:
            incumbent_obs = run_history.get_incumbents()[0]
            self.incumbent_config = incumbent_obs.config.get_dictionary().copy()
            self.incumbent_perf = -incumbent_obs.objectives[0]

        for i in range(len(newly_observed_rows)):
            row = newly_observed_rows[i]

            expand_strategy = {} if len(expand_strategys) == 0 else expand_strategys[i]

            self._record_event(
                opt=opt,
                iter_idx=iter_idx+i,
                cfg=row["config"],
                perf=row["perf"],
                eval_time=row["eval_time"],
                opt_time=row['llm_time'] if opt == 'llm' else row['smac_time'],
                iter_time=row['iter_time'],
                **expand_strategy
            )

        iter_idx = len(self.KD.bo_history) + 1
        if self.alternative_mode == 'adaptive' and iter_idx % self.alternative_interval == 0 and iter_idx > 0:
            ori_bo_prob = self.bo_prob
            self.adjust_bo_probability()
            self.logger.info(f"Adjusted BO optimizer probability from {ori_bo_prob:.2f} to {self.bo_prob:.2f} at iteration {iter_idx}.")


        iteration_cost = time.time() - t0

        # 返回 incumbent 信息以及本轮新观测到的所有行的详细数据
        return self.incumbent_perf, iteration_cost, self.incumbent_config


    def _record_event(self, opt, iter_idx, cfg, perf, eval_time=0.0, opt_time=0.0, iter_time=0.0, note="", **expand_strategy):

        wall_elapsed_sec = time.time() - self.timestamp
        algo = cfg.get('algorithm', 'default')

        strategy = expand_strategy.get('strategy', 'N/A')
        parent_node_id = expand_strategy.get('parent_id', 'N/A')

        row = {
            "ts": pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "optimizer": opt,
            "bo_prob": self.bo_prob,
            "iter_idx": iter_idx,
            "perf": perf,
            "best_so_far": max(self.perfs),
            "algorithm": algo,
            "opt_time": opt_time,
            "eval_time": eval_time,
            "iter_time": iter_time,
            "wall_elapsed_sec": wall_elapsed_sec,
            "note": note,
            "strategy": strategy,
            "parent_node_id": parent_node_id
        }
        self.traj_rows.append(row)

        try:
            columns = ["ts", "optimizer", "bo_prob", "iter_idx", "perf", "best_so_far", "algorithm",
                       "opt_time", "eval_time", "iter_time", "wall_elapsed_sec", "note", "strategy", "parent_node_id"]
            df = pd.DataFrame(self.traj_rows)
            df[columns].to_csv(self.traj_csv_path, index=False)
        except IOError as e:
            self.logger.warning(f"Failed to save trajectory to {self.traj_csv_path}. Error: {e}")

    def get_opt_trajectory(self):
        best = float(max(self.perfs)) if self.perfs else float("-inf")
        return {
            "n_evals": len(self.perfs),
            "final_best": best,
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join([str(p) for p in self.perfs]),
            "history": [
                {
                    "iter_idx": r["iter_idx"],
                    "perf": r["perf"],
                    "best_so_far": r["best_so_far"],
                    "algorithm": r["algorithm"],
                    "opt_time": r.get("opt_time", 0.0),
                    "eval_time": r.get("eval_time", 0.0),
                    "iter_time": r.get("iter_time", 0.0),
                    "wall_elapsed_sec": r["wall_elapsed_sec"],
                } for r in self.traj_rows
            ][:500]
        }