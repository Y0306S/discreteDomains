
import time
import os
import pandas as pd
from typing import Any, Dict, List, Tuple
import numpy as np
from functools import partial
from copy import deepcopy
from openbox.optimizer.parallel_smbo import pSMBO as pBO
from .generic_smbo import MySMBO as BO
from openbox.utils.constants import SUCCESS
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from ConfigSpace import Constant, Configuration
from mindware.components.optimizers.llm_optimizer_utils import _evaluate_dict, wrapper_evaluater
from mindware.components.optimizers.utils import MyConfiguration

from .mosaic_mcts import MCTSTree


class MosaicOptimizer(BaseOptimizer):
    def __init__(self, evaluator, config_space, name, eval_type, 
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, 
                 inner_iter_num_per_iter=1, timestamp=None, 
                 output_dir='./', seed=1, n_jobs=1, topk=50, optimizer_params=None,task_id='GenTrans',
                 rand_prob=0, acq_optimizer_rand_prob=0,
                 logging_enabled=True):
        
        super().__init__(evaluator=evaluator, config_space=config_space, name=name, eval_type=eval_type, task_id=task_id,
                         time_limit=time_limit, evaluation_limit=evaluation_limit, 
                         per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                         inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                         output_dir=output_dir, seed=seed, topk=topk)

        self.objective_function = partial(wrapper_evaluater, evaluator=evaluator, per_run_time_limit=per_run_time_limit)

        self.optimizer = BO(objective_function=self.evaluator,
                            config_space=config_space,
                            advisor_type='mosaic',
                            surrogate_type='prf',
                            acq_type='ei',
                            max_runs=int(1e10),
                            max_runtime_per_trial=self.per_run_time_limit,
                            random_state=self.seed,
                            logging_dir=self.output_dir,
                            task_id=task_id,
                            advisor_kwargs={'rand_prob': 0, 'acq_optimizer_rand_prob': 0})


        self.mcts_tree = MCTSTree(include_algorithms=config_space['algorithm'].choices,
                                  exploration_weight=1.3,
                                  enable_prior=True,
                                  output_dir=self.output_dir)

        

        self.trial_cnt = 0
        self.exp_output = dict()

        self.KD = KnowledgeDatabase.instance()


        self.initial_configs = []  # 迁移学习的热启动配置
        # 每个算法采样3个作为热启动
        for algo in config_space['algorithm'].choices:
            sub_cs = self.KD.bo_history_mab[algo].config_space
            sub_configs = sub_cs.sample_configuration(3)
            # 放回原始配置空间
            for tmp_config in sub_configs:
                full_config_dict = deepcopy(tmp_config.get_dictionary())
                full_config_dict['algorithm'] = algo
                config = MyConfiguration(Configuration(self.config_space, values=full_config_dict))
                self.initial_configs.append(config)

        self.configs = self.KD.configs
        self.perfs = self.KD.perfs

        self.optimizer.config_advisor.history = self.KD.bo_history
        self.bo_mode = 'bo'

        self.logging_enabled = logging_enabled
        if self.logging_enabled:
            self.traj_rows: List[Dict[str, Any]] = []
            self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_opt_smac_traj.csv")
            self.start_wall_time = time.time()
            self.outer_iter_idx = 0

        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            _threshold = int(len(set(self.config_space.sample_configuration(5000))))
            self.config_num_threshold = _threshold
        self.logger.debug('The maximum trial number in HPO is: %d' % self.config_num_threshold)
        self.maximum_config_num = min(1500, self.config_num_threshold)
        self.n_jobs = n_jobs


    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                self.logger.info("Early stop triggered, terminating optimization.")
                break
            self.iterate()
        return np.max(self.perfs) if self.perfs else -np.inf

    def iterate(self, budget=MAX_INT) -> Tuple[float, float, Any, List[Dict]]:
        _start_time = time.time()
        
        iter_idx = len(self.KD.bo_history) + 1
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

            candidate_node = self.mcts_tree.warmup_selection(cfg_d['algorithm']) # 找到算法对应的HPORootNode
            expand_strategy = self.mcts_tree.expansion(candidate_node)
            self.mcts_tree.update_results(
                config=cfg_obj,
                perf=max_perf,
                expand_strategy=expand_strategy
            )

            expand_strategy = expand_strategy.to_dict()

            self.update_saver([cfg_obj], [-max_perf])
            self.KD.update_history(obs, source='warmup', trail_id = f'{iter_idx}_warmup')

            self._record_event(
                stage="random-warmup",
                iter_idx=iter_idx,
                cfg=cfg_obj,
                perf=max_perf,
                eval_time=eval_time,
                smac_time=0,
                iter_time=0
            )

            run_history = self.KD.bo_history
            if len(run_history.get_incumbents()) > 0:
                incumbent_obs = run_history.get_incumbents()[0]
                self.incumbent_config = incumbent_obs.config.get_dictionary().copy()
                self.incumbent_perf = -incumbent_obs.objectives[0]


            iter_cost = time.time() - _start_time

            if len(self.initial_configs) == 0:
                # 热启动完毕，训练一下模型
                from openbox.core.base import build_surrogate
                model = build_surrogate(func_str='prf', config_space=self.KD.bo_history.config_space, rng=np.random.RandomState(1))
                model.train(
                    X=self.KD.bo_history.get_config_array(transform='scale'),
                    Y=self.KD.bo_history.get_objectives(transform='infeasible')[:, 0]
                )
                self.mcts_tree.update_prior_models(model=model)


            return self.incumbent_perf, iter_cost, self.incumbent_config

        # 新增: 用于存储本轮所有评估结果的列表

        if len(self.configs) == 0 and hasattr(self, 'init_hpo_iter_num') and self.init_hpo_iter_num is not None:
            inner_iter_num = self.init_hpo_iter_num
            print('initial hpo trial num is set to %d' % inner_iter_num)
        else:
            inner_iter_num = self.inner_iter_num_per_iter

        if self.n_jobs == 1:
            for i in range(inner_iter_num):
                if len(self.configs) >= self.maximum_config_num or self.early_stopped_flag or self.timeout_flag:
                    break
                if time.time() - _start_time > budget:
                    self.logger.warning('Time limit exceeded for this iteration!')
                    break

                iter_idx = len(self.KD.bo_history)

                iter_start_time = time.time()

                candidate_node = self.mcts_tree.selection()  # 选择待扩展节点
                expand_strategy = self.mcts_tree.expansion(candidate_node)  # 扩展节点，返回扩展策略和节点回溯路径
                tar_algo = expand_strategy.selected_algo
                obs = self.optimizer.iterate(tar_algo=tar_algo)

                # 更新MCTS树
                self.mcts_tree.update_results(
                    config = obs.config,
                    perf = - float(obs.objectives[0]),
                    expand_strategy = expand_strategy
                )
                # 更新先验模型
                model = self.optimizer.config_advisor.surrogate_model
                self.mcts_tree.update_prior_models(model=model)

                iter_time = time.time() - iter_start_time

                eval_time = obs.elapsed_time if obs.elapsed_time is not None else 0.0
                smac_time = iter_time - eval_time

                _config, _status, _perf = obs.config, obs.trial_state, obs.objectives
                perf = -_perf[0]

                if _status == SUCCESS:
                    self.exp_output[time.time()] = (_config, perf)

                    self.update_saver([_config], [-perf])
                    self.update_history(obs, trail_id=f'{iter_idx}')

                if self.perfs:
                    max_perf = np.max(self.perfs)
                    if max_perf > self.incumbent_perf:
                        self.incumbent_perf = max_perf
                        self.incumbent_config = self.configs[np.argmax(self.perfs)]


                # 如果日志开关打开，则执行记录 (用于独立运行smac时)
                if self.logging_enabled:
                    self._record_event(
                        stage="iterate",
                        iter_idx=iter_idx,
                        cfg=_config,
                        perf=perf,
                        eval_time=eval_time,
                        smac_time=smac_time,
                        iter_time=iter_time,
                        note=f"candidate {i+1}/{inner_iter_num}"
                    )
        else:
            pass

        if self.logging_enabled:
            self.outer_iter_idx += 1
        
        if self.time_limit is not None and time.time() - self.timestamp > self.time_limit or \
                hasattr(self, 'evaluation_num_limit') and self.evaluation_num_limit is not None and len(self.perfs) >= self.evaluation_num_limit:
            self.timeout_flag = True

        iteration_cost = time.time() - _start_time
        
        run_history = self.optimizer.get_history()
        if len(run_history.get_incumbents()) > 0:
            incumbent_obs = run_history.get_incumbents()[0]
            self.incumbent_config = incumbent_obs.config.get_dictionary().copy()
            self.incumbent_perf = -incumbent_obs.objectives[0]

        # 返回 incumbent 信息以及本轮新观测到的所有行的详细数据
        return self.incumbent_perf, iteration_cost, self.incumbent_config
    # --- END MODIFIED ---


    def update_history(self, obs, **extra_infos):
        self.KD.update_history(obs, source=self.bo_mode, **extra_infos)

    def _record_event(self, stage, iter_idx, cfg, perf, eval_time=0.0, smac_time=0.0, iter_time=0.0, note=""):

        if not self.logging_enabled:
            return
        
        wall_elapsed_sec = time.time() - self.start_wall_time
        algo = cfg.get('algorithm', 'default')

        row = {
            "ts": pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "stage": stage,
            "iter_idx": iter_idx,
            "perf": perf,
            "best_so_far": max(self.perfs) if self.perfs else float("-inf"),
            "algorithm": algo,
            "smac_time": smac_time,
            "eval_time": eval_time,
            "iter_time": iter_time,
            "wall_elapsed_sec": wall_elapsed_sec,
            "note": note,
        }
        self.traj_rows.append(row)
        
        try:
            columns = ["ts", "stage", "iter_idx", "perf", "best_so_far", "algorithm",
                       "smac_time", "eval_time", "iter_time", "wall_elapsed_sec", "note"]
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
                    "smac_time": r.get("smac_time", 0.0),
                    "eval_time": r.get("eval_time", 0.0),
                    "iter_time": r.get("iter_time", 0.0),
                    "wall_elapsed_sec": r["wall_elapsed_sec"],
                } for r in self.traj_rows
            ][:500]
        }