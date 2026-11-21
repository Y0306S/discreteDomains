import time
import os
import pandas as pd
from typing import Any, Dict, List, Tuple
import numpy as np
from openbox.optimizer.parallel_smbo import pSMBO as pBO
from .generic_smbo import MySMBO as BO
from openbox.utils.constants import SUCCESS
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from ConfigSpace import ConfigurationSpace, Constant

class SMACMABOptimizer(BaseOptimizer):
    def __init__(self, evaluator, config_space, name, eval_type, 
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, 
                 inner_iter_num_per_iter=1, timestamp=None, 
                 output_dir='./', seed=1, n_jobs=1, topk=50, optimizer_params=None, task_id='GenTrans',
                 rand_prob=0.1, acq_optimizer_rand_prob=0,
                 logging_enabled=True, surrogate_type='prf'):
        
        super().__init__(evaluator=evaluator, config_space=config_space, name=name, eval_type=eval_type, task_id=task_id,
                         time_limit=time_limit, evaluation_limit=evaluation_limit, 
                         per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                         inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                         output_dir=output_dir, seed=seed, topk=topk)

        self.n_jobs = 1
        self.KD = KnowledgeDatabase.instance()
        self.arms = list(config_space.get_hyperparameter('algorithm').choices)
        self.optimizers = dict()
        self.surrogate_type = surrogate_type
        for arm in self.arms:
            cs = self.KD.bo_history_mab[arm].config_space
            self.optimizers[arm] = BO(objective_function=self.evaluator,
                                    config_space=cs,
                                    surrogate_type=surrogate_type,
                                    acq_type='ei',
                                    max_runs=int(1e10),
                                    max_runtime_per_trial=self.per_run_time_limit,
                                    random_state=self.seed,
                                    logging_dir=self.output_dir,
                                    task_id=f"{task_id}_{arm}",
                                    advisor_kwargs={'rand_prob': rand_prob,'acq_optimizer_rand_prob': acq_optimizer_rand_prob})  # 使用25%的概率在acq_optimizer中不要local search，而是直接随机采样
            self.optimizers[arm].config_advisor.history = self.KD.bo_history_mab[arm]


        self.trial_cnt = 0
        self.exp_output = dict()

        self.configs = self.KD.configs
        self.perfs = self.KD.perfs

        self.logging_enabled = logging_enabled
        if self.logging_enabled:
            self.traj_rows: List[Dict[str, Any]] = []
            self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_opt_smac_traj.csv")
            self.start_wall_time = time.time()
            self.outer_iter_idx = 0


    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                self.logger.info("Early stop triggered, terminating optimization.")
                break
            self.iterate()
        return np.max(self.perfs) if self.perfs else -np.inf

    
    def get_surrogate_model(self, algo):
        assert algo in self.arms

        from openbox.core.base import build_surrogate
        gp_model = build_surrogate(func_str=self.surrogate_type, config_space=self.KD.bo_history_mab[algo].config_space, rng=np.random.RandomState(1))
        gp_model.train(
            X=self.KD.bo_history_mab[algo].get_config_array(transform='scale'),
            Y=self.KD.bo_history_mab[algo].get_objectives(transform='infeasible')[:, 0]
        )

        return gp_model

    def get_gp_kernel(self, algo):
        assert algo in self.arms
        from openbox.core.base import build_surrogate
        gp_model = build_surrogate(func_str='gp', config_space=self.KD.bo_history_mab[algo].config_space, rng=np.random.RandomState(1))
        gp_model.train(
            X=self.KD.bo_history_mab[algo].get_config_array(transform='scale'),
            Y=self.KD.bo_history_mab[algo].get_objectives(transform='infeasible')[:, 0]
        )
        return gp_model.kernel

    def iterate(self, selected_algo, budget=MAX_INT) -> Tuple[float, float, Any, List[Dict]]:
        _start_time = time.time()
        
        # 新增: 用于存储本轮所有评估结果的列表
        newly_observed_rows = []

        if len(self.configs) == 0 and hasattr(self, 'init_hpo_iter_num') and self.init_hpo_iter_num is not None:
            inner_iter_num = self.init_hpo_iter_num
            print('initial hpo trial num is set to %d' % inner_iter_num)
        else:
            inner_iter_num = self.inner_iter_num_per_iter

        if self.n_jobs == 1:
            for i in range(inner_iter_num):
                if self.early_stopped_flag or self.timeout_flag:
                    break
                if time.time() - _start_time > budget:
                    self.logger.warning('Time limit exceeded for this iteration!')
                    break

                iter_idx = len(self.KD.bo_history)

                iter_start_time = time.time()
                obs = self.optimizers[selected_algo].iterate()
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

                # 将结果存入列表，用于向上返回
                row_data = {
                    "perf": perf,
                    "config": _config,
                    "eval_time": eval_time,
                    "smac_time": smac_time,
                    "iter_time": iter_time,
                    "note": f"candidate {i+1}/{inner_iter_num}"
                }
                newly_observed_rows.append(row_data)

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
        
        run_history = self.KD.bo_history
        if len(run_history.get_incumbents()) > 0:
            incumbent_obs = run_history.get_incumbents()[0]
            self.incumbent_config = incumbent_obs.config.get_dictionary().copy()
            self.incumbent_perf = -incumbent_obs.objectives[0]

        # 返回 incumbent 信息以及本轮新观测到的所有行的详细数据
        return self.incumbent_perf, iteration_cost, self.incumbent_config, newly_observed_rows
    # --- END MODIFIED ---


    def update_history(self, obs, **extra_infos):
        self.KD.update_history(obs, source='bo_mab', **extra_infos)

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
            "best_so_far": max(self.perfs),
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