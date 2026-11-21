import time
import os
import pandas as pd
import numpy as np
from copy import deepcopy

from typing import Any,List, Dict

from mindware.components.optimizers.base_optimizer import BaseOptimizer
from ConfigSpace import ConfigurationSpace, Constant
from mindware.utils.constant import MAX_INT


class ConditionalOptimizer(BaseOptimizer):
    def __init__(self, node_list, node_index,
                 evaluator, cash_config_space, name, eval_type,
                 task_id: str = "test",
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024,
                 inner_iter_num_per_iter=10, timestamp=None,
                 sub_optimizer='smac', fe_config_space_dict=None,
                 output_dir='./', seed=1, n_jobs=1, topk=50):

        super(ConditionalOptimizer, self).__init__(evaluator=evaluator, config_space=(cash_config_space, fe_config_space_dict), name=name, eval_type=eval_type, task_id=task_id,
                                                   time_limit=time_limit, evaluation_limit=evaluation_limit, 
                                                   per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                                                   inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                                                   output_dir=output_dir, seed=seed, topk=topk)

        self.traj_rows: List[Dict[str, Any]] = []
        self.traj_csv_path = os.path.join(self.output_dir, f"{int(self.timestamp)}_opt_traj.csv")
        self.start_wall_time = time.time()

        self.node_index = node_index

        # Bandit settings.
        self.alpha = 4
        self.arms = list(cash_config_space.get_hyperparameter('algorithm').choices)
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()

        self.arm_cost_stats = dict()
        for _arm in self.arms:
            self.arm_cost_stats[_arm] = list()

        for arm in self.arms:

            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()

            cs = ConfigurationSpace()
            cs.add_hyperparameter(Constant('algorithm', arm))
            # Add active hyperparameters
            hps = cash_config_space.get_hyperparameters()
            for hp in hps:
                if hp.name.split(':')[0] == arm:
                    cs.add_hyperparameter(hp)
            # Add active conditions
            conds = cash_config_space.get_conditions()
            for cond in conds:
                try:
                    cs.add_condition(cond)
                except:
                    pass
            # Add active forbidden clauses
            forbids = cash_config_space.get_forbiddens()
            for forbid in forbids:
                try:
                    cs.add_forbidden_clause(forbid)
                except:
                    pass

            from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_node_type
            child_type = get_opt_node_type(node_list, node_index + 1)
            self.sub_bandits[arm] = child_type(
                node_list=node_list, node_index=node_index + 1,
                evaluator=self.evaluator, cash_config_space=deepcopy(cs), name='hpofe', eval_type=self.eval_type, 
                time_limit=time_limit, evaluation_limit=None,
                per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp, 
                sub_optimizer=sub_optimizer, fe_config_space_dict={arm: fe_config_space_dict[arm]} if fe_config_space_dict is not None else None,
                output_dir=self.output_dir,seed=self.seed, n_jobs=n_jobs, topk=topk
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.time_records = list()

        # Initialize the parameters.
        self.pull_cnt = 0
        self.pick_id = 0
        self.update_cnt = 0
        arm_num = len(self.arms)
        self.optimal_algo_id = None
        self.arm_candidate = self.arms.copy()
        self.best_lower_bounds = np.zeros(arm_num)

        if self.time_limit is None:
            if arm_num * self.alpha > self.evaluation_num_limit:
                raise ValueError('Trial number should be larger than %d.' % (arm_num * self.alpha))
        # else:
            # self.evaluation_num_limit = MAX_INT

        self.timeout_flag = False

    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                break
            self.iterate()
        return self.incumbent_perf

    def iterate(self, budget=MAX_INT):

        _start_time = time.time()

        for _arm in self.arms:
            self.sub_bandits[_arm].inner_iter_num_per_iter = self.inner_iter_num_per_iter

        arm_to_pull = None
        while self.pick_id < len(self.arm_candidate):
            arm_to_pull_candidate = self.arm_candidate[self.pick_id]
            if not self.sub_bandits[arm_to_pull_candidate].early_stopped_flag:
                arm_to_pull = arm_to_pull_candidate
                break
            self.pick_id += 1
        
        if arm_to_pull is None:
            self.early_stopped_flag = True
            self.logger.info("All arm candidates have been early-stopped.")
            return self.incumbent_perf, 0, self.incumbent_config

        if arm_to_pull:
            self.logger.info('Optimize %s in the %d-th iteration' % (arm_to_pull, self.pull_cnt))
            
            # Call the sub-optimizer, which now returns detailed rows
            reward, _, incumbent, new_rows = self.sub_bandits[arm_to_pull].iterate(budget=budget)

            if reward > self.incumbent_perf:
                self.optimal_algo_id = arm_to_pull
                self.incumbent_perf = reward
                self.incumbent_config = incumbent

            # --- NEW: Record each evaluation from the sub-optimizer ---
            for row_data in new_rows:
                self._record_event(
                    stage="iterate",
                    iter_idx=self.pull_cnt,  # Use MAB's pull_cnt as the iteration index
                    cfg=row_data["config"],
                    perf=row_data["perf"],
                    eval_time=row_data["eval_time"],
                    mab_time=row_data["smac_time"], # HPO overhead is part of MAB time
                    iter_time=row_data["iter_time"],
                    note=row_data["note"]
                )
            # --- END NEW ---

            self.perfs.extend([row['perf'] for row in new_rows])
            self.configs.extend([row['config'] for row in new_rows])

            total_iter_time_for_arm = sum(row['iter_time'] for row in new_rows)
            self.arm_cost_stats[arm_to_pull].append(total_iter_time_for_arm)
            
            
            self.eval_dict.update(self.sub_bandits[arm_to_pull].eval_dict)
            self.rewards[arm_to_pull].append(reward)
            self.action_sequence.append(arm_to_pull)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.timestamp)
            
            self.pull_cnt += 1 # Increment MAB iteration counter
            self.pick_id += 1


            # Logger output
            scores = list()
            for _arm in self.arms:
                scores.append(self.sub_bandits[_arm].incumbent_perf)
            scores = np.array(scores)
            self.logger.info('=' * 50)
            self.logger.info('Node index: %s' % str(self.node_index))
            self.logger.info('Best_algo_perf:  %s' % str(self.incumbent_perf))
            self.logger.info('Best_algo_id:    %s' % str(self.optimal_algo_id))
            self.logger.info('Arm candidates:  %s' % str(self.arms))
            self.logger.info('Best val scores: %s' % str(list(scores)))
            self.logger.info('=' * 50)

        # Eliminate arms after pulling each arm a few times.
        if self.pick_id == len(self.arm_candidate):
            self.update_cnt += 1
            self.pick_id = 0
            # Update the arms until pulling each arm for at least alpha times.
            if self.update_cnt >= self.alpha:
                # Update the upper/lower bound estimation.
                budget_left = max(self.time_limit - time.time() + self.timestamp, 0)
                avg_cost = np.array([np.mean(self.arm_cost_stats[_arm]) for _arm in self.arm_candidate]).mean()
                steps = int(budget_left / avg_cost)
                upper_bounds, lower_bounds = list(), list()

                for _arm in self.arm_candidate:
                    rewards = self.rewards[_arm]
                    slope = (rewards[-1] - rewards[-self.alpha]) / self.alpha
                    
                    steps = self.evaluation_num_limit - self.pull_cnt
                    upper_bound = np.min([1.0, rewards[-1] + slope * steps])
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(rewards[-1])
                    
                    
                    self.best_lower_bounds[self.arms.index(_arm)] = rewards[-1]

                # Reject the sub-optimal arms.
                n = len(self.arm_candidate)
                flags = [False] * n
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            if upper_bounds[i] < lower_bounds[j]:
                                flags[i] = True
                for i in range(n):
                    if np.isnan(upper_bounds[i]) or not np.isfinite(lower_bounds[i]):
                        flags[i] = True

                if np.sum(flags) == n:
                    self.logger.error('Removing all the arms simultaneously!')

                self.logger.info('=' * 50)
                self.logger.info('Candidates  : %s' % ','.join(self.arm_candidate))
                self.logger.info('Upper bound : %s' % ','.join(['%.4f' % val for val in upper_bounds]))
                self.logger.info('Lower bound : %s' % ','.join(['%.4f' % val for val in lower_bounds]))
                self.logger.info(
                    'Arms removed: %s' % [item for idx, item in enumerate(self.arm_candidate) if flags[idx]])
                self.logger.info('=' * 50)

                # Update arm_candidates.
                self.arm_candidate = [item for index, item in enumerate(self.arm_candidate) if not flags[index]]

        # Update stop flag
        self.early_stopped_flag = True
        self.timeout_flag = False
        for _arm in self.arm_candidate:
            if not self.sub_bandits[_arm].early_stopped_flag:
                self.early_stopped_flag = False
            if self.sub_bandits[_arm].timeout_flag:
                self.timeout_flag = True
        if self.early_stopped_flag:
            self.logger.info("Maximum configuration number met for each arm candidate!")
        if self.timeout_flag:
            self.logger.info('Time elapsed!')

        iteration_cost = time.time() - _start_time

        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def _record_event(self, stage, iter_idx, cfg, perf, eval_time=0.0, mab_time=0.0, iter_time=0.0, note=""):
        wall_elapsed_sec = time.time() - self.start_wall_time
        algo = cfg.get('algorithm', 'default')

        # Update incumbent performance before logging
        if perf > self.incumbent_perf:
            self.incumbent_perf = perf

        row = {
            "ts": pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "stage": stage,
            "iter_idx": iter_idx,
            "perf": perf,
            "best_so_far": self.incumbent_perf,
            "algorithm": algo,
            "mab_time": mab_time,
            "eval_time": eval_time,
            "iter_time": iter_time,
            "wall_elapsed_sec": wall_elapsed_sec,
            "note": note,
        }
        self.traj_rows.append(row)
        
        try:
            columns = ["ts", "stage", "iter_idx", "perf", "best_so_far", "algorithm",
                       "mab_time", "eval_time", "iter_time", "wall_elapsed_sec", "note"]
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
                    "mab_time": r.get("mab_time", 0.0),
                    "eval_time": r.get("eval_time", 0.0),
                    "iter_time": r.get("iter_time", 0.0),
                    "wall_elapsed_sec": r["wall_elapsed_sec"],
                } for r in self.traj_rows
            ][:500]
        }

