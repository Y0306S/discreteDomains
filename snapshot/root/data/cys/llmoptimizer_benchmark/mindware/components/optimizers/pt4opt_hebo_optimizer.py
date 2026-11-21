"""Pure HEBO (no LLM), same CASH lifecycle as classical TPE/CMA-ES."""

from __future__ import annotations

import contextlib
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import Constant, UnParametrizedHyperparameter
from openbox.utils.constants import SUCCESS

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.hebo_space import (
    build_hebo_space,
    coerce_hebo_value,
    config_to_hebo_frame,
    fixed_hp_value,
)
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.pt4opt_optuna_optimizer import OptunaConfigSpaceMixin
from mindware.components.optimizers.utils import MyConfiguration
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase


def _cpu_hebo_context():
    import torch

    return torch.device("cpu")


class HEBOOptimizer(OptunaConfigSpaceMixin, BaseOptimizer):
    optimizer_label = "hebo"

    def __init__(
        self,
        evaluator,
        config_space,
        name,
        eval_type,
        time_limit=None,
        evaluation_limit=None,
        per_run_time_limit=300,
        per_run_mem_limit=1024,
        inner_iter_num_per_iter=1,
        timestamp=None,
        output_dir="./",
        seed=1,
        n_jobs=1,
        topk=50,
        optimizer_params=None,
        task_id="GenTrans",
        logging_enabled=True,
        failure_penalty=1e12,
        **sampler_params,
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

        self.optimizer_params = dict(optimizer_params or {})
        self.optimizer_params.update(sampler_params)
        self.optimizer_params.pop("rand_prob", None)
        self.optimizer_params.pop("acq_optimizer_rand_prob", None)
        self.failure_penalty = float(failure_penalty)

        if n_jobs != 1:
            self.logger.warning(
                "%s currently uses sequential HEBO suggest/observe; n_jobs=%s will be ignored.",
                self.__class__.__name__,
                n_jobs,
            )
        self.n_jobs = 1

        self.trial_cnt = 0
        self.exp_output = {}

        self.KD = KnowledgeDatabase.instance()
        self.configs = self.KD.configs
        self.perfs = self.KD.perfs

        self._algorithm_hp = self._get_hyperparameter("algorithm")
        self._algorithm_choices = self._get_choices(self._algorithm_hp)
        self._single_algorithm = self._get_single_algorithm()
        self.bo_mode = "bo_mab" if self._single_algorithm is not None else "bo"

        n_startup = int(self.optimizer_params.get("n_startup_trials", 3) or 3)
        self.hebo_duplicate_retries = max(
            0,
            int(
                self.optimizer_params.get("hebo_duplicate_retries")
                or self.optimizer_params.get("bo_duplicate_retries")
                or 50
            ),
        )
        self.hebo_space, self._hebo_hp_by_name = build_hebo_space(
            self.config_space,
            self._single_algorithm,
            self._is_relevant_to_algorithm,
        )
        from hebo.optimizers.hebo import HEBO

        self.hebo = HEBO(
            self.hebo_space,
            model_name=str(self.optimizer_params.get("hebo_model_name") or "gp"),
            rand_sample=max(2, n_startup),
            scramble_seed=int(self.seed),
        )

        self.logging_enabled = logging_enabled
        if self.logging_enabled:
            self.traj_rows: List[Dict[str, Any]] = []
            self.traj_csv_path = os.path.join(
                self.output_dir, f"{int(self.timestamp)}_opt_{self.optimizer_label}_traj.csv"
            )
            self.start_wall_time = time.time()
            self.outer_iter_idx = 0

        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            try:
                self.config_num_threshold = int(len(set(self.config_space.sample_configuration(5000))))
            except Exception:
                self.config_num_threshold = 1500
        self.logger.debug("The maximum trial number in HPO is: %d" % self.config_num_threshold)
        self.maximum_config_num = min(1500, self.config_num_threshold)

    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                self.logger.info("Early stop triggered, terminating optimization.")
                break
            self.iterate()
        return np.max(self.perfs) if self.perfs else -np.inf

    def iterate(self, budget=MAX_INT) -> Tuple[float, float, Any, List[Dict]]:
        _start_time = time.time()
        newly_observed_rows = []

        if len(self.configs) == 0 and hasattr(self, "init_hpo_iter_num") and self.init_hpo_iter_num is not None:
            inner_iter_num = self.init_hpo_iter_num
        else:
            inner_iter_num = self.inner_iter_num_per_iter

        for i in range(inner_iter_num):
            if len(self.configs) >= self.maximum_config_num or self.early_stopped_flag or self.timeout_flag:
                break
            if time.time() - _start_time > budget:
                self.logger.warning("Time limit exceeded for this iteration!")
                break

            iter_idx = self.trial_cnt
            self.trial_cnt += 1
            iter_start_time = time.time()

            config, rec, hebo_time, note = self._suggest_hebo_config()
            obs = wrapper_evaluater(config, self.evaluator, self.per_run_time_limit)

            raw_objective = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_objective)
            tell_objective = raw_objective if success else None
            if rec is not None:
                self._tell_hebo_frame(rec, tell_objective)
            else:
                self._tell_hebo_external_config(config, tell_objective)

            iter_time = time.time() - iter_start_time
            eval_time = obs.elapsed_time if obs.elapsed_time is not None else 0.0
            smac_time = hebo_time
            perf = -raw_objective if success else float("-inf")

            if success:
                self.exp_output[time.time()] = (config, -raw_objective)
                self.update_saver([config], [raw_objective])
                self.update_history(obs, trail_id=f"{iter_idx}")

            row_data = {
                "perf": perf,
                "config": config,
                "eval_time": eval_time,
                "smac_time": smac_time,
                "iter_time": iter_time,
                "note": note or f"candidate {i + 1}/{inner_iter_num} via {self.optimizer_label}",
            }
            newly_observed_rows.append(row_data)

            if self.logging_enabled:
                self._record_event(
                    stage="iterate",
                    iter_idx=iter_idx,
                    cfg=config,
                    perf=perf,
                    eval_time=eval_time,
                    smac_time=smac_time,
                    iter_time=iter_time,
                    note=row_data["note"],
                )

        if self.logging_enabled:
            self.outer_iter_idx += 1

        if self.time_limit is not None and time.time() - self.timestamp > self.time_limit or (
            hasattr(self, "evaluation_num_limit")
            and self.evaluation_num_limit is not None
            and len(self.perfs) >= self.evaluation_num_limit
        ):
            self.timeout_flag = True

        iteration_cost = time.time() - _start_time
        if self.perfs:
            incumbent_idx = int(np.argmax(self.perfs))
            incumbent = self.configs[incumbent_idx]
            self.incumbent_config = incumbent.get_dictionary().copy()
            self.incumbent_perf = float(self.perfs[incumbent_idx])

        return self.incumbent_perf, iteration_cost, self.incumbent_config, newly_observed_rows

    def update_history(self, obs, **extra_infos):
        self._update_optuna_observation_history(obs)
        self.KD.update_history(obs, source=self.bo_mode, **extra_infos)

    def _hebo_row_to_config(self, row) -> MyConfiguration:
        values: Dict[str, Any] = {}
        if self._algorithm_hp is not None and self._single_algorithm is not None:
            values["algorithm"] = self._single_algorithm
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                values[hp.name] = fixed_hp_value(hp)
        for name in self.hebo_space.para_names:
            if name in row.index:
                values[name] = coerce_hebo_value(self._hebo_hp_by_name.get(name), row[name])
        try:
            cfg = Configuration(self.config_space, values=values)
            return MyConfiguration(cfg)
        except Exception as exc:
            self.logger.warning("HEBO generated an invalid ConfigSpace configuration: %s", exc)
            return self._sample_fallback_configuration()

    def _config_key(self, cfg) -> tuple:
        cfg_dict = cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)
        return tuple(sorted((k, str(v)) for k, v in cfg_dict.items()))

    def _is_duplicate(self, cfg) -> bool:
        key = self._config_key(cfg)
        for seen in self.configs:
            if self._config_key(seen) == key:
                return True
        return False

    def _known_objective(self, cfg) -> Optional[float]:
        key = self._config_key(cfg)
        for seen, perf in zip(self.configs, self.perfs):
            if self._config_key(seen) == key:
                # KD perfs are higher-better; HEBO observes the raw minimised objective.
                return float(-perf)
        return None

    def _suggest_hebo_config(self):
        start = time.time()
        duplicate_count = 0
        import torch

        for _ in range(self.hebo_duplicate_retries + 1):
            try:
                buf = io.StringIO()
                with torch.device(_cpu_hebo_context()), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    rec = self.hebo.suggest(n_suggestions=1)
            except Exception as exc:
                self.logger.warning("HEBO suggest failed: %s", exc)
                rec = self.hebo.quasi_sample(1)
            cfg = self._hebo_row_to_config(rec.iloc[0])
            if not self._is_duplicate(cfg):
                note = "hebo suggested configuration"
                if duplicate_count:
                    note += f" after {duplicate_count} duplicate retries"
                return cfg, rec, time.time() - start, note
            duplicate_count += 1
            self._tell_hebo_duplicate(rec, cfg)

        cfg = self._sample_fallback_configuration()
        note = f"random fallback after {duplicate_count} duplicate hebo configurations"
        return cfg, None, time.time() - start, note

    def _tell_hebo_frame(self, rec: pd.DataFrame, objective: Optional[float]):
        if objective is None or not np.isfinite(float(objective)):
            return
        try:
            y = np.asarray([[float(objective)]], dtype=float)
            self.hebo.observe(rec, y)
        except Exception as exc:
            self.logger.warning("Failed to tell HEBO observation: %s", exc)

    def _tell_hebo_duplicate(self, rec: pd.DataFrame, cfg):
        self._tell_hebo_frame(rec, self._known_objective(cfg))

    def _tell_hebo_external_config(self, cfg, objective: Optional[float]):
        frame = config_to_hebo_frame(
            cfg,
            self.hebo_space,
            self._hebo_hp_by_name,
            self._single_algorithm,
            self._is_relevant_to_algorithm,
        )
        if frame is None:
            return
        self._tell_hebo_frame(frame, objective)

    def _record_event(self, stage, iter_idx, cfg, perf, eval_time=0.0, smac_time=0.0, iter_time=0.0, note=""):
        if not self.logging_enabled:
            return

        row = {
            "ts": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "iter_idx": iter_idx,
            "perf": perf,
            "best_so_far": max(self.perfs) if self.perfs else float("-inf"),
            "algorithm": self._selected_algorithm(cfg) or "default",
            "smac_time": smac_time,
            "eval_time": eval_time,
            "iter_time": iter_time,
            "wall_elapsed_sec": time.time() - self.start_wall_time,
            "note": note,
        }
        self.traj_rows.append(row)
        try:
            columns = [
                "ts",
                "stage",
                "iter_idx",
                "perf",
                "best_so_far",
                "algorithm",
                "smac_time",
                "eval_time",
                "iter_time",
                "wall_elapsed_sec",
                "note",
            ]
            pd.DataFrame(self.traj_rows)[columns].to_csv(self.traj_csv_path, index=False)
        except IOError as e:
            self.logger.warning(f"Failed to save trajectory to {self.traj_csv_path}. Error: {e}")

    def get_opt_trajectory(self):
        best = float(max(self.perfs)) if self.perfs else float("-inf")
        traj_rows = getattr(self, "traj_rows", [])
        return {
            "n_evals": len(self.perfs),
            "final_best": best,
            "csv_path": getattr(self, "traj_csv_path", None),
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
                }
                for r in traj_rows
            ][:500],
        }
