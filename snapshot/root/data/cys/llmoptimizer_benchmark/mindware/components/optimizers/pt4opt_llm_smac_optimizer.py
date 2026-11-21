import csv
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from openbox.utils.constants import FAILED, SUCCESS
from openbox.utils.history import Observation

from mindware.components.optimizers.base_optimizer import MAX_INT
from mindware.components.optimizers.generic_smbo import MySMBO as BO
from mindware.components.optimizers.pt4opt_llm_optimizer import (
    PT4OptLLMOptimizer,
    _config_for_prompt,
    _json,
    _signature,
)


class PT4OptLLMSMACOptimizer(PT4OptLLMOptimizer):
    """LLM + SMAC hybrid with the same scheduling and tell/KD wiring as PT4OptLLMTPEOptimizer."""

    optimizer_label = "ptllm_smac"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm_prob = min(1.0, max(0.0, float(self.params.get("llm_prob", 0.5))))
        self.llm_phase_trials = max(0, int(self.params.get("llm_phase_trials", 0)))
        self.bo_duplicate_retries = max(
            0,
            int(
                self.params.get("bo_duplicate_retries")
                or self.params.get("smac_duplicate_retries")
                or self.params.get("tpe_duplicate_retries")
                or 50
            ),
        )
        self.failure_penalty = float(self.params.get("failure_penalty", 1e12))
        self.hybrid_rng = np.random.RandomState(self.seed)

        smac_params = dict(self.params.get("smac_params") or {})
        if "n_startup_trials" in self.params:
            smac_params.setdefault("initial_runs", int(self.params["n_startup_trials"]))
        smac_params.setdefault("initial_runs", 3)
        smac_params.setdefault("seed", self.seed)

        self.bo_mode = "bo_mab"
        # Isolated SMAC advisor history (like Optuna study in ptllm_tpe); fed via _tell_smac_advisor.
        self.smac_bo = BO(
            objective_function=self.objective_function,
            config_space=self.config_space,
            surrogate_type="prf",
            acq_type="ei",
            max_runs=int(1e10),
            max_runtime_per_trial=self.per_run_time_limit,
            random_state=self.seed,
            logging_dir=self.output_dir,
            task_id=self.task_id,
            initial_runs=int(smac_params["initial_runs"]),
            advisor_kwargs={
                "rand_prob": float(smac_params.get("rand_prob", 0.1)),
                "acq_optimizer_rand_prob": float(smac_params.get("acq_optimizer_rand_prob", 0)),
            },
        )
        self.trial_sources: Dict[int, str] = {}

    def iterate(self, budget=MAX_INT):
        start = time.time()
        rows = []
        for _ in range(self.inner_iter_num_per_iter):
            if self._quota_or_time_is_up() or time.time() - start > budget:
                break

            iter_idx = self.trial_cnt
            self.trial_cnt += 1
            iter_start = time.time()
            source, include_history, selection_draw = self._choose_source(iter_idx)

            llm_cost = llm_time = smac_time = 0.0
            raw, payload, meta, rationale, prompt_input = "", {}, {}, "", {}

            if source in ("llm", "llm_warm_start"):
                cfg, note, llm_cost, llm_time, raw, payload, meta, rationale, prompt_input = self._suggest_config(
                    iter_idx,
                    include_history=include_history,
                )
            else:
                cfg, smac_time, note = self._suggest_smac_config()

            obs = self.objective_function(cfg)
            raw_obj = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_obj)
            perf = -raw_obj if success else float("-inf")
            eval_time = obs.elapsed_time or 0.0

            tell_objective = raw_obj if success else None
            self._tell_smac_advisor(cfg, tell_objective, eval_time)

            if success:
                self.exp_output[time.time()] = (cfg, perf)
                self.update_saver([cfg], [raw_obj])
                if source in ("llm", "llm_warm_start"):
                    self._update_history(obs, source, iter_idx, cfg, rationale)
                else:
                    self.KD.update_history(obs, source=self.bo_mode, trail_id=str(iter_idx))
                self._mark_seen(cfg, raw_obj)

            if source in ("llm", "llm_warm_start"):
                self._finish_llm_history(cfg, perf, success, obs)
                self._dump_conversation(iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)

            iter_time = time.time() - iter_start
            row = {
                "perf": perf,
                "config": cfg,
                "eval_time": eval_time,
                "llm_time": llm_time,
                "smac_time": smac_time,
                "iter_time": iter_time,
                "note": note,
            }
            rows.append(row)
            self._record_hybrid_event(
                stage=source,
                iter_idx=iter_idx,
                cfg=cfg,
                perf=perf,
                llm_cost=llm_cost,
                llm_time=llm_time,
                smac_time=smac_time,
                eval_time=eval_time,
                iter_time=iter_time,
                selection_draw=selection_draw,
                note=note,
            )
            self.trial_sources[int(iter_idx) + 1] = source

        if self._quota_or_time_is_up():
            self.timeout_flag = True
        self._update_incumbent()
        return self.incumbent_perf, time.time() - start, self.incumbent_config, rows

    def _choose_source(self, iter_idx: int):
        include_history = iter_idx >= self.n_no_history_samples
        if not include_history:
            return "llm_warm_start", False, None
        if self.llm_phase_trials > 0 and iter_idx < self.llm_phase_trials:
            return "llm", include_history, None
        draw = float(self.hybrid_rng.rand())
        return ("llm", include_history, draw) if draw < self.llm_prob else ("smac", include_history, draw)

    def _suggest_smac_config(self):
        start = time.time()
        duplicate_count = 0
        for _ in range(self.bo_duplicate_retries + 1):
            cfg = self.smac_bo.config_advisor.get_suggestion()
            if not self._is_duplicate(cfg):
                note = "smac suggested configuration"
                if duplicate_count:
                    note += f" after {duplicate_count} duplicate retries"
                return cfg, time.time() - start, note
            duplicate_count += 1
            self._tell_smac_duplicate(cfg)

        cfg = self._sample_nonduplicate_config()
        note = f"random fallback after {duplicate_count} duplicate smac configurations"
        return cfg, time.time() - start, note

    def _tell_smac_advisor(self, cfg, objective: Optional[float], elapsed_time: float = 0.0):
        try:
            if objective is None:
                observation = Observation(
                    config=cfg,
                    objectives=[np.inf],
                    trial_state=FAILED,
                    elapsed_time=elapsed_time,
                )
            else:
                observation = Observation(
                    config=cfg,
                    objectives=[float(objective)],
                    trial_state=SUCCESS,
                    elapsed_time=elapsed_time,
                )
            self.smac_bo.config_advisor.update_observation(observation)
        except Exception as exc:
            self.logger.warning("Failed to tell SMAC advisor: %s", exc)

    def _tell_smac_duplicate(self, cfg):
        objective = self._known_objective(cfg)
        try:
            if objective is None:
                self.smac_bo.config_advisor.update_observation(
                    Observation(
                        config=cfg,
                        objectives=[np.inf],
                        trial_state=FAILED,
                        elapsed_time=0.0,
                    )
                )
            else:
                self.smac_bo.config_advisor.update_observation(
                    Observation(
                        config=cfg,
                        objectives=[float(objective)],
                        trial_state=SUCCESS,
                        elapsed_time=0.0,
                    )
                )
        except Exception as exc:
            self.logger.warning("Failed to tell duplicate SMAC config: %s", exc)

    def _record_hybrid_event(
        self,
        stage,
        iter_idx,
        cfg,
        perf,
        llm_cost,
        llm_time,
        smac_time,
        eval_time,
        iter_time,
        selection_draw,
        note,
    ):
        if not self.logging_enabled:
            return
        if llm_cost:
            self.llm_query_cost.append(float(llm_cost))
        if llm_time:
            self.llm_query_time.append(float(llm_time))
        row = {
            "ts": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "iter_idx": iter_idx,
            "perf": float(perf),
            "best_so_far": float(max(self.perfs) if self.perfs else perf),
            "algorithm": self.fixed_algorithm,
            "llm_prob": float(self.llm_prob),
            "llm_phase_trials": int(self.llm_phase_trials),
            "selection_draw": "" if selection_draw is None else float(selection_draw),
            "llm_cost": float(llm_cost),
            "cumulative_llm_cost": float(sum(self.llm_query_cost)),
            "llm_time": float(llm_time),
            "smac_time": float(smac_time),
            "eval_time": float(eval_time),
            "iter_time": float(iter_time),
            "wall_elapsed_sec": float(time.time() - self.start_wall_time),
            "note": note,
            "config_signature": _json(_signature(cfg)),
            "config_for_prompt": _json(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space)),
        }
        self.traj_rows.append(row)
        try:
            exists = os.path.exists(self.traj_csv_path)
            with open(self.traj_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except IOError as exc:
            self.logger.warning("Failed to save trajectory to %s: %s", self.traj_csv_path, exc)

    def get_opt_trajectory(self):
        best = float(max(self.perfs)) if self.perfs else float("-inf")
        stage_counts = {}
        for row in self.traj_rows:
            stage = row.get("stage", "")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        return {
            "n_evals": len(self.perfs),
            "final_best": best,
            "llm_cost_sum": float(sum(self.llm_query_cost)),
            "llm_calls": int(self.llm_call_count),
            "llm_prob": float(self.llm_prob),
            "llm_phase_trials": int(self.llm_phase_trials),
            "stage_counts": stage_counts,
            "llm_trials": int(stage_counts.get("llm", 0)),
            "llm_warm_start_trials": int(stage_counts.get("llm_warm_start", 0)),
            "smac_trials": int(stage_counts.get("smac", 0)),
            "smac_advisor_trials": len(self.smac_bo.config_advisor.history),
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join(str(p) for p in self.perfs),
            "history": self.traj_rows[:500],
        }
