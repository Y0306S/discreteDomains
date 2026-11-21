import csv
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from ConfigSpace.hyperparameters import Constant, UnParametrizedHyperparameter
from openbox.utils.constants import SUCCESS

from mindware.components.optimizers.base_optimizer import MAX_INT
from mindware.components.optimizers.pt4opt_llm_optimizer import (
    PT4OptLLMOptimizer,
    _cfg_dict,
    _config_for_prompt,
    _json,
    _signature,
)
from mindware.components.optimizers.pt4opt_optuna_optimizer import OptunaConfigSpaceMixin, PT4OptOptunaOptimizer
from mindware.components.optimizers.ml_evaluation_feedback import (
    observation_feedback, tpe_sampler_params, upgrade_runtime_study,
)


class PT4OptLLMTPEOptimizer(PT4OptLLMOptimizer, OptunaConfigSpaceMixin):
    optimizer_label = "ptllm_tpe"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm_prob = min(1.0, max(0.0, float(self.params.get("llm_prob", 0.5))))
        self.llm_phase_trials = max(0, int(self.params.get("llm_phase_trials", 0)))
        self.failure_penalty = float(self.params.get("failure_penalty", 1e12))
        self.hybrid_rng = np.random.RandomState(self.seed)
        self.hybrid_selector_mode = str(self.params.get("hybrid_selector") or "bernoulli").strip().lower()
        from mindware.components.optimizers.hybrid_selector import selector_from_params

        self.selector = selector_from_params(self.params, rng=self.hybrid_rng)
        self.tpe_duplicate_retries = max(
            0,
            int(
                self.params.get("tpe_duplicate_retries")
                or self.params.get("bo_duplicate_retries")
                or 50
            ),
        )

        self.optimizer_params = dict(self.params.get("tpe_params") or {})
        if "n_startup_trials" in self.params:
            self.optimizer_params["n_startup_trials"] = self.params["n_startup_trials"]
        self.optimizer_params.setdefault("seed", self.seed)

        self._algorithm_hp = self.algorithm_hp
        self._algorithm_choices = self._get_choices(self._algorithm_hp)
        self._single_algorithm = self.fixed_algorithm
        self.bo_mode = "bo_mab"

        self.study = optuna.create_study(
            direction="minimize",
            sampler=self._build_sampler(),
            study_name=f"{self.optimizer_label}-{self.task_id}-{int(self.timestamp)}",
        )
        self.trial_sources: Dict[int, str] = {}
        if getattr(self, "_explore_policy", None) is not None:
            self._explore_policy.tpe_peek = self._peek_tpe_config

    def _peek_tpe_config(self, llm_only: bool = False) -> Optional[Dict[str, Any]]:
        """Ghost TPE sample. Default uses all told trials; llm_only rebuilds TPE on LLM rows only."""
        if llm_only:
            return self._peek_tpe_from_llm_rows()
        import copy

        try:
            frozen = copy.deepcopy(self.study)
            trial = frozen.ask()
            cfg = self._suggest_config_from_trial(trial)
            return _config_for_prompt(cfg, self.fixed_algorithm, self.config_space)
        except Exception as exc:
            self.logger.warning("TPE ghost peek failed: %s", exc)
            return None

    def _peek_tpe_from_llm_rows(self) -> Optional[Dict[str, Any]]:
        """Fit a throwaway TPE on LLM / warm-start trials only, then ask once."""
        rows = []
        for rec in getattr(self, "traj_rows", None) or []:
            if rec.get("stage") not in ("llm", "llm_warm_start"):
                continue
            try:
                perf = float(rec.get("perf"))
            except Exception:
                continue
            if perf != perf or perf <= float("-1e18"):
                continue
            raw = rec.get("config_for_prompt")
            if isinstance(raw, str):
                try:
                    cfg = json.loads(raw)
                except Exception:
                    continue
            elif isinstance(raw, dict):
                cfg = raw
            else:
                continue
            params = self._config_to_tpe_params(cfg)
            if not params:
                continue
            rows.append((params, float(-perf)))
        if len(rows) < 8:
            return None
        try:
            sampler_kwargs = dict(self.optimizer_params)
            sampler_kwargs["n_startup_trials"] = min(8, max(3, len(rows) // 3))
            sampler_kwargs.setdefault("seed", int(getattr(self, "seed", 1) or 1) + 91)
            dummy = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(**sampler_kwargs),
            )
            for params, objective in rows:
                dummy.enqueue_trial(params)
                trial = dummy.ask()
                dummy.tell(trial, objective)
            trial = dummy.ask()
            cfg = self._suggest_config_from_trial(trial)
            return _config_for_prompt(cfg, self.fixed_algorithm, self.config_space)
        except Exception as exc:
            self.logger.warning("LLM-only TPE peek failed: %s", exc)
            return None

    def _peek_tpe_hint(self):
        self._tpe_hint_config = None
        if not self.prompt_blocks.get("tpe_hint"):
            return
        try:
            trial = self.study.ask()
            cfg = self._suggest_config_from_trial(trial)
            self._tpe_hint_config = _config_for_prompt(cfg, self.fixed_algorithm, self.config_space)
            self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as exc:
            self.logger.warning("Failed to peek TPE hint: %s", exc)
            self._tpe_hint_config = None

    def _suggest_config(self, iter_idx: int, include_history: bool = True):
        if include_history:
            self._peek_tpe_hint()
        else:
            self._tpe_hint_config = None
        return super()._suggest_config(iter_idx, include_history=include_history)

    def _build_sampler(self):
        return optuna.samplers.TPESampler(**tpe_sampler_params(self.optimizer_params))

    def iterate(self, budget=MAX_INT):
        self.study = upgrade_runtime_study(self.study)
        start = time.time()
        rows = []
        for _ in range(self.inner_iter_num_per_iter):
            if self._quota_or_time_is_up() or time.time() - start > budget:
                break

            iter_idx = self.trial_cnt
            self.trial_cnt += 1
            iter_start = time.time()
            source, include_history, selection_draw = self._choose_source(iter_idx)

            llm_cost = llm_time = tpe_time = 0.0
            raw, payload, meta, rationale, prompt_input = "", {}, {}, "", {}
            tpe_trial = None

            if source in ("llm", "llm_warm_start"):
                cfg, note, llm_cost, llm_time, raw, payload, meta, rationale, prompt_input = self._suggest_config(
                    iter_idx,
                    include_history=include_history,
                )
            else:
                cfg, tpe_trial, tpe_time, note = self._suggest_tpe_config()

            obs = self.objective_function(cfg)
            raw_obj = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_obj)
            perf = -raw_obj if success else float("-inf")
            eval_time = obs.elapsed_time or 0.0
            prev_best = float(max(self.perfs)) if self.perfs else float("-inf")

            tell_objective, tell_state, feedback = observation_feedback(obs, self.per_run_time_limit)
            if tpe_trial is not None:
                self._tell_tpe_trial(tpe_trial, tell_objective, cfg, state=tell_state, attrs=feedback)
            else:
                self._tell_tpe_external_config(cfg, tell_objective, state=tell_state, attrs=feedback)

            if success:
                self.exp_output[time.time()] = (cfg, perf)
                self.update_saver([cfg], [raw_obj])
                if source in ("llm", "llm_warm_start"):
                    self._update_history(obs, source, iter_idx, cfg, rationale)
                else:
                    PT4OptOptunaOptimizer.update_history(self, obs, trail_id=str(iter_idx))
                self._mark_seen(cfg, raw_obj)

            beat_inc = bool(success and perf > prev_best)
            if source in ("llm", "tpe"):
                arm_best = self.selector.arm_best.get(source)
                beat_arm = bool(success and (arm_best is None or perf > float(arm_best)))
                self.selector.update(
                    source,
                    perf if success else float("-inf"),
                    higher_is_better=True,
                    beat_incumbent=beat_inc,
                    beat_arm_best=beat_arm,
                )
            ee_pol = getattr(self, "_explore_policy", None)
            if ee_pol is not None and source == "llm":
                br = getattr(ee_pol, "cached_branch", None)
                if br:
                    ee_pol.update(br, perf if success else float("-inf"), beat_incumbent=beat_inc)

            if source in ("llm", "llm_warm_start"):
                self._finish_llm_history(cfg, perf, success, obs)
                self._dump_conversation(iter_idx, prompt_input, raw, payload, cfg, obs, meta, note)

            iter_time = time.time() - iter_start
            row = {
                "perf": perf,
                "config": cfg,
                "eval_time": eval_time,
                "llm_time": llm_time,
                "tpe_time": tpe_time,
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
                tpe_time=tpe_time,
                eval_time=eval_time,
                iter_time=iter_time,
                selection_draw=selection_draw,
                note=note,
                feedback=feedback,
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
        m = max(0, int(iter_idx) - int(self.n_no_history_samples))
        M = max(1, int(getattr(self, "evaluation_num_limit", 0) or 0) - int(self.n_no_history_samples))
        arm = self.selector.choose(m, M, higher_is_better=True)
        snap = self.selector.snapshot()
        draw = snap.get("selection_draw")
        return arm, include_history, draw

    def _suggest_tpe_config(self):
        start = time.time()
        duplicate_count = 0
        invalid_count = 0
        for _ in range(self.tpe_duplicate_retries + 1):
            trial = self.study.ask()
            cfg = self._try_suggest_config_from_trial(trial)
            if cfg is None:
                trial.set_user_attr("rejection_reason", "invalid_configspace_configuration")
                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                invalid_count += 1
                if invalid_count >= PT4OptOptunaOptimizer.max_suggest_retries:
                    break
                continue
            if not self._is_duplicate(cfg):
                note = "tpe suggested configuration"
                if duplicate_count:
                    note += f" after {duplicate_count} duplicate retries"
                return cfg, trial, time.time() - start, note
            duplicate_count += 1
            self._tell_tpe_duplicate_trial(trial, cfg)

        cfg = self._sample_nonduplicate_config()
        note = f"random fallback after {duplicate_count} duplicate and {invalid_count} invalid tpe configurations"
        return cfg, None, time.time() - start, note

    def _tell_tpe_trial(self, trial, objective, cfg, **feedback):
        return self._tell_config_result(cfg, objective, trial, **feedback)

    def _tell_tpe_duplicate_trial(self, trial, cfg):
        objective = self._known_objective(cfg)
        return self._tell_config_result(cfg, objective, trial, attrs={"cached_duplicate": True})

    def _tell_tpe_external_config(self, cfg, objective, **feedback):
        return self._tell_config_result(cfg, objective, **feedback)

    def _config_to_tpe_params(self, cfg) -> Dict[str, Any]:
        cfg_dict = _cfg_dict(cfg)
        algorithm = cfg_dict.get("algorithm", self.fixed_algorithm)
        params = {}
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                continue
            if not self._is_relevant_to_algorithm(name, algorithm):
                continue
            if name in cfg_dict:
                params[name] = cfg_dict[name]
        return params

    def _record_hybrid_event(
        self,
        stage,
        iter_idx,
        cfg,
        perf,
        llm_cost,
        llm_time,
        tpe_time,
        eval_time,
        iter_time,
        selection_draw,
        note,
        feedback=None,
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
            "hybrid_selector": str(getattr(self, "hybrid_selector_mode", "bernoulli")),
            "selection_draw": "" if selection_draw is None else float(selection_draw),
            "llm_cost": float(llm_cost),
            "cumulative_llm_cost": float(sum(self.llm_query_cost)),
            "llm_time": float(llm_time),
            "tpe_time": float(tpe_time),
            "eval_time": float(eval_time),
            "iter_time": float(iter_time),
            "wall_elapsed_sec": float(time.time() - self.start_wall_time),
            "note": note,
            "config_signature": _json(_signature(cfg)),
            "config_for_prompt": _json(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space)),
            **(feedback or {}),
        }
        snap = getattr(self, "selector", None)
        if snap is not None:
            info = snap.snapshot()
            row.update(
                {
                    "Q_llm": info.get("Q_llm", ""),
                    "Q_tpe": info.get("Q_tpe", ""),
                    "N_llm": info.get("N_llm", ""),
                    "N_tpe": info.get("N_tpe", ""),
                    "omega_llm": info.get("omega_llm", ""),
                    "omega_tpe": info.get("omega_tpe", ""),
                    "pucb_q": info.get("pucb_q", ""),
                }
            )
        meta = getattr(self, "_last_history_meta", None) or {}
        for key in (
            "experience_explore_branch",
            "experience_explore_p",
            "experience_explore_stagnation",
            "experience_explore_coverage",
            "experience_explore_tpe_ratio",
            "experience_explore_tpe_n_constraints",
            "experience_explore_tpe_peeked",
            "experience_explore_tpe_scope",
            "experience_self_gate",
            "experience_self_active",
            "experience_gate_latched",
        ):
            if key in meta:
                row[key] = meta[key]
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
            "tpe_trials": int(stage_counts.get("tpe", 0)),
            "tpe_study_trials": len(self.study.trials),
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join(str(p) for p in self.perfs),
            "history": self.traj_rows[:500],
        }
