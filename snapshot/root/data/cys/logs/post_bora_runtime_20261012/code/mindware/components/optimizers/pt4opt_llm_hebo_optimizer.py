"""LLM + HEBO hybrid, mirroring PT4OptLLMTPEOptimizer scheduling (H0 mix, selectors)."""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
from openbox.utils.constants import SUCCESS

try:
    from ConfigSpace.hyperparameters import OrdinalHyperparameter
except ImportError:
    OrdinalHyperparameter = ()

from mindware.components.optimizers.base_optimizer import MAX_INT
from mindware.components.optimizers.pt4opt_llm_optimizer import (
    PT4OptLLMOptimizer,
    _cfg_dict,
    _config_for_prompt,
    _json,
    _signature,
)
from mindware.components.optimizers.pt4opt_optuna_optimizer import OptunaConfigSpaceMixin
from mindware.components.optimizers.utils import MyConfiguration


def _cpu_hebo_context():
    """Keep HEBO's GPyTorch GP on CPU so it does not steal LLM VRAM."""
    import torch

    return torch.device("cpu")


class PT4OptLLMHEBOOptimizer(PT4OptLLMOptimizer, OptunaConfigSpaceMixin):
    optimizer_label = "ptllm_hebo"
    _bo_stage = "hebo"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm_prob = min(1.0, max(0.0, float(self.params.get("llm_prob", 0.5))))
        self.llm_phase_trials = max(0, int(self.params.get("llm_phase_trials", 0)))
        self.failure_penalty = float(self.params.get("failure_penalty", 1e12))
        self.hybrid_rng = np.random.RandomState(self.seed)
        self.hybrid_selector_mode = str(self.params.get("hybrid_selector") or "bernoulli").strip().lower()
        from mindware.components.optimizers.hybrid_selector import selector_from_params

        hebo_params = dict(self.params)
        hebo_params.setdefault("bo_arm", self._bo_stage)
        self.selector = selector_from_params(hebo_params, rng=self.hybrid_rng)
        self.hebo_duplicate_retries = max(
            0,
            int(
                self.params.get("hebo_duplicate_retries")
                or self.params.get("bo_duplicate_retries")
                or self.params.get("tpe_duplicate_retries")
                or 50
            ),
        )

        self._algorithm_hp = self.algorithm_hp
        self._algorithm_choices = self._get_choices(self._algorithm_hp)
        self._single_algorithm = self.fixed_algorithm
        self.bo_mode = "bo_mab"

        n_startup = int(self.params.get("n_startup_trials", 3) or 3)
        self.hebo_space, self._hebo_hp_by_name = self._build_hebo_space()
        from hebo.optimizers.hebo import HEBO

        self.hebo = HEBO(
            self.hebo_space,
            model_name=str(self.params.get("hebo_model_name") or "gp"),
            rand_sample=max(2, n_startup),
            scramble_seed=int(self.seed),
        )
        self.trial_sources: Dict[int, str] = {}
        self._pending_hebo_rec: Optional[pd.DataFrame] = None

    def _build_hebo_space(self):
        from hebo.design_space.design_space import DesignSpace

        recs = []
        hp_by_name = {}
        algorithm = self.fixed_algorithm
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                continue
            if not self._is_relevant_to_algorithm(name, algorithm):
                continue
            spec = self._hp_to_hebo_spec(hp)
            if spec is None:
                continue
            recs.append(spec)
            hp_by_name[name] = hp
        if not recs:
            raise ValueError("HEBO space is empty after filtering ConfigSpace hyperparameters")
        return DesignSpace().parse(recs), hp_by_name

    def _hp_to_hebo_spec(self, hp) -> Optional[dict]:
        name = hp.name
        if isinstance(hp, UniformIntegerHyperparameter):
            low = int(hp.lower)
            high = int(hp.upper)
            log = bool(getattr(hp, "log", False))
            step = int(getattr(hp, "q", None) or 1)
            if log:
                return {"name": name, "type": "pow_int", "lb": max(1, low), "ub": high}
            if step > 1:
                return {"name": name, "type": "step_int", "lb": low, "ub": high, "step": step}
            return {"name": name, "type": "int", "lb": low, "ub": high}
        if isinstance(hp, UniformFloatHyperparameter):
            low = float(hp.lower)
            high = float(hp.upper)
            log = bool(getattr(hp, "log", False))
            if log:
                low = max(low, 1e-12)
                return {"name": name, "type": "pow", "lb": low, "ub": high}
            return {"name": name, "type": "num", "lb": low, "ub": high}
        if isinstance(hp, CategoricalHyperparameter):
            cats = list(hp.choices)
            if cats == [False, True] or cats == [True, False]:
                return {"name": name, "type": "bool"}
            return {"name": name, "type": "cat", "categories": cats}
        if OrdinalHyperparameter and isinstance(hp, OrdinalHyperparameter):
            return {"name": name, "type": "cat", "categories": list(hp.sequence)}
        return None

    def _hebo_row_to_config(self, row) -> MyConfiguration:
        values: Dict[str, Any] = {}
        if self._algorithm_hp is not None and self.fixed_algorithm is not None:
            values["algorithm"] = self.fixed_algorithm
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                values[hp.name] = self._fixed_value(hp)
        for name in self.hebo_space.para_names:
            if name in row.index:
                values[name] = self._coerce_hebo_value(name, row[name])
        try:
            cfg = Configuration(self.config_space, values=values)
            return MyConfiguration(cfg)
        except Exception as exc:
            self.logger.warning("HEBO generated an invalid ConfigSpace configuration: %s", exc)
            return self._sample_fallback_configuration()

    def _coerce_hebo_value(self, name: str, value: Any) -> Any:
        hp = self._hebo_hp_by_name.get(name)
        if hp is None:
            return value
        if isinstance(hp, UniformIntegerHyperparameter):
            return int(round(float(value)))
        if isinstance(hp, UniformFloatHyperparameter):
            return float(value)
        if isinstance(hp, CategoricalHyperparameter):
            raw = value.item() if hasattr(value, "item") else value
            if raw in hp.choices:
                return raw
            as_str = str(raw)
            for choice in hp.choices:
                if str(choice) == as_str:
                    return choice
            if isinstance(raw, (bool, np.bool_)):
                return bool(raw)
            return raw
        if OrdinalHyperparameter and isinstance(hp, OrdinalHyperparameter):
            raw = value.item() if hasattr(value, "item") else value
            return raw
        return value

    def _config_to_hebo_frame(self, cfg) -> Optional[pd.DataFrame]:
        cfg_dict = _cfg_dict(cfg)
        algorithm = cfg_dict.get("algorithm", self.fixed_algorithm)
        row = {}
        for name in self.hebo_space.para_names:
            hp = self._hebo_hp_by_name.get(name)
            if hp is not None and not self._is_relevant_to_algorithm(name, algorithm):
                return None
            if name not in cfg_dict:
                return None
            row[name] = self._coerce_hebo_value(name, cfg_dict[name])
        return pd.DataFrame([row], columns=self.hebo_space.para_names)

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

            llm_cost = llm_time = hebo_time = 0.0
            raw, payload, meta, rationale, prompt_input = "", {}, {}, "", {}
            hebo_rec = None

            if source in ("llm", "llm_warm_start"):
                cfg, note, llm_cost, llm_time, raw, payload, meta, rationale, prompt_input = self._suggest_config(
                    iter_idx,
                    include_history=include_history,
                )
            else:
                cfg, hebo_rec, hebo_time, note = self._suggest_hebo_config()

            obs = self.objective_function(cfg)
            raw_obj = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_obj)
            perf = -raw_obj if success else float("-inf")
            eval_time = obs.elapsed_time or 0.0
            prev_best = float(max(self.perfs)) if self.perfs else float("-inf")

            tell_objective = raw_obj if success else None
            if hebo_rec is not None:
                self._tell_hebo_frame(hebo_rec, tell_objective)
            else:
                self._tell_hebo_external_config(cfg, tell_objective)

            if success:
                self.exp_output[time.time()] = (cfg, perf)
                self.update_saver([cfg], [raw_obj])
                if source in ("llm", "llm_warm_start"):
                    self._update_history(obs, source, iter_idx, cfg, rationale)
                else:
                    self.KD.update_history(obs, source=self.bo_mode, trail_id=str(iter_idx))
                self._mark_seen(cfg, raw_obj)

            beat_inc = bool(success and perf > prev_best)
            if source in ("llm", self._bo_stage):
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
                "hebo_time": hebo_time,
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
                hebo_time=hebo_time,
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
        m = max(0, int(iter_idx) - int(self.n_no_history_samples))
        M = max(1, int(getattr(self, "evaluation_num_limit", 0) or 0) - int(self.n_no_history_samples))
        arm = self.selector.choose(m, M, higher_is_better=True)
        snap = self.selector.snapshot()
        draw = snap.get("selection_draw")
        return arm, include_history, draw

    def _suggest_hebo_config(self):
        start = time.time()
        duplicate_count = 0
        import contextlib
        import io
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

        cfg = self._sample_nonduplicate_config()
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
        objective = self._known_objective(cfg)
        self._tell_hebo_frame(rec, objective)

    def _tell_hebo_external_config(self, cfg, objective: Optional[float]):
        frame = self._config_to_hebo_frame(cfg)
        if frame is None:
            return
        self._tell_hebo_frame(frame, objective)

    def _record_hybrid_event(
        self,
        stage,
        iter_idx,
        cfg,
        perf,
        llm_cost,
        llm_time,
        hebo_time,
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
            "hybrid_selector": str(getattr(self, "hybrid_selector_mode", "bernoulli")),
            "selection_draw": "" if selection_draw is None else float(selection_draw),
            "llm_cost": float(llm_cost),
            "cumulative_llm_cost": float(sum(self.llm_query_cost)),
            "llm_time": float(llm_time),
            "hebo_time": float(hebo_time),
            "tpe_time": float(hebo_time),
            "eval_time": float(eval_time),
            "iter_time": float(iter_time),
            "wall_elapsed_sec": float(time.time() - self.start_wall_time),
            "note": note,
            "config_signature": _json(_signature(cfg)),
            "config_for_prompt": _json(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space)),
        }
        snap = getattr(self, "selector", None)
        if snap is not None:
            info = snap.snapshot()
            row.update(
                {
                    "Q_llm": info.get("Q_llm", ""),
                    "Q_tpe": info.get("Q_bo", info.get("Q_tpe", "")),
                    "Q_hebo": info.get("Q_bo", ""),
                    "N_llm": info.get("N_llm", ""),
                    "N_tpe": info.get("N_bo", info.get("N_tpe", "")),
                    "N_hebo": info.get("N_bo", ""),
                    "omega_llm": info.get("omega_llm", ""),
                    "omega_tpe": info.get("omega_bo", info.get("omega_tpe", "")),
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
            "hebo_trials": int(stage_counts.get(self._bo_stage, 0)),
            "hebo_observations": int(len(getattr(self.hebo, "X", []))),
            "csv_path": self.traj_csv_path,
            "detail_perfs": ",".join(str(p) for p in self.perfs),
            "history": self.traj_rows[:500],
        }
