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
    parse_llm_json,
)
from mindware.components.optimizers.pt4opt_optuna_optimizer import OptunaConfigSpaceMixin, PT4OptOptunaOptimizer
from mindware.components.optimizers.ml_evaluation_feedback import (
    observation_feedback, tpe_sampler_params, upgrade_runtime_study,
)


class PT4OptLLMTPEOptimizer(PT4OptLLMOptimizer, OptunaConfigSpaceMixin):
    optimizer_label = "ptllm_tpe"
    bo_arm = "tpe"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm_duplicate_policy = str(self.params.get("llm_duplicate_policy", "legacy"))
        if self.llm_duplicate_policy not in ("legacy", "history_tpe5"):
            raise ValueError(f"Unknown LLM duplicate policy: {self.llm_duplicate_policy}")
        if self.llm_duplicate_policy == "history_tpe5":
            self.duplicate_prompt_rule = "Strictly do not return a configuration you have previously recommended. "
            if not self.logging_enabled:
                raise ValueError("history_tpe5 requires evaluation logging for history and budget accounting")
            if self.use_tcs_analysis:
                raise ValueError("history_tpe5 requires the selected-history prompt")
            self.history_window = max(self.history_window, self.recent_k + self.top_k + self.worst_k + 4)
            self.params["history_window"] = self.history_window

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

        self._initialize_bo()
        self.trial_sources: Dict[int, str] = {}
        if getattr(self, "_explore_policy", None) is not None:
            self._explore_policy.tpe_peek = self._peek_tpe_config

    def _initialize_bo(self):
        self.study = optuna.create_study(
            direction="minimize",
            sampler=self._build_sampler(),
            study_name=f"{self.optimizer_label}-{self.task_id}-{int(self.timestamp)}",
        )

    def _upgrade_bo(self):
        self.study = upgrade_runtime_study(self.study)

    def _suggest_bo_config(self):
        return self._suggest_tpe_config()

    def _tell_bo(self, trial, objective, cfg, state, feedback):
        if trial is not None:
            self._tell_tpe_trial(trial, objective, cfg, state=state, attrs=feedback)
        else:
            self._tell_tpe_external_config(cfg, objective, state=state, attrs=feedback)

    def _hybrid_row_details(self, row):
        return row

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
        if getattr(self, "llm_duplicate_policy", "legacy") == "history_tpe5":
            if include_history or not self._preset_warm_start_queue:
                return self._suggest_with_history_retries(iter_idx, include_history)
        return super()._suggest_config(iter_idx, include_history=include_history)

    def _retry_observations(self):
        observations = []
        for row in self.traj_rows:
            value = row.get("config_json") or row.get("config_for_prompt")
            if not value:
                raise RuntimeError("Retry history requires saved configurations for every evaluation")
            config = json.loads(value) if isinstance(value, str) else dict(value)
            config.setdefault("algorithm", self.fixed_algorithm)
            score = float(row["perf"])
            observations.append({"trial": int(row["iter_idx"]) + 1, "config": config,
                                 "score": score if np.isfinite(score) else None,
                                 "evaluation_status": row.get("evaluation_status") or
                                     ("SUCCESS" if np.isfinite(score) else "FAILED"),
                                 "successful": bool(np.isfinite(score))})
        if not observations:
            observations = [{"trial": i + 1, "config": _cfg_dict(cfg),
                             "score": float(perf) if np.isfinite(perf) else None,
                             "successful": bool(np.isfinite(perf))}
                            for i, (cfg, perf) in enumerate(zip(self.configs, self.perfs))]
        return observations

    def _is_duplicate(self, cfg):
        if getattr(self, "llm_duplicate_policy", "legacy") == "history_tpe5":
            signature = _signature(cfg)
            return any(_signature(row["config"]) == signature for row in self._retry_observations())
        return super()._is_duplicate(cfg)

    def _retry_reminder(self, cfg):
        observations = self._retry_observations()
        scores = [row["score"] for row in observations if row["successful"]]
        lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
        matches = []
        for row in observations:
            if _signature(row["config"]) != _signature(cfg):
                continue
            score = row["score"]
            matches.append({"trial": row["trial"], "successful": row["successful"],
                            "evaluation_status": row.get("evaluation_status", "FAILED"),
                            "score": score, "objective": -score if score is not None else None,
                            "rel_score": (1.0 if hi == lo else (score - lo) / (hi - lo))
                            if score is not None else None,
                            "rank": 1 + sum(s > score for s in scores) if score is not None else None})
        if not matches:
            raise RuntimeError("Duplicate proposal has no recorded evaluation to replay")
        return {"config": _config_for_prompt(cfg, self.fixed_algorithm, self.config_space),
                "previous_evaluations": matches}

    def _history_retry_prompt(self, user_prompt, reminders):
        if not reminders:
            return user_prompt
        lines = [user_prompt, "", "Selected historical trials: temporary recent reminders.",
                 "These are previously evaluated configurations, not new evaluations."]
        for index, reminder in enumerate(reminders, 1):
            lines.append(f"Retry reminder {index}: config={_json(reminder['config'])}")
            for result in reminder["previous_evaluations"]:
                if result["successful"]:
                    lines.append(f"  trial={result['trial']}, rel_score={result['rel_score']:.4f}, "
                                 f"rank={result['rank']}, tags=[\"recent\", \"retry\"]")
                else:
                    lines.append(f"  trial={result['trial']}, evaluation_status={result['evaluation_status']}, score=unavailable")
        lines.append("Return one different configuration. Keep the same output format and use every "
                     "tunable hyperparameter listed in Search space.")
        return "\n".join(lines)

    def _suggest_with_history_retries(self, iter_idx, include_history):
        start = time.time()
        system_prompt, user_prompt = self._build_prompts(include_history=include_history,
                                                        current_trial=iter_idx + 1)
        meta = dict(self._last_history_meta)
        attempts, reminders, rejected = [], [], []
        prompt_input = {"system_prompt": system_prompt, "user_prompt": user_prompt,
                        "include_history": include_history, "attempts": attempts,
                        "messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt}]}
        cost, raw, payload, rationale, error = 0.0, "", {}, "", None
        status = "PENDING"
        limit = 5 if include_history else 1
        for index in range(limit):
            prompt = self._history_retry_prompt(user_prompt, reminders)
            attempt = {"attempt": index, "messages": [{"role": "system", "content": system_prompt},
                                                       {"role": "user", "content": prompt}],
                       "retry_history": list(reminders),
                       "recent_context_limit": self.recent_k + len(reminders) if include_history else 0}
            attempt_start = time.time()
            raw, payload = "", {}
            call_cost = 0.0
            try:
                raw, call_cost = self._complete_llm(system_prompt, prompt)
                cost += call_cost
                config_dict, rationale, payload = parse_llm_json(raw)
                cfg = self.KD._dict_to_cfg(dict(config_dict, algorithm=self.fixed_algorithm))
            except Exception as exc:
                error = str(exc)
                attempt.update(raw_response=raw, parsed_payload=payload, error=error,
                               elapsed_time=time.time() - attempt_start, llm_cost=float(call_cost))
                attempts.append(attempt)
                self._save_retry_attempts(iter_idx, attempts)
                if (not raw and not self.fallback_on_transport_error) or (raw and not self.fallback_on_bad_json):
                    raise
                if raw and payload and not self.fallback_on_invalid_config:
                    raise
                cfg = self._sample_nonduplicate_config()
                status, note = "FAILED", f"random fallback after llm error: {exc}"
                break
            duplicate = self._is_duplicate(cfg)
            attempt.update(raw_response=raw, parsed_payload=payload, normalized_config=_cfg_dict(cfg),
                           duplicate=duplicate, elapsed_time=time.time() - attempt_start,
                           llm_cost=float(call_cost))
            attempts.append(attempt)
            if duplicate and include_history:
                reminder = self._retry_reminder(cfg)
                attempt["repeated_observations"] = reminder
                rejected.append(reminder["config"])
                if index + 1 < limit:
                    reminders.append(reminder)
                else:
                    status, note = "DUPLICATE", f"{self.bo_arm.upper()} required after five duplicate LLM proposals"
                    meta["llm_retry_exhausted"] = True
            else:
                note = "llm suggested configuration" if include_history else "llm warm-start configuration"
                if index:
                    note += f" after {index} duplicate retries"
            self._save_retry_attempts(iter_idx, attempts)
            if not duplicate or not include_history:
                break
        meta.update(llm_duplicate_policy="history_tpe5", llm_attempt_count=len(attempts),
                    retry_history=list(reminders))
        self._last_history_meta = meta
        self._append_llm_history(iter_idx, status, raw, payload, cfg, note, error=error,
                                 attempts=attempts, rejected_duplicate_configs=rejected)
        return cfg, note, cost, time.time() - start, raw, payload, meta, rationale, prompt_input

    def _save_retry_attempts(self, iter_idx, attempts):
        from mindware.components.optimizers.ml_checkpoint import atomic_json

        atomic_json(os.path.join(self.conversation_log_dir, f"trial_{iter_idx:04d}_proposals.json"),
                    json.loads(_json({"trial": iter_idx + 1, "llm_attempts": attempts})))

    def _sample_nonduplicate_config(self, max_attempts=50):
        if getattr(self, "llm_duplicate_policy", "legacy") != "history_tpe5":
            return super()._sample_nonduplicate_config(max_attempts)
        for _ in range(max_attempts + 1):
            cfg = self._sample_config()
            if not self._is_duplicate(cfg):
                return cfg
        raise RuntimeError(f"No unseen configuration found in {max_attempts + 1} random proposals")

    def _quota_or_time_is_up(self):
        if getattr(self, "llm_duplicate_policy", "legacy") == "history_tpe5":
            exhausted = self.evaluation_num_limit is not None and len(self.traj_rows) >= self.evaluation_num_limit
            timed_out = self.time_limit is not None and time.time() - self.timestamp > self.time_limit
            return exhausted or timed_out
        return super()._quota_or_time_is_up()

    def _build_sampler(self):
        return optuna.samplers.TPESampler(**tpe_sampler_params(self.optimizer_params))

    def iterate(self, budget=MAX_INT):
        if not hasattr(self.selector, "attempt_n"):
            # Older checkpoints counted only evaluated suppliers. Recover the
            # additional attempted LLM arms from their committed trajectory.
            self.selector.attempt_n = dict(self.selector.n)
            for row in self.traj_rows:
                if row.get("selected_arm") == "llm" and row.get("stage") == self.bo_arm:
                    self.selector.record_attempt("llm")
        self._upgrade_bo()
        start = time.time()
        rows = []
        for _ in range(self.inner_iter_num_per_iter):
            from mindware.components.optimizers import ml_checkpoint
            saved = ml_checkpoint.pending(self)
            if (self._quota_or_time_is_up() and saved is None) or time.time() - start > budget:
                break

            iter_start = time.time()
            if saved is None:
                iter_idx = self.trial_cnt
                self.trial_cnt += 1
                source, include_history, selection_draw = self._choose_source(iter_idx)
                llm_cost = llm_time = tpe_time = 0.0
                raw, payload, meta, rationale, prompt_input = "", {}, {}, "", {}
                tpe_trial = None
                if source in ("llm", "llm_warm_start"):
                    cfg, note, llm_cost, llm_time, raw, payload, meta, rationale, prompt_input = self._suggest_config(
                        iter_idx, include_history=include_history,
                    )
                    if meta.get("llm_retry_exhausted"):
                        cfg, tpe_trial, tpe_time, tpe_note = self._suggest_bo_config()
                        source = self.bo_arm
                        note = f"{self.bo_arm.upper()} fallback after five duplicate LLM proposals; {tpe_note}"
                        meta["selected_arm"] = "llm"
                else:
                    cfg, tpe_trial, tpe_time, note = self._suggest_bo_config()
                meta.setdefault("selected_arm", source)
                meta["configuration_source"] = "random" if "random fallback" in note else source
                if self.experience_retrieve_ranker == "source_gp":
                    from mindware.components.experience.source_gp import rs_position, successful_observations
                    audit = rs_position(iter_idx + 1, self.experience_retrieve_start_trial,
                                        self.experience_retrieve_stop_trial, source)
                    audit.update(meta.get("experience_retrieval") or {})
                    audit["source"] = source
                    audit.setdefault("observations", successful_observations(self.traj_rows))
                    meta["experience_retrieval"] = audit
                saved = {"iter_idx": iter_idx, "source": source, "selection_draw": selection_draw,
                         "cfg": cfg, "note": note, "llm_cost": llm_cost, "llm_time": llm_time,
                         "tpe_time": tpe_time, "raw": raw, "payload": payload, "meta": meta,
                         "rationale": rationale, "prompt_input": prompt_input,
                         "trial_id": tpe_trial._trial_id if tpe_trial else None}
            else:
                iter_idx, source, selection_draw = saved["iter_idx"], saved["source"], saved["selection_draw"]
                cfg, note = saved["cfg"], saved["note"]
                llm_cost, llm_time, tpe_time = saved["llm_cost"], saved["llm_time"], saved["tpe_time"]
                raw, payload, meta = saved["raw"], saved["payload"], saved["meta"]
                rationale, prompt_input = saved["rationale"], saved["prompt_input"]
                tpe_trial = optuna.trial.Trial(self.study, saved["trial_id"]) if saved["trial_id"] is not None else None
            self._last_history_meta = meta
            ml_checkpoint.prepare(self, saved)
            obs = self.objective_function(cfg)
            raw_obj = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_obj)
            perf = -raw_obj if success else float("-inf")
            eval_time = obs.elapsed_time or 0.0
            prev_best = float(max(self.perfs)) if self.perfs else float("-inf")

            tell_objective, tell_state, feedback = observation_feedback(obs, self.per_run_time_limit)
            meta.update(feedback)
            self._tell_bo(tpe_trial, tell_objective, cfg, tell_state, feedback)

            if success:
                self.exp_output[time.time()] = (cfg, perf)
                self.update_saver([cfg], [raw_obj])
                if source in ("llm", "llm_warm_start"):
                    self._update_history(obs, source, iter_idx, cfg, rationale)
                else:
                    PT4OptOptunaOptimizer.update_history(self, obs, trail_id=str(iter_idx))
                self._mark_seen(cfg, raw_obj)

            beat_inc = bool(success and perf > prev_best)
            if source in ("llm", self.bo_arm):
                arm_best = self.selector.arm_best.get(source)
                beat_arm = bool(success and (arm_best is None or perf > float(arm_best)))
                self.selector.update(
                    source,
                    perf if success else float("-inf"),
                    higher_is_better=True,
                    beat_incumbent=beat_inc,
                    beat_arm_best=beat_arm,
                )
                if meta.get("selected_arm", source) != source:
                    self.selector.record_attempt(meta["selected_arm"])
            ee_pol = getattr(self, "_explore_policy", None)
            if ee_pol is not None and source == "llm":
                br = getattr(ee_pol, "cached_branch", None)
                if br:
                    ee_pol.update(br, perf if success else float("-inf"), beat_incumbent=beat_inc)

            if source in ("llm", "llm_warm_start") or meta.get("llm_retry_exhausted"):
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
                meta=meta,
            )
            self.trial_sources[int(iter_idx) + 1] = source
            if getattr(self, "llm_duplicate_policy", "legacy") == "history_tpe5":
                self._update_incumbent()
                ml_checkpoint.commit(self)

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
        meta=None,
    ):
        if not self.logging_enabled:
            return
        if llm_cost:
            self.llm_query_cost.append(float(llm_cost))
        if llm_time:
            self.llm_query_time.append(float(llm_time))
        from mindware.components.optimizers.resume_traj import cfg_as_json

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
            "config_json": cfg_as_json(cfg),
            "config_for_prompt": _json(_config_for_prompt(cfg, self.fixed_algorithm, self.config_space)),
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
                    "N_attempt_llm": info.get("N_attempt_llm", ""),
                    "N_attempt_tpe": info.get("N_attempt_tpe", ""),
                    "omega_llm": info.get("omega_llm", ""),
                    "omega_tpe": info.get("omega_tpe", ""),
                    "pucb_q": info.get("pucb_q", ""),
                }
            )
        meta = meta or {}
        for key in ("evaluation_status", "evaluation_seconds", "evaluation_limit_seconds", "runtime_policy_id"):
            row[key] = meta.get(key)
        if getattr(self, "llm_duplicate_policy", "legacy") == "history_tpe5":
            row.update(selected_arm=meta.get("selected_arm", stage),
                       configuration_source=meta.get("configuration_source", stage),
                       llm_attempt_count=meta.get("llm_attempt_count", 0),
                       llm_retry_exhausted=bool(meta.get("llm_retry_exhausted")),
                       llm_retry_history=_json(meta.get("retry_history", [])),
                       llm_duplicate_policy="history_tpe5")
        audit = meta.get("experience_retrieval") or {}
        row["experience_retrieval"] = _json(audit)
        if audit:
            from pathlib import Path
            from mindware.components.optimizers.ml_checkpoint import atomic_json
            atomic_json(Path(self.output_dir) / "retrieval" / f"trial_{iter_idx + 1:04d}.json",
                        {**audit, "configuration": json.loads(row["config_json"]),
                         "performance": float(perf) if np.isfinite(perf) else None,
                         "successful": bool(np.isfinite(perf)), "eval_time": float(eval_time)})
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
        row = self._hybrid_row_details(row)
        self.traj_rows.append(row)
        try:
            exists = os.path.exists(self.traj_csv_path)
            with open(self.traj_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
        except IOError as exc:
            self.logger.warning("Failed to save trajectory to %s: %s", self.traj_csv_path, exc)
        try:
            from mindware.components.optimizers.resume_traj import (
                observations_jsonl_path,
                persist_jsonl_row,
                persist_study_pickle,
                study_pickle_path,
            )

            persist_jsonl_row(
                observations_jsonl_path(self.traj_csv_path),
                {
                    "iter_idx": iter_idx,
                    "stage": stage,
                    "perf": float(perf),
                    "config": json.loads(row["config_json"]),
                },
            )
            if self.study is not None:
                persist_study_pickle(study_pickle_path(self.traj_csv_path), self.study)
        except Exception as exc:
            self.logger.warning("Failed to persist LLM-TPE resume sidecar: %s", exc)

    def restore_from_traj(
        self,
        traj_path: str,
        history_json: Optional[str] = None,
        observations_jsonl: Optional[str] = None,
    ) -> int:
        """Replay hybrid LLM/TPE trials and continue from the last saved iter_idx."""
        from openbox.utils.constants import FAILED, SUCCESS
        from openbox.utils.history import Observation

        from mindware.components.optimizers.resume_traj import (
            copy_traj_bundle,
            load_history_json_configs,
            load_jsonl_configs,
            load_traj_rows,
            observations_jsonl_path,
            persist_study_pickle,
            resolve_row_configs,
            rewrite_observations_jsonl,
            rewrite_traj_csv,
            row_success_perf,
            study_pickle_path,
            try_load_study_pickle,
        )

        rows = load_traj_rows(traj_path)
        if not rows:
            return 0

        algo = self.fixed_algorithm
        jsonl_cfgs = load_jsonl_configs(
            observations_jsonl or observations_jsonl_path(traj_path),
            default_algorithm=algo,
        )
        hist_cfgs = load_history_json_configs(history_json, default_algorithm=algo)
        recovered, n_topk = resolve_row_configs(
            rows,
            traj_path=traj_path,
            default_algorithm=algo,
            jsonl_cfgs=jsonl_cfgs,
            hist_cfgs=hist_cfgs,
        )

        copy_traj_bundle(traj_path, self.traj_csv_path)
        self.traj_rows = [dict(r) for r in rows]
        try:
            rewrite_traj_csv(self.traj_csv_path, self.traj_rows)
        except Exception as exc:
            self.logger.warning("Resume LLM-TPE: failed to persist recovered configs: %s", exc)

        pickled = try_load_study_pickle(study_pickle_path(traj_path))
        replay_study = pickled is None
        if pickled is not None:
            self.study = pickled
            self.logger.info(
                "Restored LLM-TPE study pickle with %d trials", len(self.study.trials)
            )

        prev_best = float("-inf")
        restored_cfg = 0
        for i, row in enumerate(self.traj_rows):
            success, perf = row_success_perf(row)
            stage = str(row.get("stage") or "llm")
            cfg_dict = recovered[i] if i < len(recovered) else None
            if cfg_dict is None:
                if success:
                    prev_best = max(prev_best, perf)
                continue
            try:
                cfg = self.KD._dict_to_cfg(cfg_dict)
            except Exception as exc:
                self.logger.warning("Resume LLM-TPE: skip trial %s, invalid config: %s", i, exc)
                continue

            raw_obj = -float(perf) if success else None
            if replay_study:
                self._tell_tpe_external_config(cfg, raw_obj)

            if success:
                obs = Observation(
                    config=cfg,
                    objectives=[float(raw_obj)],
                    trial_state=SUCCESS,
                    elapsed_time=float(row.get("eval_time") or 0.0),
                )
                self.exp_output[time.time()] = (cfg, perf)
                self.update_saver([cfg], [raw_obj])
                try:
                    if stage in ("llm", "llm_warm_start"):
                        self._update_history(obs, stage, i, cfg, row.get("note") or "")
                    else:
                        PT4OptOptunaOptimizer.update_history(self, obs, trail_id=str(i))
                except Exception as exc:
                    self.logger.warning("Resume LLM-TPE history update failed at %s: %s", i, exc)
                    if cfg not in self.configs:
                        self.configs.append(cfg)
                        self.perfs.append(perf)
                self._mark_seen(cfg, raw_obj)
                restored_cfg += 1
            else:
                restored_cfg += 1

            beat_inc = bool(success and perf > prev_best)
            if stage in ("llm", "tpe"):
                arm_best = self.selector.arm_best.get(stage)
                beat_arm = bool(success and (arm_best is None or perf > float(arm_best)))
                self.selector.update(
                    stage,
                    perf if success else float("-inf"),
                    higher_is_better=True,
                    beat_incumbent=beat_inc,
                    beat_arm_best=beat_arm,
                )
                if row.get("selected_arm") == "llm" and stage == "tpe":
                    self.selector.record_attempt("llm")
            if success:
                prev_best = max(prev_best, perf)

            try:
                cost = float(row.get("llm_cost") or 0.0)
            except Exception:
                cost = 0.0
            if cost:
                self.llm_query_cost.append(cost)
            self.trial_sources[i + 1] = stage

        n_consume = min(len(rows), int(getattr(self, "n_no_history_samples", 0) or 0))
        queue = getattr(self, "_preset_warm_start_queue", None)
        if isinstance(queue, list):
            del queue[:n_consume]

        self.trial_cnt = len(rows)
        self.llm_call_count = sum(
            int(r["llm_attempt_count"]) if r.get("llm_attempt_count") not in (None, "")
            else int(str(r.get("stage") or "") in ("llm", "llm_warm_start")) for r in rows
        )
        try:
            persist_study_pickle(study_pickle_path(self.traj_csv_path), self.study)
        except Exception:
            pass
        try:
            rewrite_observations_jsonl(
                observations_jsonl_path(self.traj_csv_path),
                self.traj_rows,
                recovered,
                default_algorithm=algo,
            )
        except Exception as exc:
            self.logger.warning("Resume LLM-TPE: failed to rewrite observations jsonl: %s", exc)
        self._update_incumbent()
        self.logger.info(
            "Resume LLM-TPE: restored %d/%d trials (study replay=%s, trial_cnt=%d, topk_fill=%d) from %s",
            restored_cfg,
            len(rows),
            replay_study,
            self.trial_cnt,
            n_topk,
            traj_path,
        )
        return len(rows)

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
