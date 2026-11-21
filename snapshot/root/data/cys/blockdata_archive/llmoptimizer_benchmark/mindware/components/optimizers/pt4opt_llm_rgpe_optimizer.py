"""ckpt32 retry proposals combined with the verified RGPE(TAF) implementation."""

import json
import time

from mindware.components.optimizers.pt4opt_llm_tpe_optimizer import PT4OptLLMTPEOptimizer
from mindware.components.optimizers.pt4opt_transfer_optimizer import RGPEOptimizer
from mindware.components.optimizers.utils import MyConfiguration


class PT4OptLLMRGPEOptimizer(PT4OptLLMTPEOptimizer):
    optimizer_label = "ptllm_rgpe"
    bo_arm = "rgpe"

    def _initialize_bo(self):
        if self.selector.bo_arm != "rgpe":
            raise ValueError("LLM/RGPE requires an RGPE selector")
        if self.llm_duplicate_policy != "history_tpe5" or self.llm_phase_trials != 8:
            raise ValueError("This campaign requires retry5 and eight initial LLM selections")
        if self.prompt_blocks.get("tpe_hint") or self.experience_explore_mode not in ("off", "fifty_fifty"):
            raise ValueError("TPE prompt hints are unsupported in LLM/RGPE")
        self.study = None
        self._selection_details = {}
        self._rgpe_diagnostics = None
        bridge = self._new_rgpe_bridge()
        self.asset_identity = bridge.asset_identity
        self.schema_fingerprint = bridge.schema_fingerprint

    def _new_rgpe_bridge(self):
        # The short-lived adapter reuses the standalone worker and cache checks.
        # Only serializable identity/diagnostic data enter the hybrid checkpoint.
        params = dict(self.params.get("rgpe_params") or {})
        params.update(task_type="cls" if self.params["task_type"] == "classification" else "rgs",
                      n_trials=self.evaluation_num_limit)
        bridge = RGPEOptimizer(
            evaluator=self.evaluator, config_space=self.config_space, name=self.name,
            eval_type=self.eval_type, task_id=self.task_id, seed=self.seed,
            timestamp=self.timestamp, output_dir=self.output_dir,
            evaluation_limit=self.evaluation_num_limit, optimizer_params=params,
            logging_enabled=False,
        )
        if hasattr(self, "asset_identity") and self.asset_identity != bridge.asset_identity:
            raise RuntimeError("RGPE assets changed during the hybrid run")
        return bridge

    def _upgrade_bo(self):
        pass

    def _peek_tpe_hint(self):
        self._tpe_hint_config = None

    def _peek_tpe_config(self, llm_only=False):
        raise RuntimeError("An RGPE campaign cannot request a TPE hint")

    def _tell_bo(self, trial, objective, cfg, state, feedback):
        # RGPE consumes committed successful observations at its next selection.
        # Failed configurations remain in the attempted set via traj_rows.
        if trial is not None:
            raise RuntimeError("Unexpected Optuna trial in an RGPE run")

    def _suggest_bo_config(self):
        start = time.time()
        bridge = self._new_rgpe_bridge()
        observations = self._retry_observations()
        observed = [(self.KD._dict_to_cfg(row["config"]), row["score"])
                    for row in observations if row["successful"]]
        if not observed:
            raise RuntimeError("RGPE has no successful target observations")
        bridge.attempted_configs = {bridge._key(row["config"]) for row in observations}
        bridge.trial_cnt = self.trial_cnt
        if getattr(self, "_ml_session", None) is not None:
            bridge._ml_session = self._ml_session
        config, note = bridge._suggest_model(observed)
        config = MyConfiguration(config)
        self._rgpe_diagnostics = bridge._proposal_diagnostics
        if self._is_duplicate(config):
            raise RuntimeError("RGPE returned a previously evaluated configuration")
        return config, None, time.time() - start, note

    def _choose_source(self, iter_idx):
        trial = iter_idx + 1
        self._rgpe_diagnostics = None
        self.selector.last_snapshot = {}
        details = {"round": trial, "cooldown_active": False,
                   "cooldown_window": 9 <= trial <= 20, "pucb_proposed_arm": None,
                   "schedule_index": None, "schedule_denominator": None}
        self._selection_details = details
        if trial <= self.n_no_history_samples:
            details.update(selection_reason="preset_warm_start", selected_arm="llm_warm_start")
            return "llm_warm_start", False, None
        if trial <= 8:
            details.update(selection_reason="initial_llm_phase", selected_arm="llm")
            return "llm", True, None
        m, M = trial - 9, max(1, self.evaluation_num_limit - 9)
        arm = self.selector.choose(m, M, higher_is_better=True)
        previous = self.traj_rows[-1] if self.traj_rows else {}
        cooldown = (10 <= trial <= 20 and int(previous.get("iter_idx", -1)) == iter_idx - 1
                    and previous.get("selected_arm", previous.get("stage")) == "llm")
        details.update(pucb_proposed_arm=arm, cooldown_active=cooldown,
                       selection_reason="llm_cooldown" if cooldown else "pucb",
                       schedule_index=m, schedule_denominator=M)
        if cooldown:
            arm = "rgpe"
        details["selected_arm"] = arm
        return arm, True, self.selector.snapshot().get("selection_draw")

    def _hybrid_row_details(self, row):
        for field in ("Q", "N", "N_attempt", "omega"):
            row[field + "_rgpe"] = row.pop(field + "_tpe", "")
        row["rgpe_time"] = row.pop("tpe_time")
        row["llm_duplicate_policy"] = "history_rgpe5"
        row["rgpe_diagnostics_dir"] = self._rgpe_diagnostics
        row["rgpe_assets_json"] = json.dumps(self.asset_identity, sort_keys=True)
        row.update(self._selection_details)
        # These are post-evaluation counts; Q/omega above describe the decision.
        row["N_after_llm"] = self.selector.n["llm"]
        row["N_after_rgpe"] = self.selector.n["rgpe"]
        row["N_attempt_after_llm"] = self.selector.attempt_counts()["llm"]
        row["N_attempt_after_rgpe"] = self.selector.attempt_counts()["rgpe"]
        for key in ("experience_explore_branch", "experience_explore_p",
                    "experience_explore_stagnation", "experience_explore_coverage",
                    "experience_explore_tpe_ratio", "experience_explore_tpe_n_constraints",
                    "experience_explore_tpe_peeked", "experience_explore_tpe_scope",
                    "experience_self_gate", "experience_self_active", "experience_gate_latched"):
            row.setdefault(key, None)
        return dict(sorted(row.items()))

    def restore_from_traj(self, *args, **kwargs):
        raise RuntimeError("LLM/RGPE requires its durable checkpoint, not CSV-only resume")

    def get_opt_trajectory(self):
        from collections import Counter
        import math

        counts = dict(Counter(row["stage"] for row in self.traj_rows))
        return {"n_evals": len(self.traj_rows),
                "n_successful": sum(math.isfinite(float(r["perf"])) for r in self.traj_rows),
                "final_best": float(max(self.perfs)) if self.perfs else float("-inf"),
                "llm_cost_sum": float(sum(self.llm_query_cost)), "llm_calls": int(self.llm_call_count),
                "stage_counts": counts, "rgpe_trials": counts.get("rgpe", 0),
                "llm_trials": counts.get("llm", 0), "llm_warm_start_trials": counts.get("llm_warm_start", 0),
                "assets": self.asset_identity, "csv_path": self.traj_csv_path,
                "history": self.traj_rows, "detail_perfs": ",".join(str(p) for p in self.perfs)}
