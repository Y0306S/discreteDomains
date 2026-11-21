"""LLM + CMA-ES hybrid (Optuna CmaEsSampler), mirroring PT4OptLLMTPEOptimizer wiring."""

from __future__ import annotations

import optuna

from mindware.components.optimizers.pt4opt_llm_tpe_optimizer import PT4OptLLMTPEOptimizer


class PT4OptLLMCMAESOptimizer(PT4OptLLMTPEOptimizer):
    optimizer_label = "ptllm_cmaes"
    _bo_stage = "cmaes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_params = dict(self.params.get("cmaes_params") or {})
        if "n_startup_trials" in self.params:
            self.optimizer_params["n_startup_trials"] = self.params["n_startup_trials"]
        self.optimizer_params.setdefault("seed", self.seed)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=self._build_sampler(),
            study_name=f"{self.optimizer_label}-{self.task_id}-{int(self.timestamp)}",
        )

    def _build_sampler(self):
        return optuna.samplers.CmaEsSampler(**dict(self.optimizer_params))

    def _choose_source(self, iter_idx: int):
        include_history = iter_idx >= self.n_no_history_samples
        if not include_history:
            return "llm_warm_start", False, None
        if self.llm_phase_trials > 0 and iter_idx < self.llm_phase_trials:
            return "llm", include_history, None
        draw = float(self.hybrid_rng.rand())
        return ("llm", include_history, draw) if draw < self.llm_prob else (self._bo_stage, include_history, draw)

    def _suggest_tpe_config(self):
        cfg, trial, elapsed, note = super()._suggest_tpe_config()
        note = note.replace("tpe", self._bo_stage)
        return cfg, trial, elapsed, note

    def get_opt_trajectory(self):
        traj = super().get_opt_trajectory()
        stage_counts = {}
        for row in self.traj_rows:
            stage = row.get("stage", "")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        traj["cmaes_trials"] = int(stage_counts.get(self._bo_stage, 0))
        traj["cmaes_study_trials"] = len(self.study.trials)
        return traj
