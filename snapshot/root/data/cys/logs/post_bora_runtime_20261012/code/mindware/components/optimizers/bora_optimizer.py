"""BORA for mixed ML ConfigSpaces, using the released action policy and GP."""

import json
import math
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from mindware.components.optimizers.online_llm_optimizer import OnlineLLMOptimizer, QWEN_BASE
from mindware.components.optimizers.online_upstream import bora_policy, source_digest, UPSTREAM
from mindware.components.optimizers.strict_recommendation import RecommendationError


class BORAOptimizer(OnlineLLMOptimizer):
    optimizer_label = "bora"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args_client_invalid():
            raise ValueError("BORA requires the untouched Qwen checkpoint without an adapter")
        memory_limit = self.params.get("gpu_memory_limit_gib")
        if memory_limit is not None:
            import torch
            memory_limit = float(memory_limit)
            if not math.isfinite(memory_limit) or memory_limit <= 0 or torch.cuda.device_count() != 1:
                raise ValueError("A BORA GPU memory limit requires a positive size and exactly one visible GPU")
            total = torch.cuda.get_device_properties(0).total_memory
            if memory_limit * 2**30 > total:
                raise ValueError("BORA GPU memory limit exceeds device capacity")
            torch.cuda.set_per_process_memory_fraction(memory_limit * 2**30 / total, 0)
        if self.llm_client is None:
            from examples.benchmark.hf_qwen_client import HuggingFaceQwenClient
            self.llm_client = HuggingFaceQwenClient(
                model_path=QWEN_BASE, temperature=0.7,
                max_new_tokens=int(self.params.get("generator_max_tokens", 4096)),
                enable_thinking=False)
        self.chat_engine = QWEN_BASE
        upstream = bora_policy()
        dim = sum(not hasattr(hp, "value") for hp in self.config_space.get_hyperparameters())
        self.policy = upstream.Policy(dim=max(1, dim))
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=5, random_state=self.rng)
        self.n_initial = 5
        self.n_bo_candidates, self.n_lbo_selected, self.n_llm_suggest = 5, 2, 5
        self.n_pending = int(self.params.get("n_pending", 10000))
        if self.n_pending < self.n_bo_candidates:
            raise ValueError("BORA needs at least five BO candidates")
        self.monitoring_arrays = self._encode(self.config_space.sample_configuration(5000))
        self.queue, self.action_rewards, self.comments = [], [], []
        self.bora_assistant = None
        self.initialized = False
        self.asset_identity.update(
            policy_sha256=source_digest(UPSTREAM / "bora-the-explorer/bora/policy.py"),
            gp="sklearn Matern(2.5), alpha=1e-6, normalize_y, 5 restarts",
            acquisition="official EI(xi=0.01), random search plus L-BFGS-B; ConfigSpace projection",
            assistant_sha256=source_digest(UPSTREAM / "bora-the-explorer/bora/assistant.py"),
            acquisition_sha256=source_digest(UPSTREAM / "bora-the-explorer/bora/acquisition.py"),
            candidate_count=self.n_pending, gpu_memory_limit_gib=memory_limit,
            generator_max_tokens=int(self.params.get("generator_max_tokens", 4096)),
            history_summary="numeric correlations and categorical score summaries; failures excluded")

    def args_client_invalid(self):
        return (getattr(self.llm_client, "adapter_path", None)
                or getattr(self.llm_client, "model_path", QWEN_BASE) != QWEN_BASE)

    @staticmethod
    def _key(config):
        return json.dumps(dict(config), sort_keys=True)

    @staticmethod
    def _encode(configs):
        # Preserve inactivity outside the numeric [0,1] range.
        return np.nan_to_num(np.array([c.get_array() for c in configs]), nan=-1.0)

    def _pool(self, count, extra=()):
        excluded = {self._key(c) for c in self.configs}
        excluded.update(self._key(c) for c in extra)
        pool = []
        for _ in range(max(100, count * 20)):
            candidate = self.config_space.sample_configuration()
            key = self._key(candidate)
            if key not in excluded:
                pool.append(candidate)
                excluded.add(key)
                if len(pool) == count:
                    break
        if not pool:
            raise RuntimeError("BORA exhausted the unevaluated ConfigSpace")
        return pool

    def _fit_and_uncertainty(self):
        observed = [(c, p) for c, p in zip(self.configs, self.perfs) if np.isfinite(p)]
        if not observed:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.gp.fit(self._encode([c for c, _ in observed]), [p for _, p in observed])
        std = self.gp.predict(self.monitoring_arrays, return_std=True)[1]
        return float(np.mean(std))

    def _bo_candidates(self, count):
        from mindware.components.optimizers.bora_ml_runtime import official_candidates
        return official_candidates(self, count)

    def _prompt(self, instruction, candidates=None, proposed=()):
        history = [{"config": dict(c), "score": p if np.isfinite(p) else None}
                   for c, p in zip(self.configs, self.perfs)]
        context = dict(self.task_context, observations=history,
                       previous_comments=self.comments, already_proposed=[dict(c) for c in proposed])
        if candidates is not None:
            context["candidates"] = [{"index": i, "config": dict(c)} for i, c in enumerate(candidates)]
        return self.llm_client.complete(
            "You are the research assistant in BORA. Reflect on the current task's observations "
            "and hypotheses to maximize the validation score. Respect the supplied ConfigSpace.",
            json.dumps(context) + "\n" + instruction)

    def _assistant_suggestions(self, count):
        assistant = self._assistant()
        if self.trial_cnt < self.n_initial:
            try:
                comment = assistant.pre_optimization_comment(count)
            except ValueError as exc:
                self._record_decision({"stage": "initial_comment_failed", "error": str(exc)})
                comment = None
        else:
            cache = {tuple(dict(c).values()): p for c, p in zip(self.configs, self.perfs)}
            comment = assistant.comment_optimization(self._observations(), cache)
        if comment is None or not comment.is_valid:
            try:
                config = self.resolver.resolve(getattr(self, "bora_last_raw", ""))
            except RecommendationError:
                self._record_decision(self.resolver.last_record)
                return []
            self._record_decision(self.resolver.last_record)
            return [config] * (count if self.trial_cnt < self.n_initial else 1)
        self.comments.append(str(comment))
        from ConfigSpace import Configuration
        return [Configuration(self.config_space, values=p)
                for h in comment.hypotheses for p in h["points"]][:count]

    def _assistant(self):
        if self.bora_assistant is None:
            # Loading failures must stop visibly before upstream completion retries can hide them.
            if hasattr(self.llm_client, "_ensure_loaded"):
                self.llm_client._ensure_loaded()
            from mindware.components.optimizers.bora_ml_runtime import MLAssistant
            self.bora_assistant = MLAssistant(self)
        return self.bora_assistant

    def _observations(self):
        import pandas as pd
        return pd.DataFrame([dict(dict(c), iteration=i, score=p if np.isfinite(p) else "FAILED")
                             for i, (c, p) in enumerate(zip(self.configs, self.perfs), 1)])

    def _selection(self, candidates):
        import pandas as pd
        self.selection_candidates = candidates
        comment = self._assistant().comment_and_select_point(
            self._observations(), pd.DataFrame([dict(c) for c in candidates]))
        if comment is None:
            return []
        self.comments.append(str(comment))
        keys = {self._key(p) for h in comment.hypotheses for p in h["points"]}
        return [c for c in candidates if self._key(c) in keys]

    def _suggest(self):
        if not self.queue:
            if self.trial_cnt < self.n_initial:
                count = self.n_initial - self.trial_cnt
                if self.evaluation_num_limit is not None:
                    count = min(count, self.evaluation_num_limit - self.trial_cnt)
                configs = self._assistant_suggestions(count)
                notes = ["initial assistant proposal"] * len(configs)
                while len(configs) < count:
                    configs.append(self._pool(1, configs)[0])
                    notes.append("initial random fallback after rejected assistant proposal")
            else:
                if not self.initialized:
                    return {"config": dict(self.config_space.get_default_configuration()),
                            "note": "ConfigSpace default: no successful initialization observations"}
                action = self.policy.get_action().name
                configs = []
                if action == "ASSISTANT_SUGGESTS":
                    count = self.evaluation_num_limit - self.trial_cnt
                    configs = self._assistant_suggestions(count)
                elif action == "ASSISTANT_SELECTS":
                    try:
                        configs = self._selection(self._bo_candidates(self.n_bo_candidates))
                    except RecommendationError:
                        configs = []
                if not configs:
                    if action != "BO":
                        self.policy.skip_next_update()
                        action = "BO fallback after rejected assistant intervention"
                    configs = self._bo_candidates(1)
                notes = [action] * len(configs)
            self.queue = [{"config": dict(c), "note": note} for c, note in zip(configs, notes)]
            self.action_rewards = []
        return self.queue.pop(0)

    def _after_observation(self, proposal, perf):
        self.action_rewards.append(perf)
        if self.trial_cnt >= self.n_initial and (not self.queue or self._budget_exhausted()):
            uncertainty = self._fit_and_uncertainty()
            if uncertainty is None:
                return
            if not self.initialized:
                self.policy.initialize(self.trial_cnt, self.incumbent_perf, uncertainty)
                self.initialized = True
            else:
                self.policy.update(self.action_rewards, self.incumbent_perf,
                                   self.trial_cnt, uncertainty)

    def _checkpoint_state(self, directory):
        return self.bora_assistant.snapshot() if self.bora_assistant is not None else None

    def _checkpoint_restore(self, extra):
        super()._checkpoint_restore(extra)
        if self.asset_identity.get("assistant_sha256") != source_digest(
                UPSTREAM / "bora-the-explorer/bora/assistant.py"):
            raise RuntimeError("BORA assistant implementation changed; use a fresh task session")
        if extra is not None:
            self._assistant().restore(extra)
