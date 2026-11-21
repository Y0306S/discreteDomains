"""Target-only ML optimization with an attempt ledger shared by BORA and ToSFiT."""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from openbox.utils.constants import FAILED, SUCCESS, TIMEOUT

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.strict_recommendation import (
    GPTConfigurationExtractor, StrictRecommendationResolver,
)
from mindware.components.optimizers.transfer_hpo.schema import schema_fingerprint, configspace_spec
from mindware.components.optimizers.utils import MyConfiguration
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase


QWEN_BASE = "/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B"


class OnlineLLMOptimizer(BaseOptimizer):
    target_only = True

    def __init__(self, evaluator, config_space, name="hpo", eval_type="holdout",
                 task_id="ml", time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024,
                 inner_iter_num_per_iter=1, timestamp=None, output_dir="./",
                 seed=1, n_jobs=1, topk=50, **params):
        timestamp = time.time() if timestamp is None else timestamp
        super().__init__(evaluator, config_space, name, eval_type, task_id,
                         time_limit, evaluation_limit, per_run_time_limit,
                         per_run_mem_limit, inner_iter_num_per_iter, timestamp,
                         output_dir, seed, topk)
        algorithm = config_space.get_hyperparameter("algorithm")
        self.fixed_algorithm = getattr(algorithm, "value", None)
        if self.fixed_algorithm is None and len(getattr(algorithm, "choices", [])) == 1:
            self.fixed_algorithm = algorithm.choices[0]
        if self.fixed_algorithm is None:
            raise ValueError("%s requires one fixed ML algorithm" % self.optimizer_label)
        self.params = params
        self.schema_fingerprint = schema_fingerprint(config_space)
        self.KD = KnowledgeDatabase.instance()
        # Own lists start empty even if an embedding application reused its KD singleton.
        self.configs, self.perfs = [], []
        self.trial_cnt = 0
        self.traj_rows, self.decision_records = [], []
        self.exp_output = {}
        self.rng = np.random.RandomState(seed)
        self.config_space.seed(seed)
        import torch
        torch.manual_seed(int(self.seed))
        self.extractor_client = params.get("extractor_client")
        if self.extractor_client is None:
            limits = [params.get(key) for key in (
                "extractor_max_tokens", "extractor_timeout", "extractor_max_calls")]
            if any(value is None for value in limits):
                raise ValueError("Set explicit extractor_max_tokens, extractor_timeout and extractor_max_calls")
            self.extractor_client = GPTConfigurationExtractor(
                max_tokens=limits[0], timeout=limits[1], max_calls=limits[2],
                budget_path=Path(os.environ.get("ML_SESSION_DIR", output_dir)) / "extractor_usage.json")
        self.resolver = StrictRecommendationResolver(
            config_space, self.extractor_client,
            default_on_error=params.get("recommendation_default_fallback", False))
        self.extractor_calls = 0
        self.llm_client = params.get("llm_client")
        self.asset_identity = {"base_model": QWEN_BASE, "cross_task_sources": [],
                               "extractor": "gpt-4o-mini", "protocol": 2,
                               "recommendation_policy": self.resolver.policy}
        self.task_context = {"task": task_id, "algorithm": self.fixed_algorithm,
                             "dataset": params.get("dataset_profile", {}),
                             "objective": "maximize validation score (negative loss for regression)",
                             "configspace": configspace_spec(config_space)}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.traj_csv_path = str(Path(output_dir) / (
            "%d_opt_%s_traj.csv" % (timestamp, self.optimizer_label)))
        self.logging_enabled = params.get("logging_enabled", True)

    def _budget_exhausted(self):
        quota = self.evaluation_num_limit is not None and self.trial_cnt >= self.evaluation_num_limit
        timed_out = self.time_limit is not None and time.time() - self.timestamp >= self.time_limit
        self.timeout_flag = self.timeout_flag or timed_out
        self.early_stopped_flag = self.early_stopped_flag or quota
        return self.timeout_flag or self.early_stopped_flag

    def _record_decision(self, record):
        self.decision_records.append(dict(record))
        self.extractor_calls = getattr(self.extractor_client, "calls", self.extractor_calls)
        if self.logging_enabled:
            ml_checkpoint.atomic_json(Path(self.traj_csv_path).with_suffix(".decisions.json"),
                                      self.decision_records)

    def _checkpoint_restore(self, extra):
        if self.asset_identity.get("recommendation_policy") != self.resolver.policy:
            raise RuntimeError("Checkpoint uses a different recommendation policy; use a new run directory")
        if hasattr(self.extractor_client, "calls"):
            self.extractor_client.calls = self.extractor_calls

    def _suggest(self):
        raise NotImplementedError

    def _after_observation(self, proposal, perf):
        pass

    def _evaluate(self, proposal):
        if proposal.get("config") is None:
            if self.resolver.default_on_error:
                raise RuntimeError("Default-fallback policy requires a configuration before evaluation")
            return None, float("-inf"), FAILED, 0.0
        config = MyConfiguration(Configuration(self.config_space, values=proposal["config"]))
        self.config_space.check_configuration(config)
        obs = wrapper_evaluater(config, self.evaluator, self.per_run_time_limit, force_isolation=True)
        objective = float(obs.objectives[0]) if obs.objectives else float("inf")
        success = obs.trial_state == SUCCESS and math.isfinite(objective)
        perf = -objective if success else float("-inf")
        if not success:
            obs.objectives = [float("inf")]
        self.configs.append(config)
        self.perfs.append(perf)
        # bo_mab expects its per-algorithm history to be updated by the optimizer.
        self.KD.bo_history_mab[self.fixed_algorithm].update_observation(obs)
        self.KD.update_history(obs, source="bo_mab", trail_id=str(self.trial_cnt))
        self.eval_dict[(config, self.trial_cnt)] = (perf, obs.elapsed_time)
        if success:
            self.exp_output[time.time()] = (config, perf)
            self.update_saver([config], [objective])
            if perf > self.incumbent_perf:
                self.incumbent_perf, self.incumbent_config = perf, dict(config)
        return config, perf, obs.trial_state, float(obs.elapsed_time or 0.0)

    def iterate(self, budget=MAX_INT):
        start, rows = time.time(), []
        for _ in range(self.inner_iter_num_per_iter):
            if self._budget_exhausted() or time.time() - start >= budget:
                break
            trial_start = time.time()
            proposal = ml_checkpoint.pending(self)
            if proposal is None:
                proposal = self._suggest()
                proposal["suggest_time"] = time.time() - trial_start
            # Persist any online gradient update and its original proposal together.
            ml_checkpoint.prepare(self, proposal)
            if self._budget_exhausted() or time.time() - start >= budget:
                break
            config, perf, state, elapsed = self._evaluate(proposal)
            self.trial_cnt += 1
            self._after_observation(proposal, perf)
            row = {"ts": pd.Timestamp.now().isoformat(), "stage": "iterate",
                   "iter_idx": self.trial_cnt - 1, "perf": perf,
                   "best_so_far": self.incumbent_perf, "algorithm": self.fixed_algorithm,
                   "smac_time": proposal["suggest_time"], "eval_time": elapsed,
                   "iter_time": time.time() - trial_start,
                   "wall_elapsed_sec": time.time() - self.timestamp,
                   "note": proposal["note"], "trial_state": int(state),
                   "evaluation_status": "SUCCESS" if state == SUCCESS and math.isfinite(perf)
                   else ("TIMEOUT" if state == TIMEOUT else "FAILED"),
                   "evaluation_limit_seconds": self.per_run_time_limit,
                   "config_json": json.dumps(proposal.get("config"), sort_keys=True),
                   "oracle_called": config is not None,
                   "schema_fingerprint": self.schema_fingerprint,
                   "assets_json": json.dumps(self.asset_identity, sort_keys=True)}
            self.traj_rows.append(row)
            rows.append(row)
            ml_checkpoint.commit(self)
            if self.logging_enabled and not getattr(self, "_ml_session", None):
                pd.DataFrame(self.traj_rows).to_csv(self.traj_csv_path, index=False)
            self._budget_exhausted()
            if self._budget_exhausted() and hasattr(self, "_finalize_completed_task"):
                self._finalize_completed_task()
        return self.incumbent_perf, time.time() - start, self.incumbent_config, rows

    def run(self):
        while not self._budget_exhausted():
            self.iterate()
        if hasattr(self, "_finalize_completed_task"):
            self._finalize_completed_task()
        return self.incumbent_perf
