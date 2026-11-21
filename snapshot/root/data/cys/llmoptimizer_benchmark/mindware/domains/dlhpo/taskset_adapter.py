"""TaskSet CASH adapter: pick optimizer family, then that family's HPs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace import ConfigurationSpace

from mindware.domains.base import DomainAdapter, EvaluationResult
from mindware.domains.dlhpo.taskset_space import (
    FAMILY_HP,
    TASKSET_FAMILIES,
    decode_taskset_config,
    load_family_spaces,
    space_from_taskset,
)


class TaskSetAdapter(DomainAdapter):
    domain_id = "taskset"
    metric = "val_loss"
    higher_is_better = False

    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        *,
        dummy: bool = True,
        smoke: bool = False,
        spaces: Optional[Dict[str, Any]] = None,
        training_steps: Optional[int] = None,
        eval_every_n: Optional[int] = None,
        num_replicas: Optional[int] = None,
        objective_metric: Optional[str] = None,
        baseline_cache: Optional[Path] = None,
        oracle_work_dir: Optional[Path] = None,
        warm_timeout: int = 3600,
        replica_timeout: int = 2400,
        **kwargs: Any,
    ):
        super().__init__(task=task, **kwargs)
        self.dummy = bool(dummy) and not bool(smoke)
        self.smoke = bool(smoke)
        self._spaces = spaces or load_family_spaces()
        self._cs = space_from_taskset(self._spaces)
        self.training_steps = training_steps
        self.eval_every_n = eval_every_n
        self.num_replicas = num_replicas
        self.objective_metric = objective_metric
        self.baseline_cache = Path(baseline_cache) if baseline_cache else None
        self.oracle_work_dir = Path(oracle_work_dir) if oracle_work_dir else None
        self.warm_timeout = int(warm_timeout)
        self.replica_timeout = int(replica_timeout)

    def search_space(self, **kwargs: Any) -> ConfigurationSpace:
        return self._cs

    def decode(self, config: Any) -> Dict[str, Any]:
        return decode_taskset_config(config)

    def evaluate(self, artifact: Any, task: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        payload = artifact if isinstance(artifact, dict) and "optimizer_family" in artifact else decode_taskset_config(artifact)
        family = str(payload.get("optimizer_family") or "")
        hps = dict(payload.get("hyperparameters") or {})
        if self.dummy:
            res = _dummy_taskset_result(family, hps, task_id=self.task_id(task))
            res.artifact = payload
            return res
        from mindware.domains.dlhpo import taskset_oracle

        return taskset_oracle.evaluate_taskset_config(
            task_id=self.task_id(task),
            optimizer_family=family,
            hyperparameters=hps,
            work_dir=self.oracle_work_dir,
            baseline_cache=self.baseline_cache,
            training_steps=self.training_steps,
            eval_every_n=self.eval_every_n,
            num_replicas=self.num_replicas,
            objective_metric=self.objective_metric or taskset_oracle.DEFAULT_OBJECTIVE_METRIC,
            smoke=self.smoke,
            warm_timeout=self.warm_timeout,
            replica_timeout=self.replica_timeout,
        )

    def experience_roles(self) -> List[Tuple[str, str]]:
        return [(FAMILY_HP, "optimizer_family"), (":learning_rate", "learning_rate")]

    def semantic_keys(self) -> List[str]:
        return [FAMILY_HP]

    def semantic_choices(self) -> Dict[str, List[str]]:
        return {FAMILY_HP: list(TASKSET_FAMILIES)}

    def prompt_schema(self) -> Dict[str, Any]:
        base = super().prompt_schema()
        base["action"] = (
            "Propose JSON with optimizer_family (one of adam1p_wide_grid, adam4p_wide_grid, "
            "adam6p_wide_grid, adam8p_wide_grid, nadamw_grid) and hyperparameters containing "
            "only fields valid for that family. Minimize validation loss."
        )
        return base


def _dummy_taskset_result(family: str, hps: Dict[str, Any], *, task_id: str) -> EvaluationResult:
    """Cheap landscape: nadamw is slightly better; each family has a target lr."""
    bias = {
        "adam1p_wide_grid": 1.15,
        "adam4p_wide_grid": 1.05,
        "adam6p_wide_grid": 0.98,
        "adam8p_wide_grid": 0.92,
        "nadamw_grid": 0.80,
    }.get(family, 1.4)
    lr = float(hps.get("learning_rate") or 1e-3)
    target = 1e-2
    lr_term = (math.log10(max(lr, 1e-12)) - math.log10(target)) ** 2
    extra = 0.0
    if "beta1" in hps:
        extra += abs(float(hps["beta1"]) - 0.9) * 0.2
    if "use_nesterov" in hps:
        extra += 0.0 if hps["use_nesterov"] else 0.05
    salt = (sum(ord(c) for c in str(task_id)) % 17) * 0.001
    obj = float(bias + 0.35 * lr_term + extra + salt)
    return EvaluationResult(
        objective=obj,
        artifact={"optimizer_family": family, "hyperparameters": hps},
        feasible=True,
        info={"oracle": "dummy_taskset"},
    )
