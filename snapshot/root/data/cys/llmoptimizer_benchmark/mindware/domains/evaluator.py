"""Callable evaluator wrapping a DomainAdapter for Optuna / traj logging."""

from __future__ import annotations

from typing import Any, Dict, Optional

from mindware.domains.base import DomainAdapter, EvaluationResult, config_to_dict


class DomainEvaluator:
    def __init__(self, adapter: DomainAdapter, task: Optional[Dict[str, Any]] = None):
        self.adapter = adapter
        self.task = task if task is not None else adapter.task
        self.last: Optional[EvaluationResult] = None

    def __call__(self, config: Any, **kwargs: Any) -> float:
        result = self.adapter.evaluate_config(config, self.task)
        self.last = result
        if not result.feasible:
            return 1e10
        return float(result.objective)
"""Callable evaluator wrapping a DomainAdapter for Optuna / traj logging."""
