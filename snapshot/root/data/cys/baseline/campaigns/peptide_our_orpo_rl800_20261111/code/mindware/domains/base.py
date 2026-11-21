"""Domain adapter interface for protein, query-opt, and DL HPO.

The tabular CASH loop stays untouched. New suites live under
``benchmark_data/domain_{protein,query,dl}/`` and
``history_data/domain_{protein,query,dl}/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

LATENT_DIM = 16
LATENT_BOUND = 3.0


@dataclass
class EvaluationResult:
    """OpenBox-style minimize objective plus decoded artifact."""

    objective: float
    artifact: Any = None
    feasible: bool = True
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def score_higher_is_better(self) -> float:
        return -float(self.objective) if self.feasible else float("-inf")


def config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, Configuration):
        return dict(config.get_dictionary())
    if hasattr(config, "get_dictionary"):
        return dict(config.get_dictionary())
    return dict(config)


def add_hp(cs: ConfigurationSpace, hp) -> None:
    if hasattr(cs, "add_hyperparameter"):
        cs.add_hyperparameter(hp)
    else:
        cs.add(hp)


def latent_config_space(
    dim: int = LATENT_DIM,
    bound: float = LATENT_BOUND,
    extra: Optional[Sequence] = None,
    seed: int = 1,
) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=int(seed))
    for i in range(int(dim)):
        add_hp(
            cs,
            UniformFloatHyperparameter(f"z_{i}", -float(bound), float(bound), default_value=0.0),
        )
    for hp in extra or ():
        add_hp(cs, hp)
    return cs


def z_from_config(config: Any, dim: int = LATENT_DIM) -> List[float]:
    cfg = config_to_dict(config)
    return [float(cfg.get(f"z_{i}", 0.0) or 0.0) for i in range(int(dim))]


def z_to_config(z: Sequence[float], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {f"z_{i}": float(v) for i, v in enumerate(z)}
    if extra:
        out.update(extra)
    return out


class DomainAdapter(ABC):
    """Swap-in evaluator + search space for one application domain."""

    domain_id: str = "abstract"
    metric: str = "objective"
    higher_is_better: bool = False
    latent_dim: int = LATENT_DIM

    def __init__(self, task: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.task = dict(task or {})
        self.kwargs = dict(kwargs)

    @abstractmethod
    def search_space(self, **kwargs: Any) -> ConfigurationSpace:
        raise NotImplementedError

    def decode(self, config: Any) -> Any:
        return config_to_dict(config)

    def encode(self, artifact: Any) -> Optional[List[float]]:
        return None

    @abstractmethod
    def evaluate(self, artifact: Any, task: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        raise NotImplementedError

    def evaluate_config(
        self, config: Any, task: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        try:
            artifact = self.decode(config)
        except Exception as exc:
            return EvaluationResult(
                objective=1e10,
                artifact=None,
                feasible=False,
                info={"error": f"decode:{exc}"},
            )
        result = self.evaluate(artifact, task if task is not None else self.task)
        cfg = config_to_dict(config)
        result.info.setdefault("config", cfg)
        return result

    def profile(self, task: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        t = task if task is not None else self.task
        return {"n_keys": float(len(t or {}))}

    def experience_roles(self) -> List[Tuple[str, str]]:
        return [("z_", "latent_coord")]

    def prompt_schema(self) -> Dict[str, Any]:
        return {
            "domain": self.domain_id,
            "action": "Propose a JSON object with keys matching the search space. "
            "Prefer named semantic fields; do not dump a raw high-dimensional z vector.",
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
        }

    def semantic_keys(self) -> List[str]:
        return []

    def semantic_choices(self) -> Dict[str, List[str]]:
        return {}

    def task_id(self, task: Optional[Dict[str, Any]] = None) -> str:
        t = task if task is not None else self.task
        return str((t or {}).get("task_id") or (t or {}).get("dataset_id") or "unknown")


def dummy_result(config: Any, seed: int = 0) -> EvaluationResult:
    """Deterministic placeholder oracle for wiring tests (constant-cost loop)."""
    cfg = config_to_dict(config)
    acc = 0.0
    for i, (k, v) in enumerate(sorted(cfg.items(), key=lambda kv: kv[0])):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            acc += float(v) * (0.17 + 0.013 * i)
        else:
            acc += float(sum(ord(c) for c in str(v)) % 97) * 0.01
    acc += 0.001 * int(seed)
    return EvaluationResult(objective=float(abs(acc)), artifact=cfg, feasible=True, info={"oracle": "dummy"})
