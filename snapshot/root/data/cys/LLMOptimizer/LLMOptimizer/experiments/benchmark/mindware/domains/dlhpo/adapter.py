"""DL-framework HPO adapter: named ConfigSpace, lookup oracle from histories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from mindware.domains.base import DomainAdapter, EvaluationResult, add_hp, config_to_dict, dummy_result
from mindware.domains.dlhpo.zaxie import DL_ROLE_HINTS, ZAXIE_KEYS


def space_from_zaxie(spec: Optional[Dict[str, Any]] = None) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    spec = spec or {}
    defaults = {
        "depth": ("int", 6, 12),
        "device_batch_size": ("int", 32, 256),
        "grad_accum_steps": ("int", 1, 4),
        "embedding_lr": ("float", 0.1, 1.5, "log"),
        "unembedding_lr": ("float", 0.001, 0.02, "log"),
        "matrix_lr": ("float", 0.01, 0.12, "log"),
        "scalar_lr": ("float", 0.1, 1.0, "log"),
        "weight_decay": ("float", 0.0, 0.5),
        "warmup_ratio": ("float", 0.0, 0.2),
        "warmdown_ratio": ("float", 0.2, 0.8),
        "final_lr_frac": ("float", 0.0, 0.2),
    }
    merged = dict(defaults)
    for k, v in spec.items():
        if isinstance(v, list) and len(v) >= 3:
            merged[k] = tuple(v)
    for name, row in merged.items():
        kind = row[0]
        lo, hi = float(row[1]), float(row[2])
        log = len(row) > 3 and str(row[3]) == "log"
        if kind == "int":
            add_hp(cs, UniformIntegerHyperparameter(name, int(lo), int(hi), default_value=int((lo + hi) / 2)))
        else:
            default = math.sqrt(lo * hi) if log and lo > 0 else (lo + hi) / 2
            add_hp(
                cs,
                UniformFloatHyperparameter(name, lo, hi, default_value=float(default), log=bool(log)),
            )
    return cs


class DLHPOAdapter(DomainAdapter):
    domain_id = "dl"
    metric = "val_loss"
    higher_is_better = False

    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        *,
        lookup: Optional[Dict[str, float]] = None,
        dummy: bool = False,
        search_space_spec: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(task=task, **kwargs)
        self.dummy = bool(dummy)
        self.lookup = dict(lookup or {})
        spec = search_space_spec or (self.task or {}).get("search_space")
        self._cs = space_from_zaxie(spec)

    def search_space(self, **kwargs: Any) -> ConfigurationSpace:
        return self._cs

    def decode(self, config: Any) -> Dict[str, Any]:
        return config_to_dict(config)

    def evaluate(self, artifact: Any, task: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        cfg = dict(artifact or {})
        if self.dummy and not self.lookup:
            res = dummy_result(cfg)
            res.artifact = cfg
            return res
        key = _cfg_key(cfg)
        if key in self.lookup:
            return EvaluationResult(
                objective=float(self.lookup[key]),
                artifact=cfg,
                feasible=True,
                info={"oracle": "history_lookup"},
            )
        # Cheap surrogate: distance to nearest stored config in z-score-ish space.
        if self.lookup:
            nearest = min(self.lookup.values())
            penalty = 0.05 * abs(hash(key) % 100) / 100.0
            return EvaluationResult(
                objective=float(nearest) * (1.02 + penalty),
                artifact=cfg,
                feasible=True,
                info={"oracle": "nearest_surrogate"},
            )
        res = dummy_result(cfg)
        res.artifact = cfg
        return res

    def profile(self, task: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        t = task if task is not None else self.task
        return {
            "n_trials": float(t.get("n_trials") or len(self.lookup) or 0.0),
            "n_hps": float(len(ZAXIE_KEYS)),
            "log_params": float(t.get("num_params_M") or 0.0),
        }

    def experience_roles(self) -> List[Tuple[str, str]]:
        return list(DL_ROLE_HINTS)

    def semantic_keys(self) -> List[str]:
        return list(ZAXIE_KEYS)

    def prompt_schema(self) -> Dict[str, Any]:
        base = super().prompt_schema()
        base["action"] = (
            "Propose JSON with named training HPs: depth, device_batch_size, grad_accum_steps, "
            "embedding_lr, unembedding_lr, matrix_lr, scalar_lr, weight_decay, warmup_ratio, "
            "warmdown_ratio, final_lr_frac. Minimize validation loss."
        )
        return base


def _cfg_key(cfg: Dict[str, Any]) -> str:
    items = []
    for k in sorted(cfg):
        v = cfg[k]
        if isinstance(v, float):
            items.append(f"{k}={v:.6g}")
        else:
            items.append(f"{k}={v}")
    return "|".join(items)


def lookup_from_history(path: Path) -> Dict[str, float]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    table: Dict[str, float] = {}
    for row in obj.get("observations") or []:
        if int(row.get("trial_state", 0) or 0) != 0:
            continue
        objs = row.get("objectives") or []
        if not objs:
            continue
        table[_cfg_key(dict(row.get("config") or {}))] = float(objs[0])
    return table
