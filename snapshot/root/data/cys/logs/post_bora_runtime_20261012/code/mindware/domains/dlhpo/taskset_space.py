"""Conditional CASH space over TaskSet optimizer families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
)
from ConfigSpace.conditions import EqualsCondition

from mindware.domains.base import add_hp, config_to_dict
from mindware.domains.paths import TASKSET_ROOT

# Canonical getter names used by task_set.train_inner.
TASKSET_FAMILIES = (
    "adam1p_wide_grid",
    "adam4p_wide_grid",
    "adam6p_wide_grid",
    "adam8p_wide_grid",
    "nadamw_grid",
)

NPZ_TO_FAMILY = {
    "adam1p_wide_grid_1k": "adam1p_wide_grid",
    "adam4p_wide_grid_1k": "adam4p_wide_grid",
    "adam6p_wide_grid_1k": "adam6p_wide_grid",
    "adam8p_wide_grid_1k": "adam8p_wide_grid",
    "nadamw_grid_1k": "nadamw_grid",
}

FAMILY_HP = "optimizer_family"
DEFAULT_SPACES_PATH = TASKSET_ROOT / "scripts/optimizer_search_spaces.json"
DEFAULT_BEST_PATH = TASKSET_ROOT / "best_optimizer_per_task.json"


def load_family_spaces(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    src = Path(path or DEFAULT_SPACES_PATH)
    raw = json.loads(src.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for key, spec in raw.items():
        family = str(spec.get("family") or NPZ_TO_FAMILY.get(key) or key)
        out[family] = dict(spec)
        out[family]["npz_key"] = key
    return out


def list_taskset_ids(best_path: Optional[Path] = None) -> List[str]:
    src = Path(best_path or DEFAULT_BEST_PATH)
    payload = json.loads(src.read_text(encoding="utf-8"))
    return [str(k) for k in payload.keys()]


def _log_bounds(spec: Dict[str, Any]) -> Tuple[float, float]:
    lo = float(spec.get("low"))
    hi = float(spec.get("high"))
    if lo > hi:
        lo, hi = hi, lo
    if lo <= 0:
        lo = min(hi, 1e-12)
    return lo, hi


def _add_family_param(cs: ConfigurationSpace, family: str, name: str, spec: Dict[str, Any]) -> None:
    typ = str(spec.get("type") or "uniform")
    hp_name = f"{family}:{name}"
    if typ == "boolean":
        add_hp(cs, CategoricalHyperparameter(hp_name, [False, True], default_value=False))
        return
    if typ.startswith("transformed_log"):
        formula = str(spec.get("formula") or "")
        # 1 - log_uniform(lo, hi)  ->  store complement on log scale.
        lo, hi = 1e-4, 1.0
        if "log_uniform(" in formula:
            inside = formula.split("log_uniform(", 1)[1].split(")", 1)[0]
            parts = [p.strip() for p in inside.split(",")]
            if len(parts) >= 2:
                lo, hi = float(parts[0]), float(parts[1])
                if lo > hi:
                    lo, hi = hi, lo
        add_hp(
            cs,
            UniformFloatHyperparameter(f"{family}:{name}_inv", lo, hi, default_value=float((lo * hi) ** 0.5), log=True),
        )
        return
    if typ in {"log_uniform", "log_uniform_or_zero", "log_uniform_or_one"}:
        lo, hi = _log_bounds(spec)
        default = float((lo * hi) ** 0.5)
        add_hp(cs, UniformFloatHyperparameter(hp_name, lo, hi, default_value=default, log=True))
        if typ == "log_uniform_or_zero":
            add_hp(cs, CategoricalHyperparameter(f"{hp_name}_is_zero", [False, True], default_value=False))
        if typ == "log_uniform_or_one":
            add_hp(cs, CategoricalHyperparameter(f"{hp_name}_is_one", [False, True], default_value=False))
        return
    lo = float(spec.get("low", 0.0))
    hi = float(spec.get("high", 1.0))
    if lo > hi:
        lo, hi = hi, lo
    add_hp(cs, UniformFloatHyperparameter(hp_name, lo, hi, default_value=float((lo + hi) / 2.0)))


def space_from_taskset(spaces: Optional[Dict[str, Any]] = None, *, seed: int = 1) -> ConfigurationSpace:
    specs = spaces or load_family_spaces()
    families = [f for f in TASKSET_FAMILIES if f in specs] or list(specs.keys())
    cs = ConfigurationSpace(seed=int(seed))
    family_hp = CategoricalHyperparameter(FAMILY_HP, list(families), default_value=families[0])
    add_hp(cs, family_hp)
    for family in families:
        params = (specs.get(family) or {}).get("parameters") or {}
        before = {hp.name for hp in cs.get_hyperparameters()}
        for pname, pspec in params.items():
            _add_family_param(cs, family, pname, pspec if isinstance(pspec, dict) else {})
        for hp in list(cs.get_hyperparameters()):
            if hp.name in before or hp.name == FAMILY_HP:
                continue
            if not str(hp.name).startswith(f"{family}:"):
                continue
            try:
                cs.add_condition(EqualsCondition(hp, family_hp, family))
            except Exception:
                pass
    return cs


def decode_taskset_config(config: Any) -> Dict[str, Any]:
    cfg = config_to_dict(config)
    family = str(cfg.get(FAMILY_HP) or "")
    hps: Dict[str, Any] = {}
    prefix = f"{family}:"
    for key, val in cfg.items():
        if key == FAMILY_HP or not str(key).startswith(prefix):
            continue
        rest = str(key)[len(prefix) :]
        if rest.endswith("_is_zero") or rest.endswith("_is_one"):
            continue
        if rest.endswith("_inv"):
            pname = rest[: -len("_inv")]
            hps[pname] = float(1.0 - float(val))
            continue
        hps[rest] = val
    for key, val in cfg.items():
        if not str(key).startswith(prefix):
            continue
        rest = str(key)[len(prefix) :]
        if rest.endswith("_is_zero") and val:
            hps[rest[: -len("_is_zero")]] = 0.0
        if rest.endswith("_is_one") and val:
            hps[rest[: -len("_is_one")]] = 1.0
    return {"optimizer_family": family, "hyperparameters": hps}


def active_hp_names(cs: ConfigurationSpace, values: Dict[str, Any]) -> List[str]:
    names = []
    for hp in cs.get_hyperparameters():
        if _conditions_satisfied(cs, hp.name, values):
            names.append(hp.name)
    return names


def _conditions_satisfied(cs: ConfigurationSpace, hp_name: str, values: Dict[str, Any]) -> bool:
    try:
        conditions = cs.get_parent_conditions_of(hp_name)
    except Exception:
        return True
    for condition in conditions:
        parent = getattr(condition, "parent", None)
        parent_name = getattr(parent, "name", None)
        if parent_name is None or parent_name not in values:
            return False
        actual = values[parent_name]
        expected = getattr(condition, "value", None)
        if expected is not None and actual != expected:
            return False
    return True
