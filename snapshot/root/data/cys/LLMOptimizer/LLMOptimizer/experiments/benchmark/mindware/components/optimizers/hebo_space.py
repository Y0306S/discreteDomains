"""ConfigSpace <-> HEBO DesignSpace conversion used by pure and hybrid HEBO."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

try:
    from ConfigSpace.hyperparameters import OrdinalHyperparameter
except ImportError:
    OrdinalHyperparameter = ()


def hp_to_hebo_spec(hp) -> Optional[dict]:
    name = hp.name
    if isinstance(hp, UniformIntegerHyperparameter):
        low = int(hp.lower)
        high = int(hp.upper)
        log = bool(getattr(hp, "log", False))
        step = int(getattr(hp, "q", None) or 1)
        if log:
            return {"name": name, "type": "pow_int", "lb": max(1, low), "ub": high}
        if step > 1:
            return {"name": name, "type": "step_int", "lb": low, "ub": high, "step": step}
        return {"name": name, "type": "int", "lb": low, "ub": high}
    if isinstance(hp, UniformFloatHyperparameter):
        low = float(hp.lower)
        high = float(hp.upper)
        log = bool(getattr(hp, "log", False))
        if log:
            low = max(low, 1e-12)
            return {"name": name, "type": "pow", "lb": low, "ub": high}
        return {"name": name, "type": "num", "lb": low, "ub": high}
    if isinstance(hp, CategoricalHyperparameter):
        cats = list(hp.choices)
        if cats == [False, True] or cats == [True, False]:
            return {"name": name, "type": "bool"}
        return {"name": name, "type": "cat", "categories": cats}
    if OrdinalHyperparameter and isinstance(hp, OrdinalHyperparameter):
        return {"name": name, "type": "cat", "categories": list(hp.sequence)}
    return None


def coerce_hebo_value(hp, value: Any) -> Any:
    if hp is None:
        return value
    if isinstance(hp, UniformIntegerHyperparameter):
        return int(round(float(value)))
    if isinstance(hp, UniformFloatHyperparameter):
        return float(value)
    if isinstance(hp, CategoricalHyperparameter):
        raw = value.item() if hasattr(value, "item") else value
        if raw in hp.choices:
            return raw
        as_str = str(raw)
        for choice in hp.choices:
            if str(choice) == as_str:
                return choice
        if isinstance(raw, (bool, np.bool_)):
            return bool(raw)
        return raw
    if OrdinalHyperparameter and isinstance(hp, OrdinalHyperparameter):
        return value.item() if hasattr(value, "item") else value
    return value


def fixed_hp_value(hp):
    return getattr(hp, "value", getattr(hp, "default_value", None))


def build_hebo_space(config_space, algorithm, is_relevant) -> Tuple[Any, Dict[str, Any]]:
    from hebo.design_space.design_space import DesignSpace

    recs = []
    hp_by_name = {}
    for hp in config_space.get_hyperparameters():
        name = hp.name
        if name == "algorithm" or isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            continue
        if not is_relevant(name, algorithm):
            continue
        spec = hp_to_hebo_spec(hp)
        if spec is None:
            continue
        recs.append(spec)
        hp_by_name[name] = hp
    if not recs:
        raise ValueError("HEBO space is empty after filtering ConfigSpace hyperparameters")
    return DesignSpace().parse(recs), hp_by_name


def config_to_hebo_frame(cfg, hebo_space, hp_by_name, algorithm, is_relevant) -> Optional[pd.DataFrame]:
    cfg_dict = cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)
    algo = cfg_dict.get("algorithm", algorithm)
    row = {}
    for name in hebo_space.para_names:
        hp = hp_by_name.get(name)
        if hp is not None and not is_relevant(name, algo):
            return None
        if name not in cfg_dict:
            return None
        row[name] = coerce_hebo_value(hp, cfg_dict[name])
    return pd.DataFrame([row], columns=hebo_space.para_names)
