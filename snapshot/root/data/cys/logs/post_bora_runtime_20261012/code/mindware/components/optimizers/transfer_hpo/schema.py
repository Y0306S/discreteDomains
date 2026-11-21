"""Per-algorithm search-space grouping and numeric encoding for FSBO / RGPE.

ML HPO groups by (task_type, algorithm). Classification and regression spaces
are not mixed: lightgbm/adaboost/catboost differ across cls vs rgs, and even
when names match (extra_trees) the categorical choices (criterion) do not.
"""

from __future__ import annotations

import hashlib
import json
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from ConfigSpace.read_and_write import json as cs_json

ALGORITHMS: Tuple[str, ...] = (
    "adaboost",
    "catboost",
    "extra_trees",
    "gradient_boosting",
    "lightgbm",
    "random_forest",
    "xgboost",
)

# Prefer unordered classical BO as transfer sources. Never pull LLM trajs.
SOURCE_VARIANT_PRIORITY: Tuple[str, ...] = (
    "tpe",
    "smac",
    "cmaes",
    "hebo",
    "random_search",
)

def group_id(task_type: str, algorithm: str) -> str:
    aliases = {"cls": "cls", "classification": "cls", "rgs": "rgs", "regression": "rgs"}
    if task_type not in aliases or algorithm not in ALGORITHMS:
        raise ValueError("Unsupported ML transfer group: %s/%s" % (task_type, algorithm))
    tt = aliases[task_type]
    return f"{tt}:{algorithm}"


def parse_group_id(gid: str) -> Tuple[str, str]:
    tt, algo = gid.split(":", 1)
    return tt, algo


def all_group_ids() -> List[str]:
    return [group_id(tt, algo) for tt in ("cls", "rgs") for algo in ALGORITHMS]


def build_hpo_cs(task_type: str, algorithm: str, **cs_args) -> ConfigurationSpace:
    from mindware.components.config_space.cs_builder import get_hpo_cs
    from mindware.components.utils.constants import CLASSIFICATION, REGRESSION

    tt, _ = parse_group_id(group_id(task_type, algorithm))
    return get_hpo_cs(algorithm, CLASSIFICATION if tt == "cls" else REGRESSION, **cs_args)


def schema_spec(cs: ConfigurationSpace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for hp in cs.get_hyperparameters():
        row: Dict[str, Any] = {
            "name": hp.name, "type": type(hp).__name__, "default_value": hp.default_value,
        }
        if isinstance(hp, CategoricalHyperparameter):
            row["choices"] = list(hp.choices)
        elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            row["lower"] = hp.lower
            row["upper"] = hp.upper
            row["log"] = bool(getattr(hp, "log", False))
            row["q"] = getattr(hp, "q", None)
        elif isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            row["value"] = hp.value
        rows.append(row)
    return rows


def configspace_spec(cs: ConfigurationSpace) -> Dict[str, Any]:
    """Use ConfigSpace's serializer to preserve defaults and relational constraints."""
    spec = json.loads(cs_json.write(cs))
    # ConfigSpace 0.4/0.6 JSON silently omits numeric quantization.
    spec["transfer_hyperparameters"] = schema_spec(cs)
    return spec


def schema_fingerprint(cs: ConfigurationSpace) -> str:
    spec = configspace_spec(cs)
    canonical = {
        "hyperparameters": schema_spec(cs),
        "conditions": spec.get("conditions", []),
        "forbiddens": spec.get("forbiddens", []),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def cs_from_spec(spec: Sequence[Dict[str, Any]]) -> ConfigurationSpace:
    """Rebuild a space in older ConfigSpace (RGPE py3.7)."""
    if isinstance(spec, dict):
        parsed = cs_json.read(json.dumps(spec))
        if "transfer_hyperparameters" not in spec:
            return parsed
        cs = cs_from_spec(spec["transfer_hyperparameters"])
        replacements = {id(hp): cs.get_hyperparameter(hp.name) for hp in parsed.get_hyperparameters()}
        cs.add_conditions([copy.deepcopy(c, replacements) for c in parsed.get_conditions()])
        cs.add_forbidden_clauses([copy.deepcopy(c, replacements) for c in parsed.get_forbiddens()])
        return cs
    cs = ConfigurationSpace()
    for row in spec:
        name = row["name"]
        kind = row["type"]
        defaults = {"default_value": row["default_value"]} if "default_value" in row else {}
        if kind == "CategoricalHyperparameter":
            hp = CategoricalHyperparameter(name, row["choices"], **defaults)
        elif kind == "UniformFloatHyperparameter":
            hp = UniformFloatHyperparameter(
                name, row["lower"], row["upper"], log=bool(row.get("log")),
                q=row.get("q"), **defaults
            )
        elif kind == "UniformIntegerHyperparameter":
            hp = UniformIntegerHyperparameter(
                name, int(row["lower"]), int(row["upper"]), log=bool(row.get("log")),
                q=row.get("q"), **defaults
            )
        elif kind in ("Constant", "UnParametrizedHyperparameter"):
            cls = Constant if kind == "Constant" else UnParametrizedHyperparameter
            hp = cls(name, row["value"])
        else:
            raise ValueError("Unsupported hyperparameter type: %s" % kind)
        cs.add_hyperparameter(hp)
    return cs


def config_from_dict(cs: ConfigurationSpace, raw: Dict[str, Any], algorithm: str) -> Optional[Configuration]:
    values = dict(raw or {})
    if "algorithm" not in values:
        values["algorithm"] = algorithm
    allowed = set(cs.get_hyperparameter_names())
    if set(values) - allowed:
        return None
    try:
        config = Configuration(cs, values=values)
        cs.check_configuration(config)
        return config
    except Exception:
        return None


def encode_config(cs: ConfigurationSpace, raw: Dict[str, Any], algorithm: str) -> Optional[np.ndarray]:
    cfg = config_from_dict(cs, raw, algorithm)
    if cfg is None:
        return None
    arr = np.asarray(cfg.get_array(), dtype=np.float64)
    if np.isinf(arr).any():
        raise ValueError("Non-finite configuration encoding")
    # The inactive marker is outside normalized active coordinates.
    arr = np.nan_to_num(arr, nan=-1.0)
    return arr


def vector_dim(cs: ConfigurationSpace) -> int:
    return len(cs.get_hyperparameter_names())
