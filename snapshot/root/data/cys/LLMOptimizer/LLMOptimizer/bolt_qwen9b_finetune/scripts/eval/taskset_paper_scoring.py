"""TaskSet paper (§3.3) scoring utilities.

Reference: "Using a thousand optimization tasks to learn hyperparameter
search strategies" (Metz et al., 2020).

Training uses task_set.train_inner with defaults:
  - 10,000 steps, eval every 200 steps, 50 mini-batches per eval.
Validation for scoring is VALID_OUTER (meta-validation split).

These protocol values are fixed for all production eval runs. Import them from
this module rather than passing CLI overrides.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Paper §3.3 / train_inner protocol (Metz et al. 2020) — do not override in production.
PAPER_TRAINING_STEPS = 10_000
PAPER_EVAL_EVERY_N = 200
PAPER_NUM_REPLICAS = 5
# Per-replica cost = mean normalized VALID_OUTER over logged eval steps; task score
# = mean of replica costs (paper §3.3 averages over 5 random initializations).
PAPER_REPLICA_AGGREGATION = "mean"
PAPER_CURVE_AGGREGATION = "mean"  # mean over eval checkpoints within each replica

# CPU parallelism for eval (train_inner uses ~20 TF threads each).
# Model eval: 8 tasks in parallel on 256-core hosts; 4 on ~96-core CPU boxes.
PAPER_TASK_WORKERS = 8
CPU_TASK_WORKERS = 4
# Baseline build: fewer parallel tasks (each task runs hundreds of grid optimizers).
PAPER_BASELINE_TASK_WORKERS = 4
CPU_BASELINE_TASK_WORKERS = 2

# train_inner result tuple: [train, valid_inner, valid_outer, test]
TRAIN_IDX = 0
VALID_INNER_IDX = 1
VALID_OUTER_IDX = 2
TEST_IDX = 3

WIDE_GRID_FAMILIES = (
    "adam1p_wide_grid",
    "adam4p_wide_grid",
    "adam6p_wide_grid",
    "adam8p_wide_grid",
    "nadamw_grid",
)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def sorted_eval_steps(curves: dict[str, list[float]]) -> list[int]:
    return sorted(int(step) for step in curves.keys())


def valid_outer_at_step(curves: dict[str, list[float]], step: int | str) -> float:
    return float(curves[str(step)][VALID_OUTER_IDX])


def init_valid_outer(curves: dict[str, list[float]]) -> float:
    steps = sorted_eval_steps(curves)
    if not steps:
        raise ValueError("Empty learning curve.")
    return valid_outer_at_step(curves, steps[0])


def min_valid_outer(curves: dict[str, list[float]]) -> float:
    return min(valid_outer_at_step(curves, step) for step in sorted_eval_steps(curves))


def final_test_loss(curves: dict[str, list[float]]) -> float:
    last_step = max(sorted_eval_steps(curves))
    return float(curves[str(last_step)][TEST_IDX])


def normalize_valid_outer_value(
    value: float,
    init_val: float,
    best_val: float,
) -> float:
    """Linear normalization from paper §3.3; clip to [0, 1]."""
    if init_val <= best_val:
        return 1.0
    normed = (value - best_val) / (init_val - best_val)
    return min(max(normed, 0.0), 1.0)


def mean_normalized_valid_outer_cost(
    curves: dict[str, list[float]],
    init_val: float,
    best_val: float,
) -> float:
    """Mean normalized VALID_OUTER loss over all logged eval steps."""
    steps = sorted_eval_steps(curves)
    if not steps:
        raise ValueError("Empty learning curve.")
    normed = []
    for step in steps:
        value = valid_outer_at_step(curves, step)
        if not math.isfinite(value) or not math.isfinite(init_val) or not math.isfinite(best_val):
            continue
        normed.append(normalize_valid_outer_value(value, init_val, best_val))
    if not normed:
        return float("nan")
    return sum(normed) / len(normed)


def score_replica_curve(
    curves: dict[str, list[float]],
    best_val: float,
) -> dict[str, float]:
    init_val = init_valid_outer(curves)
    return {
        "init_valid_outer": init_val,
        "best_val_reference": best_val,
        "mean_normalized_valid_outer": mean_normalized_valid_outer_cost(
            curves, init_val, best_val
        ),
        "final_test_loss": final_test_loss(curves),
        "min_valid_outer": min_valid_outer(curves),
    }


def aggregate_replica_scores(replica_scores: list[dict[str, float]]) -> dict[str, float]:
    """Average per-replica scores (paper §3.3: mean over random initializations).

    Replicas with non-finite costs are skipped (nanmean semantics).
    """
    if not replica_scores:
        return {}
    if PAPER_REPLICA_AGGREGATION != "mean":
        raise ValueError(f"Unsupported replica aggregation: {PAPER_REPLICA_AGGREGATION!r}")
    keys = replica_scores[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(r[key]) for r in replica_scores if math.isfinite(float(r[key]))]
        if not vals:
            out[key] = float("nan")
        else:
            out[key] = sum(vals) / len(vals)
    return out


def paper_eval_protocol() -> dict[str, int | str]:
    """Return the fixed eval protocol for logging and result metadata."""
    return {
        "training_steps": PAPER_TRAINING_STEPS,
        "eval_every_n": PAPER_EVAL_EVERY_N,
        "num_replicas": PAPER_NUM_REPLICAS,
        "replica_aggregation": PAPER_REPLICA_AGGREGATION,
        "curve_aggregation": PAPER_CURVE_AGGREGATION,
        "task_workers": PAPER_TASK_WORKERS,
        "baseline_task_workers": PAPER_BASELINE_TASK_WORKERS,
        "validation_split": "VALID_OUTER",
        "scoring": "task_set_paper_section_3_3",
    }


def list_grid_optimizer_names(
    optimizer_configs_dir: Path,
    families: list[str],
    max_per_family: int | None = None,
) -> list[str]:
    names: list[str] = []
    for family in families:
        config_path = optimizer_configs_dir / f"{family}.json"
        if not config_path.exists():
            continue
        family_names = sorted(load_json(config_path).keys())
        if max_per_family is not None:
            family_names = family_names[:max_per_family]
        names.extend(family_names)
    return names


def load_baseline_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data.get("tasks"), dict):
        return data["tasks"]
    return data


def save_baseline_cache(path: Path, tasks: dict[str, dict[str, Any]], meta: dict | None = None) -> None:
    payload = {"meta": meta or {}, "tasks": tasks}
    save_json(path, payload)


def finite_best_val(per_family: dict[str, Any]) -> float | None:
    """Min VALID_OUTER across grid families, ignoring non-finite sentinel failures.

    When grid runs blow up to large negative values (common on maf/losg tasks),
    prefer the best positive finite VALID_OUTER if any exist.
    """
    finite = [
        float(v)
        for v in per_family.values()
        if isinstance(v, (int, float)) and math.isfinite(v)
    ]
    if not finite:
        return None
    positive = [v for v in finite if v > 0]
    if positive:
        return min(positive)
    return min(finite)


def get_task_best_val(
    task_name: str,
    baseline_cache: dict[str, dict[str, Any]],
) -> float | None:
    entry = baseline_cache.get(task_name)
    if not entry:
        return None
    per = entry.get("per_family_min_valid_outer") or {}
    corrected = finite_best_val(per)
    if corrected is not None:
        return corrected
    best = entry.get("best_val")
    if isinstance(best, (int, float)) and math.isfinite(best):
        return float(best)
    return None
