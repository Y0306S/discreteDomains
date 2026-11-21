"""Mint synthetic BO trajectories from TaskSet best-config dumps + search spaces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mindware.domains.history import observation, write_openbox_history


def _sample_param(spec: Dict[str, Any], rng: np.random.RandomState) -> Any:
    typ = str(spec.get("type") or "uniform")
    low = spec.get("low")
    high = spec.get("high")
    if typ in {"boolean"}:
        return bool(rng.rand() < float(spec.get("p_true", 0.5)))
    if typ in {"log_uniform", "log_uniform_or_zero", "log_uniform_or_one"} and low is not None and high is not None:
        lo, hi = float(low), float(high)
        if lo <= 0 or hi <= 0:
            return float(rng.uniform(min(lo, hi), max(lo, hi)))
        val = 10 ** rng.uniform(math.log10(lo), math.log10(hi))
        if typ == "log_uniform_or_zero" and rng.rand() < 0.25:
            return 0.0
        if typ == "log_uniform_or_one" and rng.rand() < 0.15:
            return 1.0
        return float(val)
    if typ in {"uniform"} and low is not None and high is not None:
        return float(rng.uniform(float(low), float(high)))
    if typ.startswith("transformed_log"):
        # 1 - log_uniform(...) style; sample (0, 1)
        return float(1.0 - 10 ** rng.uniform(-4.0, -0.5))
    return 0.0


def sample_config(space: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, Any]:
    params = space.get("parameters") or {}
    return {name: _sample_param(spec, rng) for name, spec in params.items()}


def convert_taskset(
    best_path: Path,
    spaces_path: Path,
    out_dir: Path,
    *,
    n_fake: int = 16,
    max_tasks: int = 80,
    seed: int = 0,
) -> List[Path]:
    best = json.loads(Path(best_path).read_text(encoding="utf-8"))
    spaces = json.loads(Path(spaces_path).read_text(encoding="utf-8"))
    rng = np.random.RandomState(int(seed))
    written: List[Path] = []
    for i, (tid, rec) in enumerate(best.items()):
        if max_tasks and i >= int(max_tasks):
            break
        family = str(rec.get("optimizer_family_npz") or "")
        space = spaces.get(family) or {}
        best_cfg = dict(rec.get("optimizer_hyperparameters") or {})
        best_loss = float(rec.get("valid_inner_loss") or 1.0)
        if not best_cfg:
            continue
        rows = [
            observation(
                best_cfg,
                best_loss,
                extra={"source": "synthetic_grid", "role": "best", "family": family},
            )
        ]
        for _ in range(max(0, int(n_fake) - 1)):
            cfg = sample_config(space, rng) if space else dict(best_cfg)
            # Worse than best (minimize).
            score = float(best_loss) * (1.05 + 0.8 * float(rng.rand()))
            rows.append(observation(cfg, score, extra={"source": "synthetic_grid", "family": family}))
        dest = Path(out_dir) / "val_loss" / str(tid) / "synthetic_grid.json"
        write_openbox_history(
            dest,
            task_id=str(tid),
            observations=rows,
            meta={"source": "taskset_synthetic_grid", "family": family, "best_optimizer": rec.get("best_optimizer_name")},
        )
        written.append(dest)
    return written
"""Mint synthetic BO trajectories from TaskSet best-config dumps + search spaces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mindware.domains.history import observation, write_openbox_history


def _sample_param(spec: Dict[str, Any], rng: np.random.RandomState) -> Any:
    typ = str(spec.get("type") or "uniform")
    low = spec.get("low")
    high = spec.get("high")
    if typ in {"boolean"}:
        return bool(rng.rand() < float(spec.get("p_true", 0.5)))
    if typ in {"log_uniform", "log_uniform_or_zero", "log_uniform_or_one"} and low is not None and high is not None:
        lo, hi = float(low), float(high)
        if lo <= 0 or hi <= 0:
            return float(rng.uniform(min(lo, hi), max(lo, hi)))
        val = 10 ** rng.uniform(math.log10(lo), math.log10(hi))
        if typ == "log_uniform_or_zero" and rng.rand() < 0.25:
            return 0.0
        if typ == "log_uniform_or_one" and rng.rand() < 0.15:
            return 1.0
        return float(val)
    if typ in {"uniform"} and low is not None and high is not None:
        return float(rng.uniform(float(low), float(high)))
    if typ.startswith("transformed_log"):
        # 1 - log_uniform(...) style; sample (0, 1)
        return float(1.0 - 10 ** rng.uniform(-4.0, -0.5))
    return 0.0


def sample_config(space: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, Any]:
    params = space.get("parameters") or {}
    return {name: _sample_param(spec, rng) for name, spec in params.items()}


def convert_taskset(
    best_path: Path,
    spaces_path: Path,
    out_dir: Path,
    *,
    n_fake: int = 16,
    max_tasks: int = 80,
    seed: int = 0,
) -> List[Path]:
    best = json.loads(Path(best_path).read_text(encoding="utf-8"))
    spaces = json.loads(Path(spaces_path).read_text(encoding="utf-8"))
    rng = np.random.RandomState(int(seed))
    written: List[Path] = []
    for i, (tid, rec) in enumerate(best.items()):
        if max_tasks and i >= int(max_tasks):
            break
        family = str(rec.get("optimizer_family_npz") or "")
        space = spaces.get(family) or {}
        best_cfg = dict(rec.get("optimizer_hyperparameters") or {})
        best_loss = float(rec.get("valid_inner_loss") or 1.0)
        if not best_cfg:
            continue
        rows = [
            observation(
                best_cfg,
                best_loss,
                extra={"source": "synthetic_grid", "role": "best", "family": family},
            )
        ]
        for _ in range(max(0, int(n_fake) - 1)):
            cfg = sample_config(space, rng) if space else dict(best_cfg)
            # Worse than best (minimize).
            score = float(best_loss) * (1.05 + 0.8 * float(rng.rand()))
            rows.append(observation(cfg, score, extra={"source": "synthetic_grid", "family": family}))
        dest = Path(out_dir) / "val_loss" / str(tid) / "synthetic_grid.json"
        write_openbox_history(
            dest,
            task_id=str(tid),
            observations=rows,
            meta={"source": "taskset_synthetic_grid", "family": family, "best_optimizer": rec.get("best_optimizer_name")},
        )
        written.append(dest)
    return written
