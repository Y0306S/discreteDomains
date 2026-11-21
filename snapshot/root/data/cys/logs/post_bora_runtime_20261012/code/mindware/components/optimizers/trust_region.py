"""TuRBO-style trust region state (adapted from BOLT LOLBO turbo.py for minimization)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TrustRegionState:
    """Local trust region around incumbent latent point (minimize objective)."""

    dim: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = float("inf")
    restart_triggered: bool = False
    center: Optional[np.ndarray] = field(default=None, repr=False)

    def reset_length(self, length: float = 0.8) -> None:
        self.length = float(length)
        self.failure_counter = 0
        self.success_counter = 0
        self.restart_triggered = False


def latent_bound_from_values(
    bounds_low: Sequence[float],
    bounds_high: Sequence[float],
) -> np.ndarray:
    low = np.asarray(bounds_low, dtype=np.float64)
    high = np.asarray(bounds_high, dtype=np.float64)
    return np.maximum(np.abs(low), np.abs(high))


def trust_region_bounds(
    center: np.ndarray,
    bounds_low: Sequence[float],
    bounds_high: Sequence[float],
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """BOLT TuRBO box: half-width = (ub - lb) * length / 2 per dimension."""
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    lb_abs = np.asarray(bounds_low, dtype=np.float64)
    ub_abs = np.asarray(bounds_high, dtype=np.float64)
    width = ub_abs - lb_abs
    half = width * float(length) / 2.0
    tr_lb = np.maximum(center - half, lb_abs)
    tr_ub = np.minimum(center + half, ub_abs)
    return tr_lb, tr_ub


def update_trust_region_minimize(state: TrustRegionState, objective: float) -> TrustRegionState:
    """Update counters and TR length after one evaluation (minimize)."""
    y = float(objective)
    if not np.isfinite(state.best_value):
        improved = True
    else:
        improved = y < state.best_value - 1e-3 * abs(state.best_value)
    if improved:
        state.success_counter += 1
        state.failure_counter = 0
        state.best_value = y
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def z_params_from_config(config: Dict[str, Any], dim: int) -> np.ndarray:
    return np.asarray(
        [float(config.get(f"z_{i}", 0.0) or 0.0) for i in range(int(dim))],
        dtype=np.float64,
    )


def incumbent_from_observations(
    observations: List[Dict[str, Any]],
    dim: int,
    bad_value: float = 1e10,
) -> tuple[Optional[np.ndarray], float]:
    best_z: Optional[np.ndarray] = None
    best_val = float("inf")
    for row in observations:
        cfg = row.get("config") or {}
        objs = row.get("objectives") or []
        try:
            val = float(objs[0])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(val) or val >= bad_value:
            continue
        if val < best_val:
            best_val = val
            best_z = z_params_from_config(cfg, dim)
    if best_z is None:
        return None, float("inf")
    return best_z, best_val


def reseed_trust_region_from_observations(
    state: TrustRegionState,
    observations: List[Dict[str, Any]],
    *,
    initial_length: float = 0.8,
    bad_value: float = 1e10,
) -> TrustRegionState:
    """Replay observation stream to restore TR length and incumbent."""
    state.reset_length(initial_length)
    state.best_value = float("inf")
    state.center = None
    for row in observations:
        cfg = row.get("config") or {}
        objs = row.get("objectives") or []
        try:
            val = float(objs[0])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(val) or val >= bad_value:
            continue
        z = z_params_from_config(cfg, state.dim)
        update_trust_region_minimize(state, val)
        if val <= state.best_value:
            state.center = z.copy()
        if state.restart_triggered:
            state.reset_length(initial_length)
    return state
