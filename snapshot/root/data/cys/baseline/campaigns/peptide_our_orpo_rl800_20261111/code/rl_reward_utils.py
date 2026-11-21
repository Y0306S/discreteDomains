"""Reward helpers for P5 RL rollout buffers (Δbest + optional absolute term)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def finite_scores(perfs: List[float]) -> List[float]:
    return [float(p) for p in perfs if np.isfinite(p)]


def rel_score(score: float, pool: List[float]) -> float:
    """Min-max rel score for `score` against finite values in `pool` (matches prompt scale)."""
    if not np.isfinite(score):
        return 0.0
    finite = finite_scores(pool)
    if not finite:
        return 1.0
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span <= 0:
        if score > hi:
            return 1.0
        if score < lo:
            return 0.0
        return 1.0
    return float((score - lo) / span)


def best_rel_score(pool: List[float]) -> float:
    finite = finite_scores(pool)
    if not finite:
        return 0.0
    return max(rel_score(s, finite) for s in finite)


def delta_vs_best_rel(perf: float, prior_perfs: List[float]) -> tuple[float, float, float]:
    """Return (rel_on_prior_scale, best_rel_before, delta_best)."""
    prior = finite_scores(prior_perfs)
    if not np.isfinite(perf):
        best_before = best_rel_score(prior)
        return 0.0, best_before, -best_before

    if not prior:
        return 1.0, 0.0, 1.0

    lo, hi = min(prior), max(prior)
    span = hi - lo
    best_before = best_rel_score(prior)
    if span <= 0:
        if perf > hi:
            return 1.0, best_before, 1.0
        if perf < lo:
            return 0.0, best_before, -best_before
        return 1.0, best_before, 0.0

    rel_on_prior = float((perf - lo) / span)
    return rel_on_prior, best_before, rel_on_prior - best_before

@dataclass
class RewardWeights:
    delta_best: float = 1.0
    absolute: float = 0.1
    parse_fail: float = 0.5
    duplicate: float = 0.1
    eval_fail: float = 1.0


@dataclass
class TrialReward:
    perf: float
    eval_ok: bool
    parse_ok: bool
    duplicate: bool
    rel_score: float
    best_rel_before: float
    delta_best: float
    reward: float
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perf": self.perf,
            "eval_ok": self.eval_ok,
            "parse_ok": self.parse_ok,
            "duplicate": self.duplicate,
            "rel_score": self.rel_score,
            "best_rel_before": self.best_rel_before,
            "delta_best": self.delta_best,
            "reward": self.reward,
            "reward_components": dict(self.components),
        }


def compute_trial_reward(
    *,
    perf: float,
    prior_perfs: List[float],
    parse_ok: bool,
    duplicate: bool,
    eval_ok: bool,
    weights: Optional[RewardWeights] = None,
) -> TrialReward:
    """Primary signal: improvement vs best rel_score seen before this trial.

    ``rel_score`` for the absolute term uses min-max over prior + current trial.
    ``delta_best`` uses the prior-only scale (what the model saw in the prompt).
    """
    w = weights or RewardWeights()
    prior = finite_scores(prior_perfs)

    rel_on_prior, best_before, delta = delta_vs_best_rel(perf if eval_ok else float("-inf"), prior)
    if not eval_ok:
        rel_on_prior = 0.0
    rel_with_trial = rel_score(perf, prior + [perf]) if eval_ok else 0.0

    components = {
        "delta_best": w.delta_best * delta,
        "absolute": w.absolute * rel_with_trial if eval_ok else 0.0,
        "parse_fail": -w.parse_fail if not parse_ok else 0.0,
        "duplicate": -w.duplicate if duplicate else 0.0,
        "eval_fail": -w.eval_fail if not eval_ok else 0.0,
    }
    reward = float(sum(components.values()))

    return TrialReward(
        perf=float(perf),
        eval_ok=eval_ok,
        parse_ok=parse_ok,
        duplicate=duplicate,
        rel_score=rel_with_trial,
        best_rel_before=best_before,
        delta_best=delta,
        reward=reward,
        components=components,
    )
