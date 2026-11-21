"""Fit a tiny linear scorer: cold features → observed Spearman transfer."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from mindware.components.experience.retrieve import (
    DEFAULT_WEIGHTS,
    ExperienceBank,
    cold_features,
    spearman_transfer,
)


FEATURE_NAMES = ("importance_cos", "algo_match", "profile_cos", "role_jaccard")


def _query_from_experience(exp: Dict[str, Any]) -> Dict[str, Any]:
    ret = exp.get("retrieval") or {}
    prob = exp.get("problem") or {}
    flt = (exp.get("delivery") or {}).get("filters") or {}
    return {
        "dataset_id": prob.get("dataset_id"),
        "algorithm_id": prob.get("algorithm_id"),
        "filters": {
            "domain": flt.get("domain"),
            "task_type": flt.get("task_type"),
            "representation": flt.get("representation") or "structured_config",
        },
        "importance_by_role": ret.get("importance_by_role") or {},
        "roles": ret.get("roles") or [],
        "profile_vec": ret.get("profile_vec") or {},
    }


def pair_labels(
    bank: ExperienceBank,
    *,
    max_pairs: int = 400,
    seed: int = 0,
    same_algorithm_frac: float = 0.5,
) -> Tuple[List[List[float]], List[float], List[Dict[str, Any]]]:
    """Pairs across datasets. Mix same-algorithm and cross-algorithm sources."""
    rng = random.Random(seed)
    usable = [
        exp
        for exp in bank.experiences
        if len((exp.get("retrieval") or {}).get("trials") or []) >= 8
    ]
    by_algo: Dict[str, List[Dict[str, Any]]] = {}
    for exp in usable:
        algo = str(((exp.get("problem") or {}).get("algorithm_id") or "")).lower()
        by_algo.setdefault(algo, []).append(exp)

    X: List[List[float]] = []
    y: List[float] = []
    meta: List[Dict[str, Any]] = []
    n_same = int(max_pairs * same_algorithm_frac)
    n_cross = max_pairs - n_same

    def dataset_id(exp: Dict[str, Any]) -> str:
        return str(((exp.get("problem") or {}).get("dataset_id") or "")).lower()

    def take_pair(src, tgt):
        tgt_trials = (tgt.get("retrieval") or {}).get("trials") or []
        rho = spearman_transfer(src, tgt_trials)
        q = _query_from_experience(tgt)
        feats = cold_features(q, src)
        X.append([float(feats[k]) for k in FEATURE_NAMES])
        y.append(float(rho))
        meta.append({"src": src.get("id"), "tgt": tgt.get("id"), "rho": rho, "features": feats})

    # same-algorithm, different dataset
    same_cands: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for group in by_algo.values():
        if len(group) < 2:
            continue
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if dataset_id(a) == dataset_id(b):
                    continue
                same_cands.append((a, b))
    rng.shuffle(same_cands)
    for src, tgt in same_cands[:n_same]:
        take_pair(src, tgt)

    # cross-algorithm
    cross_cands: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    algos = [a for a, g in by_algo.items() if g]
    if len(algos) >= 2:
        for _ in range(n_cross * 4):
            a1, a2 = rng.sample(algos, 2)
            src = rng.choice(by_algo[a1])
            tgt = rng.choice(by_algo[a2])
            if dataset_id(src) == dataset_id(tgt):
                continue
            cross_cands.append((src, tgt))
            if len(cross_cands) >= n_cross:
                break
    for src, tgt in cross_cands[:n_cross]:
        take_pair(src, tgt)
    return X, y, meta


def fit_linear_weights(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    ridge: float = 1e-2,
) -> Dict[str, float]:
    """Non-negative ridge regression, then L1-normalize to a weight simplex."""
    try:
        import numpy as np
    except ImportError:
        return dict(DEFAULT_WEIGHTS)

    if len(X) < 8:
        return dict(DEFAULT_WEIGHTS)
    A = np.asarray(X, dtype=float)
    b = np.asarray(y, dtype=float)
    # Shift labels to [0, 1] so negative Spearman is "do not retrieve"
    b = np.clip(b, 0.0, 1.0)
    d = A.shape[1]
    gram = A.T @ A + ridge * np.eye(d)
    w = np.linalg.solve(gram, A.T @ b)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 1e-12:
        return dict(DEFAULT_WEIGHTS)
    w = w / s
    return {name: float(val) for name, val in zip(FEATURE_NAMES, w.tolist())}


def fit_bank_weights(bank: ExperienceBank, *, max_pairs: int = 400, seed: int = 0) -> Dict[str, float]:
    X, y, _ = pair_labels(bank, max_pairs=max_pairs, seed=seed)
    weights = fit_linear_weights(X, y)
    bank.weights = {**DEFAULT_WEIGHTS, **weights}
    return bank.weights


def save_weights(weights: Dict[str, float], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
