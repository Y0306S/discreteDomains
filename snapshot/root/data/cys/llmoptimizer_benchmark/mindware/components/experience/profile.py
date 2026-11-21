"""Numeric dataset fingerprints for cold-start similarity (Layer A)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

PROFILE_KEYS = (
    "log_n",
    "log_p",
    "n_classes",
    "imbalance",
    "missing_rate",
    "numeric_frac",
)


def _log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))


def profile_from_info_row(row: Dict[str, Any], *, task_type: str) -> Dict[str, float]:
    n = float(row.get("n") or row.get("n_samples") or 0.0)
    p = float(row.get("p") or row.get("n_features") or 0.0)
    n_classes = float(row.get("C") or row.get("n_classes") or (2.0 if task_type == "classification" else 0.0))
    imb = float(row.get("class ratio") or row.get("class_ratio") or row.get("imbalance") or 0.0)
    missing = float(row.get("missing_rate") or 0.0)
    n_numeric = row.get("n_numeric")
    numeric_frac = float(n_numeric) / max(p, 1.0) if n_numeric is not None else 0.5
    return {
        "log_n": _log1p(n),
        "log_p": _log1p(p),
        "n_classes": n_classes,
        "imbalance": imb,
        "missing_rate": missing,
        "numeric_frac": numeric_frac,
    }


def profile_from_dataset_profile(prof: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Compress the optimizer's dataset_profile dict into a fixed numeric vector."""
    if not prof:
        return {k: 0.0 for k in PROFILE_KEYS}
    if all(k in prof for k in ("log_n", "log_p")):
        return {k: float(prof.get(k, 0.0) or 0.0) for k in PROFILE_KEYS}
    n = float(prof.get("n_samples") or 0.0)
    p = float(prof.get("n_features") or 0.0)
    n_numeric = float(prof.get("n_numeric") or 0.0)
    missing = float(prof.get("missing_rate") or 0.0)
    dist = (prof.get("class_distribution") or {}).get("distribution") or {}
    n_classes = float((prof.get("class_distribution") or {}).get("n_classes") or 0.0)
    imb = 0.0
    if dist:
        # minority share among named classes (exclude 'others')
        vals = [float(v) for k, v in dist.items() if k != "others"]
        imb = min(vals) if vals else 0.0
    return {
        "log_n": _log1p(n),
        "log_p": _log1p(p),
        "n_classes": n_classes,
        "imbalance": imb,
        "missing_rate": missing,
        "numeric_frac": n_numeric / max(p, 1.0),
    }


def profile_vector(p: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(float(p.get(k, 0.0) or 0.0) for k in PROFILE_KEYS)


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if len(av) != len(bv) or not av:
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def load_dataset_info_table(repo: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """(task_type, dataset_name) -> profile dict from datasets_bundle/dataset_info."""
    import csv

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    info = repo / "datasets_bundle" / "dataset_info"
    cls_path = info / "classification.csv"
    rgs_path = info / "regression.csv"
    if cls_path.is_file():
        with cls_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                out[("classification", name)] = profile_from_info_row(row, task_type="classification")
                out[("cls", name)] = out[("classification", name)]
    if rgs_path.is_file():
        with rgs_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                out[("regression", name)] = profile_from_info_row(row, task_type="regression")
                out[("rgs", name)] = out[("regression", name)]
    return out
