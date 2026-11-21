"""Optional TabPFN dataset embedding. Missing install is a no-op."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

_TABPFN_ERR: Optional[str] = None


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401

        return True
    except Exception as exc:  # pragma: no cover
        global _TABPFN_ERR
        _TABPFN_ERR = str(exc)
        return False


def embed_dataset_xy(
    X,
    y,
    *,
    max_samples: int = 256,
) -> Optional[List[float]]:
    """Return a TabPFN embedding if the package is installed, else None.

    Uses a small row subsample so bank-build stays cheap. The cascade already
    has a 6-d profile vector as the always-on meta-feature fallback.
    """
    if not tabpfn_available():
        return None
    try:
        import numpy as np
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except Exception:
        return None

    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    if n == 0:
        return None
    if n > max_samples:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    # Heuristic: integer labels with few uniques → classifier
    unique = np.unique(y)
    try:
        if unique.size <= max(20, int(0.05 * len(y))) and np.allclose(unique, unique.astype(int)):
            model = TabPFNClassifier()
        else:
            model = TabPFNRegressor()
        if hasattr(model, "get_embeddings"):
            emb = model.get_embeddings(X, y)
        else:
            return None
        vec = np.asarray(emb).reshape(-1)
        if vec.size == 0:
            return None
        # Average over rows if 2d
        if vec.ndim > 1:
            vec = vec.mean(axis=0)
        return [float(x) for x in vec.tolist()[:64]]
    except Exception:
        return None


def last_import_error() -> Optional[str]:
    return _TABPFN_ERR
