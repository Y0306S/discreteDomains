"""Deterministic target-only expected improvement for MIC minimization."""
import time
import numpy as np
from scipy.special import ndtr


def shortlist(target, candidates, best_mic, mean, scale, size=200):
    started = time.monotonic()
    x = np.asarray(candidates, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not 0 < size <= len(x):
        raise ValueError('Invalid screening pool')
    if not np.isfinite(best_mic) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid target output units')
    # SMAC fits the underlying skopt GP on standardized MIC. Avoid loading any
    # source GP, and compute only marginal variances, never a 1500x1500 matrix.
    mus, stds = [], []
    for start in range(0, len(x), 128):
        mu, std = target.gp.predict(x[start:start+128], return_std=True)
        mus.extend(np.asarray(mu).reshape(-1))
        stds.extend(np.asarray(std).reshape(-1))
    mu, std = np.asarray(mus), np.asarray(stds)
    if not np.isfinite(mu).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError('Invalid target marginal predictions')
    improvement = (float(best_mic)-mean)/scale-mu
    z = np.divide(improvement, std, out=np.zeros_like(std), where=std>0)
    ei = np.where(std>0, improvement*ndtr(z)+std*np.exp(-z*z/2)/np.sqrt(2*np.pi),
                  np.maximum(improvement, 0.))
    ei = np.maximum(ei, 0.)
    indices = np.lexsort((np.arange(len(x)), -ei))[:size]
    return dict(indices=indices, info=dict(method='target_GP_expected_improvement',
        direction='minimize_MIC', candidates=len(x), shortlisted=size,
        seconds=time.monotonic()-started, best_mic=float(best_mic),
        ei_max=float(ei.max()), ei_min_selected=float(ei[indices].min()),
        source_models_evaluated=0))
