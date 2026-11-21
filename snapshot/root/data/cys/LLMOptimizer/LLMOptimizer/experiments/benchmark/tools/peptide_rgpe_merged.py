"""Exact independent RGPE posterior: tiled source covariances, one joint draw.

Source posteriors must still be evaluated. Only the merged posterior is
factorized for acquisition; ranking and target fitting are unchanged.
"""
import numbers
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from peptide_rgpe_posterior import full_posterior

VERSION = 'tiled-independent-rgpe-v1'
BLOCK = 256


def validate_covariance(covariance):
    n = len(covariance)
    if covariance.shape != (n, n) or not n:
        raise ValueError('Invalid covariance shape')
    # Avoid N x N boolean and difference temporaries during validation.
    for start in range(0, n, BLOCK):
        row = covariance[start:start+BLOCK]
        if not np.isfinite(row).all() or not np.allclose(
                row, covariance[:, start:start+BLOCK].T, rtol=1e-8, atol=1e-10):
            raise ValueError('Nonfinite or asymmetric covariance')
    magnitude = max(1., float(np.max(np.abs(np.diag(covariance)))))
    if np.any(np.diag(covariance) < -1e-10*magnitude):
        raise ValueError('Negative posterior variance')
    return magnitude


def _add_source(model, x, weight, mean, covariance, source_output):
    if source_output != 'standardized':
        source_mean, source_covariance = full_posterior(model, x, source_output)
        source_mean = np.asarray(source_mean, dtype=np.float64).reshape(-1)
        source_covariance = np.asarray(source_covariance, dtype=np.float64)
        if source_mean.shape != mean.shape or not np.isfinite(source_mean).all():
            raise ValueError('Invalid source mean')
        if source_covariance.shape != covariance.shape:
            raise ValueError('Invalid source covariance')
        validate_covariance(source_covariance)
        mean += weight*source_mean
        for start in range(0, len(x), BLOCK):
            covariance[start:start+BLOCK] += weight**2*source_covariance[start:start+BLOCK]
        return 'full_fallback'
    # This path mirrors the recipe-pinned skopt GPR.predict(return_cov=True).
    # The SMAC wrapper normalizes labels; its underlying GPR does not.
    gp = getattr(model, 'gp', None)
    if (gp is None or not model.normalize_y or gp.normalize_y
            or np.asarray(gp.alpha_).ndim != 1
            or float(gp.y_train_std_) != 1. or float(gp.y_train_mean_) != 0.):
        raise ValueError('Tiled acquisition requires the pinned standardized scalar skopt GP')
    locations = model._impute_inactive(x)
    n, trained = len(x), len(gp.X_train_)
    projected = np.empty((trained, n), dtype=np.float64, order='F')
    for start in range(0, n, BLOCK):
        stop = min(n, start+BLOCK)
        cross = gp.kernel_(locations[start:stop], gp.X_train_)
        source_mean = cross.dot(gp.alpha_)
        if source_mean.shape != (stop-start,) or not np.isfinite(source_mean).all():
            raise ValueError('Invalid source mean')
        mean[start:stop] += weight*source_mean
        projected[:, start:stop] = solve_triangular(gp.L_, cross.T, lower=True,
                                                   check_finite=True)
    # K_t(X,X) - (L_t^-1 K_t(train,X)).T (L_t^-1 K_t(train,X)).
    # Diagonal kernel blocks use Y=None to retain WhiteKernel's diagonal.
    # Off-diagonal blocks use explicit Y, as in the full kernel matrix.
    for start in range(0, n, BLOCK):
        stop = min(n, start+BLOCK)
        for other in range(0, start+1, BLOCK):
            end = min(n, other+BLOCK)
            tile = (gp.kernel_(locations[start:stop]) if start == other else
                    gp.kernel_(locations[start:stop], locations[other:end]))
            tile -= projected[:, start:stop].T.dot(projected[:, other:end])
            if not np.isfinite(tile).all():
                raise ValueError('Invalid source covariance block')
            if start == other:
                validate_covariance(tile)
            tile *= weight**2
            covariance[start:stop, other:end] += tile
            if other != start:
                covariance[other:end, start:stop] += tile.T
    return 'tiled_exact'


def merge_posterior(ensemble, candidates, source_output='standardized'):
    if ensemble.variance_mode != 'average':
        raise ValueError('Independent merging requires average variance mode')
    if source_output not in ('ensemble', 'standardized'):
        raise ValueError('Unknown source output units')
    x = np.asarray(candidates, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Candidates must be a finite nonempty matrix')
    weights = np.asarray(ensemble.weights_, dtype=np.float64)
    total = weights.sum()
    if (weights.shape != (len(ensemble.model_list_),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(total) or total <= 0):
        raise ValueError('Invalid RGPE weights')
    scale, offset = float(ensemble.Y_std_), float(ensemble.Y_mean_)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError('Invalid target output scaling')
    weights = weights/total
    mean = np.zeros(len(x), dtype=np.float64)
    covariance = np.zeros((len(x), len(x)), dtype=np.float64, order='F')
    modes = set()
    for index in np.flatnonzero(weights):
        model = ensemble.model_list_[int(index)]
        modes.add(_add_source(model, x, weights[index], mean, covariance, source_output))
        del model
    mean *= scale
    mean += offset
    covariance *= scale**2
    validate_covariance(covariance)
    return mean, covariance, dict(implementation=VERSION, contributing_models=int(np.count_nonzero(weights)),
        candidates=len(x), covariance_bytes=covariance.nbytes, covariance_mode='full',
        independence_assumption=True, source_output=source_output, block_size=BLOCK,
        source_covariance_modes=sorted(modes), retained_source_covariances=0 if modes == {'tiled_exact'} else 1)


def sample_posterior(mean, covariance, count, seed, maximize_negative_mic=True):
    mean, covariance = np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    if (not isinstance(count, numbers.Integral) or isinstance(count, bool) or count < 1
            or mean.ndim != 1 or not len(mean) or not np.isfinite(mean).all()
            or covariance.shape != (len(mean), len(mean))):
        raise ValueError('Invalid merged posterior or sample count')
    magnitude = validate_covariance(covariance)
    attempts = 0
    for relative in (0., 1e-12, 1e-10, 1e-8):
        attempts += 1
        factor = np.array(covariance, dtype=np.float64, order='F', copy=True)
        jitter = relative*magnitude
        factor.flat[::len(mean)+1] += jitter
        try:
            factor = cholesky(factor, lower=True, overwrite_a=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            del factor
    else:
        raise ValueError('Merged covariance is not positive semidefinite within numerical tolerance')
    draws = np.random.RandomState(seed).normal(size=(count, len(mean))).dot(factor.T)
    draws += mean
    if maximize_negative_mic:
        draws *= -1
    return draws, dict(samples=int(count), seed=int(seed), covariance_jitter=jitter,
        covariance_factorizations=1, cholesky_attempts=attempts,
        score_direction='negative_MIC' if maximize_negative_mic else 'MIC')


def sample_ensemble(ensemble, candidates, count, seed):
    started = time.monotonic()
    mean, covariance, info = merge_posterior(ensemble, candidates)
    merged_at = time.monotonic()
    draws, sampling = sample_posterior(mean, covariance, count, seed)
    info.update(sampling, method='rgpe', posterior='merged_independent_GP', higher_is_better=True,
        merge_seconds=merged_at-started, sample_seconds=time.monotonic()-merged_at)
    return dict(draws=draws, info=info)
