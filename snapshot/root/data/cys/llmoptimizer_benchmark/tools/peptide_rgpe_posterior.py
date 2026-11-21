#!/usr/bin/env python3
"""Stream an RGPE full posterior into one covariance for correlated batch draws."""

import numpy as np
from scipy.linalg import cholesky


def full_posterior(model, candidates, source_output='ensemble'):
    if source_output not in ('ensemble', 'standardized'):
        raise ValueError('Specify ensemble or standardized source output units')
    if hasattr(model, 'gp') and hasattr(model.gp, 'predict'):
        x = model._impute_inactive(np.asarray(candidates, dtype=np.float64))
        mean, covariance = model.gp.predict(x, return_cov=True)
        if source_output == 'ensemble' and model.normalize_y:
            mean, covariance = model._untransform_y(mean, covariance)
        elif source_output == 'standardized' and not model.normalize_y:
            raise ValueError('Standardized bank predictions require a normalized source GP')
    elif source_output == 'ensemble':
        mean, covariance = model._predict(candidates, cov_return_type='full_cov')
    else:
        raise ValueError('Cannot extract standardized predictions from this model')
    # SMAC clips every covariance entry in _predict, including valid negative
    # correlations. Use the underlying GP and retain its signed covariance.
    return np.asarray(mean, dtype=np.float64).reshape(-1), np.asarray(covariance, dtype=np.float64)


def merge_posterior(ensemble, candidates, source_output='ensemble'):
    if ensemble.variance_mode != 'average':
        raise ValueError('Weighted independent-GP merging requires average variance mode')
    x = np.asarray(candidates, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Candidates must be a finite nonempty matrix')
    weights = np.asarray(ensemble.weights_, dtype=np.float64)
    if (weights.shape != (len(ensemble.model_list_),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Invalid RGPE weights')
    weights = weights / weights.sum()
    mean = np.zeros(len(x), dtype=np.float64)
    covariance = np.zeros((len(x), len(x)), dtype=np.float64)
    contributing = 0
    # Retain one source covariance at a time instead of stacking T dense matrices.
    for index in np.flatnonzero(weights):
        weight = weights[index]
        model = ensemble.model_list_[int(index)]
        source_mean, source_covariance = full_posterior(model, x, source_output=source_output)
        source_mean = np.asarray(source_mean, dtype=np.float64).reshape(-1)
        source_covariance = np.array(source_covariance, dtype=np.float64, copy=True)
        if (source_mean.shape != mean.shape or source_covariance.shape != covariance.shape
                or not np.isfinite(source_mean).all() or not np.isfinite(source_covariance).all()):
            raise ValueError('Invalid source full posterior')
        if not np.allclose(source_covariance, source_covariance.T, rtol=1e-8, atol=1e-10):
            raise ValueError('Asymmetric source covariance')
        mean += weight * source_mean
        source_covariance *= weight ** 2
        covariance += source_covariance
        del model, source_mean, source_covariance
        contributing += 1
    scale, offset = float(ensemble.Y_std_), float(ensemble.Y_mean_)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError('Invalid RGPE target output scaling')
    mean = mean * scale + offset
    covariance *= scale ** 2
    return mean, covariance, dict(contributing_models=contributing,
                                  candidates=len(x), covariance_bytes=covariance.nbytes,
                                  covariance_mode='full', independence_assumption=True,
                                  source_output=source_output)


def sample_rank_posterior(model, locations, num_samples, engine_seed):
    from botorch.sampling.qmc import NormalQMCEngine
    mean, covariance = full_posterior(model, locations)
    magnitude = max(1.0, float(np.max(np.abs(np.diag(covariance)))))
    for relative_jitter in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            stabilized = covariance.copy()
            stabilized.flat[::len(mean)+1] += relative_jitter*magnitude
            factor = cholesky(stabilized, lower=True, check_finite=True)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        raise ValueError('Ranking posterior covariance is not numerically positive semidefinite')
    engine = NormalQMCEngine(len(mean), seed=engine_seed)
    # A lower Cholesky factor gives K = L L.T, so row-wise draws use L.T.
    return mean + engine.draw(num_samples).numpy().dot(factor.T)


def sample_posterior(mean, covariance, count, seed, maximize_negative_mic=True):
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if (count < 1 or mean.ndim != 1 or not len(mean) or covariance.shape != (len(mean), len(mean))
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10)):
        raise ValueError('Invalid merged posterior or sample count')
    magnitude = max(1.0, float(np.max(np.abs(np.diag(covariance)))))
    if np.any(np.diag(covariance) < -1e-10 * magnitude):
        raise ValueError('Negative posterior variance')
    factor, jitter = None, 0.0
    for relative_jitter in (0.0, 1e-12, 1e-10, 1e-8):
        jitter = relative_jitter * magnitude
        stabilized = covariance.copy()
        stabilized.flat[::len(mean) + 1] += jitter
        try:
            factor = cholesky(stabilized, lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            continue
    if factor is None:
        raise ValueError('Merged covariance is not positive semidefinite within numerical tolerance')
    draws = np.random.RandomState(seed).normal(size=(count, len(mean))).dot(factor.T) + mean
    if maximize_negative_mic:
        draws = -draws
    return draws, dict(samples=count, seed=seed, covariance_jitter=jitter,
                       score_direction='negative_MIC' if maximize_negative_mic else 'MIC')
