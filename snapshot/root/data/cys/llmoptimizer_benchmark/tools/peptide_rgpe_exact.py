"""Exact SMAC Matern-5/2 marginal gradients without an N*N*D tensor."""

import contextlib
import time
from types import MethodType

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import pdist, squareform


def kernel_parts(kernel, dimensions):
    from smac.epm.gp_kernels import ConstantKernel, Matern, Product, Sum, WhiteKernel
    if (not isinstance(kernel, Sum) or not isinstance(kernel.k1, Product)
            or not isinstance(kernel.k1.k1, ConstantKernel)
            or not isinstance(kernel.k1.k2, Matern)
            or not isinstance(kernel.k2, WhiteKernel)):
        raise ValueError('Exact contraction supports Constant * Matern + White only')
    amplitude, matern, noise = kernel.k1.k1, kernel.k1.k2, kernel.k2
    if (matern.nu != 2.5 or np.asarray(matern.length_scale).shape != (dimensions,)
            or len(kernel.theta) != dimensions + 2
            or any(getattr(k, 'has_conditions', False) for k in (kernel, kernel.k1, amplitude, matern, noise))
            or not np.array_equal(matern.operate_on, np.arange(dimensions))):
        raise ValueError('Exact contraction requires unconditional continuous ARD Matern-5/2')
    return amplitude, matern, noise


def marginal_likelihood(gp, theta):
    x = np.asarray(gp.X_train_, dtype=np.float64)
    y = np.asarray(gp.y_train_, dtype=np.float64)
    if y.ndim != 1 or gp.alpha != 0 or gp.normalize_y:
        raise ValueError('Expected a scalar-output SMAC GP with kernel-governed noise')
    kernel = gp.kernel_.clone_with_theta(theta)
    amplitude, matern, noise = kernel_parts(kernel, x.shape[1])
    covariance = kernel(x)
    try:
        factor = cholesky(covariance, lower=True, overwrite_a=True)
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros_like(theta)
    alpha = cho_solve((factor, True), y)
    value = -0.5 * np.dot(y, alpha) - np.log(np.diag(factor)).sum() - len(x) / 2.0 * np.log(2 * np.pi)
    inverse = cho_solve((factor, True), np.eye(len(x)))
    sensitivity = np.outer(alpha, alpha)
    sensitivity -= inverse
    del inverse, factor, covariance

    distances = squareform(pdist(x / matern.length_scale))
    radial = distances * np.sqrt(5.0)
    exponential = np.exp(-radial)
    gradient = np.empty(len(theta), dtype=np.float64)
    signal = (1.0 + radial + radial ** 2 / 3.0) * exponential
    gradient[0] = 0.5 * amplitude.constant_value * np.einsum('ij,ij->', sensitivity, signal)
    gradient[-1] = 0.5 * noise.noise_level * np.trace(sensitivity)
    del signal, distances

    # Contract squared coordinate differences through a matrix product. The
    # diagonal derivative is zero; removing it also avoids cancellation noise.
    sensitivity *= (5.0 / 3.0) * amplitude.constant_value * (1.0 + radial) * exponential
    np.fill_diagonal(sensitivity, 0.0)
    coordinate_squares = x * x
    edge_sums = sensitivity.sum(axis=0) + sensitivity.sum(axis=1)
    gradient[1:-1] = (0.5 * np.dot(edge_sums, coordinate_squares)
                       - np.einsum('ij,ij->j', x, sensitivity.dot(x))) / (matern.length_scale ** 2)
    return value, gradient


def efficient_nll(model, theta):
    model._n_ll_evals += 1
    started = time.monotonic()
    try:
        value, gradient = marginal_likelihood(model.gp, theta)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros(theta.shape)
    for dimension, priors in enumerate(model._all_priors):
        for prior in priors:
            value += prior.lnprob(theta[dimension])
            gradient[dimension] += prior.gradient(theta[dimension])
    state = model._exact_contraction_state
    state['evaluations'] += 1
    state['seconds'] += time.monotonic() - started
    if state['callback'] is not None and state['evaluations'] % 25 == 0:
        state['callback'](dict(evaluations=state['evaluations'], nll_seconds=state['seconds']))
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return 1e25, np.zeros(theta.shape)
    return -value, -gradient


@contextlib.contextmanager
def exact_objective(model, callback=None):
    kernel_parts(model.kernel, len(model.types))
    if '_nll' in model.__dict__ or '_exact_contraction_state' in model.__dict__:
        raise ValueError('Model already has an overridden marginal objective')
    state = dict(evaluations=0, seconds=0.0, callback=callback)
    model._exact_contraction_state = state
    model._nll = MethodType(efficient_nll, model)
    try:
        yield state
    finally:
        del model._nll
        del model._exact_contraction_state
