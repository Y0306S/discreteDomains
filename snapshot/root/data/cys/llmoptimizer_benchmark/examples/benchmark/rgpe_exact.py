"""SMAC GP likelihood gradients with bounded, two-dimensional working arrays.

The kernel formulas, including upstream categorical/conditional conventions,
are preserved. No training observations or optimization restarts are removed.
"""

import contextlib

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import pdist, squareform
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_kernels import ConstantKernel, HammingKernel, Matern, Product, Sum, WhiteKernel

ENGINE = "exact_streamed_kernel_gradients_v1"


def _conditional_mask(x):
    mask = np.ones((len(x), len(x)), dtype=bool)
    for column in x.T:
        inactive = column <= -1
        mask &= inactive[:, None] == inactive[None, :]
    return mask


def _length_scales(kernel, dimensions):
    values = np.asarray(kernel.length_scale, dtype=np.float64)
    if values.ndim > 1 or (values.ndim == 1 and len(values) not in (1, dimensions)):
        raise ValueError("Kernel length scales do not match its input dimensions")
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("Invalid kernel length scale")
    return np.broadcast_to(values, (dimensions,))


def _kernel_tree(kernel, x, active=None):
    selected = x if kernel.operate_on is None else x[:, kernel.operate_on]
    if active is None and kernel.has_conditions:
        active = _conditional_mask(selected)
    node = dict(kernel=kernel, x=selected, active=active)
    if isinstance(kernel, (Sum, Product)):
        left = _kernel_tree(kernel.k1, selected, active)
        right = _kernel_tree(kernel.k2, selected, active)
        node["children"] = (left, right)
        node["covariance"] = (left["covariance"] + right["covariance"]
                              if isinstance(kernel, Sum)
                              else left["covariance"] * right["covariance"])
    elif isinstance(kernel, ConstantKernel):
        node["covariance"] = np.full((len(x), len(x)), kernel.constant_value)
    elif isinstance(kernel, WhiteKernel):
        node["covariance"] = np.eye(len(x)) * kernel.noise_level
        if active is not None:
            node["covariance"] *= active
    elif isinstance(kernel, Matern):
        if kernel.nu != 2.5:
            raise ValueError("Streamed RGPE gradients currently require Matern nu=2.5")
        scales = _length_scales(kernel, selected.shape[1])
        radial = squareform(pdist(selected / scales)) * np.sqrt(5.0)
        node["covariance"] = (1 + radial + radial ** 2 / 3.0) * np.exp(-radial)
        np.fill_diagonal(node["covariance"], 1.0)
        if active is not None:
            node["covariance"] *= active
    elif isinstance(kernel, HammingKernel):
        scales = _length_scales(kernel, selected.shape[1])
        exponent = np.zeros((len(x), len(x)))
        for column, scale in zip(selected.T, scales):
            exponent += (-1 / (2 * scale ** 2)) * (column[:, None] != column[None, :])
        node["covariance"] = np.exp(exponent)
        if active is not None:
            node["covariance"] *= active
    else:
        raise ValueError("Unsupported kernel for streamed RGPE gradients: " + type(kernel).__name__)
    return node


def _contract(node, weight):
    kernel, x = node["kernel"], node["x"]
    if isinstance(kernel, (Sum, Product)):
        left, right = node["children"]
        if isinstance(kernel, Product):
            left_weight = weight * right["covariance"]
            yield from _contract(left, left_weight)
            del left_weight
            right_weight = weight * left["covariance"]
            yield from _contract(right, right_weight)
        else:
            yield from _contract(left, weight)
            yield from _contract(right, weight)
    elif isinstance(kernel, ConstantKernel):
        if not kernel.hyperparameter_constant_value.fixed:
            yield float(np.sum(weight) * kernel.constant_value)
    elif isinstance(kernel, WhiteKernel):
        if not kernel.hyperparameter_noise_level.fixed:
            yield float(np.trace(weight) * kernel.noise_level)
    elif isinstance(kernel, Matern):
        if kernel.hyperparameter_length_scale.fixed:
            return
        scales = _length_scales(kernel, x.shape[1])
        squared_radius = np.zeros((len(x), len(x)))
        for column, scale in zip(x.T, scales):
            squared_radius += ((column[:, None] - column[None, :]) / scale) ** 2
        radial = np.sqrt(5 * squared_radius)
        coefficient = (5.0 / 3.0) * (radial + 1) * np.exp(-radial)
        # Upstream masks the conditional covariance but leaves this leaf's
        # Matern derivatives unmasked. Preserve that convention for parity.
        contracted = weight * coefficient
        isotropic = 0.0
        for column, scale in zip(x.T, scales):
            differences = ((column[:, None] - column[None, :]) / scale) ** 2
            value = float(np.sum(contracted * differences))
            if kernel.anisotropic:
                yield value
            else:
                isotropic += value
        if not kernel.anisotropic:
            yield isotropic
    elif isinstance(kernel, HammingKernel):
        if kernel.hyperparameter_length_scale.fixed:
            return
        scales = _length_scales(kernel, x.shape[1])
        contracted = weight * node["covariance"]
        anisotropic = np.ndim(kernel.length_scale) > 0 and len(kernel.length_scale) > 1
        isotropic = 0.0
        for column, scale in zip(x.T, scales):
            mismatch = column[:, None] != column[None, :]
            # Retain SMAC's existing 1/l^3 gradient convention.
            value = float(np.sum(contracted * mismatch) * (1 / scale ** 3))
            if anisotropic:
                yield value
            else:
                isotropic += value
        if not anisotropic:
            yield isotropic


def marginal_likelihood(gp, theta):
    x = np.asarray(gp.X_train_, dtype=np.float64)
    y = np.asarray(gp.y_train_, dtype=np.float64)
    if y.ndim != 1 or gp.alpha != 0 or gp.normalize_y:
        raise ValueError("Expected a scalar-output SMAC GP with kernel-governed noise")
    kernel = gp.kernel_.clone_with_theta(theta)
    tree = _kernel_tree(kernel, x)
    try:
        factor = cholesky(tree["covariance"], lower=True)
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros_like(theta)
    alpha = cho_solve((factor, True), y)
    value = (-0.5 * np.dot(y, alpha) - np.log(np.diag(factor)).sum()
             - len(x) / 2.0 * np.log(2 * np.pi))
    sensitivity = np.outer(alpha, alpha)
    sensitivity -= cho_solve((factor, True), np.eye(len(x)))
    sensitivity *= 0.5
    gradient = np.asarray(list(_contract(tree, sensitivity)))
    if gradient.shape != np.asarray(theta).shape:
        raise ValueError("Kernel gradient ordering/dimensions differ from theta")
    return value, gradient


def efficient_nll(model, theta):
    model._n_ll_evals += 1
    try:
        value, gradient = marginal_likelihood(model.gp, theta)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros(theta.shape)
    for dimension, priors in enumerate(model._all_priors):
        for prior in priors:
            value += prior.lnprob(theta[dimension])
            gradient[dimension] += prior.gradient(theta[dimension])
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return 1e25, np.zeros(theta.shape)
    return -value, -gradient


@contextlib.contextmanager
def exact_gp_fitting():
    """Use the bounded-memory objective for this isolated worker's GP fits."""
    previous = GaussianProcess._nll
    if previous is efficient_nll:
        raise ValueError("Streamed GP fitting is already active")
    GaussianProcess._nll = efficient_nll
    try:
        yield
    finally:
        GaussianProcess._nll = previous

