#!/usr/bin/env python3
"""Compare compact exact derivatives with upstream gradients and full-size memory."""

import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

os.sched_setaffinity(0, {18})
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, str(TOOLS))
import numpy as np
import peptide_transfer as transfer
from peptide_rgpe_exact import marginal_likelihood, exact_objective
sys.path.insert(0, str(transfer.RGPE_REPO))
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.epm.util_funcs import get_types
from rgpe.utils import get_gaussian_process

ROOT = Path(__file__).resolve().parent
CORPUS = Path('/root/data/logs/peptide_transfer_train800_20261027_v2/corpus')


def model(dimensions, seed=123):
    cs = ConfigurationSpace()
    cs.add_hyperparameters([UniformFloatHyperparameter('z%03d' % i, 0, 1) for i in range(dimensions)])
    types, bounds = get_types(cs, None)
    return get_gaussian_process(cs, types, bounds, np.random.RandomState(seed), None)


def main():
    manifest, db = transfer.load_corpus(CORPUS)
    data = transfer.read_task(db, 'pep_002')
    db.close()
    x, y = transfer.normalize(data['X'], manifest).astype(np.float64), -data['y']
    small = model(256)
    small._train(x[:32], y[:32], do_optimize=False)
    rng = np.random.RandomState(73)
    cases = []
    for index in range(6):
        theta = small.gp.kernel_.theta.copy()
        theta[0] += rng.uniform(-1, 0.5)
        theta[1:-1] += rng.uniform(-0.5, 0.05, 256)
        theta[-1] = np.log(0.001 if index % 2 else 0.05)
        expected, gradient = small.gp.log_marginal_likelihood(theta, eval_gradient=True)
        actual, compact = marginal_likelihood(small.gp, theta)
        assert np.isclose(actual, expected, rtol=1e-11, atol=1e-9), (actual, expected)
        assert np.allclose(compact, gradient, rtol=2e-7, atol=2e-8), float(np.max(np.abs(compact-gradient)))
        small._all_priors = small._get_all_priors(add_bound_priors=False)
        expected_nll, expected_grad = small._nll(theta)
        with exact_objective(small):
            actual_nll, actual_grad = small._nll(theta)
        assert np.isclose(actual_nll, expected_nll, rtol=1e-11, atol=1e-8)
        assert np.allclose(actual_grad, expected_grad, rtol=2e-7, atol=2e-8)
        cases.append(dict(case=index, value_error=abs(actual-expected), gradient_max_error=float(np.max(np.abs(compact-gradient)))))
    # End-to-end comparison retains all ten random optimization restarts.
    sx = rng.uniform(0, 1, (24, 6)); sy = np.sin(sx[:,0]*4) + sx[:,1]**2
    original, optimized = model(6, seed=33), model(6, seed=33)
    original.train(sx, sy)
    with exact_objective(optimized):
        optimized.train(sx, sy)
    probe = rng.uniform(0, 1, (20, 6))
    first, second = original.predict(probe), optimized.predict(probe)
    assert np.allclose(first[0], second[0], rtol=2e-4, atol=2e-5)
    assert np.allclose(first[1], second[1], rtol=5e-4, atol=2e-5)
    restored = pickle.loads(pickle.dumps(optimized, protocol=4))
    assert '_nll' not in restored.__dict__ and '_exact_contraction_state' not in restored.__dict__
    assert np.allclose(restored.predict(probe)[0], second[0])
    print(json.dumps(dict(small_gradient_and_fit_checks=True, cases=cases)), flush=True)
    # Full-size derivative probe only; no new oracle data or production models.
    full = model(256)
    started = time.monotonic()
    full._train(x, y, do_optimize=False)
    theta = full.gp.kernel_.theta.copy()
    value, gradient = marginal_likelihood(full.gp, theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    assert peak < 8, peak
    report = dict(passed=True, real_source='pep_002', source_observations=len(x), dimensions=256,
        gradient_cases=cases, full_size_seconds=time.monotonic()-started, peak_rss_gib=peak,
        upstream_kernel_priors_restarts_preserved=True, end_to_end_posterior_matches=True,
        ten_optimizer_restarts=True, pickle_compatible_with_upstream=True, oracle_calls=0,
        objective_sha256=transfer.digest(TOOLS/'peptide_rgpe_exact.py'),
        runner_sha256=transfer.digest(TOOLS/'peptide_rgpe_fit.py'))
    transfer.atomic_json(ROOT/'exact_validation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
