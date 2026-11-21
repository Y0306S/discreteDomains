#!/usr/bin/env python3
"""Check the merger against upstream RGPE using small, real train800 GP fits."""

import json
import os
from pathlib import Path
import resource
import sqlite3
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
os.sched_setaffinity(0, {12})
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, '/root/data/cys/baseline/repos/transfer-hpo-framework')

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.epm.util_funcs import get_types
from rgpe.utils import get_gaussian_process
from rgpe.methods.rgpe import RGPE
from peptide_rgpe_posterior import merge_posterior, sample_posterior
import peptide_transfer as transfer


def main():
    started = time.monotonic()
    corpus = Path('/root/data/logs/peptide_transfer_train800_20261027_v2/corpus')
    recipe = json.loads((transfer.SOURCE / 'recipe.json').read_text())
    task = recipe['train'][0]
    assert task not in recipe['test']
    db = sqlite3.connect('file:' + str(corpus / 'corpus.sqlite') + '?mode=ro', uri=True)
    data = transfer.read_task(db, task)
    db.close()
    raw = data['X'][:96].astype(np.float64)
    low, high = raw.min(axis=0), raw.max(axis=0)
    x = (raw - low) / np.where(high > low, high - low, 1.0)
    y = -data['y'][:96]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([UniformFloatHyperparameter('z%03d' % i, 0.0, 1.0) for i in range(256)])
    types, bounds = get_types(cs, None)
    models = []
    for index in range(3):
        model = get_gaussian_process(cs, types, bounds, np.random.RandomState(index), None)
        # Only an isolated posterior check: no production source cache is written.
        model._train(x[index * 32:(index + 1) * 32], y[index * 32:(index + 1) * 32], do_optimize=False)
        models.append(model)
    ensemble = RGPE.__new__(RGPE)
    ensemble.model_list_ = models
    ensemble.weights_ = np.array([0.2, 0.3, 0.5])
    ensemble.variance_mode = 'average'
    ensemble.Y_mean_, ensemble.Y_std_ = 12.5, 3.25
    candidates = np.concatenate([x[:32], np.clip(x[:32] + 1e-5, 0, 1)])
    before = [m._predict(candidates, cov_return_type='full_cov') for m in models]
    expected_mean, expected_covariance = ensemble._predict(candidates, cov_return_type='full_cov')
    mean, covariance, merging = merge_posterior(ensemble, candidates)
    assert np.allclose(mean, expected_mean.flatten(), atol=1e-10, rtol=1e-10)
    assert np.allclose(covariance, expected_covariance, atol=1e-10, rtol=1e-10)
    for model, (old_mean, old_covariance) in zip(models, before):
        after_mean, after_covariance = model._predict(candidates, cov_return_type='full_cov')
        assert np.array_equal(old_mean, after_mean) and np.array_equal(old_covariance, after_covariance)
    draws, sampling = sample_posterior(mean, covariance, 20000, 123)
    empirical = np.cov(draws, rowvar=False)
    error = float(np.linalg.norm(empirical - covariance) / np.linalg.norm(covariance))
    assert error < 0.06
    standard_error = np.sqrt(np.diag(covariance) / len(draws))
    mean_z_error = float(np.max(np.abs(draws.mean(axis=0) + mean) / np.maximum(standard_error, 1e-12)))
    assert mean_z_error < 6.0
    raw_draws, _ = sample_posterior(mean, covariance, 20000, 123, maximize_negative_mic=False)
    assert np.array_equal(draws, -raw_draws)
    batch, batch_info = sample_posterior(mean, covariance, 50, 456)
    repeated, _ = sample_posterior(mean, covariance, 50, 456)
    assert batch.shape == (50, 64) and np.array_equal(batch, repeated)
    off_diagonal = covariance.copy()
    np.fill_diagonal(off_diagonal, 0)
    assert np.linalg.norm(off_diagonal) > 0
    rejected_indefinite = False
    try:
        sample_posterior(np.zeros(2), np.array([[1.0, 2.0], [2.0, 1.0]]), 50, 1)
    except ValueError:
        rejected_indefinite = True
    assert rejected_indefinite
    result = dict(passed=True, source_task=task, source_payload_sha256=data['payload_sha256'],
        source_models=3, observations_per_component=32, dimensions=256,
        purpose='isolated_component_check_not_production_training', oracle_calls=0,
        matches_upstream_full_posterior=True, source_posteriors_unchanged=True,
        empirical_covariance_relative_error=error, candidate_correlations_retained=True,
        empirical_mean_max_standard_errors=mean_z_error, score_sign_exactly_verified=True,
        batch50_shape=list(batch.shape), seeded_draws_identical=True,
        indefinite_covariance_rejected=rejected_indefinite, merging=merging,
        sampling=sampling, batch_sampling=batch_info,
        seconds=time.monotonic() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        merger_sha256=transfer.digest(Path('/root/data/cys/llmoptimizer_benchmark/tools/peptide_rgpe_posterior.py')),
        full_800_source_5000_candidate_benchmark_complete=False)
    transfer.atomic_json(Path(__file__).with_name('rgpe_merge_check.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
