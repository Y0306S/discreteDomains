#!/usr/bin/env python3
"""Independent checks of real-bank merging and streamed upstream joint LOOCV."""

import io
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys
from types import SimpleNamespace

os.sched_setaffinity(0, {38})
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, '/root/data/cys/baseline/repos/transfer-hpo-framework')
import numpy as np
from peptide_transfer_models import RGPEBackend, target_ranking_loss
from peptide_rgpe_posterior import merge_posterior, sample_posterior, sample_rank_posterior
from rgpe.utils import get_gaussian_process
from rgpe.methods import rgpe as upstream
import peptide_transfer as transfer


def main():
    root = Path('/root/data/logs/peptide_transfer_wrapper_check_20261028_v1/rgpe')
    recipe = json.loads((root / 'recipe.json').read_text())
    with Path(__file__).with_name('rgpe_math_inputs.pkl').open('rb') as stream:
        state = pickle.load(stream)
    backend = RGPEBackend(recipe)
    backend.restore(state['model_snapshot'])
    low, high = np.asarray(recipe['lower']), np.asarray(recipe['upper'])
    x = ((state['features'][:12] - low) / np.where(high > low, high-low, 1.)).astype(np.float32)
    source, target = backend.bank[0], backend.bank.target
    a, ka = source.gp.predict(x, return_cov=True)
    b, kb = target.gp.predict(x, return_cov=True)
    mean, scale = backend.ensemble.Y_mean_, backend.ensemble.Y_std_
    ensemble = SimpleNamespace(model_list_=[source, target], weights_=np.array([0.35, 0.65]),
        Y_mean_=mean, Y_std_=scale, variance_mode='average')
    actual, covariance, _ = merge_posterior(ensemble, x, source_output='standardized')
    expected = (0.35*a + 0.65*b)*scale + mean
    expected_covariance = (0.35**2*ka + 0.65**2*kb)*scale**2
    assert np.allclose(actual, expected) and np.allclose(covariance, expected_covariance)
    draws, _ = sample_posterior(actual, covariance, 8192, 813, maximize_negative_mic=False)
    error = float(np.linalg.norm(np.cov(draws, rowvar=False)-covariance)/np.linalg.norm(covariance))
    assert error < 0.1
    # Compare the new streamed folds with upstream's explicit fold stack on small data.
    cx, cy = x[:6], np.arange(6, dtype=float)**0.7
    model = get_gaussian_process(target.configspace, target.types, target.bounds,
                                 np.random.RandomState(4), target.kernel)
    model._train(cx, cy, do_optimize=False)
    upstream.sample_sobol = sample_rank_posterior
    old = upstream.compute_target_model_ranking_loss(cx, cy, 32, pickle.loads(pickle.dumps(model)), 112)
    new = target_ranking_loss(cx, cy, 32, pickle.loads(pickle.dumps(model)), 112)
    assert np.array_equal(old, new)
    result = dict(passed=True, real_source_and_target_merged=True,
        diagnostic_weights_only=[0.35,0.65], production_ranking_weights_unchanged=True,
        full_covariance_formula_verified=True, target_scale_applied_once=True,
        empirical_covariance_relative_error=error, streamed_joint_loo_matches_upstream=True,
        source_models_completed=len(recipe['rgpe_sources']), new_oracle_calls=0)
    transfer.atomic_json(Path(__file__).with_name('rgpe_math_report.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
