"""Read-only production-bank and 5000-candidate checks in isolated processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, '/root/data/cys/baseline/repos/transfer-hpo-framework')
import numpy as np
from peptide_transfer_models import RGPEBackend, SourceBank
from peptide_rgpe_posterior import merge_posterior as original_merge, sample_posterior as original_sample
from peptide_rgpe_merged import merge_posterior, sample_posterior, sample_ensemble
import peptide_rgpe_merged as merged_module

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('old5000', 'new5000', 'bank800', 'dense100', 'recovery'), required=True)
    p.add_argument('--cpu', type=int, default=120)
    a = p.parse_args()
    os.sched_setaffinity(0, {a.cpu})
    recipe = json.loads((CAMPAIGN/'recipe.json').read_text())
    fixture = Path('/root/reports/peptide_transfer_wrapper_check_20261028_v1/rgpe_math_inputs.pkl')
    value = pickle.loads(fixture.read_bytes())
    backend = RGPEBackend(recipe)
    backend.restore(value['model_snapshot'])
    low, high = np.asarray(recipe['lower']), np.asarray(recipe['upper'])
    x = ((value['features']-low)/np.where(high>low, high-low, 1.)).astype(np.float64)
    rng = np.random.RandomState(1102)
    base = x[0]
    candidates = np.tile(base, (5000, 1))
    mask = rng.uniform(size=candidates.shape) < 20./256
    candidates[mask] += rng.normal(0, .02, size=np.count_nonzero(mask))
    candidates[-1] = candidates[0]
    started = time.monotonic()
    result = dict(mode=a.mode, oracle_calls=0, vae_updates=False)
    if a.mode in ('old5000', 'new5000'):
        # Same real target GP and candidate set in separate processes for RSS.
        ensemble = SimpleNamespace(model_list_=[backend.bank.target], weights_=np.array([1.]),
            Y_mean_=backend.ensemble.Y_mean_, Y_std_=backend.ensemble.Y_std_, variance_mode='average')
        merger, sampler = ((original_merge, original_sample) if a.mode == 'old5000'
                           else (merge_posterior, sample_posterior))
        mean, covariance, merging = merger(ensemble, candidates, 'standardized')
        merged_at = time.monotonic()
        draws, sampling = sampler(mean, covariance, 50, 1102)
        result.update(merging=merging, sampling=sampling, merge_seconds=merged_at-started,
                      sample_seconds=time.monotonic()-merged_at, draw_shape=list(draws.shape))
        np.savez(ROOT/(a.mode+'.npz'), mean=mean, covariance_probe=covariance[:64, :64], draws=draws)
        assert draws.shape == (50, 5000) and np.isfinite(draws).all()
    elif a.mode in ('bank800', 'dense100'):
        count, size = ((800, 32) if a.mode == 'bank800' else (100, 512))
        chosen = candidates[:size]
        short = dict(recipe, rgpe_sources=recipe['rgpe_sources'][:count])
        bank = SourceBank(short)
        weights = rng.uniform(.1, 1., count)
        weights /= weights.sum()
        ensemble = SimpleNamespace(model_list_=bank, weights_=weights,
            Y_mean_=backend.ensemble.Y_mean_, Y_std_=backend.ensemble.Y_std_, variance_mode='average')
        refmean, refcov = np.zeros(size), np.zeros((size, size))
        probe = 8 if count == 100 else size
        original_add = merged_module._add_source
        checked = [0]
        def checked_add(model, locations, weight, mean, covariance, units):
            m, c = model.gp.predict(model._impute_inactive(chosen[:probe]), return_cov=True)
            refmean[:probe] += weight*m
            refcov[:probe, :probe] += weight**2*c
            mode = original_add(model, locations, weight, mean, covariance, units)
            checked[0] += 1
            if checked[0] % 25 == 0:
                print(json.dumps(dict(checked_sources=checked[0], total=count,
                                      seconds=time.monotonic()-started)), flush=True)
            return mode
        merged_module._add_source = checked_add
        mean, covariance, info = merge_posterior(ensemble, chosen)
        merged_module._add_source = original_add
        refmean = refmean*ensemble.Y_std_+ensemble.Y_mean_
        refcov *= ensemble.Y_std_**2
        np.testing.assert_allclose(mean[:probe], refmean[:probe], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(covariance[:probe, :probe], refcov[:probe, :probe], atol=1e-8, rtol=1e-8)
        draws, sampling = sample_posterior(mean, covariance, 50, 1102)
        result.update(merging=info, sampling=sampling, source_models=count,
                      mean_max_error=float(np.max(np.abs(mean[:probe]-refmean[:probe]))),
                      covariance_max_error=float(np.max(np.abs(covariance[:probe,:probe]-refcov[:probe,:probe]))))
    else:
        # Compact fitted-target snapshots restore without refitting/ranking.
        backend.ensemble.weights_ = np.zeros(len(backend.bank))
        backend.ensemble.weights_[-1] = 1.
        before = sample_ensemble(backend.ensemble, x[:24], 50, 413)
        snapshot = backend.snapshot()
        restored = RGPEBackend(recipe)
        restored.restore(snapshot)
        after = sample_ensemble(restored.ensemble, x[:24], 50, 413)
        np.testing.assert_allclose(before['draws'], after['draws'], atol=1e-8, rtol=1e-8)
        result.update(snapshot_bytes=len(snapshot), compact_restore_passed=True,
                      weights_unchanged=bool(np.array_equal(backend.ensemble.weights_, restored.ensemble.weights_)))
    result.update(passed=True, seconds=time.monotonic()-started,
                  cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    (ROOT/(a.mode+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
