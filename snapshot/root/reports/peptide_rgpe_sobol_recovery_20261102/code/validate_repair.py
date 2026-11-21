"""Regression checks in the actual legacy RGPE model environment."""
import hashlib
import json
import pickle
from pathlib import Path
import sqlite3
import sys
import time
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
import botorch.sampling.qmc as qmc
from peptide_rgpe_posterior import sample_rank_posterior
from sobol_repair import ROOT, install

OUT = ROOT.parent
legacy = qmc.NormalQMCEngine
client = install()


class Fixed:
    def __init__(self, mean, covariance):
        self.mean, self.covariance = mean, covariance
    def _predict(self, x, cov_return_type):
        return self.mean, self.covariance


class RepairTest(unittest.TestCase):
    def test_only_legacy_sampler_failure_is_retried(self):
        from repair_dispatch import recoverable_sobol_exit
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'worker.log'
            record = dict(command=['python','/tools/peptide_rgpe_merged_launch.py'], log=str(log))
            log.write_text('ValueError: Supported range of dimensionality for SobolEngine is [1, 1111]')
            self.assertTrue(recoverable_sobol_exit(record))
            record['command'][1] = '/tools/repair_launch.py'
            self.assertFalse(recoverable_sobol_exit(record))
            record['command'][1] = '/tools/peptide_rgpe_merged_launch.py'
            log.write_text('RuntimeError: unrelated fit failure')
            self.assertFalse(recoverable_sobol_exit(record))

    def test_legacy_draws_are_bit_identical(self):
        for d in (3, 64, 1110):
            expected = legacy(d, seed=99).draw(256)
            actual = qmc.NormalQMCEngine(d, seed=99).draw(256)
            self.assertTrue(torch.equal(actual, expected))

    def test_failure_boundary_and_seed_recovery(self):
        with self.assertRaises(ValueError):
            legacy(1150, seed=21)
        a = qmc.NormalQMCEngine(1150, seed=21).draw(256)
        b = qmc.NormalQMCEngine(1150, seed=21).draw(256)
        c = qmc.NormalQMCEngine(1150, seed=22).draw(256)
        self.assertEqual(tuple(a.shape), (256, 1150))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_later_budget_and_direction_table_boundary(self):
        for d in (5100, 25000):
            a = qmc.NormalQMCEngine(d, seed=7).draw(16)
            self.assertEqual(tuple(a.shape), (16, d))
            self.assertTrue(torch.isfinite(a).all())

    def test_joint_covariance_not_partitioned(self):
        n = 1150
        cov = np.eye(n)
        cov[0, 1] = cov[1, 0] = -.6
        model = Fixed(np.linspace(-1, 1, n), cov)
        result = sample_rank_posterior(model, np.zeros((n, 1)), 4096, 91)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[:, :2].mean(0), model.mean[:2], atol=.015)
        np.testing.assert_allclose(np.cov(result[:, :2], rowvar=False), cov[:2, :2], atol=.025)

    def test_real_saved_coordinates_and_pinned_source_gp(self):
        recipe = json.loads(Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe/recipe.json').read_text())
        sys.path.insert(0, '/root/data/cys/baseline/repos/transfer-hpo-framework')
        with sqlite3.connect('file:'+recipe['rgpe_bank']+'?mode=ro', uri=True) as db:
            payload = db.execute('select payload from models where task=?', (recipe['rgpe_sources'][0],)).fetchone()[0]
        model = pickle.loads(payload)
        x = np.load(str(OUT/'saved_coordinates.npz'))['features']
        low, high = np.asarray(recipe['lower']), np.asarray(recipe['upper'])
        x = ((x-low)/np.where(high>low,high-low,1.)).astype(np.float32)
        self.assertEqual(len(x), 1150)
        result = sample_rank_posterior(model, x, 256, 19)
        self.assertEqual(result.shape, (256, 1150))
        self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepairTest))
    code = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    (OUT/'validation.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        seconds=time.time()-started, code=code), indent=2)+'\n')
    client.close()
    raise SystemExit(0 if result.wasSuccessful() else 1)
