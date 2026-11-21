"""Acquisition math/streaming/sampling checks; no oracle or VAE execution."""
import gc
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import peptide_rgpe_merged as merged
from peptide_rgpe_posterior import merge_posterior as old_merge


class Fixed:
    def __init__(self, mean, cov):
        self.mean, self.cov = mean, cov
    def _predict(self, x, cov_return_type):
        return self.mean, self.cov


def ensemble(models, weights, scale=1., offset=0.):
    return SimpleNamespace(model_list_=models, weights_=np.array(weights),
        variance_mode='average', Y_mean_=offset, Y_std_=scale)


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.x = np.ones((3, 2))
        self.a = Fixed(np.array([1., 2., 3.]), np.array([[2., -.4, .1], [-.4, 1., .2], [.1, .2, 1.]]))
        self.b = Fixed(np.array([4., 1., -2.]), np.array([[1., .5, -.2], [.5, 2., -.3], [-.2, -.3, 2.]]))
    def test_weighted_means_squared_weights_and_units(self):
        e = ensemble([self.a, self.b], [2., 3.], 3.25, 12.5)
        mean, cov, info = merged.merge_posterior(e, self.x, 'ensemble')
        np.testing.assert_allclose(mean, (.4*self.a.mean+.6*self.b.mean)*3.25+12.5)
        np.testing.assert_allclose(cov, (.16*self.a.cov+.36*self.b.cov)*3.25**2)
        self.assertEqual(info['contributing_models'], 2)
        self.assertLess(cov[0, 2], 0)
    def test_skip_zero_weight_without_loading(self):
        class Bank:
            def __len__(s): return 3
            def __getitem__(s, i):
                if i != 1: raise AssertionError('Zero weight source loaded')
                return self.a
        actual = merged.merge_posterior(ensemble(Bank(), [0, 1, 0]), self.x, 'ensemble')
        np.testing.assert_equal(actual[1], self.a.cov)
    def test_sources_not_mutated(self):
        before = self.a.cov.copy()
        merged.merge_posterior(ensemble([self.a], [1]), self.x, 'ensemble')
        np.testing.assert_array_equal(before, self.a.cov)
    def test_800_models_are_released_between_sources(self):
        refs = []
        class Bank:
            def __len__(s): return 800
            def __getitem__(s, i):
                gc.collect()
                self.assertFalse(any(ref() is not None for ref in refs))
                model = Fixed(self.a.mean, self.a.cov)
                refs.append(weakref.ref(model))
                return model
        actual = merged.merge_posterior(ensemble(Bank(), np.ones(800)), self.x, 'ensemble')
        np.testing.assert_allclose(actual[1], self.a.cov/800.)
        self.assertFalse(any(ref() is not None for ref in refs))
    def test_empirical_moments_match_independent_source_sum(self):
        e = ensemble([self.a, self.b], [.4, .6], 2., 5.)
        mean, cov, _ = merged.merge_posterior(e, self.x, 'ensemble')
        draws, _ = merged.sample_posterior(mean, cov, 80000, 91, False)
        rng = np.random.RandomState(40)
        independent = (.4*rng.multivariate_normal(self.a.mean, self.a.cov, 80000)
                       +.6*rng.multivariate_normal(self.b.mean, self.b.cov, 80000))*2.+5.
        for sample in (draws, independent):
            np.testing.assert_allclose(sample.mean(0), mean, atol=.025)
            np.testing.assert_allclose(np.cov(sample, rowvar=False), cov, atol=.04)
    def test_one_factorization_for_batch50(self):
        with patch.object(merged, 'cholesky', wraps=merged.cholesky) as factor:
            draws, info = merged.sample_posterior(self.a.mean, self.a.cov, 50, 1)
        self.assertEqual(factor.call_count, 1)
        self.assertEqual(info['covariance_factorizations'], 1)
        self.assertEqual(draws.shape, (50, 3))
    def test_seed_and_score_sign(self):
        a, _ = merged.sample_posterior(self.a.mean, self.a.cov, 50, 3)
        b, _ = merged.sample_posterior(self.a.mean, self.a.cov, 50, 3)
        c, _ = merged.sample_posterior(self.a.mean, self.a.cov, 50, 3, False)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, -c)
    def test_duplicate_candidates_and_jitter(self):
        _, info = merged.sample_posterior(np.zeros(2), np.ones((2, 2)), 50, 3)
        self.assertGreater(info['covariance_jitter'], 0)
    def test_sampling_does_not_change_covariance(self):
        before = self.a.cov.copy()
        merged.sample_posterior(self.a.mean, self.a.cov, 50, 3)
        np.testing.assert_array_equal(before, self.a.cov)
    def test_invalid_weights_and_scale(self):
        for weights in ([0, 0], [-1, 2], [np.nan, 1], [np.inf, 0], [1]):
            with self.assertRaises(ValueError):
                merged.merge_posterior(ensemble([self.a, self.b], weights), self.x, 'ensemble')
        for scale in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                merged.merge_posterior(ensemble([self.a], [1], scale), self.x, 'ensemble')
    def test_invalid_inputs_rejected(self):
        for x in (np.empty((0, 2)), np.array([[np.nan]]), np.ones(4)):
            with self.assertRaises(ValueError):
                merged.merge_posterior(ensemble([self.a], [1]), x, 'ensemble')
        for cov in (np.array([[1., 2.], [2., 1.]]), np.array([[1., 0.], [.1, 1.]]),
                    np.array([[1., np.nan], [np.nan, 1.]]), np.diag([-1., 1.])):
            with self.assertRaises(ValueError):
                merged.sample_posterior(np.zeros(2), cov, 50, 1)
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                merged.sample_posterior(self.a.mean, self.a.cov, count, 1)


class RealGPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, '/root/data/cys/baseline/repos/transfer-hpo-framework')
        from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
        from smac.epm.util_funcs import get_types
        from rgpe.utils import get_gaussian_process
        rng = np.random.RandomState(100)
        cls.x = rng.uniform(size=(40, 256))
        cls.x[-2:] = cls.x[0]
        cs = ConfigurationSpace()
        cs.add_hyperparameters([UniformFloatHyperparameter('z%03d'%i, 0., 1.) for i in range(256)])
        types, bounds = get_types(cs, None)
        cls.models = []
        for seed in range(3):
            model = get_gaussian_process(cs, types, bounds, np.random.RandomState(seed), None)
            model._train(cls.x[:12], rng.normal(size=12), do_optimize=False)
            cls.models.append(model)
    def test_tiled_matches_original_across_block_boundaries(self):
        e = ensemble(self.models, [.2, .3, .5], 2.5, 14.)
        before = old_merge(e, self.x, 'standardized')
        for block in (1, 7, 16, 256):
            with patch.object(merged, 'BLOCK', block):
                actual = merged.merge_posterior(e, self.x)
            np.testing.assert_allclose(actual[0], before[0], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(actual[1], before[1], rtol=1e-8, atol=1e-9)
        self.assertEqual(actual[2]['retained_source_covariances'], 0)
    def test_tiled_never_calls_full_predict_or_source_sampler(self):
        with patch.object(self.models[0].gp, 'predict', side_effect=AssertionError('full predict')):
            actual = merged.sample_ensemble(ensemble(self.models[:1], [1]), self.x, 50, 43)
        self.assertEqual(actual['draws'].shape, (50, 40))
        self.assertEqual(actual['info']['covariance_factorizations'], 1)
    def test_100_contributors_match_original(self):
        models = [self.models[i%3] for i in range(100)]
        e = ensemble(models, np.arange(1, 101), 1.5, 2.)
        expected = old_merge(e, self.x, 'standardized')
        actual = merged.merge_posterior(e, self.x)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-10)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
