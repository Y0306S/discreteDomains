"""Numerical and worker-level checks for the bounded-memory ML RGPE path."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import tracemalloc
import unittest

import numpy as np
from ConfigSpace import (CategoricalHyperparameter, ConfigurationSpace, EqualsCondition,
                         UniformFloatHyperparameter)
from smac.configspace import convert_configurations_to_array
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.util_funcs import get_types
from rgpe.utils import get_gaussian_process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/benchmark'))
import rgpe_exact as exact
import rgpe_worker as worker

ASSETS = ROOT / 'benchmark_data/ml_test_20_fixed_fsbo_rgpe_20260928/assets'
REPORT = {}
OUTPUT = None


def model(cs, seed=71, kernel=None):
    types, bounds = get_types(cs, None)
    return get_gaussian_process(cs, types, bounds, np.random.RandomState(seed), kernel)


def scenario(kind):
    cs = ConfigurationSpace(seed=15)
    if kind in ('continuous', 'mixed_conditional'):
        cs.add_hyperparameters([UniformFloatHyperparameter('x%d' % i, 0, 1) for i in range(3)])
    if kind in ('categorical', 'mixed_conditional'):
        gate = CategoricalHyperparameter('gate', ['a', 'b', 'c'])
        option = CategoricalHyperparameter('option', [False, True])
        cs.add_hyperparameters([gate, option])
        if kind == 'mixed_conditional':
            cs.add_condition(EqualsCondition(cs.get_hyperparameter('x1'), gate, 'b'))
            cs.add_condition(EqualsCondition(option, gate, 'c'))
    configs = cs.sample_configuration(18)
    x = convert_configurations_to_array(configs)
    filled = np.where(np.isfinite(x), x, -1)
    y = np.sin(filled[:, 0] * 3) + 0.2 * filled.sum(axis=1)
    return cs, x, y


class ExactRGPEChecks(unittest.TestCase):
    def test_conditional_gradients(self):
        cs, x, y = scenario('mixed_conditional')
        gp = model(cs)
        gp._train(x, y, do_optimize=False)
        results = []
        for index in range(4):
            theta = gp.gp.kernel_.theta.copy() + (index - 1.5) * 0.2
            theta[-1] = np.log(0.01 + index * 0.02)
            expected, grad = gp.gp.log_marginal_likelihood(theta, eval_gradient=True)
            actual, bounded = exact.marginal_likelihood(gp.gp, theta)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(bounded, grad, rtol=1e-7, atol=2e-8)
            results.append(float(np.max(np.abs(grad - bounded))))
        REPORT['conditional_gradient_max_errors'] = results

    def test_large_categorical_objective_memory(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameters([CategoricalHyperparameter('c%02d' % i, [0, 1, 2])
                                for i in range(64)])
        rng = np.random.RandomState(53)
        x = rng.randint(0, 3, (256, 64)).astype(float)
        y = rng.normal(size=256)
        gp = model(cs)
        gp._train(x, y, do_optimize=False)
        theta = gp.gp.kernel_.theta.copy()
        tracemalloc.start()
        expected, grad = gp.gp.log_marginal_likelihood(theta, eval_gradient=True)
        original_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        tracemalloc.start()
        actual, bounded = exact.marginal_likelihood(gp.gp, theta)
        bounded_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(bounded, grad, rtol=1e-7, atol=2e-8)
        self.assertLess(bounded_peak, original_peak * 0.5)
        REPORT['objective_memory_check'] = dict(observations=256, categorical_dimensions=64,
            upstream_peak_mib=original_peak / 2**20, bounded_peak_mib=bounded_peak / 2**20)

    def test_all_ml_source_kernel_gradients(self):
        results = []
        rng = np.random.RandomState(120)
        for path in sorted((ASSETS / 'dumps').glob('*.json')):
            dump, tasks, cs = worker.load_dump(path)
            configs = worker.configs_from_rows(cs, dump['tasks'][tasks[0]]['configs'][:16], dump['algorithm'])
            x = convert_configurations_to_array(configs + configs[:2])
            y = np.asarray(dump['tasks'][tasks[0]]['y_minimize'][:16], dtype=float)
            y = np.concatenate((y, y[:2]))
            gp = model(cs)
            gp._train(x, y, do_optimize=False)
            gp._all_priors = gp._get_all_priors(add_bound_priors=False)
            max_gradient_error, max_value_error = 0.0, 0.0
            for index in range(4):
                theta = gp.gp.kernel_.theta.copy() + rng.uniform(-0.5, 0.3, len(gp.kernel.theta))
                theta[-1] = np.log(0.02 + 0.01 * index)
                expected, grad = gp.gp.log_marginal_likelihood(theta, eval_gradient=True)
                actual, bounded = exact.marginal_likelihood(gp.gp, theta)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(bounded, grad, rtol=1e-7, atol=2e-8)
                expected_nll, expected_grad = gp._nll(theta)
                with exact.exact_gp_fitting():
                    actual_nll, actual_grad = gp._nll(theta)
                np.testing.assert_allclose(actual_nll, expected_nll, rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-7, atol=2e-8)
                max_gradient_error = max(max_gradient_error, float(np.max(np.abs(grad - bounded))))
                max_value_error = max(max_value_error, abs(actual - expected))
            results.append(dict(group=path.stem, cases=4, observations=len(x),
                                conditional=bool(cs.get_conditions()),
                                max_gradient_error=max_gradient_error, max_value_error=max_value_error))
        self.assertEqual(len(results), 16)
        REPORT['ml_source_gradient_checks'] = results

    def test_continuous_categorical_and_conditional_fits(self):
        results = []
        for kind in ('continuous', 'categorical', 'mixed_conditional'):
            cs, x, y = scenario(kind)
            original, bounded = model(cs), model(cs)
            original.train(x, y)
            previous = GaussianProcess._nll
            with exact.exact_gp_fitting():
                bounded.train(x, y)
            self.assertIs(GaussianProcess._nll, previous)
            probe = convert_configurations_to_array(cs.sample_configuration(32))
            first, second = original.predict(probe), bounded.predict(probe)
            np.testing.assert_allclose(first[0], second[0], rtol=5e-4, atol=5e-5)
            np.testing.assert_allclose(first[1], second[1], rtol=1e-3, atol=5e-5)
            self.assertEqual(bounded.n_opt_restarts, 10)
            restored = pickle.loads(pickle.dumps(bounded, protocol=4))
            self.assertNotIn('_nll', restored.__dict__)
            np.testing.assert_array_equal(restored.predict(probe)[0], second[0])
            results.append(dict(kind=kind, optimizer_restarts=10,
                                mean_max_difference=float(np.max(np.abs(first[0] - second[0]))),
                                variance_max_difference=float(np.max(np.abs(first[1] - second[1]))),
                                pickle_reload_verified=True))
        REPORT['full_fit_checks'] = results

    def test_default_worker_with_existing_and_new_sources(self):
        if OUTPUT is None:
            self.skipTest('Provide --output for isolated CLI checks')
        original_dump = ASSETS / 'dumps/rgs_adaboost.json'
        dump = json.loads(original_dump.read_text())
        task = sorted(dump['tasks'])[0]
        _, _, cs = worker.load_dump(original_dump)
        record = dump['tasks'][task]
        request = dict(target_configs=record['configs'][:12], y_minimize=record['y_minimize'][:12],
                       pending_configs=[c.get_dictionary() for c in cs.sample_configuration(64)],
                       seed=1, n_trials=100, schema_fingerprint=dump['schema_fingerprint'])
        path = OUTPUT / 'request.json'
        path.write_text(json.dumps(request))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=os.pathsep.join([str(ROOT), '/root/data/cys/baseline/repos/transfer-hpo-framework']))
        script = ROOT / 'examples/benchmark/rgpe_worker.py'
        def run(flags, name):
            with (OUTPUT / (name + '.log')).open('w') as log:
                subprocess.run([sys.executable, str(script)] + flags, env=env, cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        cache = ASSETS / 'rgpe/rgs_adaboost_seed1.pkl'
        original_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
        response = OUTPUT / 'old_cache_response.json'
        run(['suggest', '--dump', str(original_dump), '--cache', str(cache),
             '--request', str(path), '--out', str(response)], 'old_cache')
        payload = json.loads(response.read_text())
        self.assertEqual(payload['fitting_engine'], exact.ENGINE)
        self.assertTrue(0 <= payload['index'] < 64)
        self.assertEqual(original_sha, hashlib.sha256(cache.read_bytes()).hexdigest())
        derived = copy.deepcopy(dump)
        derived['tasks'] = {task: record}
        derived_path = OUTPUT / 'one_source_dump.json'
        derived_path.write_text(json.dumps(derived))
        new_cache = OUTPUT / 'new_sources.pkl'
        run(['fit-sources', '--dump', str(derived_path), '--cache', str(new_cache), '--seed', '1'], 'new_fit')
        metadata = json.loads(new_cache.with_suffix('.json').read_text())
        self.assertEqual(metadata['fitting_engine'], exact.ENGINE)
        self.assertEqual(metadata['source_tasks'], [task])
        response = OUTPUT / 'new_cache_response.json'
        run(['suggest', '--dump', str(derived_path), '--cache', str(new_cache),
             '--request', str(path), '--out', str(response)], 'new_cache')
        payload = json.loads(response.read_text())
        self.assertEqual(payload['fitting_engine'], exact.ENGINE)
        self.assertEqual(payload['n_sources'], 1)
        REPORT['worker_checks'] = dict(existing_cache_unchanged=True,
                                      existing_cache_target_fit_passed=True,
                                      fresh_source_fit_passed=True, fresh_cache_target_fit_passed=True,
                                      source_observations=len(record['y_minimize']),
                                      source_task=task, evaluator_calls=0)

    def test_context_restored_after_error_and_unsupported_kernel_rejected(self):
        previous = GaussianProcess._nll
        with self.assertRaises(RuntimeError):
            with exact.exact_gp_fitting():
                raise RuntimeError('diagnostic interruption')
        self.assertIs(GaussianProcess._nll, previous)
        from smac.epm.gp_kernels import RBF
        with self.assertRaisesRegex(ValueError, 'Unsupported kernel'):
            exact._kernel_tree(RBF(), np.ones((3, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args, remaining = parser.parse_known_args()
    OUTPUT = args.output
    if OUTPUT is not None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExactRGPEChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(passed=result.wasSuccessful(), tests=result.testsRun,
                  engine=exact.ENGINE, elapsed_seconds=time.monotonic() - started)
    if OUTPUT is not None:
        (OUTPUT / 'report.json').write_text(json.dumps(REPORT, indent=2) + '\n')
    print(json.dumps(REPORT), flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
