"""Exercise SemanticOpt through CASH, including the complete evaluation budget."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.datasets import make_classification, make_regression
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.history import Observation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmark'))
import run_toy_categorical_optimizers as runner
from mindware import CASH
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.optimizers.semanticopt_optimizer import SemanticOptOptimizer
from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION, NUMERICAL
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.base import BaseAutoML


def client_for(config):
    def respond(messages, **kwargs):
        if 'initial_configuration_1' in messages[0]['content']:
            return json.dumps({f'initial_configuration_{i}': config for i in range(1, 6)})
        return json.dumps({'next_configuration': config, 'mean': 0.1, 'std': .2,
                           'failure_probability': .1})
    return SimpleNamespace(complete_messages=Mock(side_effect=respond))


class SemanticEntryPointTests(unittest.TestCase):
    def test_cash_dispatch_and_real_evaluation_for_both_task_types(self):
        for label, kind in (('cls', CLASSIFICATION), ('rgs', REGRESSION)):
            x, y = (make_classification(n_samples=80, n_features=6, random_state=3)
                    if label == 'cls' else make_regression(n_samples=80, n_features=6, random_state=3))
            node = DataNode([x, y], feature_type=[NUMERICAL] * 6, task_type=kind)
            config = dict(build_hpo_cs(label, 'random_forest').get_default_configuration())
            with self.subTest(task_type=label), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ), patch.object(KnowledgeDatabase, '_instance', None):
                os.environ.pop('ML_SESSION_DIR', None)
                cash = CASH(include_algorithms=['random_forest'], task_type=kind,
                    metric='bal_acc' if label == 'cls' else 'mse', data_node=node,
                    optimizer='semanticopt', amount_of_resource=100, time_limit=None,
                    per_run_time_limit=2400, output_dir=directory,
                    resampling_params={'test_size': .25}, optimizer_params={
                        'task_type': label, 'llm_client': client_for(config),
                        'extractor_client': SimpleNamespace(calls=0),
                        'recommendation_default_fallback': True})
                self.assertIsInstance(cash.optimizer, SemanticOptOptimizer)
                cash.optimizer.iterate()
                row = cash.optimizer.traj_rows[0]
                self.assertEqual(row['evaluation_status'], 'SUCCESS')
                self.assertTrue(np.isfinite(row['perf']))
                self.assertEqual(cash.optimizer.semantic_history[0]['objective'], -row['perf'])

    def test_full_runner_counts_100_calls_and_resumes_without_extra_fit(self):
        x, y = make_classification(n_samples=80, n_features=6, random_state=3)
        node = DataNode([x, y], feature_type=[NUMERICAL] * 6, task_type=CLASSIFICATION)
        config = dict(build_hpo_cs('cls', 'random_forest').get_default_configuration())
        client = client_for(config)
        states = iter([SUCCESS] * 98 + [TIMEOUT, FAILED])

        def evaluate(configuration, evaluator, timeout, **kwargs):
            state = next(states)
            self.assertEqual(timeout, 2400)
            return Observation(config=configuration, trial_state=state, elapsed_time=.01,
                               objectives=[-.6 if state == SUCCESS else float('inf')])

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), \
                patch.object(KnowledgeDatabase, '_instance', None), \
                patch('examples.benchmark.hf_qwen_client.HuggingFaceQwenClient', return_value=client), \
                patch('mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater',
                      side_effect=evaluate) as oracle, \
                patch.object(BaseAutoML, '_refit_config', side_effect=AssertionError('Extra fit')):
            with patch.object(sys, 'argv', ['run', '--optimizer', 'semanticopt', '--dataset', 'fixture',
                    '--algorithm', 'random_forest', '--n-trials', '100', '--time-limit', 'none',
                    '--per-run-time-limit', '2400', '--skip-test-eval',
                    '--extractor-max-tokens', '2048', '--extractor-max-calls', '100',
                    '--extractor-timeout', '60', '--output-dir', directory + '/outputs',
                    '--history-dir', directory + '/history', '--session-dir', directory + '/session']):
                args = runner.parse_args()
            self.assertNotIn('semanticopt', runner.parse_optimizers(['all'], []))
            first = runner.run_one_optimizer(args, 'semanticopt', node, node, ['random_forest'], {})
            self.assertEqual(oracle.call_count, 100)
            self.assertEqual(first['progress']['attempts'], 100)
            self.assertEqual(first['progress']['successful'], 98)
            self.assertEqual(first['progress']['failed'], 2)
            self.assertTrue(first['progress']['budget_exhausted'])
            self.assertFalse(first['model_info']['fitted_model_saved'])
            calls = client.complete_messages.call_count
            args.resume = True
            restored = runner.run_one_optimizer(args, 'semanticopt', node, node, ['random_forest'], {})
            self.assertEqual(restored['run_id'], first['run_id'])
            self.assertEqual(restored['progress'], first['progress'])
            self.assertEqual(oracle.call_count, 100)
            self.assertEqual(client.complete_messages.call_count, calls)
            args.skip_test_eval = False
            with self.assertRaisesRegex(ValueError, 'skip-test-eval'):
                runner.build_optimizer_params(args, 'semanticopt', args.algorithm, {})


if __name__ == '__main__':
    unittest.main()
