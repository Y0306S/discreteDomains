"""Verify ToSFiT's public runner budget and cached incumbent accounting."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from sklearn.datasets import make_classification
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.history import Observation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmark'))
import run_toy_categorical_optimizers as runner
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs
from mindware.components.utils.constants import CLASSIFICATION, NUMERICAL
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.base import BaseAutoML
from test_online_llm_baselines import MockRuntime


class RunnerRuntime(MockRuntime):
    def propose(self, prompt, step):
        self.proposals += 1
        self.policy_version = max(0, step - 15)
        return {'raw_text': json.dumps(dict(self.cs.get_default_configuration())),
                'tokens': [step], 'features': [float(step), 1.0],
                'policy_version': self.policy_version, 'gradient_step': step >= 16}


class ToSFiTEntryPointTests(unittest.TestCase):
    def test_100_actual_calls_cached_incumbent_and_completed_resume(self):
        x, y = make_classification(n_samples=80, n_features=6, random_state=3)
        node = DataNode([x, y], feature_type=[NUMERICAL] * 6, task_type=CLASSIFICATION)
        runtime = RunnerRuntime(build_hpo_cs('cls', 'random_forest'))
        states = iter([SUCCESS] * 98 + [TIMEOUT, FAILED])

        def evaluate(configuration, evaluator, timeout, **kwargs):
            state = next(states)
            self.assertEqual(timeout, 2400)
            objective = -.8 if runtime.proposals == 1 else -.6
            return Observation(config=configuration, trial_state=state, elapsed_time=.01,
                               objectives=[objective if state == SUCCESS else float('inf')])

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), \
                patch.object(KnowledgeDatabase, '_instance', None), \
                patch('mindware.components.optimizers.tosfit_optimizer.ToSFiTRuntime', return_value=runtime), \
                patch('mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater',
                      side_effect=evaluate) as oracle, \
                patch.object(BaseAutoML, '_refit_config', side_effect=AssertionError('Extra fit')):
            with patch.object(sys, 'argv', ['run', '--optimizer', 'tosfit', '--dataset', 'fixture',
                    '--algorithm', 'random_forest', '--n-trials', '100', '--time-limit', 'none',
                    '--per-run-time-limit', '2400', '--skip-test-eval',
                    '--tosfit-device', 'cpu', '--tosfit-reference-device', 'cpu',
                    '--recommendation-fallback', 'default',
                    '--extractor-max-tokens', '2048', '--extractor-max-calls', '100',
                    '--extractor-timeout', '60', '--output-dir', directory + '/outputs',
                    '--history-dir', directory + '/history', '--session-dir', directory + '/session']):
                args = runner.parse_args()
            first = runner.run_one_optimizer(args, 'tosfit', node, node, ['random_forest'], {})
            self.assertEqual(oracle.call_count, 100)
            self.assertEqual(runtime.proposals, 100)
            self.assertEqual(runtime.policy_version, 84)
            self.assertEqual(len(runtime.observations), 98)
            self.assertEqual(first['progress']['successful'], 98)
            self.assertEqual(first['progress']['failed'], 2)
            self.assertTrue(first['progress']['budget_exhausted'])
            self.assertFalse(first['model_info']['fitted_model_saved'])
            trajectory = Path(first['progress']['trajectory'])
            decisions = json.loads(trajectory.with_suffix('.decisions.json').read_text())
            checks = [r for r in decisions if r.get('stage') == 'incumbent_cached']
            self.assertEqual(len(checks), 99)
            self.assertTrue(all(r['source_trial'] == 0 and r['score'] == .8
                                and not r['oracle_called'] and not r['evaluation_slot_consumed'] for r in checks))
            args.resume = True
            restored = runner.run_one_optimizer(args, 'tosfit', node, node, ['random_forest'], {})
            self.assertEqual(restored['run_id'], first['run_id'])
            self.assertEqual(restored['progress'], first['progress'])
            self.assertEqual(oracle.call_count, 100)
            self.assertEqual(runtime.proposals, 100)
            args.skip_test_eval = False
            with self.assertRaisesRegex(ValueError, 'skip-test-eval'):
                runner.build_optimizer_params(args, 'tosfit', args.algorithm, {})


if __name__ == '__main__':
    unittest.main()
