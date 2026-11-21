import json
import unittest

import numpy as np

from ConfigSpace import ConfigurationSpace, Constant, UniformIntegerHyperparameter
from ConfigSpace import ForbiddenEqualsClause

from mindware.components.optimizers.semanticopt_protocol import (
    build_messages, normalize_response, objective_feedback,
)


class SemanticOptProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters([Constant('algorithm', 'adaboost'),
                                    UniformIntegerHyperparameter('adaboost:max_depth', 1, 8)])
        self.payload = {'next_configuration': {'adaboost:max_depth': 3}, 'mean': -.7,
                        'std': .04, 'failure_probability': .02}

    def parse(self, payload=None, failed=False):
        return normalize_response(json.dumps(self.payload if payload is None else payload),
                                  self.cs, 'adaboost', failed)

    def test_only_fixed_identifier_and_inapplicable_metadata_change(self):
        value, changes = self.parse()
        self.assertEqual(value, {'next_configuration': {'algorithm': 'adaboost', 'adaboost:max_depth': 3},
                                 'mean': -.7, 'std': .04})
        self.assertEqual(changes, ['inserted_fixed_algorithm', 'removed_failure_probability_before_failure'])
        self.assertNotIn('algorithm', self.payload['next_configuration'])

    def test_failure_probability_is_required_and_preserved_after_failure(self):
        value, changes = self.parse(failed=True)
        self.assertEqual(value['failure_probability'], .02)
        for probability in (None, -1, 2, True, float('nan')):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self.parse(dict(self.payload, failure_probability=probability), failed=True)
        with self.assertRaises(ValueError):
            self.parse({k: v for k, v in self.payload.items() if k != 'failure_probability'}, failed=True)

    def test_rejects_bad_tunables_without_defaults_or_clipping(self):
        for configuration in ({}, {'adaboost:max_depth': 100}, {'adaboost:max_depth': True},
                              {'adaboost:max_depth': 3.0}, {'adaboost:max_depth': '3'},
                              {'adaboost:max_depth': 3, 'algorithm': 'xgboost'},
                              {'adaboost:max_depth': 3, 'unknown': 1}):
            with self.subTest(configuration=configuration), self.assertRaises((ValueError, KeyError)):
                self.parse(dict(self.payload, next_configuration=configuration))
        self.cs.add_forbidden_clause(ForbiddenEqualsClause(self.cs['adaboost:max_depth'], 3))
        with self.assertRaises(ValueError):
            self.parse()

    def test_rejects_malformed_or_nonfinite_predictions(self):
        for key, value in (('mean', True), ('std', -.1), ('std', float('inf')), ('mean', float('nan'))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse(dict(self.payload, **{key: value}))
        with self.assertRaises(ValueError):
            normalize_response('{"mean": 0, "mean": 1}', self.cs, 'adaboost', False)

    def test_objective_signs_and_failed_feedback(self):
        result = {'result': {'objectives': [np.float64(-.8)]}, 'timeout': False, 'traceback': None}
        feedback, state = objective_feedback(result, 'cls')
        self.assertEqual((feedback['objective'], state), (-.8, 'SUCCESS'))
        result['result']['objectives'] = [4.2]
        self.assertEqual(objective_feedback(result, 'rgs')[0]['objective'], 4.2)
        result.update(timeout=True, result=None)
        self.assertEqual(objective_feedback(result, 'rgs'), ({'status': 'failure'}, 'TIMEOUT'))
        result.update(timeout=False, traceback='worker failed')
        self.assertEqual(objective_feedback(result, 'rgs'), ({'status': 'failure'}, 'FAILED'))

    def test_prompt_uses_only_recorded_past_results_and_normalized_answers(self):
        proposal, _ = self.parse()
        initial = [{'configuration': proposal['next_configuration'], 'status': 'success', 'objective': -.623456}]
        adaptive = [{'proposal': proposal, 'status': 'failure'}]
        messages = build_messages('system', {'task_type': 'classification'}, self.cs, 'adaboost', 'cls', 8,
                                  initial, adaptive)
        context, _ = json.JSONDecoder().raw_decode(messages[1]['content'])
        self.assertEqual(context['initial_evaluations'][0]['objective'], -.6235)
        self.assertEqual(json.loads(messages[2]['content']), proposal)
        feedback, _ = json.JSONDecoder().raw_decode(messages[3]['content'])
        self.assertEqual(feedback, {'evaluation': 2, 'status': 'failure'})


if __name__ == '__main__':
    unittest.main()
