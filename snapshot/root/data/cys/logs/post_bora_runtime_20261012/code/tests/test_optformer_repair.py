import json
from types import SimpleNamespace
import unittest

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, Constant
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause
from vizier import pyvizier as vz

from mindware.components.optimizers.transfer_hpo.optformer_bridge import config_from_suggestion


class SuggestionRepairTests(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace(seed=1)
        self.cs.add_hyperparameters([
            Constant('algorithm', 'example'),
            UniformIntegerHyperparameter('depth', 1, 10, default_value=3),
            UniformFloatHyperparameter('rate', 0.01, 1.0, default_value=0.1),
            CategoricalHyperparameter('criterion', ['squared', 'absolute'], default_value='squared'),
        ])
        self.valid = dict(algorithm='"example"', depth=8, rate=0.4, criterion='"absolute"')

    def resolve(self, values):
        return config_from_suggestion(self.cs, vz.TrialSuggestion(parameters=values),
                                      return_diagnostics=True)

    def test_valid_unchanged(self):
        config, record = self.resolve(self.valid)
        self.assertEqual(config, dict(algorithm='example', depth=8, rate=0.4, criterion='absolute'))
        self.assertFalse(record['configuration_changed'])
        self.assertTrue(record['direct_valid'])

    def test_missing_and_out_of_range_preserve_other_values(self):
        values = dict(self.valid, rate=2.0)
        del values['depth']
        config, record = self.resolve(values)
        self.assertEqual(config['rate'], 0.1)
        self.assertEqual(config['depth'], 3)
        self.assertEqual(config['criterion'], 'absolute')
        self.assertEqual(record['repairs']['rate']['reason'], 'out_of_range')
        self.assertIn('depth', record['defaulted'])
        self.assertTrue(record['configuration_changed'])

    def test_fractional_integer_and_unknown_category_use_defaults(self):
        config, record = self.resolve(dict(self.valid, depth=2.5, criterion='"unknown"'))
        self.assertEqual(config['depth'], 3)
        self.assertEqual(config['criterion'], 'squared')
        self.assertEqual(config['rate'], 0.4)
        self.assertEqual(record['repairs']['depth']['reason'], 'fractional_integer')

    def test_broken_category_json_is_logged_and_defaulted(self):
        config, record = self.resolve(dict(self.valid, criterion='broken-json'))
        self.assertEqual(config['criterion'], 'squared')
        self.assertEqual(config['depth'], 8)
        self.assertEqual(record['raw_suggestion']['criterion'], 'broken-json')
        self.assertIn('criterion', record['decode_errors'])

    def test_nonfinite_value_does_not_discard_valid_parameters(self):
        suggestion = SimpleNamespace(parameters={key: SimpleNamespace(value=value)
            for key, value in dict(self.valid, rate=float('nan')).items()})
        config, record = config_from_suggestion(self.cs, suggestion, return_diagnostics=True)
        self.assertEqual(config['rate'], 0.1)
        self.assertEqual(config['depth'], 8)
        json.dumps(record, allow_nan=False)

    def test_forbidden_combination_uses_labelled_whole_default(self):
        self.cs.add_forbidden_clause(ForbiddenAndConjunction(
            ForbiddenEqualsClause(self.cs['depth'], 8),
            ForbiddenEqualsClause(self.cs['criterion'], 'absolute')))
        config, record = self.resolve(self.valid)
        self.assertEqual(config, dict(self.cs.get_default_configuration()))
        self.assertEqual(record['fallback'], 'configspace_default')

    def test_no_suggestion_uses_labelled_whole_default(self):
        config, record = config_from_suggestion(self.cs, None, return_diagnostics=True)
        self.assertEqual(config, dict(self.cs.get_default_configuration()))
        self.assertEqual(record['fallback'], 'configspace_default')


if __name__ == '__main__':
    unittest.main()
