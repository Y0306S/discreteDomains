import copy
import json
from pathlib import Path
import sys
import unittest

from ConfigSpace import CategoricalHyperparameter, Configuration, ConfigurationSpace, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

SCRIPTS = Path(__file__).resolve().parents[2] / 'examples/benchmark/domains'
sys.path.insert(0, str(SCRIPTS))
import export_bolt_taskset_top10 as exporter


class BoltTaskSetExportTest(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        family = CategoricalHyperparameter('optimizer_family', ['a', 'b'])
        rate = UniformFloatHyperparameter('a:learning_rate', 0.01, 1.0)
        self.cs.add_hyperparameters([family, rate])
        self.cs.add_condition(EqualsCondition(rate, family, 'a'))

    def observation(self, rate, score, state=0):
        return dict(config={'optimizer_family': 'a', 'a:learning_rate': rate},
                    objectives=[score], trial_state=state)

    def test_minimization_duplicate_outcomes_and_stable_ties(self):
        rows = [self.observation(.1, .8), self.observation(.2, .2),
                self.observation(.1, .1), self.observation(.3, .2)]
        selected, excluded, distinct = exporter.select_top_k(rows, self.cs, k=3)
        self.assertEqual([r['observation_index'] for r in selected], [2, 1, 3])
        self.assertEqual(distinct, 3)
        self.assertEqual(excluded, [{'observation_index': 0, 'reason': 'duplicate_configuration'}])

    def test_invalid_failed_missing_and_inactive_fields_are_never_repaired(self):
        missing = {'config': {'optimizer_family': 'a'}, 'objectives': [0], 'trial_state': 0}
        inactive = {'config': {'optimizer_family': 'b', 'a:learning_rate': .1}, 'objectives': [0], 'trial_state': 0}
        rows = [self.observation(-1, 0), self.observation(.2, 0, state=1),
                self.observation(.3, float('nan')), missing, inactive, self.observation(.4, .5)]
        original = copy.deepcopy(rows)
        selected, excluded, distinct = exporter.select_top_k(rows, self.cs)
        self.assertEqual([r['observation_index'] for r in selected], [5])
        self.assertEqual(distinct, 1)
        self.assertEqual(len(excluded), 5)
        self.assertEqual(rows[0], original[0])
        self.assertEqual(rows[3], original[3])

    def test_top_ten_are_separate_examples_with_identical_context(self):
        rows = [self.observation(.05 + i * .05, 1 - i * .01) for i in range(12)]
        selected, excluded, _ = exporter.select_top_k(rows, self.cs)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(excluded), 2)
        task = dict(task_family='mlp_family', spec={'dataset': 'example', 'hidden_units': [32]})
        messages = [exporter.build_messages('mlp_family_seed1', task, self.cs, r['config']) for r in selected]
        self.assertTrue(all(m[:2] == messages[0][:2] for m in messages))
        self.assertEqual(len({m[-1]['content'] for m in messages}), 10)
        self.assertEqual([m['role'] for m in messages[0]], ['system', 'user', 'assistant'])
        for result, record in zip(messages, selected):
            self.assertEqual(json.loads(result[-1]['content']), record['config'])
            self.assertNotIn('objective_minimize', result[1]['content'])
            self.assertNotIn('observation_index', result[1]['content'])

    def nesterov_space(self):
        cs = ConfigurationSpace()
        family = CategoricalHyperparameter('optimizer_family', ['nadamw_grid', 'a'])
        nesterov = CategoricalHyperparameter(exporter.NESTEROV_HP, [False, True])
        rate = UniformFloatHyperparameter('nadamw_grid:learning_rate', .01, 1.)
        cs.add_hyperparameters([family, nesterov, rate])
        cs.add_conditions([EqualsCondition(hp, family, 'nadamw_grid') for hp in (nesterov, rate)])
        return cs

    def test_nesterov_projection_preserves_source_and_deduplicates_effective_configs(self):
        source = self.nesterov_space()
        effective = exporter.effective_space(source)
        config = {'optimizer_family': 'nadamw_grid', exporter.NESTEROV_HP: True,
                  'nadamw_grid:learning_rate': .1}
        rows = [dict(config=config, objectives=[.1], trial_state=0),
                dict(config=dict(config, **{exporter.NESTEROV_HP: False}), objectives=[.2], trial_state=0)]
        original = copy.deepcopy(rows)
        selected, excluded, distinct = exporter.select_top_k(rows, source, effective_cs=effective)
        self.assertEqual(rows, original)
        self.assertEqual(distinct, 1)
        self.assertEqual(selected[0]['objective_minimize'], .1)
        self.assertIs(selected[0]['config'][exporter.NESTEROV_HP], False)
        self.assertEqual(selected[0]['original_config'], original[0]['config'])
        self.assertEqual(selected[0]['original_config_sha256'], exporter.value_digest(original[0]['config']))
        self.assertEqual(selected[0]['configuration_adjustments'], [dict(
            parameter=exporter.NESTEROV_HP, recorded_value=True, effective_value=False,
            reason='legacy_factory_ignored_use_nesterov')])
        self.assertEqual(excluded, [dict(observation_index=1, reason='duplicate_configuration')])
        with self.assertRaises(ValueError):
            effective.check_configuration(Configuration(effective, values=config))
        Configuration(source, values={'optimizer_family': 'a'}).is_valid_configuration()
        Configuration(effective, values={'optimizer_family': 'a'}).is_valid_configuration()

    def test_original_nesterov_values_are_validated_before_projection(self):
        source = self.nesterov_space()
        missing = {'optimizer_family': 'nadamw_grid', 'nadamw_grid:learning_rate': .1}
        invalid = dict(missing, **{exporter.NESTEROV_HP: 'invalid'})
        rows = [dict(config=c, objectives=[.1], trial_state=0) for c in (missing, invalid)]
        selected, excluded, distinct = exporter.select_top_k(rows, source, effective_cs=exporter.effective_space(source))
        self.assertEqual(selected, [])
        self.assertEqual(distinct, 0)
        self.assertEqual([r['reason'] for r in excluded], ['invalid_configuration'] * 2)

    def test_prompt_describes_historical_schedule_scoring_and_encoded_fields(self):
        task = dict(task_family='quadratic_family', spec={'dims': 10})
        prompt = exporter.build_messages('quadratic_family_seed1', task, self.cs,
                                         {'optimizer_family': 'a', 'a:learning_rate': .1})[1]['content']
        for detail in ('Evaluation budget: 1000', 'schedule horizon: 10000',
                       'initialization (step 0) and step 1000', 'minimum normalized VALID_OUTER',
                       'type EQ means equality', 'numeric field is still active',
                       'actual positive value, not its logarithm', 'synthetic optimization problem'):
            self.assertIn(detail, prompt)


if __name__ == '__main__':
    unittest.main()
