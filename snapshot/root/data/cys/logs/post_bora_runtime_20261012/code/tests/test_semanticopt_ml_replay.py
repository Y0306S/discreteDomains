import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Constant, UniformIntegerHyperparameter

from examples.benchmark.replay_semanticopt_ml import (
    InsufficientHistory, RF_SETTINGS, allocation_leaf_cap, load_record, replay_steps, save_record,
)
from examples.benchmark.semanticopt_ml_sft import digest, history_prefix_digest, original_history_length


class SemanticOptReplayTests(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters([Constant('algorithm', 'lightgbm'),
                                    UniformIntegerHyperparameter('lightgbm:max_depth', 3, 15)])
        self.observations = [
            {'config': {'algorithm': 'lightgbm', 'lightgbm:max_depth': 3 + i},
             'objectives': [1e30 if i == 2 else -0.5 - 0.04 * i],
             'trial_state': 2 if i == 2 else 0}
            for i in range(9)
        ]

    def replay(self, observations=None, task_type='cls'):
        return replay_steps(self.observations if observations is None else observations,
                            self.cs, 'lightgbm', task_type, seed=42)

    def test_current_and_future_outcomes_cannot_change_a_prediction(self):
        warmup, baseline = self.replay()
        changed = copy.deepcopy(self.observations)
        changed[6]['objectives'] = [-0.99]
        changed[7]['trial_state'] = 2
        changed[7]['objectives'] = [1e30]
        _, later = self.replay(changed)
        self.assertEqual(warmup, 5)
        self.assertEqual(baseline[:2], later[:2])
        self.assertNotEqual(baseline[2]['mean'], later[2]['mean'])
        self.assertEqual(baseline[0]['objective_fit_observations'], 4)
        self.assertEqual(baseline[0]['failure_fit_observations'], 5)
        self.assertEqual(baseline[0]['prefix_sha256'], digest(self.observations[:5]))
        self.assertTrue(all(-1 <= s['mean'] <= 0 for s in baseline))
        self.assertTrue(all(0 <= s['failure_probability'] <= 1 for s in baseline))

    def test_failure_probability_only_after_an_observed_failure(self):
        observations = copy.deepcopy(self.observations)
        observations[2].update(trial_state=0, objectives=[-0.58])
        observations[6].update(trial_state=2, objectives=[1e30])
        _, steps = self.replay(observations)
        self.assertNotIn('failure_probability', steps[0])
        self.assertNotIn('failure_probability', steps[1])
        self.assertIn('failure_probability', steps[2])
        self.assertEqual(steps[2]['objective_fit_observations'], 6)

    def test_archived_infinite_failure_scores_are_hashed_but_not_fit(self):
        _, baseline = self.replay()
        observations = copy.deepcopy(self.observations)
        observations[2]['objectives'] = [float('inf')]
        _, steps = self.replay(observations)
        self.assertEqual(steps[0]['prefix_sha256'], history_prefix_digest(observations[:5]))
        self.assertNotEqual(steps[0]['prefix_sha256'], baseline[0]['prefix_sha256'])
        for before, after in zip(baseline, steps):
            for key in ('mean', 'std', 'failure_probability'):
                self.assertEqual(before[key], after[key])

    def test_leaf_allocation_bound_preserves_regression_and_failure_predictions(self):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        rng = np.random.RandomState(123)
        X = rng.random((65, 7))
        y = rng.random(65)
        target = rng.random((100, 7))
        for cls, values in ((RandomForestRegressor, y), (RandomForestClassifier, y > .6)):
            predictions = []
            for cap in (2**20, allocation_leaf_cap(len(y))):
                model = cls(n_estimators=10, min_samples_leaf=3, min_samples_split=3,
                            random_state=42, max_leaf_nodes=cap, n_jobs=1).fit(X, values)
                predictions.append([tree.predict(target).tolist() for tree in model.estimators_])
            self.assertEqual(*predictions)

    def test_reported_uncertainty_is_tree_standard_deviation(self):
        from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
        from openbox.utils.config_space import convert_configurations_to_array
        from openbox.utils.util_funcs import get_types
        _, steps = self.replay()
        prefix = [o for o in self.observations[:5] if o['trial_state'] == 0]
        X = convert_configurations_to_array([Configuration(self.cs, values=o['config']) for o in prefix])
        target = convert_configurations_to_array([Configuration(self.cs, values=self.observations[5]['config'])])
        model = skRandomForestWithInstances(*get_types(self.cs), seed=42, **RF_SETTINGS)
        model.train(X, np.asarray([o['objectives'][0] for o in prefix]))
        tree_predictions = [float(tree.predict(target)[0]) for tree in model.rf.estimators_]
        self.assertAlmostEqual(steps[0]['std'], float(np.std(tree_predictions)))
        self.assertAlmostEqual(steps[0]['mean'], float(np.mean(tree_predictions)))

    def test_topups_are_never_fit_or_annotated_and_sources_are_not_mutated(self):
        baseline = self.replay()
        observations = copy.deepcopy(self.observations)
        observations.append({'config': {'invalid_topup_config': True}, 'trial_state': 0,
                             'objectives': [-999], 'extra_info': {'source': 'topup_rationale_G'}})
        before = copy.deepcopy(observations)
        self.assertEqual(self.replay(observations), baseline)
        self.assertEqual(observations, before)
        self.assertEqual(original_history_length(observations), 9)
        observations.append(copy.deepcopy(self.observations[0]))
        with self.assertRaisesRegex(ValueError, 'trailing segment'):
            self.replay(observations)

    def test_short_histories_and_empty_success_prefix_are_explicit(self):
        with self.assertRaises(InsufficientHistory):
            self.replay(self.observations[:2])
        observations = copy.deepcopy(self.observations)
        for obs in observations[:6]:
            obs.update(trial_state=2, objectives=[1e30])
        warmup, steps = self.replay(observations)
        self.assertEqual(warmup, 7)
        self.assertEqual(steps[0]['objective_fit_observations'], 1)
        self.assertAlmostEqual(steps[0]['mean'], observations[6]['objectives'][0])
        self.assertAlmostEqual(steps[0]['std'], 0)
        for obs in observations:
            obs.update(trial_state=2, objectives=[1e30])
        with self.assertRaisesRegex(InsufficientHistory, 'No successful'):
            self.replay(observations)

    def test_regression_keeps_full_precision_original_mse_scale(self):
        observations = copy.deepcopy(self.observations)
        for i, obs in enumerate(observations):
            if obs['trial_state'] == 0:
                obs['objectives'] = [123.456789 + i * 10000.0]
        _, steps = self.replay(observations, task_type='rgs')
        self.assertGreater(steps[0]['mean'], 1000)
        self.assertGreater(steps[0]['std'], 1)
        self.assertEqual(self.replay(observations, task_type='rgs')[1], steps)

    def test_resume_checks_recipe_source_and_record_integrity(self):
        recipe = {'setting': 42}
        source = {'task_id': 'cls/train/lightgbm', 'source_sha256': 'source'}
        record = {**source, 'recipe_sha256': digest(recipe), 'status': 'complete'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task.json'
            save_record(path, record)
            self.assertEqual(load_record(path, source, recipe), record)
            with self.assertRaisesRegex(ValueError, 'Stale or changed'):
                load_record(path, source, {'setting': 43})
            altered = json.loads(path.read_text())
            altered['record']['status'] = 'excluded'
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'Stale or changed'):
                load_record(path, source, recipe)


if __name__ == '__main__':
    unittest.main()
