import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ConfigSpace import ConfigurationSpace, Constant, UniformIntegerHyperparameter

from examples.benchmark.semanticopt_ml_sft import (
    CONFIG, canonical, digest, load_config, load_training_data,
    tokenize_trajectory, trajectory_messages, validate_coverage, validate_steps,
)


class SemanticOptDataTests(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters([Constant('algorithm', 'lightgbm'),
                                    UniformIntegerHyperparameter('lightgbm:max_depth', 3, 15)])
        self.observations = [
            {'config': {'algorithm': 'lightgbm', 'lightgbm:max_depth': depth},
             'objectives': [score], 'trial_state': state}
            for depth, score, state in ((3, -0.6, 0), (4, -0.7, 0), (5, 1e30, 2), (6, -0.8, 0))
        ]
        self.row = {
            'initial_observations': 1, 'objective_scale': 'archive_minimize',
            'teacher': {'name': 'fixture', 'causal_prefix_only': True,
                        'settings': {}, 'code_sha256': 'fixture_only'},
            'steps': [{'observation_index': i, 'prefix_sha256': digest(self.observations[:i]),
                       'mean': -0.65, 'std': 0.05, **({'failure_probability': 0.2} if i == 3 else {})}
                      for i in range(1, 4)],
        }

    def validate(self, row=None, observations=None):
        return validate_steps(row or self.row, observations or self.observations,
                              self.cs, 'lightgbm', 'cls')

    def test_failure_is_feedback_not_a_sentinel_regression_target(self):
        recorded, answers = self.validate()
        self.assertEqual(recorded[2]['status'], 'failure')
        self.assertNotIn('objective', recorded[2])
        self.assertNotIn('failure_probability', answers[1])
        self.assertEqual(answers[2]['failure_probability'], 0.2)

    def test_missing_mean_std_and_nonfinite_labels_are_rejected(self):
        for field, value in [('mean', None), ('std', None), ('std', -0.1),
                             ('mean', float('nan')), ('mean', True)]:
            row = copy.deepcopy(self.row)
            row['steps'][0][field] = value
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_failure_prediction_requires_a_prior_failure(self):
        row = copy.deepcopy(self.row)
        row['steps'][0]['failure_probability'] = 0.1
        with self.assertRaisesRegex(ValueError, 'before an observed failure'):
            self.validate(row)
        for value in (None, -0.1, 1.1):
            row = copy.deepcopy(self.row)
            row['steps'][2]['failure_probability'] = value
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_configurations_cannot_be_repaired_while_keeping_their_scores(self):
        for config in ({'algorithm': 'lightgbm', 'lightgbm:max_depth': -1},
                       {'algorithm': 'lightgbm'}):
            observations = copy.deepcopy(self.observations)
            observations[0]['config'] = config
            with self.assertRaises(ValueError):
                self.validate(observations=observations)

    def test_future_outcomes_do_not_appear_before_their_action(self):
        messages = trajectory_messages(self.row, self.observations, self.cs,
                                       {'dataset': 'fixture'}, 'lightgbm', 'cls')
        changed = copy.deepcopy(self.observations)
        changed[-1]['objectives'] = [-0.9123]
        later = trajectory_messages(self.row, changed, self.cs,
                                    {'dataset': 'fixture'}, 'lightgbm', 'cls')
        self.assertEqual(messages[:-1], later[:-1])
        self.assertNotEqual(messages[-1], later[-1])
        self.assertNotIn('-0.9123', canonical(messages[:-1]))

    def test_prefix_hash_cannot_include_the_next_outcome(self):
        row = copy.deepcopy(self.row)
        row['steps'][0]['prefix_sha256'] = digest(self.observations[:2])
        with self.assertRaisesRegex(ValueError, 'preceding history'):
            self.validate(row)

    def test_topups_cannot_enter_targets_or_warmup_context(self):
        observations = copy.deepcopy(self.observations)
        for obs in observations[2:]:
            obs['extra_info'] = {'source': 'topup_rationale_G'}
        with self.assertRaisesRegex(ValueError, 'Top-up observations'):
            self.validate(observations=observations)
        row = copy.deepcopy(self.row)
        row['initial_observations'] = 3
        row['steps'] = row['steps'][-1:]
        with self.assertRaisesRegex(ValueError, 'Top-up observations'):
            self.validate(row, observations)

    def test_chronology_and_metric_scale_are_checked(self):
        row = copy.deepcopy(self.row)
        row['steps'][0]['observation_index'] = 2
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            self.validate(row)
        row = copy.deepcopy(self.row)
        row['objective_scale'] = 'maximize'
        with self.assertRaisesRegex(ValueError, 'original'):
            self.validate(row)
        observations = copy.deepcopy(self.observations)
        observations[0]['objectives'] = [0.8]
        with self.assertRaisesRegex(ValueError, 'allowed range'):
            self.validate(observations=observations)

    def test_heldout_task_rejected_before_reading_its_history(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'data.jsonl'
            dataset.write_text(json.dumps({'task_id': 'cls/heldout/lightgbm'}) + '\n')
            config = {'dataset': str(dataset)}
            with patch.dict(os.environ, {}, clear=True), \
                    patch('examples.benchmark.semanticopt_ml_sft.allowed_task_ids', return_value={'cls/train/lightgbm'}):
                with self.assertRaisesRegex(ValueError, 'fixed train80'):
                    load_training_data(config, None)

    def test_paper_hyperparameters_and_default_does_not_train(self):
        config = load_config(CONFIG)
        self.assertEqual(config['effective_batch_size'], 64)
        self.assertEqual(config['max_seq_length'], 32768)
        result = subprocess.run([sys.executable, str(CONFIG.parents[2] / 'examples/benchmark/semanticopt_ml_sft.py')],
                                capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['training_started'])
        self.assertFalse(plan['automatic_launch_after_evaluations'])

    def test_sequence_length_is_configurable_within_native_model_context(self):
        config = load_config(CONFIG)
        model_config = json.loads((Path(config['model_name_or_path']) / 'config.json').read_text())
        native_limit = model_config['text_config']['max_position_embeddings']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for limit in (24576, 32768, native_limit):
                path.write_text(json.dumps(dict(config, max_seq_length=limit)))
                self.assertEqual(load_config(path)['max_seq_length'], limit)
            for limit in (None, True, 0, -1, 32768.0):
                path.write_text(json.dumps(dict(config, max_seq_length=limit)))
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    load_config(path)
            path.write_text(json.dumps(dict(config, max_seq_length=native_limit + 1)))
            with self.assertRaisesRegex(ValueError, 'exceeds the local model context limit'):
                load_config(path)

    def test_partial_annotation_corpus_cannot_start_full_training(self):
        manifest = {'format_version': 1, 'status': 'complete', 'dataset_sha256': 'data',
                    'train_manifest_sha256': 'train', 'represented_tasks': ['a'],
                    'excluded_tasks': {'b': 'insufficient causal prefix'},
                    'trajectories': 1, 'horizon_policy': 'complete available trajectories'}
        validate_coverage(manifest, {'a', 'b'}, {'a'}, 1, 'data', 'train')
        partial = dict(manifest, excluded_tasks={})
        with self.assertRaisesRegex(ValueError, 'Full train80 coverage'):
            validate_coverage(partial, {'a', 'b'}, {'a'}, 1, 'data', 'train')
        with self.assertRaisesRegex(ValueError, 'marked complete'):
            validate_coverage(dict(manifest, status='running'), {'a', 'b'}, {'a'}, 1, 'data', 'train')


class SemanticOptTokenizerTests(unittest.TestCase):
    setUp = SemanticOptDataTests.setUp

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(
            '/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B', local_files_only=True)

    def test_real_qwen_masks_every_user_turn_and_preserves_every_answer(self):
        messages = trajectory_messages(self.row, self.observations, self.cs,
                                       {'dataset': 'fixture'}, 'lightgbm', 'cls')
        encoded = tokenize_trajectory(self.tokenizer, messages, 24576)
        supervised = self.tokenizer.decode([v for v in encoded['labels'] if v != -100],
                                           skip_special_tokens=False)
        expected = ''.join(m['content'] + '<|im_end|>\n' for m in messages if m['role'] == 'assistant')
        self.assertEqual(supervised, expected)
        self.assertEqual(len(encoded['input_ids']), len(encoded['labels']))
        self.assertIn(-100, encoded['labels'])
        from transformers import DataCollatorForSeq2Seq
        batch = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=-100)([encoded])
        self.assertEqual(batch['labels'][0].tolist(), encoded['labels'])

    def test_token_limit_rejects_instead_of_truncating(self):
        messages = trajectory_messages(self.row, self.observations, self.cs,
                                       {'dataset': 'fixture'}, 'lightgbm', 'cls')
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            tokenize_trajectory(self.tokenizer, messages, 10)


if __name__ == '__main__':
    unittest.main()
