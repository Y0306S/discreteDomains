import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ConfigSpace import ConfigurationSpace, Constant, UniformIntegerHyperparameter

from examples.benchmark.export_bolt_ml_top10 import build_messages, select_top_k
from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs
from examples.benchmark.supervise_bolt_ml_sft import checkpoint_step, gpu_launch_requirements, parse_cpu_cores, ram_resources, stop_child, stop_reason
from examples.benchmark.bolt_ml_train_rank import configure_checkpoint_shards, load_training_config, restore_training_policy


class BoltSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters([Constant('algorithm', 'lightgbm'),
                                    UniformIntegerHyperparameter('lightgbm:max_depth', 3, 15)])

    def observation(self, depth, score, state=0):
        return {'config': {'algorithm': 'lightgbm', 'lightgbm:max_depth': depth},
                'objectives': [score], 'trial_state': state}

    def test_invalid_failed_missing_and_nonfinite_cannot_be_targets(self):
        rows = [self.observation(-1, -0.99), self.observation(4, -0.98, 2),
                self.observation(5, float('nan')), self.observation(6, -0.8)]
        rows.append({'config': {'algorithm': 'lightgbm'}, 'objectives': [-0.95], 'trial_state': 0})
        selected, rejected, distinct = select_top_k(rows, self.cs, 'lightgbm', 'cls')
        self.assertEqual([r['observation_index'] for r in selected], [3])
        self.assertEqual(distinct, 1)
        self.assertEqual(len(rejected), 4)
        self.assertEqual(rows[0]['config']['lightgbm:max_depth'], -1)

    def test_rank_deduplicate_then_take_top_k(self):
        rows = [self.observation(3, 20), self.observation(4, 10),
                self.observation(3, 5), self.observation(5, 10), self.observation(6, 12)]
        selected, rejected, distinct = select_top_k(rows, self.cs, 'lightgbm', 'rgs', k=2)
        self.assertEqual([r['observation_index'] for r in selected], [2, 1])
        self.assertEqual(distinct, 4)
        self.assertEqual(len(rejected), 3)

    def test_real_schema_excludes_poisson_for_negative_regression(self):
        for algorithm in ('random_forest', 'extra_trees'):
            cs = build_hpo_cs('rgs', algorithm, y_neg_mask=False)
            config = dict(cs.get_default_configuration())
            config[f'{algorithm}:criterion'] = 'poisson'
            selected, rejected, _ = select_top_k(
                [{'config': config, 'objectives': [0.1], 'trial_state': 0}], cs, algorithm, 'rgs')
            self.assertFalse(selected)
            self.assertEqual(rejected[0]['reason'], 'invalid_configuration')

    def test_prompt_carries_schema_but_no_target_or_score(self):
        cfg = dict(self.cs.get_default_configuration())
        messages = build_messages({'dataset': 'fixture'}, 'lightgbm', 'cls', self.cs, cfg)
        self.assertEqual(json.loads(messages[-1]['content']), cfg)
        self.assertIn('"lower": 3', messages[1]['content'])
        self.assertIn('"upper": 15', messages[1]['content'])
        changed = dict(cfg, **{'lightgbm:max_depth': 12})
        self.assertEqual(messages[:2], build_messages({'dataset': 'fixture'}, 'lightgbm', 'cls', self.cs, changed)[:2])


class BoltSupervisorTests(unittest.TestCase):
    def test_cpu_allocation_requires_two_disjoint_four_core_ranks(self):
        cores = parse_cpu_cores('0,1,2,3,4,5,6,7')
        self.assertEqual(cores[:4], (0, 1, 2, 3))
        self.assertEqual(cores[4:], (4, 5, 6, 7))
        for invalid in ('0,1,2,3', '0,1,2,3,4,5,6,6', '-1,1,2,3,4,5,6,7'):
            with self.assertRaises(ValueError):
                parse_cpu_cores(invalid)

    def test_explicit_unlimited_policy_disables_memory_stops(self):
        self.assertIsNone(stop_reason({'available_ram_gib': 0.001, 'container_ram_limit_gib': 400,
                                       'gpu_free_gib': {'0': 0.001}}, disable_memory_limits=True))

    def test_checkpoint_sharding_preserves_save_arguments(self):
        from unittest.mock import Mock
        original = Mock()
        model = SimpleNamespace(save_pretrained=original)
        configure_checkpoint_shards(model)
        state = {'weight': object()}
        model.save_pretrained('/checkpoint', state_dict=state)
        original.assert_called_once_with('/checkpoint', state_dict=state, max_shard_size='1GB')

    def test_launch_requirement_tracks_allocator_budget(self):
        self.assertEqual(gpu_launch_requirements({'gpu_total_gib': {'0': 80, '1': 80}}),
                         {'0': 26, '1': 26})

    def test_json_config_preserves_scientific_notation_as_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_config.json'
            path.write_text(json.dumps({'learning_rate': 1e-5, 'adam_epsilon': 1e-8}))
            config = load_training_config(path)
            self.assertIsInstance(config['learning_rate'], float)
            self.assertIsInstance(config['adam_epsilon'], float)
            self.assertEqual(config['learning_rate'], 0.00001)

    def test_resumed_trainer_uses_new_checkpoint_interval(self):
        from transformers.trainer_callback import DefaultFlowCallback, TrainerControl, TrainerState
        from transformers.trainer_utils import IntervalStrategy
        state = TrainerState(global_step=324, max_steps=1620, save_steps=500)
        args = SimpleNamespace(save_steps=50, logging_steps=1, eval_steps=None,
                               save_strategy=IntervalStrategy.STEPS,
                               logging_strategy=IntervalStrategy.STEPS,
                               logging_first_step=False, eval_strategy=IntervalStrategy.NO)
        restore_training_policy(args, state)
        flow = DefaultFlowCallback()
        state.global_step = 349
        self.assertFalse(flow.on_step_end(args, state, TrainerControl()).should_save)
        state.global_step = 350
        self.assertTrue(flow.on_step_end(args, state, TrainerControl()).should_save)

    def test_container_limit_overrides_host_free_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'memory.max').write_text(str(400 * 2**30))
            (root / 'memory.current').write_text(str(390 * 2**30))
            (root / 'memory.stat').write_text(
                f'file {80 * 2**30}\nshmem {70 * 2**30}\ninactive_file {5 * 2**30}\n')
            resources = ram_resources(root, host_available=600 * 2**30)
            self.assertEqual(resources['available_ram_gib'], 15)
            self.assertEqual(resources['container_ram_limit_gib'], 400)
            self.assertEqual(stop_reason(resources), 'Available CPU RAM below 40 GiB')

    def test_ram_stop_uses_ten_percent_of_container_limit(self):
        resources = {'container_ram_limit_gib': 400, 'available_ram_gib': 95}
        self.assertIsNone(stop_reason(resources))
        resources['available_ram_gib'] = 39
        self.assertEqual(stop_reason(resources), 'Available CPU RAM below 40 GiB')

    def test_shared_memory_is_not_counted_as_reclaimable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'memory.max').write_text(str(400 * 2**30))
            (root / 'memory.current').write_text(str(390 * 2**30))
            (root / 'memory.stat').write_text(
                f'file {80 * 2**30}\nshmem {80 * 2**30}\ninactive_file {80 * 2**30}\n')
            self.assertEqual(ram_resources(root, 600 * 2**30)['available_ram_gib'], 10)

    def test_unlimited_cgroup_uses_host_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'memory.max').write_text('max')
            self.assertEqual(ram_resources(root, 600 * 2**30)['available_ram_gib'], 600)

    def test_gpu_free_memory_does_not_stop_training(self):
        self.assertIsNone(stop_reason({'gpu_free_gib': {'0': 0.5, '1': 9.9},
                                       'available_ram_gib': 120}))

    def test_resume_requires_both_ranks_optimizer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'trainer_state.json').write_text('{"global_step": 324}')
            (root / 'latest').write_text('global_step324')
            (root / 'global_step324').mkdir()
            (root / 'model.safetensors').write_bytes(b'model')
            for rank in (0, 1):
                (root / f'rng_state_{rank}.pth').write_bytes(b'rng')
                (root / 'global_step324' / f'zero_pp_rank_{rank}_mp_rank_00_model_states.pt').write_bytes(b'model')
                (root / 'global_step324' / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'optimizer')
            self.assertEqual(checkpoint_step(root), 324)
            (root / 'model.safetensors').unlink()
            (root / 'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'weight': 'model-00001-of-00001.safetensors'}}))
            (root / 'model-00001-of-00001.safetensors').write_bytes(b'shard')
            self.assertEqual(checkpoint_step(root), 324)
            (root / 'model-00001-of-00001.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                checkpoint_step(root)
            (root / 'model-00001-of-00001.safetensors').write_bytes(b'shard')
            (root / 'global_step324' / 'bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                checkpoint_step(root)

    def test_stop_affects_only_owned_process_group(self):
        command = [sys.executable, '-c', 'import time; time.sleep(60)']
        owned = subprocess.Popen(command, start_new_session=True)
        unrelated = subprocess.Popen(command, start_new_session=True)
        try:
            stop_child(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
            stop_child(owned)
        finally:
            stop_child(owned)
            unrelated.terminate()
            unrelated.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
