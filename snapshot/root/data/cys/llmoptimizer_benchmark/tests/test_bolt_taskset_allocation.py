import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / 'examples/benchmark/domains/train_bolt_taskset_sft.py'
spec = importlib.util.spec_from_file_location('bolt_taskset_training', SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class AllocationTests(unittest.TestCase):
    def configure(self, gpus, cpus):
        args = argparse.Namespace(host='LLMOptimizer2', gpus=gpus, cpus=cpus)
        with patch.object(training.os, 'sched_getaffinity', return_value=set(range(40))):
            training.configure_allocation(args)

    def test_single_gpu_allocation_is_forwarded_to_children(self):
        self.configure('1', '21,22,23')
        self.assertEqual(training.HOST, 'cs-94bac-0693e-server')
        self.assertEqual(training.GPUS, (1,))
        self.assertEqual(training.CORES, (21, 22, 23))
        self.assertEqual(training.allocation_args(),
            ['--host', 'LLMOptimizer2', '--gpus', '1', '--cpus', '21,22,23'])

    def test_invalid_or_overlapping_allocation_is_rejected(self):
        for gpus, cpus in [('1,1', '21,22'), ('1', '21,21'), ('1,2', '21,22,23'), ('1', '40')]:
            with self.subTest(gpus=gpus, cpus=cpus), self.assertRaises(ValueError):
                self.configure(gpus, cpus)

    def test_checkpoint_validation_uses_actual_world_size(self):
        for gpus, cpus in [('1', '21,22,23'), ('0,1', '20,21,22,23')]:
            with self.subTest(gpus=gpus), tempfile.TemporaryDirectory() as directory:
                self.configure(gpus, cpus)
                root = Path(directory)
                (root / 'trainer_state.json').write_text(json.dumps({'global_step': 1}))
                (root / 'latest').write_text('global_step1')
                (root / 'model.safetensors').write_bytes(b'weights')
                (root / 'global_step1').mkdir()
                rng = ['rng_state.pth'] if len(training.GPUS) == 1 else ['rng_state_0.pth', 'rng_state_1.pth']
                for name in rng:
                    (root / name).write_bytes(b'rng')
                for rank in range(len(training.GPUS)):
                    for kind in ('optim', 'model'):
                        name = (f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
                                if kind == 'optim' else f'zero_pp_rank_{rank}_mp_rank_00_model_states.pt')
                        with zipfile.ZipFile(root / 'global_step1' / name, 'w') as archive:
                            archive.writestr('state', b'state')
                training.verify_checkpoint(root, 1)
                (root / rng[-1]).unlink()
                with self.assertRaises(ValueError):
                    training.verify_checkpoint(root, 1)


if __name__ == '__main__':
    unittest.main()
