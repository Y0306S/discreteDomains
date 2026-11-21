"""Training-position and untouched-source checks for the optimizer migration."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('bolt_peptide_adamw8bit', TOOLS / 'bolt_peptide_adamw8bit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OptimizerMigration(unittest.TestCase):
    def test_same_effective_batch_and_five_epoch_budget(self):
        old = dict(num_train_epochs=5, learning_rate=1e-5, cutoff_len=1024,
                   dataset='unchanged', finetuning_type='full', train_on_prompt=False)
        config = module.config_for_migration(old, '/verified/checkpoint', '/new/output')
        batch = config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
        self.assertEqual(batch, 16)
        self.assertEqual(8000 // batch * config['num_train_epochs'], 2500)
        self.assertEqual(config['learning_rate'], old['learning_rate'])
        self.assertEqual(config['dataset'], old['dataset'])
        self.assertTrue(config['pure_bf16'])
        self.assertIsNone(config['deepspeed'])
        self.assertEqual(config['optim'], 'adamw_bnb_8bit')

    def test_cosine_schedule_continues_without_repeating_warmup(self):
        import torch
        from transformers import get_scheduler

        def setup():
            parameter = torch.nn.Parameter(torch.tensor([0.25]))
            optimizer = torch.optim.AdamW([parameter], lr=1e-5)
            scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=20, num_training_steps=2500)
            return parameter, optimizer, scheduler

        parameter, optimizer, scheduler = setup()
        for _ in range(100):
            parameter.grad = torch.tensor([0.1])
            optimizer.step()
            scheduler.step()
        _, fresh_optimizer, fresh_scheduler = setup()
        rates = module.restore_schedule(SimpleNamespace(optimizer=fresh_optimizer,
                                                        lr_scheduler=fresh_scheduler), 100)
        self.assertEqual(rates, scheduler.get_last_lr())
        self.assertEqual(fresh_scheduler.last_epoch, 100)
        self.assertFalse(fresh_optimizer.state)
        optimizer.step()
        scheduler.step()
        fresh_optimizer.step()
        fresh_scheduler.step()
        self.assertEqual(fresh_scheduler.get_last_lr(), scheduler.get_last_lr())

    def checkpoint(self, path):
        import torch
        path.mkdir()
        (path / 'trainer_state.json').write_text(json.dumps(dict(global_step=100, train_batch_size=1)))
        (path / 'latest').write_text('global_step100')
        (path / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'weight': 'part.safetensors'})))
        (path / 'part.safetensors').write_bytes(b'unchanged-test-shard')
        (path / 'config.json').write_text('{}')
        torch.save(dict(cpu=torch.random.get_rng_state(), cuda=[torch.tensor([1]), torch.tensor([2])]),
                   path / 'rng_state_0.pth')
        (path / 'rng_state_1.pth').write_bytes(b'rank-one-rng')
        tag = path / 'global_step100'
        tag.mkdir()
        for rank in range(2):
            for name in ('bf16_zero_pp_rank_%d_mp_rank_00_optim_states.pt',
                         'zero_pp_rank_%d_mp_rank_00_model_states.pt'):
                (tag / (name % rank)).write_bytes(b'state')

    def test_source_preserved_rng_adapted_and_old_optimizer_excluded(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'resume'
            self.checkpoint(source)
            before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            module.prepare_resume(source, target)
            self.assertEqual(before, {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()})
            state = json.loads((target / 'trainer_state.json').read_text())
            self.assertEqual(state, dict(global_step=100, train_batch_size=4))
            rng = torch.load(target / 'rng_state.pth', weights_only=False)
            self.assertEqual(rng['cuda'].tolist(), [1])
            self.assertTrue((target / 'part.safetensors').is_symlink())
            self.assertFalse((target / 'global_step100').exists())
            self.assertFalse((target / 'optimizer.pt').exists())

    def test_incomplete_two_rank_checkpoint_rejected_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'resume'
            self.checkpoint(source)
            (source / 'global_step100/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                module.prepare_resume(source, target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
