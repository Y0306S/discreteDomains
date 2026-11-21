"""CPU checks for accumulation, no-thinking labels and verified LoRA loading."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from examples.benchmark import train_p5_reinforce as trainer
from examples.benchmark.hf_qwen_client import HuggingFaceQwenClient, load_verified_adapter


class OfflineTrainingRepairs(unittest.TestCase):
    def test_provenance_pins_inputs_and_effective_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('buffer', 'base', 'adapter'):
                (root / name).mkdir()
            (root / 'buffer/rollouts_0.jsonl').write_text('{}\n')
            (root / 'base/config.json').write_text('{}')
            (root / 'train.jsonl').write_text('{}\n')
            args = SimpleNamespace(buffer_root=root / 'buffer', base_model=str(root / 'base'),
                init_adapter=root / 'adapter', train_manifest=root / 'train.jsonl',
                dataset_order_file=None, grad_accum=8, max_seq_len=8192, output_dir=root / 'output')
            tokenizer = SimpleNamespace(name_or_path=str(root / 'base'), chat_template='template', eos_token_id=2)
            report = trainer.training_provenance(args, tokenizer, [{'completion': '{}'}])
            self.assertIn(str(root / 'buffer/rollouts_0.jsonl'), report['files_sha256'])
            self.assertEqual(report['arguments']['grad_accum'], 8)
            self.assertFalse(report['enable_thinking'])
            self.assertEqual(report['answer_eos_token_id'], 2)
            trainer.freeze_training_inputs(args, report)
            with tarfile.open(report['input_archive']['path']) as archive:
                self.assertEqual(archive.extractfile('buffer/rollouts_0.jsonl').read(), b'{}\n')
            json.dumps(report)

    def test_partial_batches_and_skipped_boundaries_apply_every_valid_gradient(self):
        for values, expected_steps, expected_skips in (
            ([1] * 8, 1, 0), ([1] * 7 + [0], 1, 1), ([0] + [1] * 7, 1, 1),
            ([1] * 7 + [0] + [1] * 3 + [0], 2, 2), ([0, 0], 0, 2), ([], 0, 0)):
            with self.subTest(values=values):
                policy, ref = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
                torch.nn.init.zeros_(policy.weight)
                opt = torch.optim.SGD(policy.parameters(), lr=1)
                rows = [{"messages": [], "completion": value, "advantage": 1.0} for value in values]
                with patch.object(trainer, 'build_batch_ids', side_effect=lambda t, m, c, d, l: (None, None) if c else None), \
                     patch.object(trainer, 'mean_token_logprob', side_effect=lambda model, ids, labels: model.weight.sum()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = trainer.train_on_order(policy=policy, ref=ref, opt=opt, tokenizer=None,
                        records=rows, order=list(range(len(rows))),
                        args=SimpleNamespace(max_seq_len=100, grad_accum=8, kl_beta=0, max_grad_norm=1000))
                self.assertEqual(result, (expected_steps, expected_skips))
                self.assertAlmostEqual(policy.weight.item(), expected_steps)
                self.assertIsNone(policy.weight.grad)

    def test_actual_qwen_template_masks_prompt_and_trains_eos(self):
        tokenizer = AutoTokenizer.from_pretrained('/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B', local_files_only=True)
        messages = [{'role': 'user', 'content': 'Return a JSON configuration.'}]
        completion = '{"config":{"n_estimators":100}}'
        ids, labels = trainer.build_batch_ids(tokenizer, messages, completion, torch.device('cpu'), 512)
        prompt = tokenizer(trainer.render_s1b_prompt(tokenizer, messages), add_special_tokens=False)['input_ids']
        self.assertEqual(ids[0, :len(prompt)].tolist(), prompt)
        self.assertTrue((labels[0, :len(prompt)] == -100).all())
        self.assertEqual(labels[0, -1].item(), tokenizer.eos_token_id)
        self.assertNotIn('<think>', tokenizer.decode(prompt))
        self.assertEqual(tokenizer.decode(ids[0, len(prompt):-1]), completion)
        self.assertIsNone(trainer.build_batch_ids(tokenizer, messages, completion, torch.device('cpu'), len(prompt)))

    def test_rl_prompt_ids_exactly_match_llamafactory_sft_orpo_template(self):
        from llamafactory.data.template import TEMPLATES
        tokenizer = AutoTokenizer.from_pretrained('/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B', local_files_only=True)
        messages = [{'role': 'system', 'content': 'A system.'}, {'role': 'user', 'content': 'A task.'}]
        completion = 'Rationale: Brief.\n\nJSON:\n{"config":{"model:param":1}}'
        lf_prompt, lf_answer = TEMPLATES['qwen3_5_nothink'].encode_oneturn(
            tokenizer, [messages[1], {'role': 'assistant', 'content': completion}], system=messages[0]['content'])
        ids, labels = trainer.build_batch_ids(tokenizer, messages, completion, torch.device('cpu'), 512)
        self.assertEqual(ids[0, :len(lf_prompt)].tolist(), lf_prompt)
        self.assertTrue((labels[0, :len(lf_prompt)] == -100).all())
        answer_ids = labels[0, len(lf_prompt):].tolist()
        self.assertEqual(lf_answer[:len(answer_ids)], answer_ids)
        self.assertEqual(tokenizer.decode(lf_answer[len(answer_ids):]), '\n')

    def test_adapter_inference_uses_the_same_answer_boundary_as_rl(self):
        tokenizer = AutoTokenizer.from_pretrained('/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B', local_files_only=True)
        client = HuggingFaceQwenClient(adapter_path='unused-in-mock', temperature=0)
        client._tokenizer = tokenizer
        client._ensure_loaded = lambda: None
        client._device = lambda: torch.device('cpu')
        captured = {}
        def generate(**kwargs):
            captured['input_ids'] = kwargs['input_ids'][0].tolist()
            return kwargs['input_ids']
        client._model = SimpleNamespace(generate=generate)
        client.complete('A system.', 'A task.')
        messages = [{'role': 'system', 'content': 'A system.'}, {'role': 'user', 'content': 'A task.'}]
        expected = tokenizer(trainer.render_s1b_prompt(tokenizer, messages), add_special_tokens=False)['input_ids']
        self.assertEqual(captured['input_ids'], expected)
        self.assertNotIn('<think>', tokenizer.decode(captured['input_ids']))

    def test_rationale_and_json_are_both_trained_with_the_sft_format(self):
        from examples.benchmark.p5_response_format import format_response, parse_response
        tokenizer = AutoTokenizer.from_pretrained('/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B', local_files_only=True)
        completion = format_response('More trees can reduce variance at additional runtime cost.',
                                     {'extra_trees:n_estimators': 100})
        messages = [{'role': 'user', 'content': 'Return Rationale: followed by JSON: and the config object.'}]
        ids, labels = trainer.build_batch_ids(tokenizer, messages, completion, torch.device('cpu'), 512)
        supervised = labels[0][labels[0] != -100].tolist()
        self.assertEqual(supervised[-1], tokenizer.eos_token_id)
        decoded = tokenizer.decode(supervised[:-1])
        self.assertEqual(decoded, completion)
        self.assertEqual(parse_response(decoded)[1], {'extra_trees:n_estimators': 100})

    def test_real_peft_load_verifies_and_preserves_trainability(self):
        config = LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
        original = get_peft_model(LlamaForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
                                                                    target_modules=['q_proj']))
        with torch.no_grad():
            for name, value in original.named_parameters():
                if 'lora_B' in name:
                    value.fill_(0.2)
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            for trainable in (False, True):
                loaded, report = load_verified_adapter(LlamaForCausalLM(config), directory, is_trainable=trainable)
                self.assertEqual(report['verified_tensors'], 2)
                self.assertTrue(all(v.requires_grad == trainable for n, v in loaded.named_parameters() if 'lora_' in n))
                self.assertFalse(any(v.requires_grad for n, v in loaded.named_parameters() if 'lora_' not in n))


if __name__ == '__main__':
    unittest.main()
