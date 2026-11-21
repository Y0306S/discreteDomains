"""CPU tests of the production token objective, GP update and weight checkpoints."""

import copy
import tempfile
from types import SimpleNamespace
import unittest

import torch

from mindware.components.optimizers.tosfit_runtime import ToSFiTRuntime, published_recipe


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "fixed prompt"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens, **kwargs):
        return "original tokens " + str(tokens.tolist())


class SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)
        self.backward_inputs, self.generated = [], []

    def forward(self, input_ids, logits_to_keep=None, **kwargs):
        hidden = self.embeddings(input_ids)
        logits = self.head(hidden)
        if self.training and torch.is_grad_enabled():
            self.backward_inputs.append(input_ids.detach().clone())
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=None)

    def generate(self, input_ids, max_new_tokens, temperature, **kwargs):
        result = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(result).logits[:, -1] / temperature
            token = torch.multinomial(logits.softmax(-1), 1)
            result = torch.cat((result, token), dim=1)
            if token.item() == 0:
                break
        self.generated.append(result.detach().clone())
        return result

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def gradient_checkpointing_disable(self):
        pass


class RuntimeTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(42)
        model = SmallPolicy()
        reference = copy.deepcopy(model)
        recipe = published_recipe()
        # Smaller sequences/groups only for this CPU fixture. Published LR is retained.
        recipe.update(batch_size=4, max_new_tokens=4)
        runtime = ToSFiTRuntime("cpu", "cpu", model=model, reference=reference,
                                tokenizer=Tokenizer(), recipe=recipe)
        return runtime

    def test_recipe_uses_released_faq_qwen_settings(self):
        recipe = published_recipe()
        self.assertEqual(recipe["learning_rate"], 2e-7)
        self.assertEqual(recipe["batch_size"], 16)
        self.assertEqual(recipe["n_marginal_likelihood_warmup_steps"], 16)
        self.assertEqual(recipe["fine_tune_steps/bo_step"], 1)

    def test_original_tokens_drive_updates_and_reference_stays_frozen(self):
        runtime = self.make()
        original = {k: v.detach().clone() for k, v in runtime.model.state_dict().items()}
        initial = runtime.propose("task prompt", step=0)
        self.assertFalse(initial["gradient_step"])
        self.assertTrue(all(torch.equal(original[k], v) for k, v in runtime.model.state_dict().items()))
        runtime.observe(initial["features"], -3.0)
        self.assertEqual(runtime.gp.n_observations, 1)
        sample = runtime.propose("task prompt", step=16)
        self.assertTrue(sample["gradient_step"])
        delta = sum((original[k] - v).abs().sum().item() for k, v in runtime.model.state_dict().items())
        self.assertGreater(delta, 0)
        self.assertTrue(all(torch.isfinite(v).all() for v in runtime.model.parameters()))
        self.assertTrue(all(torch.equal(original[k], v) for k, v in runtime.reference.state_dict().items()))
        self.assertEqual(len(runtime.model.backward_inputs), 4)
        self.assertEqual(sample["tokens"], runtime.model.backward_inputs[0][0, 2:].tolist())
        self.assertEqual(len(sample["generation_log_probs"]), 4)
        self.assertTrue(all(p.grad is None for p in runtime.model.parameters()))

    def test_snapshot_restores_policy_gp_and_version_without_training(self):
        runtime = self.make()
        sample = runtime.propose("task prompt", step=16)
        runtime.observe(sample["features"], -2.0)
        with tempfile.TemporaryDirectory() as directory:
            state = runtime.snapshot(directory)
            fresh = self.make()
            fresh.restore(state)
            self.assertEqual(fresh.policy_version, 1)
            self.assertEqual(fresh.gp.n_observations, 1)
            self.assertEqual(fresh.model.backward_inputs, [])
            self.assertTrue(all(torch.equal(v, fresh.model.state_dict()[k])
                                for k, v in runtime.model.state_dict().items()))
            feature = torch.tensor([sample["features"]], dtype=torch.float64)
            self.assertTrue(torch.equal(runtime.gp.posterior_mean(feature), fresh.gp.posterior_mean(feature)))

    def test_three_training_rounds_continue_after_checkpoint_reload(self):
        runtime = self.make()
        runtime.recipe["n_marginal_likelihood_warmup_steps"] = 2
        original = {k: v.detach().clone() for k, v in runtime.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            for step in range(5):
                before = {k: v.detach().clone() for k, v in runtime.model.state_dict().items()}
                sample = runtime.propose("task prompt", step=step)
                changed = any(not torch.equal(before[k], v) for k, v in runtime.model.state_dict().items())
                self.assertEqual(changed, step >= 2)
                self.assertEqual(runtime.policy_version, max(0, step - 1))
                self.assertTrue(all(torch.equal(original[k], v)
                                    for k, v in runtime.reference.state_dict().items()))
                self.assertTrue(all(torch.isfinite(v).all() for v in runtime.model.parameters()))
                if step == 2:
                    state = runtime.snapshot(directory)
                    fresh = self.make()
                    fresh.recipe["n_marginal_likelihood_warmup_steps"] = 2
                    fresh.restore(state)
                    self.assertEqual(fresh.model.backward_inputs, [])
                    self.assertTrue(all(torch.equal(v, fresh.model.state_dict()[k])
                                        for k, v in runtime.model.state_dict().items()))
                    runtime = fresh
                runtime.observe(sample["features"], float(step - 4))
                self.assertEqual(runtime.gp.n_observations, step + 1)
        self.assertEqual(runtime.policy_version, 3)

    def test_score_includes_only_first_eos_even_when_padding_equals_eos(self):
        runtime = self.make()
        runtime._ensure_loaded()
        runtime._set_prompt("task prompt")
        one, _ = runtime._score(runtime.model, torch.tensor([[3, 0]]))
        padded, _ = runtime._score(runtime.model, torch.tensor([[3, 0, 0, 0]]))
        self.assertTrue(torch.allclose(one, padded))

    def test_actual_qwen35_architecture_forward_generation_and_backward_on_cpu(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        config = Qwen3_5Config(
            text_config={"vocab_size": 8, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                         "head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                         "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                         "eos_token_id": 0, "pad_token_id": 0,
                         "layer_types": ["linear_attention", "full_attention"]},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 2, "out_hidden_size": 32})
        model = Qwen3_5ForConditionalGeneration(config)
        reference = copy.deepcopy(model)
        recipe = published_recipe()
        recipe.update(batch_size=4, max_new_tokens=4)
        runtime = ToSFiTRuntime("cpu", "cpu", model=model, reference=reference,
                                tokenizer=Tokenizer(), recipe=recipe)
        sample = runtime.propose("task prompt", step=16)
        runtime.observe(sample["features"], -1.0)
        self.assertEqual(runtime.policy_version, 1)
        self.assertEqual(runtime.gp.n_observations, 1)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
