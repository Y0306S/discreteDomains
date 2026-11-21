import unittest

import torch

from examples.benchmark.train_verified_orpo import check_adapter_states, validate_recipe


class VerifiedOrpoTests(unittest.TestCase):
    def test_changed_training_hyperparameter_is_rejected(self):
        source = {"stage": "dpo", "pref_loss": "orpo", "learning_rate": 5e-6}
        config = dict(source, dataset="p5_attr_t2_rationale", overwrite_output_dir=False, learning_rate=1e-5)
        with self.assertRaisesRegex(ValueError, "Unapproved recipe"):
            validate_recipe(source, config)

    def test_operational_paths_can_change_without_changing_recipe(self):
        source = {"stage": "dpo", "pref_loss": "orpo", "dataset": "old", "overwrite_output_dir": True}
        config = dict(source, dataset="p5_attr_t2_rationale", overwrite_output_dir=False, output_dir="new")
        self.assertEqual(set(validate_recipe(source, config)), {"dataset", "overwrite_output_dir", "output_dir"})

    def test_missing_or_randomized_parent_weights_are_rejected(self):
        saved = {"lora_A": torch.tensor([0.15]), "lora_B": torch.tensor([0.2])}
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            check_adapter_states(saved, {"lora_A": saved["lora_A"]})
        with self.assertRaisesRegex(ValueError, "tensor mismatch"):
            check_adapter_states(saved, dict(saved, lora_B=torch.zeros(1)))

    def test_bfloat16_loading_is_compared_at_actual_precision(self):
        saved = {"lora_A": torch.tensor([0.15]), "lora_B": torch.tensor([0.2])}
        self.assertEqual(check_adapter_states(saved, {key: value.bfloat16() for key, value in saved.items()}), 2)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            check_adapter_states(saved, dict(saved, lora_B=torch.tensor([float("nan")])))


if __name__ == "__main__":
    unittest.main()
