import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, UniformIntegerHyperparameter

from mindware.components.optimizers.pt4opt_llm_optimizer import PT4OptLLMOptimizer


class LLMDuplicatePolicyTest(unittest.TestCase):
    def optimizer(self):
        space = ConfigurationSpace(seed=1)
        space.add_hyperparameters([
            Constant("algorithm", "toy"),
            UniformIntegerHyperparameter("toy:depth", 1, 10),
        ])
        config = Configuration(space, values={"algorithm": "toy", "toy:depth": 3})
        opt = PT4OptLLMOptimizer.__new__(PT4OptLLMOptimizer)
        opt.config_space = space
        opt.fixed_algorithm = "toy"
        opt.configs, opt.perfs = [config], [0.5]
        opt.seen_signatures, opt.seen_objectives = set(), {}
        opt._seen_config_count = 0
        opt.KD = SimpleNamespace(_dict_to_cfg=lambda values: Configuration(space, values=values))
        opt.llm_client = Mock()
        opt.llm_client.complete.return_value = json.dumps({"config": {"toy:depth": 3}})
        opt.llm_call_count = 0
        opt.llm_history = []
        opt._last_history_meta = {}
        opt._preset_warm_start_queue = []
        opt.use_tcs_analysis = False
        opt.max_duplicate_reprompts = 1
        opt.fallback_on_transport_error = False
        opt.fallback_on_bad_json = False
        opt.fallback_on_invalid_config = False
        opt._build_prompts = Mock(return_value=("Return JSON only.", "No rationale."))
        opt._sample_nonduplicate_config = Mock(side_effect=AssertionError("Unexpected random replacement"))
        return opt, config

    def test_repeated_duplicate_gets_one_retry_then_random_replacement(self):
        opt, original = self.optimizer()
        replacement = Configuration(opt.config_space, values={"algorithm": "toy", "toy:depth": 5})
        opt._sample_nonduplicate_config = Mock(return_value=replacement)
        result = opt._suggest_config(3)
        self.assertEqual(result[0], replacement)
        self.assertEqual(opt.llm_call_count, 2)
        self.assertEqual(opt.llm_client.complete.call_count, 2)
        opt._sample_nonduplicate_config.assert_called_once()
        self.assertIn("reprompt_user_prompt", result[-1])
        self.assertEqual(len(opt.llm_history[-1]["attempts"]), 2)
        self.assertTrue(opt.llm_history[-1]["rejected_duplicate_configs"])

    def test_unique_proposal_uses_one_call(self):
        opt, _ = self.optimizer()
        opt.llm_client.complete.return_value = json.dumps({"config": {"toy:depth": 4}})
        opt._suggest_config(3)
        opt.llm_client.complete.assert_called_once()
        opt._sample_nonduplicate_config.assert_not_called()

    def test_presets_accept_duplicates_without_model_calls(self):
        opt, original = self.optimizer()
        opt._preset_warm_start_queue = [original.get_dictionary()]
        result = opt._suggest_config(0, include_history=False)
        self.assertEqual(result[0], original)
        opt.llm_client.complete.assert_not_called()
        opt._sample_nonduplicate_config.assert_not_called()

    def test_existing_retry_policy_remains_available(self):
        opt, original = self.optimizer()
        opt.llm_client.complete.side_effect = [
            json.dumps({"config": {"toy:depth": 3}}),
            json.dumps({"config": {"toy:depth": 4}}),
        ]
        result = opt._suggest_config(3)
        self.assertNotEqual(result[0], original)
        self.assertEqual(opt.llm_call_count, 2)
        self.assertIn("reprompt_user_prompt", result[-1])
        opt._sample_nonduplicate_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
