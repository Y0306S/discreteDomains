"""Generation failures must not corrupt history or spend evaluation slots."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from test_online_llm_baselines import AdapterTests
from mindware.components.optimizers.bora_ml_runtime import BORA_CONTEXT_TOKENS, QwenBridge


class BORARecoveryTests(unittest.TestCase):
    def make(self):
        fixture = AdapterTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture.make(limit=100, recommendation_default_fallback=True)

    def test_failed_overview_and_legacy_history_allow_later_prompts(self):
        opt = self.make()
        assistant = opt._assistant()
        response = json.dumps({"comment": "A trial", "hypotheses": [{
            "name": "Depth", "rationale": "Try the default", "confidence": "medium",
            "points": [dict(opt.config_space.get_default_configuration())]}]})
        with patch.object(assistant, "_chat_completion", side_effect=[None, response]):
            self.assertTrue(assistant.pre_optimization_comment(1).is_valid)
        self.assertTrue(all(m["content"] is not None for m in assistant._chat_history))
        opt.configs, opt.perfs = [opt.config_space.get_default_configuration()], [0.5]
        self.assertIsInstance(assistant._generate_comment_prompt(opt._observations()), str)
        state = assistant.snapshot()
        state["chat_history"].append({"role": "assistant", "content": None})
        assistant.restore(state)
        self.assertTrue(all(m["content"] is not None for m in assistant._chat_history))
        self.assertIsInstance(assistant._generate_comment_prompt(opt._observations()), str)
        self.assertEqual(opt.trial_cnt, 0)

    def test_context_guard_uses_rendered_prompt_and_allows_exact_boundary(self):
        opt = self.make()
        output = 4096
        tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value="rendered"),
            encode=Mock(return_value=range(BORA_CONTEXT_TOKENS - output + 1)))
        opt.llm_client = SimpleNamespace(_tokenizer=tokenizer, max_new_tokens=output,
                                       complete_messages=Mock(return_value="fresh"))
        opt.bora_last_raw = "stale"
        bridge = QwenBridge(opt)
        messages = [{"role": "user", "content": "query"}]
        self.assertEqual(bridge.create(messages=messages).choices[0].message.content, "")
        opt.llm_client.complete_messages.assert_not_called()
        tokenizer.apply_chat_template.assert_called_once_with(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        record = next(r for r in opt.decision_records if r.get("reason") == "context_limit")
        self.assertFalse(record["evaluation_slot_consumed"])
        self.assertEqual(opt.bora_last_raw, "")
        tokenizer.encode.return_value = range(BORA_CONTEXT_TOKENS - output)
        self.assertEqual(bridge.create(messages=messages).choices[0].message.content, "fresh")
        self.assertEqual(opt.trial_cnt, 0)

    def test_generation_oom_clears_stale_output_and_next_request_succeeds(self):
        opt = self.make()
        opt.llm_client.complete = Mock(side_effect=[torch.cuda.OutOfMemoryError("allocation"), "fresh"])
        opt.bora_last_raw = "stale"
        bridge = QwenBridge(opt)
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "query"}]
        with patch("torch.cuda.empty_cache") as clear:
            self.assertEqual(bridge.create(messages=messages).choices[0].message.content, "")
            self.assertEqual(opt.bora_last_raw, "")
            self.assertEqual(bridge.create(messages=messages).choices[0].message.content, "fresh")
        clear.assert_called_once_with()
        record = next(r for r in opt.decision_records if r.get("reason") == "cuda_out_of_memory")
        self.assertEqual(record["error"], "allocation")
        self.assertFalse(record["evaluation_slot_consumed"])
        self.assertEqual(opt.trial_cnt, 0)

    def test_model_load_errors_propagate_before_upstream_retries(self):
        opt = self.make()
        opt.llm_client._ensure_loaded = Mock(side_effect=RuntimeError("checkpoint load failed"))
        with self.assertRaisesRegex(RuntimeError, "checkpoint load failed"):
            opt._suggest()
        self.assertIsNone(opt.bora_assistant)
        opt.llm_client.complete.assert_not_called()
        self.assertEqual(opt.trial_cnt, 0)

    def test_all_initial_failures_recover_with_default_then_initialize_gp(self):
        opt = self.make()
        defaults = [opt.config_space.get_default_configuration()] * 5
        opt.configs, opt.perfs, opt.trial_cnt = defaults, [float("-inf")] * 5, 5
        proposal = opt._suggest()
        self.assertEqual(proposal["config"], dict(defaults[0]))
        self.assertIn("no successful initialization", proposal["note"])
        opt.configs.append(defaults[0])
        opt.perfs.append(0.5)
        opt.trial_cnt += 1
        opt.incumbent_perf = 0.5
        opt._after_observation(proposal, 0.5)
        self.assertTrue(opt.initialized)


if __name__ == "__main__":
    unittest.main()
