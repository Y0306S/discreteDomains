"""Initialization recovery and mixed-data summaries at the real adapter boundary."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from ConfigSpace import Configuration

from test_online_llm_baselines import Extractor, space
from mindware.components.optimizers.bora_optimizer import BORAOptimizer
from mindware.components.optimizers.semanticopt_optimizer import SemanticOptOptimizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))
import run_toy_categorical_optimizers as runner


class ParserSummaryTests(unittest.TestCase):
    def make(self, cls, raw="", extractor=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with patch("mindware.components.optimizers.online_llm_optimizer.KnowledgeDatabase.instance",
                   return_value=SimpleNamespace()):
            return cls(evaluator=SimpleNamespace(), config_space=space(),
                output_dir=directory.name, evaluation_limit=100, task_type="rgs",
                recommendation_default_fallback=True, extractor_client=extractor or Extractor(),
                llm_client=SimpleNamespace(complete_messages=Mock(return_value=raw)), logging_enabled=False)

    @staticmethod
    def envelope():
        return {"initial_configuration_" + str(i):
                dict(dict(space().get_default_configuration()), **{"fixture:depth": i + 3})
                for i in range(1, 6)}

    def test_prose_and_fences_keep_all_five_distinct_configs(self):
        payload = json.dumps(self.envelope())
        for raw in (payload, "Here are five configurations:\n" + payload,
                    "Here they are:\n```json\n" + payload + "\n```\nEnd."):
            with self.subTest(raw=raw[:30]):
                opt = self.make(SemanticOptOptimizer, raw)
                opt._suggest()
                self.assertEqual([c["config"]["fixture:depth"] for c in opt.semantic_initial], [4, 5, 6, 7, 8])
                self.assertEqual(opt.extractor_client.calls, 0)
                self.assertEqual(opt.llm_client.complete_messages.call_count, 1)
                self.assertEqual(opt.trial_cnt, 0)

    def test_missing_or_invalid_slot_defaults_only_that_slot(self):
        for bad in (None, [], 17):
            payload = self.envelope()
            payload["initial_configuration_3"] = bad
            opt = self.make(SemanticOptOptimizer, "Proposals:\n" + json.dumps(payload))
            opt._suggest()
            self.assertEqual([c["config"]["fixture:depth"] for c in opt.semantic_initial], [4, 5, 9, 7, 8])
            self.assertEqual(opt.extractor_client.calls, 0)
            self.assertIn("labelled default", opt.semantic_initial[2]["note"])

    def test_prose_slot_is_extracted_in_isolation(self):
        payload = self.envelope()
        prose = "Try depth = 6."
        payload["initial_configuration_3"] = prose
        extractor = Extractor([("depth", 6, "depth = 6")])
        extractor.extract = Mock(wraps=extractor.extract)
        opt = self.make(SemanticOptOptimizer, "Proposals:\n" + json.dumps(payload), extractor)
        opt._suggest()
        self.assertEqual([c["config"]["fixture:depth"] for c in opt.semantic_initial], [4, 5, 6, 7, 8])
        self.assertEqual(extractor.extract.call_args.args[0], prose)
        self.assertEqual(extractor.calls, 1)

    def test_ambiguous_or_duplicate_key_envelopes_are_not_silently_chosen(self):
        payload = json.dumps(self.envelope())
        duplicate = 'Proposals: {"initial_configuration_1": {}, "initial_configuration_1": {}}'
        for raw in (payload + "\nAlternatively:\n" + payload, duplicate, "No configurations."):
            opt = self.make(SemanticOptOptimizer, raw)
            opt._suggest()
            self.assertTrue(all(c["config"] == dict(space().get_default_configuration())
                                for c in opt.semantic_initial))
            self.assertEqual(opt.extractor_client.calls, 0)
            record = next(r for r in opt.decision_records if r.get("stage") == "initialization_parsing")
            self.assertEqual(record["route"], "labelled_defaults")

    def test_bora_summary_handles_categories_failures_and_both_prompt_paths(self):
        opt = self.make(BORAOptimizer)
        defaults = dict(space().get_default_configuration())
        opt.configs = [Configuration(opt.config_space, values=dict(defaults, **{
            "fixture:depth": depth, "fixture:criterion": category}))
            for depth, category in ((4, "gini"), (8, "entropy"), (9, "gini"))]
        opt.perfs = [.2, .8, float("-inf")]
        data = opt._observations()
        original = data.copy(deep=True)
        self.assertEqual(data.iteration.tolist(), [1, 2, 3])
        assistant = opt._assistant()
        with patch.object(assistant, "_would_context_be_full", return_value=True):
            prompts = [assistant._generate_comment_prompt(data),
                       assistant._generate_comment_selection_prompt(data, data)]
        pd.testing.assert_frame_equal(data, original)
        for prompt in prompts:
            self.assertIn("successful: 2; failed: 1", prompt)
            self.assertIn("Categorical measured-score summary", prompt)
            self.assertIn("entropy", prompt)
            self.assertIn("Numeric correlations", prompt)
            self.assertIn("parameter", prompt)
        opt.perfs = [float("-inf")] * 3
        summary = assistant._summarize_dataset(opt._observations()).to_string()
        self.assertIn("successful: 0; failed: 3", summary)

    def test_bora_context_check_reserves_history_and_output_tokens(self):
        opt = self.make(BORAOptimizer)
        assistant = opt._assistant()
        assistant._model_info["max_tokens"] = 10000
        assistant._chat_history = [{"role": "system", "content": "a" * 1000}]
        self.assertTrue(assistant._would_context_be_full("b" * 5000))
        self.assertFalse(assistant._would_context_be_full("b" * 1000))

    def test_bora_cli_default_and_explicit_override_leave_other_defaults_alone(self):
        for extra, expected in (([], 4096), (["--llm-max-tokens", "2048"], 2048),
                                (["--llm-max-tokens=3072"], 3072)):
            with patch.object(sys, "argv", ["run", "--optimizer", "bora", "--algorithm", "random_forest",
                    "--extractor-max-tokens", "2048", "--extractor-timeout", "60",
                    "--extractor-max-calls", "100", "--skip-test-eval"] + extra):
                args = runner.parse_args()
            self.assertEqual(runner.build_optimizer_params(args, "bora", args.algorithm, {})[
                "generator_max_tokens"], expected)
            self.assertEqual(runner.build_optimizer_params(args, "semanticopt", args.algorithm, {})[
                "generator_max_tokens"], args.llm_max_tokens)
        opt = self.make(BORAOptimizer)
        self.assertEqual(opt.asset_identity["generator_max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
