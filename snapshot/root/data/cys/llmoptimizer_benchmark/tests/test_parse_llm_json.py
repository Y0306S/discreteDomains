#!/usr/bin/env python3
"""Unit tests for strict LLM config JSON extraction."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mindware.components.optimizers.pt4opt_llm_optimizer import parse_llm_json  # noqa: E402

CONFIG = {
    "random_forest:bootstrap": "True",
    "random_forest:criterion": "entropy",
    "random_forest:max_features": 0.6,
    "random_forest:min_samples_leaf": 1,
    "random_forest:min_samples_split": 4,
}
WRAPPED = json.dumps({"config": CONFIG})


class TestParseLlmJson(unittest.TestCase):
    def test_json_only(self):
        cfg, rationale, payload = parse_llm_json(WRAPPED)
        self.assertEqual(cfg, CONFIG)
        self.assertEqual(rationale, "")
        self.assertIn("config", payload)

    def test_rationale_before_json(self):
        text = (
            "Rationale: Build on best trials with entropy and lower max_features.\n\n"
            f"JSON:\n{WRAPPED}"
        )
        cfg, rationale, _ = parse_llm_json(text)
        self.assertEqual(cfg, CONFIG)
        self.assertIn("entropy", rationale)

    def test_trailing_junk_after_json(self):
        text = f"{WRAPPED}\n\nHope this helps!"
        cfg, _, _ = parse_llm_json(text)
        self.assertEqual(cfg, CONFIG)

    def test_leaked_second_json_object(self):
        leak = json.dumps({"config": {"random_forest:bootstrap": "False"}})
        text = f"Rationale: first config only.\n\nJSON:\n{WRAPPED}\n\nuser\n{leak}"
        cfg, _, _ = parse_llm_json(text)
        self.assertEqual(cfg, CONFIG)

    def test_leaked_im_start_suffix(self):
        text = f"Rationale: ok\n\nJSON:\n{WRAPPED}<|im_start|>user\n{WRAPPED}"
        cfg, _, _ = parse_llm_json(text)
        self.assertEqual(cfg, CONFIG)

    def test_fenced_json_block(self):
        text = f"Rationale: use fenced output.\n\n```json\n{WRAPPED}\n```"
        cfg, rationale, _ = parse_llm_json(text)
        self.assertEqual(cfg, CONFIG)
        self.assertIn("fenced", rationale)

    def test_bare_hp_map_without_wrapper(self):
        bare = json.dumps(CONFIG)
        cfg, _, payload = parse_llm_json(bare)
        self.assertEqual(cfg, CONFIG)
        self.assertNotIn("config", payload)

    def test_skips_non_config_json_then_parses_wrapped(self):
        noise = '{"trial": 9, "rel_score": 1.0}'
        text = f"{noise}\n{WRAPPED}"
        cfg, _, _ = parse_llm_json(text)
        self.assertEqual(cfg, CONFIG)
        text2 = f"Rationale: ignore metadata.\n\nJSON:\n{WRAPPED}"
        cfg, _, _ = parse_llm_json(text2)
        self.assertEqual(cfg, CONFIG)

    def test_multiple_wrapped_objects_takes_first(self):
        second = json.dumps(
            {
                "config": {
                    "random_forest:bootstrap": "False",
                    "random_forest:criterion": "gini",
                    "random_forest:max_features": 0.1,
                    "random_forest:min_samples_leaf": 2,
                    "random_forest:min_samples_split": 5,
                }
            }
        )
        text = f"{WRAPPED}\n{second}"
        cfg, _, _ = parse_llm_json(text)
        self.assertEqual(cfg["random_forest:criterion"], "entropy")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_llm_json("   ")

    def test_real_p5_h4_yeast_trial_shape(self):
        text = (
            "Rationale: To maximize the validation score, I will propose a configuration that "
            "builds on the best-performing trials while introducing slight variations. Given that "
            "the best trials utilized a criterion of \"entropy\" and a bootstrap of \"True,\" I "
            "will maintain these settings.\n\n"
            "JSON:\n"
            '{"config": {"random_forest:bootstrap": "True", "random_forest:criterion": "entropy", '
            '"random_forest:max_features": 0.6, "random_forest:min_samples_leaf": 1, '
            '"random_forest:min_samples_split": 4}}'
        )
        cfg, rationale, _ = parse_llm_json(text)
        self.assertEqual(cfg["random_forest:max_features"], 0.6)
        self.assertTrue(rationale.startswith("To maximize"))


if __name__ == "__main__":
    unittest.main()
