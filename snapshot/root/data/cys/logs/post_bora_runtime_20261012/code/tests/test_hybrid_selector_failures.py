"""Nonfinite evaluations consume selections without earning improvement credit."""

import math
import unittest

from mindware.components.optimizers.hybrid_selector import HybridSelector


class SelectorFailureTest(unittest.TestCase):
    def test_unrewarded_attempt_changes_exploration_without_changing_q(self):
        selector = HybridSelector(mode="pucb", pucb_q="arm_best", pucb_p_end=0.5)
        selector.update("llm", 0.8)
        selector.update("tpe", 0.8)
        q = selector.q_values()
        self.assertEqual(selector.choose(97, 97), "llm")
        selector.record_attempt("llm")
        self.assertEqual(selector.q_values(), q)
        self.assertEqual(selector.n, {"llm": 1, "tpe": 1})
        self.assertEqual(selector.attempt_n, {"llm": 2, "tpe": 1})
        self.assertEqual(selector.choose(97, 97), "tpe")

    def test_first_failure_is_not_an_arm_improvement(self):
        for value in (-math.inf, math.inf, math.nan):
            for direction in (True, False):
                selector = HybridSelector(mode="pucb", pucb_q="arm_best")
                selector.update("llm", value, higher_is_better=direction,
                                beat_arm_best=True, beat_incumbent=True)
                self.assertEqual(selector.n["llm"], 1)
                self.assertEqual(selector.successes["llm"], 0)
                self.assertEqual(selector.failures["llm"], 1)
                self.assertIsNone(selector.arm_best["llm"])
                self.assertEqual(selector.q_values()["llm"], 1 / 3)
                self.assertEqual(selector.disc_sum["llm"], 0)

    def test_success_after_failure_sets_best_and_later_failure_preserves_it(self):
        for direction, good, better in ((True, -5.0, -4.0), (False, 5.0, 4.0)):
            selector = HybridSelector(mode="pucb", pucb_q="arm_best")
            for value in (math.nan, good, math.inf, better, better):
                selector.update("tpe", value, higher_is_better=direction)
            self.assertEqual(selector.arm_best["tpe"], better)
            self.assertEqual(selector.successes["tpe"], 2)
            self.assertEqual(selector.failures["tpe"], 3)
            self.assertEqual(selector.n["tpe"], 5)

    def test_window_scores_and_priority_remain_finite_after_failures(self):
        selector = HybridSelector(mode="pucb", pucb_q="window_minmax")
        selector.update("llm", -math.inf)
        selector.update("tpe", 0.7)
        selector.choose(2, 97)
        self.assertTrue(all(math.isfinite(v) for v in selector.q_values().values()))
        self.assertEqual(selector.successes["llm"], 0)


if __name__ == "__main__":
    unittest.main()
