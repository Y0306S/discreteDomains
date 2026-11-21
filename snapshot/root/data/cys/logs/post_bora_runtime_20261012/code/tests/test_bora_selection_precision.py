"""BORA must preserve candidate identity through JSON float round trips."""

import json
import unittest
from unittest.mock import patch

import numpy as np
from ConfigSpace import Configuration, UniformFloatHyperparameter

from test_paper_parser_summary import ParserSummaryTests
from mindware.components.optimizers.bora_optimizer import BORAOptimizer
from mindware.components.optimizers.bora_ml_runtime import (
    ExpectedImprovement, official_candidates, same_configuration,
)
from mindware.components.optimizers.strict_recommendation import StrictRecommendationResolver


class BORASelectionTests(unittest.TestCase):
    def make(self):
        fixture = ParserSummaryTests()
        self.addCleanup(fixture.doCleanups)
        return fixture.make(BORAOptimizer)

    def test_roundtrip_maps_to_original_but_changed_values_are_rejected(self):
        opt = self.make()
        opt.config_space.add_hyperparameter(UniformFloatHyperparameter("fixture:rate", .1, 1., default_value=.9))
        opt.resolver = StrictRecommendationResolver(opt.config_space, opt.extractor_client, default_on_error=True)
        original = dict(opt.config_space.get_default_configuration())
        original["fixture:rate"] = .8999999999999999
        opt.selection_candidates = [Configuration(opt.config_space, values=original)]
        rounded = dict(original, **{"fixture:rate": .9})
        wrong = dict(original, **{"fixture:rate": .89})
        self.assertTrue(same_configuration(original, rounded, opt.config_space))
        self.assertFalse(same_configuration(original, wrong, opt.config_space))
        comment = {"comment": "Select", "hypotheses": [
            {"name": name, "rationale": "test", "confidence": "high", "points": [point]}
            for name, point in (("roundtrip", rounded), ("changed", wrong))]}
        assistant = opt._assistant()
        with patch.object(assistant, "_chat_completion", return_value=json.dumps(comment)):
            selected = assistant.comment_and_select_point(opt._observations(), None)
        self.assertEqual(selected.hypotheses[0]["points"], [original])
        self.assertEqual(len(selected.hypotheses), 1)
        self.assertEqual(opt.trial_cnt, 0)
        self.assertTrue(any(r.get("stage") == "bora_selection_float_roundtrip" for r in opt.decision_records))

    def test_collapsed_optima_fill_from_existing_ei_scored_pool(self):
        opt = self.make()
        offered = []

        def collapse(gp, target_space, **kwargs):
            vectors = target_space.random_points(10)
            offered.extend(vectors)
            return np.repeat(vectors[:1], kwargs["n"], axis=0)

        with patch.object(ExpectedImprovement, "suggest", side_effect=collapse), \
                patch.object(ExpectedImprovement, "_get_acq", return_value=lambda x: -x[:, 1]):
            candidates = official_candidates(opt, 5)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len({opt._key(c) for c in candidates}), 5)
        self.assertEqual(len(offered), 10)
        self.assertTrue(all(any(np.array_equal(c.get_array(), v) for v in offered) for c in candidates))
        self.assertEqual(opt.trial_cnt, 0)


if __name__ == "__main__":
    unittest.main()
