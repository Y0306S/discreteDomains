"""Verify worker limits reach CatBoost's real native estimator."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mindware.components.evaluators import cls_evaluator, rgs_evaluator


class CatBoostWorkerThreadsTests(unittest.TestCase):
    def test_caps_and_affinity_reach_real_classification_and_regression_fits(self):
        rng = np.random.RandomState(1)
        x = rng.normal(size=(48, 4))
        for module, regression in ((cls_evaluator, False), (rgs_evaluator, True)):
            for requested, available, expected in ((1, 3, 1), (3, 3, 3), (3, 1, 1)):
                with self.subTest(regression=regression, requested=requested, available=available):
                    config = {"catboost:iterations": 5, "catboost:learning_rate": 0.1,
                              "catboost:depth": 2, "catboost:l2_leaf_reg": 3}
                    if regression:
                        config["catboost:loss_function"] = "RMSE"
                    with patch.dict(os.environ, {"EVAL_N_JOBS": str(requested)}), \
                            patch.object(module, "_available_cpu_count", return_value=available):
                        _, estimator = module.get_estimator(config, "catboost")
                    self.assertEqual(estimator.n_jobs, expected)
                    y = x[:, 0] if regression else (x[:, 0] > 0).astype(int)
                    previous = os.getcwd()
                    with tempfile.TemporaryDirectory() as directory:
                        try:
                            os.chdir(directory)
                            estimator.fit(x, y)
                            self.assertEqual(estimator.estimator.get_params()["thread_count"], expected)
                            self.assertTrue(np.isfinite(estimator.predict(x)).all())
                        finally:
                            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
