import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples/benchmark"))
from ml_runtime_policy import apply_policy, load_policy, optional_seconds
from mindware.modules.base import BaseAutoML


class TrialOptimizer:
    def __init__(self):
        self.budgets = []
        self.outcomes = []
        self.early_stopped_flag = False
        self.timeout_flag = False
        self.incumbent_perf = 0.5
        self.incumbent_config = {"algorithm": "extra_trees"}
        self.eval_dict = {}

    def iterate(self, budget):
        self.budgets.append(budget)
        self.outcomes.append(len(self.budgets) % 3 != 0)


class RuntimePolicyTest(unittest.TestCase):
    def controller(self, overall, directory):
        controller = BaseAutoML.__new__(BaseAutoML)
        controller.optimizer = TrialOptimizer()
        controller.time_limit = overall
        controller.timestamp = 0
        controller.inner_iter_num_per_iter = 1
        controller.amount_of_resource = 100
        controller.timeout_flag = False
        controller.early_stop_flag = False
        controller.ensemble_method = None
        controller.refit_incumbent = Mock()
        controller.output_dir = directory
        controller.datetime = "test"
        controller.logger = Mock()
        return controller

    def test_no_overall_timeout_counts_failed_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(None, directory)
            with patch("mindware.modules.base.time.time", return_value=10**9):
                controller.run()
            self.assertEqual(len(controller.optimizer.budgets), 100)
            self.assertTrue(all(math.isinf(value) for value in controller.optimizer.budgets))
            self.assertEqual(controller.optimizer.outcomes.count(False), 33)
            self.assertFalse(controller.timeout_flag)

    def test_explicit_finite_overall_limit_still_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(1, directory)
            with patch("mindware.modules.base.time.time", return_value=2):
                controller.run()
            self.assertTrue(controller.timeout_flag)
            self.assertEqual(len(controller.optimizer.budgets), 1)

    def test_shared_policy_sets_budget_and_enforcement(self):
        args = SimpleNamespace(runtime_policy=REPO / "configs/ml_runtime_policy.json",
                               n_trials=3, time_limit=600, per_run_time_limit=30)
        with patch.dict(os.environ, {"ML_ISOLATED_EVALUATION": "0"}):
            apply_policy(args)
            self.assertEqual(args.n_trials, 100)
            self.assertEqual(args.per_run_time_limit, 172800)
            self.assertIsNone(args.time_limit)
            self.assertEqual(os.environ["ML_ISOLATED_EVALUATION"], "1")
        self.assertEqual(optional_seconds("none"), None)
        self.assertEqual(optional_seconds("60"), 60)


if __name__ == "__main__":
    unittest.main()
