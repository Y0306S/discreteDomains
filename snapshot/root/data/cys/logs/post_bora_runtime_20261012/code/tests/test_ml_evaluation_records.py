import sys
import json
import logging
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))
from audit_ml_evaluation_records import from_result, from_topk


class EvaluationRecordsTest(unittest.TestCase):
    def test_ckpt_records_fixed_parameters_and_failed_configuration(self):
        from ConfigSpace import ConfigurationSpace, Constant, UniformIntegerHyperparameter
        from mindware.components.optimizers.pt4opt_llm_tpe_optimizer import PT4OptLLMTPEOptimizer

        space = ConfigurationSpace()
        space.add_hyperparameters([Constant("algorithm", "extra_trees"),
                                   Constant("extra_trees:max_depth", "None"),
                                   UniformIntegerHyperparameter("extra_trees:min_samples_leaf", 1, 5)])
        config = space.get_default_configuration()
        with tempfile.TemporaryDirectory() as directory:
            opt = SimpleNamespace(logging_enabled=True, llm_query_cost=[], llm_query_time=[],
                                  start_wall_time=time.time(), traj_rows=[], perfs=[],
                                  fixed_algorithm="extra_trees", llm_prob=0.5, llm_phase_trials=0,
                                  config_space=space, traj_csv_path=str(Path(directory) / "test_traj.csv"),
                                  study=None, logger=logging.getLogger(__name__))
            for index, score in enumerate([0.5, float("-inf")]):
                PT4OptLLMTPEOptimizer._record_hybrid_event(
                    opt, "tpe", index, config, score, 0, 0, 0, 2, 2, None, "test")
            for row in opt.traj_rows:
                self.assertEqual(json.loads(row["config_json"]), config.get_dictionary())
            sidecar = Path(directory) / "test_observations.jsonl"
            rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
            self.assertEqual(rows[1]["config"], config.get_dictionary())
            self.assertEqual(rows[1]["perf"], float("-inf"))

    def test_tied_scores_preserve_pairs_without_inventing_trial_order(self):
        rows = [{"perf": 0.5}, {"perf": 0.5}]
        items = [({"depth": 2}, 0.5), ({"depth": 9}, 0.5)]
        pairs = from_topk(rows, items)
        self.assertEqual([p["config"] for p in pairs], [{"depth": 2}, {"depth": 9}])
        self.assertTrue(all(p["trial_index"] is None for p in pairs))
        with self.assertRaises(ValueError):
            from_topk(rows, items[:1])

    def test_success_failure_and_timeout_keep_configuration(self):
        row = {"config_json": '{"depth": 2}', "perf": 0.75, "eval_time": 2, "iter_idx": 0}
        result = {"result": {"objectives": [-0.75]}, "elapsed_time": 2, "timeout": False, "traceback": None}
        self.assertEqual(from_result(row, result)["status"], "success")
        with self.assertRaises(ValueError):
            from_result({**row, "perf": 0.5}, result)
        for timeout, error, status in [(True, None, "timeout"), (False, "training error", "failed")]:
            pair = from_result({**row, "perf": float("-inf")},
                               {**result, "result": None, "timeout": timeout, "traceback": error})
            self.assertEqual(pair["status"], status)
            self.assertEqual(pair["config"], {"depth": 2})
            self.assertEqual(pair["error"], error)


if __name__ == "__main__":
    unittest.main()
