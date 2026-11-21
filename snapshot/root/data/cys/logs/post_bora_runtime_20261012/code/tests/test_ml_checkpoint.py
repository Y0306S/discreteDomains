import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ConfigSpace import ConfigurationSpace
import numpy as np

from mindware.components.optimizers.ml_checkpoint import MLSession, progress
from mindware.components.optimizers.isolated_evaluator import run_isolated


class CheckpointTest(unittest.TestCase):
    def optimizer(self, directory):
        space = ConfigurationSpace(seed=7)
        kd = SimpleNamespace(configs=[], perfs=[], bo_history_mab={})
        return SimpleNamespace(config_space=space, KD=kd, configs=kd.configs, perfs=kd.perfs,
                               evaluator=SimpleNamespace(output_dir=directory),
                               evaluation_num_limit=100, traj_rows=[],
                               traj_csv_path=str(Path(directory) / "trial_traj.csv"),
                               trial_cnt=1, state={"pending_batch": [1, 2, 3]})

    def test_pending_round_restores_state_and_random_generators(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ML_RUN_ID": "test"}):
            first = self.optimizer(directory)
            session = MLSession(first, directory)
            session.save({"configuration": {"depth": 2}, "trial_id": 0})
            expected = (random.random(), np.random.rand(), first.config_space.random.rand())
            first.state["pending_batch"].clear()
            second = self.optimizer(directory)
            resumed = MLSession(second, directory)
            self.assertEqual(resumed.restore(), 0)
            self.assertEqual(second.state["pending_batch"], [1, 2, 3])
            self.assertEqual(resumed.pending["configuration"], {"depth": 2})
            self.assertEqual(expected, (random.random(), np.random.rand(), second.config_space.random.rand()))

    def test_checkpoint_controls_completion_and_repairs_export(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ML_RUN_ID": "test"}):
            first = self.optimizer(directory)
            first.traj_rows = [{"iter_idx": 0, "perf": 0.5}]
            session = MLSession(first, directory)
            session.save()
            Path(first.traj_csv_path).write_text("iter_idx,perf\n0,0.5\n1,0.9\n")
            (Path(directory) / "progress.json").write_text('{"attempts": 100, "identity": "wrong"}')
            second = self.optimizer(directory)
            resumed = MLSession(second, directory)
            self.assertEqual(resumed.restore(), 1)
            self.assertEqual(len(Path(second.traj_csv_path).read_text().splitlines()), 2)
            self.assertIsNone(resumed.pending)
            repaired = json.loads((Path(directory) / "progress.json").read_text())
            self.assertEqual(repaired["attempts"], 1)
            self.assertEqual(repaired["identity"], "test")
            with patch.dict(os.environ, {"ML_RUN_ID": "different"}):
                with self.assertRaises(RuntimeError):
                    MLSession(self.optimizer(directory), directory).restore()

    def test_success_and_budget_exhaustion_are_distinct(self):
        rows = [{"iter_idx": i, "perf": 0.5 if i else -math.inf} for i in range(100)]
        value = progress(rows, 100)
        self.assertTrue(value["budget_exhausted"])
        self.assertFalse(value["clean_complete"])
        self.assertEqual(value["successful"], 99)
        self.assertEqual(value["status"], "complete_with_failures")
        with self.assertRaises(ValueError):
            progress(rows + [rows[0]], 100)

    def test_consumed_allowance_is_not_reset_on_retry(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"ML_SESSION_DIR": directory, "ML_EVALUATION_ID": "0"}):
            folder = Path(directory) / "evaluations/0"
            folder.mkdir(parents=True)
            (folder / "execution.json").write_text(json.dumps({"status": "interrupted", "elapsed_seconds": 12}))
            result = run_isolated(None, (), {}, 10)
            self.assertTrue(result["timeout"])
            self.assertEqual(result["elapsed_time"], 12)
            self.assertEqual(run_isolated(None, (), {}, 10), result)


if __name__ == "__main__":
    unittest.main()
