import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))
import ml_transfer_campaign as campaign
from mindware.components.optimizers.transfer_hpo.protocol import checked_source_path, sha256_file, validate_dump


class CampaignTests(unittest.TestCase):
    def task(self, algorithm="extra_trees", dataset="car"):
        return {"task_type": "cls", "dataset": dataset, "algorithm": algorithm,
                "metric": "bal_acc", "memory_reservation_gib": 4}

    def test_cpu_budget_and_serial_backfill(self):
        spec = campaign.HOSTS["LLMOptimizer3"]
        tasks = [self.task(), self.task("adaboost")]
        active = {("fsbo", "one"): {"state": {"method": "fsbo", "memory_reservation_gib": 4}},
                  ("rgpe", "two"): {"state": {"method": "rgpe", "memory_reservation_gib": 4}},
                  ("fsbo", "three"): {"state": {"method": "fsbo", "memory_reservation_gib": 4}}}
        self.assertIsNone(campaign.choose_task(spec, tasks, active, set(), [11], 0, 0))
        method, task, cpus = campaign.choose_task(spec, tasks, active, set(), [11], 0, 181)
        self.assertEqual(task["algorithm"], "adaboost")
        self.assertEqual(cpus, [11])
        self.assertIsNone(campaign.choose_task(spec, tasks, active, set(), [], 0, 181))

    def test_memory_pressure_stops_admission(self):
        spec = campaign.HOSTS["LLMOptimizer3"]
        self.assertIsNone(campaign.choose_task(spec, [self.task()], {}, set(), spec["cpus"], 50 * 2**30, 181))
        heavy = self.task()
        heavy["memory_reservation_gib"] = 43
        self.assertIsNone(campaign.choose_task(spec, [heavy], {}, set(), spec["cpus"], 0, 181))

    def test_method_balance_and_claim_exclusion(self):
        spec = campaign.HOSTS["LLMOptimizer3"]
        task = self.task()
        active = {("fsbo", "another"): {"state": {"method": "fsbo"}}}
        result = campaign.choose_task(spec, [task], active, set(), spec["cpus"], 0, 0)
        self.assertEqual(result[0], "rgpe")
        unavailable = {(m, campaign.task_key(task)) for m in campaign.METHODS}
        self.assertIsNone(campaign.choose_task(spec, [task], active, unavailable, spec["cpus"], 0, 0))

    def test_commands_keep_methods_algorithms_and_preflight_separate(self):
        root = Path("/tmp/transfer_campaign_test")
        a, env, session = campaign.command(root, "fsbo", self.task(), [36, 37, 38, 39])
        b, _, other = campaign.command(root, "rgpe", self.task(), [36])
        c, _, _ = campaign.command(root, "fsbo", self.task("adaboost"), [36])
        self.assertNotEqual(session, other)
        self.assertNotEqual(a[a.index("--output-dir") + 1], b[b.index("--output-dir") + 1])
        self.assertNotEqual(a[a.index("--output-dir") + 1], c[c.index("--output-dir") + 1])
        self.assertEqual(env["EVAL_N_JOBS"], "4")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("ML_TRANSFER_HISTORY_MIRROR", env)

    def test_completion_rejects_other_identity_and_failed_budget(self):
        state = {"status": "complete", "run_identity": "fsbo_A"}
        info = {"identity": "fsbo_A", "attempts": 100, "successful": 100,
                "clean_complete": True, "pending": False}
        self.assertEqual(campaign.terminal_state(state, info), "complete")
        for changes in ({"identity": "rgpe_A"}, {"successful": 99}, {"pending": True}):
            self.assertEqual(campaign.terminal_state(state, {**info, **changes}), "incomplete")

    def test_worker_inherits_claim_after_scheduler_closes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.lock"
            lock = campaign.claim(path)
            child = subprocess.Popen([sys.executable, "-u", "-c", "print('ready'); input()"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     pass_fds=(lock.fileno(),))
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                lock.close()
                self.assertIsNone(campaign.claim(path))
                child.communicate("done\n", timeout=5)
                again = campaign.claim(path)
                self.assertIsNotNone(again)
                again.close()
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()

    def test_portable_source_keeps_hash_and_boundary_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            mirror = root / "mirror"
            mirror.mkdir()
            source = mirror / "history.json"
            source.write_text('{"observations": [1, 2, 3]}')
            dump = {"format_version": 2, "schema_fingerprint": "schema", "task_type": "cls",
                    "algorithm": "extra_trees", "tasks": {"cls/car/extra_trees": {
                        "dataset": "car", "source_path": str(original / "history.json"),
                        "source_sha256": sha256_file(source), "configs": [{}], "X": [[0]],
                        "y_minimize": [-1], "y_maximize": [1]}}}
            with patch.dict(os.environ, {"ML_TRANSFER_HISTORY_MIRROR": str(mirror)}):
                self.assertEqual(validate_dump(dump, set(dump["tasks"]), original), ["cls/car/extra_trees"])
                source.write_text("modified")
                with self.assertRaisesRegex(ValueError, "Source history changed"):
                    validate_dump(dump, set(dump["tasks"]), original)
                with self.assertRaises(ValueError):
                    checked_source_path(original / "../outside.json", original)
                outside = root / "outside.json"
                outside.write_text("outside")
                (mirror / "escape.json").symlink_to(outside)
                with self.assertRaises(ValueError):
                    checked_source_path(original / "escape.json", original)


if __name__ == "__main__":
    unittest.main()
