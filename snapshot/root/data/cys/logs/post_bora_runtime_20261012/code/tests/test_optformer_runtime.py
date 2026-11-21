import copy
import unittest

from examples.benchmark.optformer_worker import validate_resume
from examples.benchmark.benchmark_optformer_runtime import choose_configuration, runtime_command


class RuntimeTests(unittest.TestCase):
    def test_runtime_change_cannot_bypass_training_identity(self):
        original = {"model_size": "base", "batch_size": 256, "microbatches": 64,
                    "seed": 123, "sequences_sha256": "original", "train_tasks": ["allowed"]}
        changed = dict(original, microbatches=16)
        with self.assertRaises(ValueError):
            validate_resume(original, changed)
        self.assertEqual(set(validate_resume(original, changed, True)), {"microbatches"})
        for key, value in (("model_size", "tiny"), ("batch_size", 128), ("seed", 1),
                           ("sequences_sha256", "changed"), ("train_tasks", ["other"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume(original, dict(changed, **{key: value}), True)

    def test_selection_requires_successful_baseline_and_ten_percent_speedup(self):
        baseline = {"name": "1gpu_b4_log1", "status": "passed", "seconds_per_update": 10}
        other = {"name": "4gpu_b8_log20", "status": "passed", "seconds_per_update": 9.5}
        self.assertIs(choose_configuration([baseline, other]), baseline)
        other["seconds_per_update"] = 5
        self.assertIs(choose_configuration([baseline, other]), other)
        other["status"] = "failed"
        self.assertIs(choose_configuration([baseline, other]), baseline)
        self.assertIsNone(choose_configuration([other]))

    def test_benchmark_command_preserves_sources_and_global_batch(self):
        original = ["python", "worker.py", "train", "--sequences", "frozen.jsonl",
                    "--batch-size", "256", "--model-dir", "production", "--steps", "100000"]
        saved = copy.copy(original)
        command = runtime_command(original, "benchmark", 6760, 8, 20, benchmark=True)
        self.assertEqual(original, saved)
        self.assertEqual(command[command.index("--sequences") + 1], "frozen.jsonl")
        self.assertEqual(command[command.index("--batch-size") + 1], "256")
        self.assertIn("--benchmark", command)
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
