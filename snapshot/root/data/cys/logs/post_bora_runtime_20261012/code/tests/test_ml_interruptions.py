"""Integration check for orphan cleanup and cumulative retry allowances."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class InterruptionTest(unittest.TestCase):
    def test_owner_kill_retries_with_remaining_allowance(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            env = {**os.environ, "PYTHONPATH": str(root), "ML_SESSION_DIR": directory,
                   "ML_EVALUATION_ID": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            command = [sys.executable, str(root / "tests/ml_crash_probe.py")]
            owner = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 25
                while not (folder / "objective_pid").exists():
                    self.assertIsNone(owner.poll())
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.1)
                child = int((folder / "objective_pid").read_text())
                owner.kill()
                owner.communicate(timeout=10)
                journal = folder / "evaluations/0/execution.json"
                deadline = time.monotonic() + 10
                while True:
                    prior = json.loads(journal.read_text())
                    stat = Path(f"/proc/{child}/stat")
                    exited = not stat.exists() or stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z"
                    if prior["status"] == "interrupted" and exited:
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.1)
                self.assertGreater(prior["elapsed_seconds"], 0)
                rerun = subprocess.run(command, env=env, capture_output=True, text=True, timeout=40)
                self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
                self.assertIn("'timeout': True", rerun.stdout)
                final = json.loads(journal.read_text())
                self.assertEqual(final["execution_number"], 2)
                self.assertEqual(final["interrupted_executions"], 1)
                self.assertEqual(len(final["executions"]), 1)
                self.assertEqual(final["previous_seconds"], prior["elapsed_seconds"])
                self.assertGreaterEqual(final["elapsed_seconds"], 30)
                self.assertLess(final["elapsed_seconds"], 33)
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
