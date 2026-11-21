"""Dispatch exclusion, crash recovery and cgroup admission checks."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/benchmark/domains"))
from queue_taskset_llm_fsbo import claim_task, resource_limits


def hold_claim(path, ready):
    with claim_task(path, "task") as acquired:
        ready.put(acquired)
        time.sleep(60)


class SharedDispatchTest(unittest.TestCase):
    def test_concurrent_workers_claim_once_and_skip_committed_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            def work(_):
                with claim_task(tmp, "task") as acquired:
                    if acquired:
                        time.sleep(.03)
                        Path(tmp, "task.done.json").write_text("{}")
                    return acquired
            with ThreadPoolExecutor(max_workers=32) as pool:
                results = list(pool.map(work, range(64)))
            self.assertEqual(sum(results), 1)

    def test_dead_process_claim_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context("spawn")
            ready = ctx.Queue()
            process = ctx.Process(target=hold_claim, args=(tmp, ready))
            process.start()
            try:
                self.assertTrue(ready.get(timeout=15))
                with claim_task(tmp, "task") as acquired:
                    self.assertFalse(acquired)
            finally:
                process.terminate()
                process.join(timeout=15)
            with claim_task(tmp, "task") as acquired:
                self.assertTrue(acquired)

    def test_failed_task_is_preserved_for_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "task.failed.json").write_text(json.dumps({"returncode": 1}))
            with claim_task(tmp, "task") as acquired:
                self.assertFalse(acquired)

    def test_cpu_quota_and_cgroup_memory_reject_excess(self):
        config = {"reserved_cpu_cores": 1000000, "workers": [{"cpus": [0]}],
                  "min_available_memory_gib": 0, "memory_per_worker_gib": 1}
        with self.assertRaisesRegex(ValueError, "exceeds quota"):
            resource_limits(config)
        config.update(reserved_cpu_cores=0, memory_per_worker_gib=1000000)
        with self.assertRaisesRegex(ValueError, "Memory headroom"):
            resource_limits(config)


if __name__ == "__main__":
    unittest.main()
