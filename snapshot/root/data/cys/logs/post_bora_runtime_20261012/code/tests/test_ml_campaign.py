import contextlib
import csv
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))
import ml_fixed_campaign as campaign
import monitor_ckpt32_campaign as monitor


class CampaignTest(unittest.TestCase):
    def test_bolt_command_uses_approved_runtime_and_explicit_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign.save(root / "protocol.json", {
                "baseline_python": "/approved/python",
                "method_args": {"bolt_ml": ["--extractor-max-tokens", "2048",
                                             "--recommendation-fallback", "default"]},
                "worker_env": {"HF_HUB_OFFLINE": "1"}})
            task = {"task_type": "cls", "dataset": "toy", "algorithm": "lightgbm", "metric": "bal_acc"}
            argv, env, _ = campaign.command(root, "bolt_ml", task, [4, 5, 6], 1)
            self.assertEqual(argv[3], "/approved/python")
            self.assertIn("--skip-test-eval", argv)
            self.assertEqual(argv[argv.index("--optimizer") + 1], "bolt_ml")
            self.assertEqual(argv[argv.index("--recommendation-fallback") + 1], "default")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(env["EVAL_N_JOBS"], "3")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_bolt_scheduler_handles_one_and_three_core_tasks(self):
        cpus = sorted(os.sched_getaffinity(0))[:4]
        starts = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code").mkdir()
            campaign.save(root / "tasks_132.json", [])
            campaign.save(root / "runtime_policy.json", {})
            tasks = [{"task_type": "cls", "dataset": "toy", "algorithm": algorithm,
                      "metric": "bal_acc"} for algorithm in ("adaboost", "lightgbm")]
            campaign.save(root / "queues/bolt_ml.json", tasks)
            campaign.save(root / "reused/bolt_ml.json", [])
            spec = {"hostname": socket.gethostname(), "cpus": cpus, "gpu_slots": [1, 1],
                    "gpu_methods": ["bolt_ml"], "limits": {"bolt_ml": 2}, "memory_gib": 10000,
                    "serial_worker_cores": 1, "parallel_worker_cores": 3}
            campaign.save(root / "protocol.json", {"hosts": {"LLMOptimizer3": spec},
                "methods": ["bolt_ml"], "task_count": 2, "ckpt32_policy": "rerun-all",
                "manifest_sha256": campaign.digest(root / "tasks_132.json"),
                "policy_sha256": campaign.digest(root / "runtime_policy.json")})

            def command(folder, method, task, assigned, gpu):
                starts.append((task["algorithm"], assigned, gpu))
                session = folder / "sessions" / method / campaign.task_key(task)
                program = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[2]); "
                    "p.mkdir(parents=True,exist_ok=True); "
                    "p.joinpath('progress.json').write_text(json.dumps(dict(attempts=100,"
                    "successful=100,failed=0,clean_complete=True,budget_exhausted=True,status='complete')))")
                return [sys.executable, "-c", program, "--session-dir", str(session)], os.environ.copy(), session

            original_sleep = time.sleep
            with patch.object(campaign, "command", command), \
                    patch.object(campaign.time, "sleep", lambda seconds: original_sleep(min(seconds, .05))), \
                    contextlib.redirect_stdout(io.StringIO()):
                campaign.run(root, "LLMOptimizer3")
            self.assertEqual(starts, [("adaboost", cpus[:1], 1), ("lightgbm", cpus[1:], 1)])
            self.assertEqual(campaign.status(root, quiet=True)["bolt_ml"]["new_clean"], 2)

    def test_single_method_campaign_has_132_records_and_uses_selected_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"ckpt_python": sys.executable, "base_model": "base", "adapter": "ckpt32",
                        "methods": ["ckpt32"], "task_count": 132, "ckpt32_policy": "rerun-all",
                        "ckpt32_hybrid_variant": "HPUCB_arm_p05_retry5"}
            campaign.save(root / "protocol.json", protocol)
            tasks = [{"task_type": "cls", "dataset": f"task_{i}", "algorithm": "extra_trees", "metric": "bal_acc"}
                     for i in range(132)]
            campaign.save(root / "queues/ckpt32.json", tasks)
            campaign.save(root / "reused/ckpt32.json", [])
            status = campaign.status(root, quiet=True)
            self.assertEqual(status["ckpt32"]["pending"], 132)
            self.assertEqual(len(campaign.load(root / "task_index.json")["tasks"]), 132)
            argv, _, _ = campaign.command(root, "ckpt32", tasks[0], [0], 0)
            self.assertEqual(argv[argv.index("--hybrid-variant") + 1], "HPUCB_arm_p05_retry5")

    def test_ckpt_command_uses_json_only_and_existing_duplicate_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign.save(root / "protocol.json",
                          {"ckpt_python": sys.executable, "base_model": "base", "adapter": "ckpt32"})
            task = {"task_type": "cls", "dataset": "toy", "algorithm": "extra_trees", "metric": "bal_acc"}
            argv, _, _ = campaign.command(root, "ckpt32", task, [0], 0)
            self.assertEqual(argv[argv.index("--llm-response-format") + 1], "json_only")
            self.assertNotIn("--llm-accept-duplicates", argv)
            self.assertEqual(argv[argv.index("--qwen-adapter-path") + 1], "ckpt32")

    def test_monitor_reads_large_retrieval_fields_and_checks_provenance(self):
        old_limit = csv.field_size_limit(131072)
        self.addCleanup(csv.field_size_limit, old_limit)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"task_type": "cls", "dataset": "toy", "algorithm": "extra_trees"}
            key = campaign.task_key(task)
            run_dir = root / "outputs"
            run_dir.mkdir()
            trajectory = run_dir / "trajectory.csv"
            with trajectory.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["iter_idx", "config_json", "experience_retrieval"])
                writer.writeheader()
                writer.writerow({"iter_idx": 0, "config_json": '{"n_estimators": 100}',
                                 "experience_retrieval": json.dumps({"evidence": "x" * 180000})})
            campaign.save(root / "protocol.json", {"adapter_sha256": "adapter", "source_gp_manifest_sha256": "gp",
                          "warm_start_sha256": "warm", "ckpt32_policy": "rerun-all"})
            campaign.save(root / "queues/ckpt32.json", [task])
            campaign.save(root / "state/ckpt32" / f"{key}.json",
                          {"host": "LLMOptimizer2", "pid": 1, "gpu": 0, "cpus": [0]})
            campaign.save(root / "sessions/ckpt32" / key / "progress.json",
                          {"attempts": 1, "successful": 1, "failed": 0, "trajectory": str(trajectory)})
            config = {"optimizer_params": {"adapter_verification": {"sha256": "adapter", "verified_tensors": 496},
                      "source_gp_manifest_sha256": "gp", "preset_warm_start_sha256": "warm"}}
            campaign.save(run_dir / "config.json", config)
            campaign.save(run_dir / "retrieval/trial_0001.json", {"observations": [], "selected_ids": []})
            with patch.object(campaign, "status", return_value={}), patch.object(campaign, "process_alive", return_value=True):
                result = monitor.inspect(root, {"ckpt32_revision": "superseded",
                                               "tasks": [{"task": key, "attempts": 100}]})
                self.assertEqual(result["issues"], [])
                self.assertEqual(result["tasks"][0]["successful"], 1)
                config["optimizer_params"]["adapter_verification"]["sha256"] = "wrong"
                campaign.save(run_dir / "config.json", config)
                result = monitor.inspect(root, {})
                self.assertIn(f"{key}: missing or mismatched ckpt32 adapter verification", result["issues"])

    def test_claim_excludes_second_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.lock"
            first = campaign.claim(path)
            try:
                self.assertIsNone(campaign.claim(path))
            finally:
                first.close()
            second = campaign.claim(path)
            self.assertIsNotNone(second)
            second.close()

    def test_scheduler_admits_without_cpu_overlap_and_indexes_each_pair(self):
        cpus = sorted(os.sched_getaffinity(0))[:12]
        starts = []
        children = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code").mkdir()
            campaign.save(root / "tasks_132.json", [])
            campaign.save(root / "runtime_policy.json", {})
            tasks = [{"task_type": "cls", "dataset": f"dataset_{i}", "algorithm": "extra_trees", "metric": "bal_acc"}
                     for i in range(132)]
            for method in campaign.inventory.METHODS:
                campaign.save(root / "queues" / f"{method}.json", tasks[:2])
                campaign.save(root / "reused" / f"{method}.json",
                              [{**task, "path": "legacy.csv", "sha256": "verified"} for task in tasks[2:]])
            spec = {"hostname": socket.gethostname(), "cpus": cpus, "gpu_slots": [0, 0, 1, 1],
                    "gpu_methods": ["ckpt32"],
                    "limits": {method: 8 for method in campaign.inventory.METHODS}, "memory_gib": 10000}
            campaign.save(root / "protocol.json", {"hosts": {"LLMOptimizer2": spec}, "ckpt32_policy": "rerun-all",
                          "manifest_sha256": campaign.digest(root / "tasks_132.json"),
                          "policy_sha256": campaign.digest(root / "runtime_policy.json")})

            def command(folder, method, task, assigned, gpu):
                occupied = {cpu for process, used in children if process.poll() is None for cpu in used}
                self.assertFalse(occupied.intersection(assigned))
                starts.append((method, task["dataset"], assigned, gpu))
                session = folder / "sessions" / method / campaign.task_key(task)
                program = ("import json,pathlib,time,sys; time.sleep(0.15); "
                           "p=pathlib.Path(sys.argv[2]); p.mkdir(parents=True,exist_ok=True); "
                           "p.joinpath('progress.json').write_text(json.dumps(dict(attempts=100,successful=100,failed=0,"
                           "clean_complete=True,budget_exhausted=True,status='complete')))")
                return [sys.executable, "-c", program, "--session-dir", str(session)], os.environ.copy(), session

            original_popen = campaign.subprocess.Popen
            original_sleep = time.sleep

            def popen(*args, **kwargs):
                process = original_popen(*args, **kwargs)
                children.append((process, starts[-1][2]))
                return process

            with patch.object(campaign, "command", command), patch.object(campaign.subprocess, "Popen", popen), \
                    patch.object(campaign.time, "sleep", lambda seconds: original_sleep(min(seconds, 0.1))), \
                    contextlib.redirect_stdout(io.StringIO()):
                campaign.run(root, "LLMOptimizer2")
                summary = campaign.status(root)
            self.assertEqual(len(starts), 8)
            self.assertTrue(all(gpu is None for method, _, _, gpu in starts if method == "lolbo"))
            self.assertEqual(len({(method, task) for method, task, _, _ in starts}), 8)
            self.assertTrue(all(value["clean_total"] == 132 for value in summary.values()))
            self.assertEqual(len(json.loads((root / "task_index.json").read_text())["tasks"]), 528)
            task = tasks[0]
            progress_path = root / "sessions/tpe" / campaign.task_key(task) / "progress.json"
            progress_path.unlink()
            state_path = root / "state/tpe" / f"{campaign.task_key(task)}.json"
            state = campaign.load(state_path)
            state["status"] = "running"
            campaign.save(state_path, state)
            summary = campaign.status(root, quiet=True)
            self.assertEqual(summary["tpe"]["running"], 0)
            self.assertEqual(summary["tpe"]["liveness_unverified"], 1)


if __name__ == "__main__":
    unittest.main()
