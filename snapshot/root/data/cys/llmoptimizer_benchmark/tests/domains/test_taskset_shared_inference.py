"""Offline checks: ML scheduling parity, concurrent HTTP clients and BOLT context."""
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from mindware.domains.dlhpo.taskset_llm.inference import (
    InferenceServiceError, SharedQwenClient, fingerprint, make_server,
)
from mindware.domains.dlhpo.taskset_llm import loop as L, p5_prompt as P, spec as S
from mindware.domains.dlhpo.taskset_llm.llm_client import ScriptedClient
from tests.domains import test_taskset_llm_rgpe as fixtures

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/benchmark/domains"))


class SharedInferenceTest(unittest.TestCase):
    def setUp(self):
        class Model:
            active = maximum_active = calls = 0

            def generate(self, messages, *, n, temperature, seed):
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                self.calls += 1
                time.sleep(0.01)
                self.active -= 1
                return [f"{seed}:{messages[-1]['content']}:{i}" for i in range(n)]

        self.model = Model()
        self.identity = {"model": "offline-test", "max_new_tokens": 768}
        self.server = make_server(("127.0.0.1", 0), self.model, self.identity, "unit-test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.token = Path(self.tmp.name) / "token"
        self.token.write_text("unit-test-token")
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def client(self, **kwargs):
        return SharedQwenClient(self.url, self.token, timeout=5, **kwargs)

    def test_many_cpu_clients_one_generation_at_a_time(self):
        clients = [self.client(seed=i) for i in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda item: item[1].complete("sys", f"task{item[0]}"), enumerate(clients)))
        self.assertEqual(self.model.calls, 8)
        self.assertEqual(self.model.maximum_active, 1)
        self.assertEqual(len(set(results)), 8)
        # Repeat in a different order: scheduling order does not change sampling.
        for i in reversed(range(8)):
            self.assertEqual(clients[i].complete("sys", f"task{i}"), results[i])

    def test_identity_auth_and_bad_request_do_not_generate(self):
        with self.assertRaises(InferenceServiceError):
            self.client(expected_identity="wrong-checkpoint")
        self.token.write_text("wrong-token")
        with self.assertRaises(InferenceServiceError):
            self.client()
        self.token.write_text("unit-test-token")
        client = self.client(expected_identity=fingerprint(self.identity))
        with self.assertRaises(InferenceServiceError):
            client.generate([{"role": "user", "content": "test"}], n=4)
        client.identity_sha256 = "different-checkpoint"
        with self.assertRaises(InferenceServiceError):
            client.complete("sys", "user")
        self.assertEqual(self.model.calls, 0)

    def test_expanded_queue_serves_more_than_sixteen_clients(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server = make_server(("127.0.0.1", 0), self.model, self.identity,
                                  "unit-test-token", max_pending=128)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        clients = [self.client(seed=i) for i in range(32)]
        barrier = threading.Barrier(len(clients))

        def generate(item):
            barrier.wait()
            i, client = item
            return client.complete("sys", f"task{i}")

        with ThreadPoolExecutor(max_workers=32) as pool:
            results = list(pool.map(generate, enumerate(clients)))
        self.assertEqual(self.model.calls, 32)
        self.assertEqual(self.model.maximum_active, 1)
        self.assertEqual(len(set(results)), 32)

    def test_server_failure_has_no_hidden_retry(self):
        calls = []

        def fail(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("offline simulated model error")

        self.model.generate = fail
        with self.assertRaises(InferenceServiceError):
            self.client().complete("sys", "user")
        self.assertEqual(len(calls), 1)


class MLParityTest(unittest.TestCase):
    def test_all_100_rounds_against_actual_ml_method(self):
        path = Path("/root/data/cys/blockdata_archive/llmoptimizer_benchmark/mindware/components/optimizers/pt4opt_llm_rgpe_optimizer.py")
        if not path.exists():
            self.skipTest("Archived ML reference is not installed")
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_choose_source")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        choose_ml = namespace["_choose_source"]
        for seed in (1, 7, 42):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as tmp:
                loop, _ = fixtures.LoopTest().make(Path(tmp), ScriptedClient([]), budget=100)
                loop.selector.rng.seed(seed)
                ml = SimpleNamespace(selector=copy.deepcopy(loop.selector), traj_rows=[],
                                     n_no_history_samples=3, evaluation_num_limit=100)
                for trial in range(1, 101):
                    arm, details = loop.choose_source(trial)
                    expected, _, _ = choose_ml(ml, trial - 1)
                    self.assertEqual(arm, expected, f"seed={seed}, round={trial}")
                    for key in ("round", "cooldown_active", "cooldown_window", "selection_reason", "selected_arm"):
                        self.assertEqual(details[key], ml._selection_details[key])
                    loop.observations.append({"extra_info": {"selected_arm": arm}})
                    ml.traj_rows.append({"iter_idx": trial - 1, "selected_arm": arm})
                    if trial > 3:
                        # Include fallback credits and failed observations.
                        credited = "rgpe" if trial % 5 == 0 else arm
                        perf = -float("inf") if trial % 13 == 0 else trial / 100
                        for selector in (loop.selector, ml.selector):
                            selector.update(credited, perf, higher_is_better=True,
                                            beat_incumbent=trial % 3 == 0, beat_arm_best=trial % 2 == 0)
                            if arm != credited:
                                selector.record_attempt(arm)

    def test_prompt_boundary_and_counts_include_failed_and_zero_families(self):
        cs = fixtures.space_from_taskset()
        config = fixtures.sample(cs, 1)
        rows = [{"trial": 1, "config": config, "loss": .3, "successful": True},
                {"trial": 2, "config": config, "loss": None, "successful": False}]
        _, user49, meta49 = P.build_p5_prompt(fixtures.TASK, fixtures.SPECS, cs, rows, trial=49, seed=1)
        self.assertIsNone(meta49["experience_branch"])
        self.assertNotIn("Exploration counts", user49)
        for trial in (50, 51, 100):
            _, _, meta = P.build_p5_prompt(fixtures.TASK, fixtures.SPECS, cs, rows, trial=trial, seed=1)
            self.assertIsNotNone(meta["experience_branch"])
        lines = "\n".join(P.explore_block(rows, S.families(cs)))
        self.assertIn(f"{config['optimizer_family']}: 2 evaluations (1 successful)", lines)
        self.assertIn(": 0 evaluations (0 successful)", lines)

    def test_service_outage_does_not_evaluate_or_randomly_fallback(self):
        def outage(*_args):
            raise InferenceServiceError("offline outage")

        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fixtures.LoopTest().make(Path(tmp), ScriptedClient(outage))
            for _ in range(3):
                loop.step()
            evaluated = []
            loop.evaluate = lambda config: evaluated.append(config)
            with self.assertRaises(InferenceServiceError):
                loop.step()
            self.assertEqual(len(loop.observations), 3)
            self.assertEqual(evaluated, [])

    def test_bolt_uses_full_sft_training_context_for_frozen_test200(self):
        from gen_taskset_ws3_presets import bolt_records
        from export_bolt_taskset_top10 import build_messages
        path = ROOT / "benchmark_data/taskset_s1b_s2g_train838_20261029_v2/test200.json"
        cs, records = bolt_records(path)
        split, specs = json.loads(path.read_text()), S.load_task_specs()
        self.assertEqual([r["metadata"]["task_id"] for r in records], split["test"])
        self.assertEqual(len(records), 200)
        self.assertEqual(list(cs.get_hyperparameter("nadamw_grid:use_nesterov").choices), [False])
        for record in records:
            task = record["metadata"]["task_id"]
            self.assertEqual(record["messages"], build_messages(task, specs[task], cs, {})[:2])
            self.assertEqual([m["role"] for m in record["messages"]], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
