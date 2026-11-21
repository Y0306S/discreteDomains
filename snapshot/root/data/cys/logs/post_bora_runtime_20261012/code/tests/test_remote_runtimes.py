"""Exercise real ToSFiT math through durable RPC, including lost acknowledgments."""

import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import gc
import weakref

import torch

from mindware.components.optimizers.runtime_rpc import call, server
from mindware.components.optimizers.tosfit_remote import DurableToSFiTActor, RemoteToSFiTRuntime
from mindware.components.optimizers.optformer_remote import OptFormerService
from test_tosfit_runtime import RuntimeTests


class ActorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.factory = RuntimeTests().make
        self.actor = DurableToSFiTActor(self.root, "code-v1", self.factory)
        self.httpd = server(self.actor, 0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.config = {"url": self.url, "identity": "code-v1", "timeout_seconds": 10}
        self.client = RemoteToSFiTRuntime(self.config, self.root / "task-a", "run-a", 42)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        self.actor._drop()
        self.tmp.cleanup()

    def test_gradient_rounds_reload_and_new_task_reset(self):
        first = self.client.propose("task prompt", 0)
        self.client.observe(first["features"], -3.0)
        for step in (16, 17, 18):
            sample = self.client.propose("task prompt", step)
            self.client.observe(sample["features"], -2.0)
        self.assertEqual(self.client.policy_version, 3)
        state = self.client.snapshot(self.root / "task-a")
        weights = {k: v.clone() for k, v in self.actor.runtime.model.state_dict().items()}
        gp_count = self.actor.runtime.gp.n_observations
        self.client.detach()
        resumed = RemoteToSFiTRuntime(self.config, self.root / "task-a", "run-a", 42)
        resumed.restore(state)
        with self.actor.lock:
            self.actor._load()
        self.assertEqual(self.actor.runtime.gp.n_observations, gp_count)
        self.assertTrue(all(torch.equal(v, self.actor.runtime.model.state_dict()[k]) for k, v in weights.items()))
        resumed.detach()
        other = RemoteToSFiTRuntime(self.config, self.root / "task-b", "run-b", 42)
        self.assertEqual(other.propose("task prompt", 0), first)
        self.assertEqual(other.policy_version, 0)

    def test_lost_acknowledgment_does_not_repeat_gradient(self):
        fired = []
        def fault(where):
            if where == "after_commit" and not fired:
                fired.append(True)
                raise RuntimeError("lost acknowledgment")
        self.actor.fault = fault
        with self.assertRaisesRegex(RuntimeError, "lost acknowledgment"):
            self.client.propose("task prompt", 16)
        self.assertEqual(self.client.next_seq, 0)
        result = self.client.propose("task prompt", 16)
        self.assertEqual(result["policy_version"], 1)
        count = self.actor.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        self.assertEqual(count, 1)

    def test_uncommitted_gradient_is_rolled_back_before_retry(self):
        first = self.client.propose("task prompt", 0)
        self.client.observe(first["features"], -3.0)
        fired = []
        def fault(where):
            if where == "before_commit" and not fired:
                fired.append(True)
                raise RuntimeError("crash before commit")
        self.actor.fault = fault
        with self.assertRaisesRegex(RuntimeError, "crash before commit"):
            self.client.propose("task prompt", 16)
        result = self.client.propose("task prompt", 16)
        self.assertEqual(result["policy_version"], 1)
        self.assertEqual(self.actor.runtime.gp.n_observations, 1)

    def test_checkpoint_cursor_replays_server_ahead_of_client(self):
        sample = self.client.propose("task prompt", 16)
        state = self.client.snapshot(self.root / "task-a")
        self.client.observe(sample["features"], -2.0)
        self.client.detach()
        resumed = RemoteToSFiTRuntime(self.config, self.root / "task-a", "run-a", 42)
        resumed.restore(state)
        resumed.observe(sample["features"], -2.0)
        with self.actor.lock:
            self.actor._load()
        self.assertEqual(self.actor.runtime.gp.n_observations, 1)

    def test_reused_sequence_and_cross_task_requests_are_rejected(self):
        self.client.propose("task prompt", 0)
        stale = RemoteToSFiTRuntime(self.config, self.root / "task-a", "run-a", 42)
        with self.assertRaisesRegex(RuntimeError, "different input"):
            stale.propose("different prompt", 0)
        other = RemoteToSFiTRuntime(self.config, self.root / "task-b", "run-b", 42)
        with self.assertRaisesRegex(RuntimeError, "another task"):
            other.propose("task prompt", 0)

    def test_completed_weights_retire_and_cannot_extend(self):
        self.client.propose("task prompt", 16)
        self.client.weights_retired = True
        state = self.client.snapshot(self.root / "task-a")
        self.assertTrue(state["weights_retired"])
        self.assertFalse(list((self.root / "task-a/tosfit-policy").glob("weights-*.pt")))
        with self.assertRaises(RuntimeError):
            self.client.propose("task prompt", 17)

    def test_detach_releases_policy_parameters(self):
        self.client.propose("task prompt", 16)
        parameters = [weakref.ref(p) for p in self.actor.runtime.model.parameters()]
        self.client.detach()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in parameters))


class OptFormerTests(unittest.TestCase):
    def test_durable_results_and_identity(self):
        class Engine:
            meta = {"checkpoint_sha256": "checkpoint"}
            calls = 0
            def suggest(self, request):
                self.calls += 1
                return {"checkpoint_sha256": "checkpoint", "config": {"x": request["seed"]}}
        with tempfile.TemporaryDirectory() as root:
            engine = Engine()
            journal = Path(root) / "requests.sqlite"
            service = OptFormerService(engine, journal)
            request = {"identity": "checkpoint", "request_id": "a", "request": {"seed": 2}}
            result = service.handle(request)
            self.assertEqual(service.handle(request), result)
            self.assertEqual(engine.calls, 1)
            service.connection.close()
            service = OptFormerService(engine, journal)
            self.assertEqual(service.handle(request), result)
            self.assertEqual(engine.calls, 1)
            with self.assertRaises(ValueError):
                service.handle(dict(request, request={"seed": 3}))
            with self.assertRaises(ValueError):
                service.handle(dict(request, identity="different checkpoint"))
            service.connection.close()


if __name__ == "__main__":
    unittest.main()
