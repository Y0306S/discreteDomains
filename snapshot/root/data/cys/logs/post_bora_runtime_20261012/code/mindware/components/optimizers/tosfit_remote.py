"""Task-owned ToSFiT actors with durable, replayable mutation receipts."""

import contextlib
import fcntl
import gc
import json
import os
from pathlib import Path
import pickle
import random
import sqlite3
import threading

from .runtime_rpc import call, fingerprint


def capture_rng():
    import numpy as np
    import torch
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class DurableToSFiTActor:
    def __init__(self, root, identity, factory, *, checkpoint_lock=None, fault=None):
        self.root = Path(root).resolve()
        self.identity, self.factory = identity, factory
        self.checkpoint_lock, self.fault = checkpoint_lock, fault
        self.lock = threading.Lock()
        self.runtime = self.connection = self.session = self.run_id = None
        self.task_lock = None
        self.busy = False

    def health(self):
        return {"kind": "tosfit", "identity": self.identity, "ready": True,
                "busy": self.busy, "run_id": self.run_id,
                "session": str(self.session) if self.session is not None else None}

    def _drop(self):
        if self.runtime is not None:
            self.runtime.close()
        self.runtime = None
        if self.connection is not None:
            self.connection.close()
        self.connection = self.session = self.run_id = None
        if self.task_lock is not None:
            self.task_lock.close()
            self.task_lock = None
        gc.collect()
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def _bind(self, request):
        session = Path(request["session"]).resolve()
        if not session.is_relative_to(self.root) or session == self.root:
            raise ValueError("Session must be within the configured campaign sessions")
        run_id = request["run_id"]
        if not isinstance(run_id, str) or not 1 <= len(run_id) <= 256:
            raise ValueError("Invalid run identity")
        if self.session is not None:
            if (session, run_id) != (self.session, self.run_id):
                raise ValueError("This model slot belongs to another task; detach it after its worker exits")
            return
        session.mkdir(parents=True, exist_ok=True)
        task_lock = (session / "tosfit-runtime.lock").open("a")
        try:
            fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            task_lock.close()
            raise
        connection = sqlite3.connect(session / "tosfit-runtime.sqlite", timeout=60, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), "
                           "identity TEXT NOT NULL, run_id TEXT NOT NULL, seed INTEGER NOT NULL, "
                           "next_seq INTEGER NOT NULL, payload BLOB)")
        connection.execute("CREATE TABLE IF NOT EXISTS receipts (seq INTEGER PRIMARY KEY, "
                           "request_hash TEXT NOT NULL, response TEXT NOT NULL)")
        row = connection.execute("SELECT identity, run_id, seed FROM state WHERE id=1").fetchone()
        expected = (self.identity, run_id, int(request["seed"]))
        if row is not None and row != expected:
            connection.close()
            task_lock.close()
            raise ValueError("Remote task checkpoint identity or seed changed")
        if row is None:
            connection.execute("INSERT INTO state VALUES (1, ?, ?, ?, 0, NULL)", expected)
        connection.commit()
        self.connection, self.session, self.run_id = connection, session, run_id
        self.task_lock = task_lock

    def _load(self):
        if self.runtime is not None:
            return
        row = self.connection.execute("SELECT seed, payload FROM state WHERE id=1").fetchone()
        import numpy as np
        import torch
        random.seed(row[0])
        np.random.seed(row[0])
        torch.manual_seed(row[0])
        self.runtime = self.factory()
        if row[1] is not None:
            state = pickle.loads(row[1])
            self.runtime.restore(state["runtime"])
            restore_rng(state["rng"])

    @contextlib.contextmanager
    def _writer(self):
        if self.checkpoint_lock is None:
            yield
            return
        path = Path(self.checkpoint_lock)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _prune(self, state):
        keep = state.get("weights")
        for path in (self.session / "tosfit-policy").glob("weights-*.pt"):
            if str(path) != keep:
                path.unlink()

    def handle(self, request):
        with self.lock:
            self.busy = True
            try:
                if request.get("identity") != self.identity:
                    raise ValueError("ToSFiT service identity mismatch")
                self._bind(request)
                operation = request["operation"]
                next_seq = self.connection.execute("SELECT next_seq FROM state WHERE id=1").fetchone()[0]
                if operation == "status":
                    return {"next_seq": next_seq, "identity": self.identity}
                if operation == "detach":
                    self._drop()
                    return {"detached": True}
                seq = request["seq"]
                if type(seq) is not int or seq < 0:
                    raise ValueError("Invalid sequence")
                digest = fingerprint(request)
                previous = self.connection.execute(
                    "SELECT request_hash, response FROM receipts WHERE seq=?", (seq,)).fetchone()
                if previous is not None:
                    if previous[0] != digest:
                        raise ValueError("A mutation sequence was reused with different input")
                    return json.loads(previous[1])
                if seq != next_seq:
                    raise ValueError("Missing earlier remote mutations")
                self._load()
                runtime = self.runtime
                if request["policy_version"] != runtime.policy_version:
                    raise ValueError("Remote policy version mismatch")
                body = request["body"]
                try:
                    if operation == "propose":
                        result = runtime.propose(body["prompt"], int(body["step"]))
                    elif operation == "observe":
                        import math
                        if not math.isfinite(body["reward"]):
                            raise ValueError("Remote observations must have a finite reward")
                        runtime.observe(body["features"], body["reward"])
                        result = {"observed": True}
                    elif operation == "retire":
                        runtime.saved_weights = None
                        runtime.saved_version = runtime.policy_version
                        runtime.weights_retired = True
                        runtime.gp = None
                        result = {"retired": True}
                    else:
                        raise ValueError("Unknown ToSFiT operation")
                    result = dict(result, policy_version=runtime.policy_version)
                    response = json.dumps(result, sort_keys=True, allow_nan=False)
                    if self.fault:
                        self.fault("before_commit")
                    # Commit weights, RNG, GP and receipt before acknowledging a mutation.
                    with self._writer():
                        snapshot = runtime.snapshot(self.session)
                        result['gpu_peak_reserved_gib'] = snapshot.get('gpu_peak_reserved_gib', {})
                        response = json.dumps(result, sort_keys=True, allow_nan=False)
                        payload = pickle.dumps({"runtime": snapshot, "rng": capture_rng()},
                                               protocol=pickle.HIGHEST_PROTOCOL)
                        with self.connection:
                            self.connection.execute("UPDATE state SET next_seq=?, payload=? WHERE id=1",
                                                    (seq + 1, payload))
                            self.connection.execute("INSERT INTO receipts VALUES (?, ?, ?)",
                                                    (seq, digest, response))
                    self._prune(snapshot)
                    if operation == "retire":
                        self.connection.execute("VACUUM")
                        runtime.close()
                    if self.fault:
                        self.fault("after_commit")
                    return result
                except BaseException:
                    # A partial in-memory update must never be the starting point of a retry.
                    runtime.close()
                    self.runtime = None
                    gc.collect()
                    raise
            finally:
                self.busy = False


class RemoteToSFiTRuntime:
    def __init__(self, config, session, run_id, seed):
        self.config = dict(config)
        self.session, self.run_id, self.seed = str(Path(session).resolve()), run_id, int(seed)
        self.next_seq = self.policy_version = 0
        self.saved_version, self.saved_weights = -1, None
        self.ready = self.weights_retired = self.retired_committed = False
        self.gpu_peaks = {}

    @classmethod
    def from_environment(cls, seed):
        config = json.loads(Path(os.environ["TOSFIT_REMOTE_CONFIG"]).read_text())
        return cls(config, os.environ["ML_SESSION_DIR"], os.environ["ML_RUN_ID"], seed)

    def _request(self, operation, body=None, *, mutation=True):
        request = {"identity": self.config["identity"], "session": self.session,
                   "run_id": self.run_id, "seed": self.seed, "operation": operation}
        if mutation:
            request.update(seq=self.next_seq, policy_version=self.policy_version, body=body or {})
        result = call(self.config["url"] + "/rpc", request,
                      timeout=self.config.get("timeout_seconds", 7200))
        if mutation:
            self.next_seq += 1
            self.policy_version = result["policy_version"]
            self.gpu_peaks = result.get("gpu_peak_reserved_gib", {})
        return result

    def propose(self, prompt, step):
        if self.weights_retired:
            raise RuntimeError("Completed task weights were retired")
        result = self._request("propose", {"prompt": prompt, "step": step})
        self.ready = True
        return result

    def observe(self, features, reward):
        return self._request("observe", {"features": features, "reward": reward})

    def snapshot(self, directory):
        if str(Path(directory).resolve()) != self.session:
            raise ValueError("Remote checkpoint session changed")
        if self.weights_retired and not self.retired_committed:
            self._request("retire")
            self.retired_committed = True
        return {"remote": True, "identity": self.config["identity"], "run_id": self.run_id,
                "session": self.session, "next_seq": self.next_seq, "ready": self.ready,
                "version": self.policy_version, "weights": None,
                "weights_retired": self.weights_retired, "gpu_peak_reserved_gib": self.gpu_peaks}

    def restore(self, state):
        if state is None:
            return
        if not state.get("remote") or any(state[key] != value for key, value in (
                ("identity", self.config["identity"]), ("run_id", self.run_id), ("session", self.session))):
            raise ValueError("Remote ToSFiT checkpoint identity changed")
        remote = self._request("status", mutation=False)
        if remote["next_seq"] < state["next_seq"]:
            raise RuntimeError("Remote durable state is behind the committed ML checkpoint")
        self.next_seq, self.policy_version = state["next_seq"], state["version"]
        self.ready = state["ready"]
        self.gpu_peaks = state.get("gpu_peak_reserved_gib", {})
        self.weights_retired = self.retired_committed = state["weights_retired"]

    def detach(self):
        return self._request("detach", mutation=False)
