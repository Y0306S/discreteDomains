"""Persistent OptFormer requests with seeded task isolation and durable replies."""

import json
import os
from pathlib import Path
import sqlite3
import threading

from .runtime_rpc import call, fingerprint


class OptFormerService:
    def __init__(self, engine, journal):
        self.engine = engine
        self.identity = engine.meta["checkpoint_sha256"]
        self.lock = threading.Lock()
        self.pending = 0
        Path(journal).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(journal, timeout=60, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS replies "
                                "(request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, response TEXT NOT NULL)")
        self.connection.commit()

    def health(self):
        return {"kind": "optformer", "identity": self.identity, "ready": True,
                "pending": self.pending}

    def handle(self, envelope):
        self.pending += 1
        try:
            with self.lock:
                if envelope["identity"] != self.identity:
                    raise ValueError("OptFormer checkpoint mismatch")
                digest = fingerprint(envelope["request"])
                prior = self.connection.execute("SELECT digest, response FROM replies WHERE request_id=?",
                                                (envelope["request_id"],)).fetchone()
                if prior:
                    if prior[0] != digest:
                        raise ValueError("OptFormer request ID reused with different data")
                    return json.loads(prior[1])
                response = self.engine.suggest(envelope["request"])
                if response["checkpoint_sha256"] != self.identity:
                    raise ValueError("OptFormer result checkpoint mismatch")
                with self.connection:
                    self.connection.execute("INSERT INTO replies VALUES (?, ?, ?)",
                        (envelope["request_id"], digest, json.dumps(response, allow_nan=False)))
                return response
        finally:
            self.pending -= 1


def suggest_remote(request, expected_checkpoint, config=None):
    if config is None:
        config = json.loads(Path(os.environ["OPTFORMER_REMOTE_CONFIG"]).read_text())
    if config["identity"] != expected_checkpoint:
        raise ValueError("OptFormer client checkpoint mismatch")
    envelope = {"identity": expected_checkpoint, "request": request,
                "request_id": fingerprint({"run_id": os.environ.get("ML_RUN_ID"), "request": request,
                                           "identity": expected_checkpoint})}
    urls = list(config["urls"])
    offset = int(envelope["request_id"][:8], 16) % len(urls)
    urls = urls[offset:] + urls[:offset]
    healthy = []
    for index, url in enumerate(urls):
        try:
            status = call(url + "/health", timeout=3, retries=0)
            if status.get("ready") and status.get("identity") == expected_checkpoint:
                healthy.append((status.get("pending", 0), index, url))
        except RuntimeError:
            continue
    if not healthy:
        raise RuntimeError("No verified OptFormer replica is available")
    url = min(healthy)[2]
    response = call(url + "/rpc", envelope, timeout=config.get("timeout_seconds", 1800))
    if response.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("OptFormer response checkpoint mismatch")
    return response, {"url": url, "request_id": envelope["request_id"], "transport": "persistent_rpc"}
