"""One verified Qwen replica per GPU, behind a durable host-local request broker."""

import argparse
import base64
from contextlib import nullcontext
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import sqlite3
import threading
import time
from urllib.error import URLError

from qwen_inference_transport import (SharedInferenceError, digest, encoded, encode_rng,
                                      http_json, model_identity)


def storage_failure(error):
    return any(message in str(error).lower() for message in
               ('disk quota exceeded', 'database or disk is full', 'unable to open database file',
                # A journal lock is transient storage contention, not a model failure (2026-10-09).
                'database is locked'))


class Replica:
    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer
        from hf_qwen_client import HuggingFaceQwenClient

        os.environ.pop("ML_QWEN_POOL_CONFIG", None)
        self.lock = threading.Lock()
        self.config = config
        self.identities = config["identities"]
        self.name = config["name"]
        self.failed = None
        self.completed = 0
        self.client = HuggingFaceQwenClient(config["base"], config.get("adapter"))
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(config["memory_gib"] * 2**30 / total, 0)
        self.client._ensure_loaded()
        self.placement = dict(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                              embedding_device=str(self.client._model.get_input_embeddings().weight.device),
                              language_devices=sorted({str(p.device) for name,p in self.client._model.named_parameters() if 'visual' not in name}))
        if self.placement['embedding_device'] != 'cuda:0' or self.placement['language_devices'] != ['cuda:0']:
            raise SharedInferenceError('Language model must reside entirely on its assigned GPU: '+str(self.placement))
        if config.get('prompt_cache', {}).get('enabled'):
            from prompt_kv_cache import PromptKVCache
            self.client.prompt_kv_cache = PromptKVCache(
                max_bytes=int(config['prompt_cache']['max_bytes']),
                min_tokens=int(config['prompt_cache'].get('min_tokens', 128)))
        verified = model_identity(config["base"], config.get("adapter"))
        if verified not in self.identities:
            raise SharedInferenceError("Loaded model differs from the pinned service identity")
        self.adapter_tokenizer = self.client._tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(config["base"], trust_remote_code=True)
        if self.base_tokenizer.pad_token_id is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        for identity in self.identities:
            if identity != model_identity(config["base"], identity["adapter"]):
                raise SharedInferenceError("Service model/tokenizer fingerprint changed")
            if identity["adapter"] not in (None, config.get("adapter")):
                raise SharedInferenceError("This replica supports only its verified adapter and untouched base")

    def health(self):
        verification = self.client.adapter_verification
        return {"ready": self.failed is None, "name": self.name, "identities": self.identities,
                "adapter_verifications": [verification] if verification else [],
                "completed": self.completed, "busy": self.lock.locked(), "error": self.failed,
                "placement": self.placement,
                "prompt_cache": self.config.get('prompt_cache', {'enabled': False})}

    def generate(self, body):
        import torch

        with self.lock:
            if self.failed:
                raise SharedInferenceError(self.failed)
            if body.get("identity") not in self.identities:
                raise SharedInferenceError("Model identity mismatch")
            expected = digest({k: v for k, v in body.items() if k != "request_id"})
            if expected != body.get("request_id"):
                raise SharedInferenceError("Request digest mismatch")
            settings = body["settings"]
            if (set(settings) != {"temperature", "max_new_tokens", "top_p", "top_k", "enable_thinking"}
                    or not 1 <= int(settings["max_new_tokens"]) <= self.config["max_new_tokens"]):
                raise SharedInferenceError("Generation settings exceed the validated service envelope")
            started = time.time()
            client = self.client
            for key, value in settings.items():
                setattr(client, key, value)
            use_adapter = body["identity"]["adapter"] is not None
            client.adapter_path = self.config.get("adapter") if use_adapter else None
            client._tokenizer = self.adapter_tokenizer if use_adapter else self.base_tokenizer
            client.max_input_tokens = self.config["max_input_tokens"]
            context = client._model.disable_adapter() if self.config.get("adapter") and not use_adapter else nullcontext()
            try:
                with torch.random.fork_rng(devices=[0]), context:
                    torch.manual_seed(int(body["rng"]["seed"]))
                    state = body["rng"].get("cuda_rng")
                    if state:
                        torch.cuda.set_rng_state(torch.tensor(list(base64.b64decode(state, validate=True)),
                                                             dtype=torch.uint8), 0)
                    torch.cuda.reset_peak_memory_stats(0)
                    text = client.complete_messages(body["messages"],
                        use_s1b_template=body["use_s1b_template"])
                    cuda_rng = encode_rng(torch.cuda.get_rng_state(0))
                    peak = torch.cuda.max_memory_reserved(0) / 2**30
            except torch.cuda.OutOfMemoryError as exc:
                self.failed = "GPU allocation exhausted; replica requires inspection/restart"
                raise SharedInferenceError(self.failed) from exc
            self.completed += 1
            result = {"request_id": body["request_id"], "identity": body["identity"], "text": text,
                      "cuda_rng": cuda_rng, "replica": self.name, "started_at": started,
                      "finished_at": time.time(), "peak_reserved_gib": peak,
                      "generation": client.last_generation}
            # Avoid retaining long-history KV-cache allocator blocks between requests.
            torch.cuda.empty_cache()
            return result


class Broker:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.health_lock = threading.Lock()
        self.health_cached = None
        self.health_cached_at = 0
        self.pending = {}
        # More coordinators than queue slots used to fail healthy requests with
        # queue.Full. Cover the entire 200-task campaign, including 88 workers.
        self.queue = queue.Queue(maxsize=256)
        self.task_routes = {}
        self.replica_queues = {}
        # Wait for a concurrent reader instead of failing after SQLite's default 5 s.
        self.db = sqlite3.connect(config["journal"], check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS requests "
                        "(id TEXT PRIMARY KEY, body TEXT NOT NULL, response TEXT, error TEXT, created REAL)")
        self.db.commit()
        self.replica_health = []
        for url in config["replicas"]:
            health = http_json(url + "/health", timeout=10)
            if not health.get("ready") or health.get("identities") != config["identities"]:
                raise SharedInferenceError("Replica readiness/identity verification failed")
            self.replica_health.append(health)
            self.replica_queues[url] = queue.Queue(maxsize=256)
            threading.Thread(target=self.worker, args=(url,), daemon=True).start()

    def health(self):
        # Coalesce the startup burst from CPU coordinators. Without this, each
        # caller fans out to every replica and healthy workers time out.
        with self.health_lock:
            if self.health_cached is not None and time.monotonic()-self.health_cached_at < 2:
                return self.health_cached
            result=self._health()
            self.health_cached=result
            self.health_cached_at=time.monotonic()
            return result

    def _health(self):
        current = []
        for url in self.config['replicas']:
            try:
                current.append(http_json(url + '/health', timeout=2))
            except Exception:
                current.append({'ready': False})
        return {"ready": any(row.get('ready') for row in current), "identities": self.config["identities"],
                "adapter_verifications": self.replica_health[0]["adapter_verifications"],
                "task_affinity": bool(self.config.get('task_affinity')),
                "replicas": len(self.config["replicas"]),
                "queued": (sum(q.qsize() for q in self.replica_queues.values())
                           if self.config.get('task_affinity') else self.queue.qsize()),
                'replica_health': current}

    def add_replica(self, url):
        if not isinstance(url, str) or not url.startswith('http://127.0.0.1:'):
            raise SharedInferenceError('Only a local verified replica can join')
        health = http_json(url + '/health', timeout=10)
        if not health.get('ready') or health.get('identities') != self.config['identities']:
            raise SharedInferenceError('Handoff replica readiness/identity verification failed')
        with self.lock:
            if url not in self.config['replicas']:
                self.config['replicas'].append(url)
                self.replica_health.append(health)
                self.replica_queues[url] = queue.Queue(maxsize=256)
                threading.Thread(target=self.worker, args=(url,), daemon=True).start()
        return {'added': url, 'replicas': len(self.config['replicas'])}

    def worker(self, url):
        while True:
            # An unavailable replica must not drain the shared queue into errors.
            # Failed in-flight requests remain explicit; never silently retry them.
            try:
                health = http_json(url + '/health', timeout=2)
                ready = (health.get('ready')
                         and health.get('identities') == self.config['identities'])
            except Exception:
                ready = False
            if not ready:
                threading.Event().wait(1)
                continue
            try:
                body, result = (self.replica_queues[url] if self.config.get('task_affinity')
                                else self.queue).get(timeout=1)
            except queue.Empty:
                continue
            try:
                response = http_json(url + "/generate", body, timeout=3600)
                response["queue_wait_seconds"] = response["started_at"] - result["enqueued_at"]
                with self.lock:
                    self.db.execute("UPDATE requests SET response=?, error=NULL WHERE id=?",
                                    (encoded(response).decode(), body["request_id"]))
                    self.db.commit()
                result["response"] = response
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                with self.lock:
                    try:
                        self.db.rollback()
                        self.db.execute("UPDATE requests SET error=? WHERE id=?",
                                        (result["error"], body["request_id"]))
                        self.db.commit()
                    except sqlite3.Error as persistence_error:
                        self.db.rollback()
                        result['error'] += f'; journal write failed: {persistence_error}'
            finally:
                result["event"].set()
                (self.replica_queues[url] if self.config.get('task_affinity')
                 else self.queue).task_done()

    def generate(self, body):
        request_id = body.get("request_id")
        if digest({k: v for k, v in body.items() if k != "request_id"}) != request_id:
            raise SharedInferenceError("Request digest mismatch")
        if body.get("identity") not in self.config["identities"]:
            raise SharedInferenceError("Unregistered model identity")
        data = encoded(body).decode()
        with self.lock:
            saved = self.db.execute("SELECT body,response,error FROM requests WHERE id=?", (request_id,)).fetchone()
            if saved:
                if saved[0] != data:
                    raise SharedInferenceError("Request ID reused with different content")
                if saved[1]:
                    return dict(json.loads(saved[1]), replayed=True)
                pending = self.pending.get(request_id, {})
                error = saved[2] or pending.get('error')
                if error:
                    if not storage_failure(error):
                        raise SharedInferenceError(error)
                    # A quota outage must not permanently poison this deterministic request.
                    self.db.execute('CREATE TABLE IF NOT EXISTS request_recoveries '
                                    '(id TEXT, error TEXT, recovered REAL)')
                    self.db.execute('INSERT INTO request_recoveries VALUES (?,?,?)',
                                    (request_id, error, time.time()))
                    self.db.execute('UPDATE requests SET error=NULL WHERE id=?', (request_id,))
                    self.db.commit()
                    self.pending.pop(request_id, None)
            else:
                self.db.execute("INSERT INTO requests VALUES (?,?,NULL,NULL,?)", (request_id, data, time.time()))
                self.db.commit()
            if request_id not in self.pending:
                result = {"event": threading.Event(), "enqueued_at": time.time()}
                if self.config.get('task_affinity'):
                    # Keep consecutive generations for a task on its cache-owning
                    # replica. First requests spread across the shortest queues.
                    task = body['task']
                    url = self.task_routes.get(task)
                    if url is None:
                        counts = {u: list(self.task_routes.values()).count(u)
                                  for u in self.replica_queues}
                        url = min(self.replica_queues,
                                  key=lambda u: (counts[u], self.replica_queues[u].qsize()))
                        self.task_routes[task] = url
                    self.replica_queues[url].put_nowait((body, result))
                else:
                    self.queue.put_nowait((body, result))
                self.pending[request_id] = result
            result = self.pending[request_id]
        result["event"].wait()
        if "error" in result:
            raise SharedInferenceError(result["error"])
        with self.lock:
            self.pending.pop(request_id, None)
        return result["response"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, payload):
        data = encoded(payload)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.reply(200, self.server.service.health()) if self.path == "/health" else self.reply(404, {"error": "unknown path"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.path not in ('/generate', '/add_replica') or not 0 < length <= 8 * 1024**2:
                raise ValueError("Invalid request path/size")
            body = json.loads(self.rfile.read(length))
            if self.path == '/add_replica':
                if not isinstance(self.server.service, Broker):
                    raise ValueError('Only the broker accepts replica registration')
                result = self.server.service.add_replica(body['url'])
            else:
                result = self.server.service.generate(body)
            self.reply(200, result)
        except Exception as exc:
            self.reply(503, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("replica", "broker"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.role == 'replica':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
        os.environ['HF_QWEN_DEVICE_MAP'] = 'cuda:0'
    os.sched_setaffinity(0, config["cpus"])
    lock = args.config.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Bind first so a conflicting port cannot waste a model load or acquire the wrong service.
    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer(("127.0.0.1", config["port"]), Handler)
    server.service = Replica(config) if args.role == "replica" else Broker(config)
    print(json.dumps({"role": args.role, "pid": os.getpid(), "port": config["port"],
                      "health": server.service.health()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
