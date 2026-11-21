"""Shared TaskSet inference: one loaded model, serialized GPU calls, CPU clients.

Only model generation is shared. Each evaluation worker owns its oracle and GP
state. There are no automatic request retries or hidden extra model calls.
"""

from __future__ import annotations

import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler


class InferenceServiceError(RuntimeError):
    """Stop the task visibly rather than record an outage as an LLM result."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_identity(client):
    """Hash actual weights once at service startup, including full BOLT shards."""
    paths = {Path(client.base_path).resolve()}
    if client.adapter_path:
        paths.add(Path(client.adapter_path).resolve())
    files = {}
    for directory in sorted(paths):
        weights = sorted(directory.glob("*.safetensors"))
        if not weights:
            raise ValueError(f"No safetensors weights in {directory}")
        for path in weights + sorted(directory.glob("*.json")) + sorted(directory.glob("*.jinja")):
            files[str(path)] = sha256_file(path)
    return {"client": client.identity(), "files": files,
            "loader_sha256": sha256_file(Path(__file__).with_name("llm_client.py")),
            "template_sha256": sha256_file(Path(__file__).with_name("spec.py"))}


def make_server(address, client, identity, token, *, max_pending=16):
    """Create a server; caller explicitly starts it. Also supports offline tests."""
    if not token or max_pending < 1:
        raise ValueError("A nonempty bearer token and positive queue capacity are required")
    identity_sha = fingerprint(identity)
    generation_lock = threading.Lock()
    capacity = threading.BoundedSemaphore(max_pending)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Never put prompts or authorization headers in HTTP logs.

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self.reply(401, {"error": "Unauthorized"})
                return False
            return True

        def do_GET(self):
            if self.authorized():
                if self.path != "/health":
                    return self.reply(404, {"error": "Unknown endpoint"})
                self.reply(200, {"ready": True, "identity": identity, "identity_sha256": identity_sha})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/generate":
                return self.reply(404, {"error": "Unknown endpoint"})
            if not capacity.acquire(blocking=False):
                return self.reply(429, {"error": "Inference queue full; no generation performed"})
            try:
                self.connection.settimeout(30)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Request body must be at most 1 MiB")
                request = json.loads(self.rfile.read(size))
                if request.get("identity_sha256") != identity_sha:
                    return self.reply(409, {"error": "Model identity changed"})
                messages, n = request["messages"], request.get("n", 1)
                temperature, seed = request.get("temperature"), request.get("seed")
                if (not isinstance(messages, list) or not messages
                        or any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                               or not isinstance(m.get("content"), str) for m in messages)
                        or type(n) is not int or not 1 <= n <= 3
                        or (temperature is not None and (type(temperature) not in (int, float)
                                                        or not 0 <= temperature <= 2))
                        or type(seed) is not int or not 0 <= seed < 2**63):
                    raise ValueError("Invalid messages, sample count, temperature or seed")
                with generation_lock:
                    replies = client.generate(messages, n=n, temperature=temperature, seed=seed)
                self.reply(200, {"replies": replies, "identity_sha256": identity_sha})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.reply(503, {"error": f"Inference failed: {type(exc).__name__}"})
            finally:
                capacity.release()

    class Server(ThreadingHTTPServer):
        # CPU pools can reach a generation boundary together. Queue TCP
        # connections before handlers apply the generation admission limit.
        request_queue_size = max(128, max_pending)

    server = Server(address, Handler)
    server.daemon_threads = True
    return server


class SharedQwenClient:
    def __init__(self, url, token_file, *, expected_identity=None, temperature=0.2,
                 seed=1, timeout=1800):
        self.url = url.rstrip("/")
        self._token = Path(token_file).read_text().strip()
        self.temperature, self.seed, self.timeout = float(temperature), int(seed), float(timeout)
        self._opener = build_opener(ProxyHandler({}))
        health = self._request("/health")
        self.model_identity, self.identity_sha256 = health["identity"], health["identity_sha256"]
        if (not health.get("ready") or fingerprint(self.model_identity) != self.identity_sha256
                or expected_identity is not None and expected_identity != self.identity_sha256):
            raise InferenceServiceError("Inference service identity/readiness mismatch")

    def _request(self, path, payload=None):
        request = Request(self.url + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise InferenceServiceError(f"Shared inference request failed ({type(exc).__name__})") from exc

    def identity(self):
        return {"client": "shared_qwen", "model_identity_sha256": self.identity_sha256,
                "temperature": self.temperature, "seed": self.seed}

    def generate(self, messages, *, n=1, temperature=None, seed=None):
        temperature = self.temperature if temperature is None else float(temperature)
        payload = {"messages": list(messages), "n": n, "temperature": temperature}
        # Same prompt/task seed -> same sampling stream regardless of arrival order.
        payload["seed"] = int(fingerprint({"seed": self.seed, **payload})[:15], 16) if seed is None else int(seed)
        result = self._request("/generate", {"identity_sha256": self.identity_sha256, **payload})
        replies = result.get("replies")
        if (result.get("identity_sha256") != self.identity_sha256 or not isinstance(replies, list)
                or len(replies) != n or any(not isinstance(x, str) for x in replies)):
            raise InferenceServiceError("Invalid inference response or model identity changed")
        return replies

    def complete(self, system_prompt, user_prompt):
        return self.generate([{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}])[0]
