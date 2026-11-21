"""Local shared Qwen transport with replayable requests and task-owned RNG state."""

import base64
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler


class SharedInferenceError(RuntimeError):
    """Infrastructure failure; must never become a configuration or evaluation."""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_identity(model_path, adapter_path):
    base = Path(model_path).resolve()
    adapter = Path(adapter_path).resolve() if adapter_path else None
    tokenizer = adapter or base
    return {
        "base": str(base), "base_config_sha256": file_digest(base / "config.json"),
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": file_digest(adapter / "adapter_model.safetensors") if adapter else None,
        "tokenizer": {name: file_digest(tokenizer / name) for name in
                      ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                      if (tokenizer / name).exists()},
    }


def http_json(url, value=None, timeout=120):
    # Local services must not be routed through an environment HTTP proxy.
    request = Request(url, data=encoded(value) if value is not None else None,
                      headers={"Content-Type": "application/json"})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            error = json.loads(exc.read()).get("error", "request rejected")
        except (ValueError, AttributeError):
            error = "request rejected"
        raise SharedInferenceError(f"Shared inference HTTP {exc.code}: {error}") from exc


def encode_rng(state):
    return base64.b64encode(bytes(state.tolist())).decode("ascii")


class SharedQwenClient:
    def __init__(self, config_path, model_path, adapter_path):
        self.config = json.loads(Path(config_path).read_text())
        self.identity = model_identity(model_path, adapter_path)
        if self.identity not in self.config["identities"]:
            raise SharedInferenceError("Requested model identity is not registered in the inference pool")
        self.url = self.config["url"].rstrip("/")
        if not self.url.startswith("http://127.0.0.1:"):
            raise SharedInferenceError("Only a host-local inference pool is supported")
        self.state = {"version": 1, "identity": self.identity, "counter": 0,
                      "cuda_rng": None, "seed": None}
        self.adapter_verification = None
        self.last_response = None

    def ensure_loaded(self):
        for attempt in range(3):
            try:
                health = http_json(self.url + "/health", timeout=30)
                break
            except (URLError, OSError, ValueError) as exc:
                if attempt==2:
                    raise SharedInferenceError("Shared inference pool is unavailable") from exc
                time.sleep(1)
        if not health.get("ready") or self.identity not in health.get("identities", []):
            raise SharedInferenceError("Shared inference pool is not ready for this model")
        self.adapter_verification = next((item for item in health.get("adapter_verifications", [])
            if item.get("sha256") == self.identity["adapter_sha256"]), None)
        if self.identity["adapter"] and not self.adapter_verification:
            raise SharedInferenceError("Shared inference adapter tensor verification is missing")

    def import_state(self, state=None, legacy_cuda=None):
        if state is not None:
            if state.get("version") != 1 or state.get("identity") != self.identity:
                raise SharedInferenceError("Checkpoint inference identity differs from the requested model")
            self.state = dict(state)
        elif legacy_cuda:
            if len(legacy_cuda) != 1:
                raise SharedInferenceError("Expected one legacy Qwen CUDA generator")
            self.state["cuda_rng"] = encode_rng(legacy_cuda[0])

    def complete_messages(self, messages, *, use_s1b_template, settings):
        import torch

        if self.state["seed"] is None:
            self.state["seed"] = int(torch.initial_seed())
        task = os.environ.get("ML_SESSION_DIR") or os.environ.get("ML_QWEN_REQUEST_NAMESPACE")
        if not task:
            raise SharedInferenceError("A durable task/session namespace is required")
        body = {"identity": self.identity, "task": task, "messages": messages,
                "use_s1b_template": bool(use_s1b_template), "settings": settings,
                "rng": self.state, "run_id": os.environ.get("ML_RUN_ID")}
        body["request_id"] = digest(body)
        response = None
        for attempt in range(3):
            try:
                response = http_json(self.url + "/generate", body,
                                     timeout=self.config.get('request_timeout_seconds', 120))
                break
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise SharedInferenceError("Shared inference transport failed; evaluation not started") from exc
                time.sleep(1)
        if (not isinstance(response, dict) or response.get("request_id") != body["request_id"]
                or response.get("identity") != self.identity or not isinstance(response.get("text"), str)
                or not isinstance(response.get("cuda_rng"), str)):
            raise SharedInferenceError("Shared inference response identity/state mismatch")
        self.state.update(counter=self.state["counter"] + 1, cuda_rng=response["cuda_rng"])
        self.last_response = response
        return response["text"]
