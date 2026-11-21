"""Task-isolated VAE inference and APEX RPC over a loopback SSH tunnel."""

from __future__ import annotations

import hashlib
import io
import time
import urllib.error
import urllib.request

import numpy as np
import torch


def dumps(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def loads(data):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


class RemoteObjective:
    def __init__(self, objective, url, identity):
        self.objective = objective
        self.url = url.rstrip("/")
        self.identity = identity
        self.version = None
        self.native_forward = objective.vae_forward
        objective.vae_forward = self.forward
        objective.vae_decode = self.decode
        objective.query_oracle = self.oracle

    def request(self, value):
        value = dict(value, identity=self.identity)
        data = dumps(value)
        request_id = hashlib.sha256(data).hexdigest()
        error = None
        for attempt in range(3):
            request = urllib.request.Request(self.url + "/rpc", data=data,
                headers={"Content-Type": "application/octet-stream", "X-Request-ID": request_id})
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = loads(response.read())
                if result.get("error"):
                    raise RuntimeError("GPU service: " + result["error"])
                return result
            except (OSError, urllib.error.URLError) as exc:
                error = exc
                time.sleep(attempt + 1)
        raise RuntimeError("GPU service unavailable after retries") from error

    def dirty(self):
        self.version = None

    def sync(self):
        if self.version is None:
            weights = {k: v.detach().cpu().clone() for k, v in self.objective.vae.state_dict().items()}
            self.version = hashlib.sha256(dumps(weights)).hexdigest()
            self.request(dict(op="weights", version=self.version, weights=weights))

    def inference(self, op, **values):
        self.sync()
        seed = int(torch.randint(0, 2**62, (1,)).item())
        result = self.request(dict(op=op, version=self.version, seed=seed, **values))
        if result.get("missing_weights"):
            self.dirty()
            self.sync()
            result = self.request(dict(op=op, version=self.version, seed=seed, **values))
        return result

    def forward(self, sequences):
        # Gradient-bearing joint updates stay local and retain the released objective.
        if torch.is_grad_enabled():
            return self.native_forward(sequences)
        result = self.inference("encode", sequences=list(sequences))
        return result["z"], torch.tensor(result["loss"])

    def decode(self, z):
        return self.inference("decode", z=z.detach().cpu())["sequences"]

    def oracle(self, sequences):
        result = self.request(dict(op="oracle", sequences=list(sequences),
            nonce=int(torch.randint(0, 2**62, (1,)).item())))
        return np.asarray(result["scores"], dtype=float)

    def close(self):
        self.request(dict(op="release"))
