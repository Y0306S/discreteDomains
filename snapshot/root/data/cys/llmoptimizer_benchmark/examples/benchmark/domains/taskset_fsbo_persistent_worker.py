#!/usr/bin/env python3
"""Persistent CPU FSBO min-max EI worker; reset target model for each request.

The target fitting and EI recipe matches the frozen TaskSet paper worker.
Only imports and the original checkpoint tensors are cached between requests.
"""
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from fsbo_modules import DeepKernelGP
from fsbo_utils import EI, totorch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Engine:
    def __init__(self, checkpoint, log):
        self.path = Path(checkpoint)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.digest = sha(self.path / "weights")
        if (self.meta["status"] != "complete" or self.meta["method"] != "fsbo_paper"
                or self.meta["dim"] != 43 or self.meta["config"]["kernel"] != "rbf"
                or self.meta["target_scaling"] != "minmax" or self.digest != self.meta["weights_sha256"]):
            raise ValueError("Expected the completed 43D TaskSet FSBO min-max RBF checkpoint")
        if torch.cuda.is_available():
            raise ValueError("This parity-preserving worker requires CUDA_VISIBLE_DEVICES='' (CPU)")
        self.weights = torch.load(str(self.path / "weights"), map_location="cpu")
        self.log = log

    def suggest(self, request):
        if (request["schema_fingerprint"] != self.meta["schema_fingerprint"]
                or request["target_scaling"] != "minmax" or request["finetune_epochs"] != 100):
            raise ValueError("FSBO schema, target scaling or fitting recipe mismatch")
        x, y, pending = (np.asarray(request[key], dtype=float) for key in ("X_obs", "y_obs", "X_pen"))
        if (x.shape != (len(y), 43) or len(y) < 2 or pending.ndim != 2 or pending.shape[1] != 43
                or not len(pending) or not all(np.isfinite(a).all() for a in (x, y, pending))):
            raise ValueError("Invalid FSBO observations or candidates")
        initial_weights = self.weights

        class StrictDeepKernelGP(DeepKernelGP):
            def load_checkpoint(self, _checkpoint):
                self.model.load_state_dict(initial_weights["gp"], strict=False)
                self.likelihood.load_state_dict(initial_weights["likelihood"], strict=False)
                self.feature_extractor.load_state_dict(initial_weights["net"], strict=False)

            def train(self):
                self.load_checkpoint(None)
                self.model.train()
                self.feature_extractor.train()
                self.likelihood.train()
                optimizer = torch.optim.Adam([
                    {"params": self.model.parameters()}, {"params": self.feature_extractor.parameters()}
                ], lr=self.lr)
                best, previous, patience, best_weights = float("inf"), float("inf"), 0, None
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    features = self.feature_extractor(self.X_obs)
                    self.model.set_train_data(features, self.y_obs, strict=False)
                    loss = -self.mll(self.model(features), self.y_obs)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite FSBO target adaptation loss")
                    loss.backward()
                    optimizer.step()
                    value = float(loss.detach())
                    if value < best:
                        best, best_weights = value, copy.deepcopy(self.state_dict())
                    patience = patience + 1 if np.allclose(value, previous, atol=self.loss_tol) else 0
                    previous = value
                    if patience > self.max_patience:
                        break
                if best_weights is None:
                    raise RuntimeError("No successful FSBO adaptation updates")
                self.load_state_dict(best_weights)
                self.epochs_used = epoch + 1

        model = StrictDeepKernelGP(input_size=43, log_dir=self.log,
                                  hidden_size=self.meta["config"]["hidden_size"], kernel="rbf", ard=False,
                                  seed=int(request["seed"]), checkpoint=str(self.path), load_model=True,
                                  epochs=100, verbose=False)
        lo, hi = float(y.min()), float(y.max())
        y = (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)
        model.X_obs = totorch(x, model.device)
        model.y_obs = totorch(y, model.device).reshape(-1)
        model.train()
        mean, std = model.predict(totorch(pending, model.device))
        scores = EI(max(y), mean, std)
        if not all(np.isfinite(a).all() for a in (mean, std, scores)):
            raise RuntimeError("Non-finite FSBO posterior or EI")
        return {"index": int(np.argmax(scores)), "checkpoint_sha256": self.digest,
                "acquisition": {"name": "EI", "values": scores.reshape(-1).tolist()},
                "posterior_mean": mean.reshape(-1).tolist(), "posterior_stddev": std.reshape(-1).tolist(),
                "incumbent": float(max(y)), "target_scaling": "minmax", "finetune_epochs": 100,
                "epochs_used": model.epochs_used, "observations": len(y), "device": "cpu",
                "worker_sha256": sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    engine = Engine(args.ckpt, args.log)
    for line in sys.stdin:
        request_id = None
        started = time.monotonic()
        try:
            message = json.loads(line)
            request_id = message["id"]
            with contextlib.redirect_stdout(sys.stderr):
                result = engine.suggest(message["request"])
            response = {"id": request_id, "ok": True, "result": result, "seconds": time.monotonic() - started}
        except Exception as exc:
            response = {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
