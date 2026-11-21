#!/usr/bin/env python3
"""Serialized GPU inference; each request specifies its task's VAE weights and RNG seed."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--vae", required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    torch.set_num_threads(1)
    from mindware.domains.protein.peptide_rpc import dumps, loads
    upstream = Path("/root/data/cys/BOLT/optimization/peptides")
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
    objective = ApexConstrainedDiverseObjective(task_id="apex", task_specific_args=["bacteria_0"],
        path_to_vae_statedict=args.vae, max_string_length=30)
    original = {k: v.detach().cpu().clone() for k, v in objective.vae.state_dict().items()}
    banks, replies = OrderedDict(), OrderedDict()
    mutex = threading.Lock()
    stats = dict(requests=0, oracle_calls=0, errors=0, started=time.time(), gpu=args.gpu)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            payload = json.dumps(dict(stats, weights=len(banks),
                allocated_mib=torch.cuda.memory_allocated() / 2**20,
                peak_mib=torch.cuda.max_memory_allocated() / 2**20)).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 128 * 2**20:
                self.send_error(413)
                return
            value = loads(self.rfile.read(length))
            key = self.headers.get("X-Request-ID")
            with mutex:
                try:
                    if key in replies and value["op"] not in ("weights", "release"):
                        payload = replies[key]
                    else:
                        stats["requests"] += 1
                        identity, op = value["identity"], value["op"]
                        bank_key = (identity, value.get("version"))
                        result = {}
                        if op == "weights":
                            weights = value["weights"]
                            if weights.keys() != original.keys() or any(
                                weights[k].shape != original[k].shape or not torch.isfinite(weights[k]).all()
                                for k in weights):
                                raise ValueError("Invalid task VAE weights")
                            banks[bank_key] = weights
                            banks.move_to_end(bank_key)
                            while len(banks) > 8:
                                banks.popitem(last=False)
                        elif op == "release":
                            for k in list(banks):
                                if k[0] == identity:
                                    del banks[k]
                        elif op == "oracle":
                            sequences = value["sequences"]
                            with torch.no_grad():
                                scores = objective.query_oracle(sequences) if sequences else []
                            result = dict(scores=list(scores))
                            stats["oracle_calls"] += len(sequences)
                        elif op in ("encode", "decode"):
                            if bank_key not in banks:
                                result = dict(missing_weights=True)
                            else:
                                banks.move_to_end(bank_key)
                                objective.vae.load_state_dict(banks[bank_key], strict=True)
                                objective.vae.eval()
                                # Serialize and restore both RNGs so callers cannot affect one another.
                                import random
                                python_rng = random.getstate()
                                try:
                                    with torch.random.fork_rng(devices=[0]), torch.no_grad():
                                        torch.manual_seed(value["seed"])
                                        random.seed(value["seed"])
                                        if op == "encode":
                                            z, loss = objective.vae_forward(value["sequences"])
                                            result = dict(z=z.detach().cpu(), loss=float(loss))
                                        else:
                                            result = dict(sequences=objective.vae_decode(value["z"]))
                                finally:
                                    random.setstate(python_rng)
                        else:
                            raise ValueError("Unknown RPC operation")
                        payload = dumps(result)
                        if not result.get("missing_weights") and op not in ("weights", "release"):
                            replies[key] = payload
                            while len(replies) > 1024:
                                replies.popitem(last=False)
                except Exception as exc:
                    stats["errors"] += 1
                    payload = dumps(dict(error=type(exc).__name__ + ": " + str(exc)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

    print(json.dumps(dict(status="ready", **stats)), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
