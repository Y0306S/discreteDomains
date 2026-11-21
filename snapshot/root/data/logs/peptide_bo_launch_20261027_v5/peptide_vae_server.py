#!/usr/bin/env python3
"""Serialized GPU inference; each request specifies its task's VAE weights and RNG seed."""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/root/data/logs/peptide_bo_release_20261027_v5')
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--max-weight-banks", type=int, default=64)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.05)
    parser.add_argument('--skip-identical-weights', action='store_true')
    parser.add_argument('--http-keepalive', action='store_true')
    args = parser.parse_args()
    if args.max_weight_banks < 48:
        parser.error('The expanded pool requires at least 48 task weight banks')
    if not 0 < args.gpu_memory_fraction <= 0.05:
        parser.error('Shared inference must use at most five percent of GPU allocator capacity')
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    from mindware.domains.protein.peptide_rpc import dumps, loads
    upstream = Path("/root/data/cys/BOLT/optimization/peptides")
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
    objective = ApexConstrainedDiverseObjective(task_id="apex", task_specific_args=["bacteria_0"],
        path_to_vae_statedict=args.vae, max_string_length=30)
    original = {k: v.detach().cpu().clone() for k, v in objective.vae.state_dict().items()}
    banks, replies = OrderedDict(), OrderedDict()
    fingerprints, active_fingerprint = {}, None
    mutex = threading.Lock()
    stats = dict(requests=0, oracle_calls=0, errors=0, started=time.time(), gpu=args.gpu,
                 weight_loads=0, weight_loads_skipped=0, reply_cache_hits=0,
                 skip_identical_weights=args.skip_identical_weights,
                 http_keepalive=args.http_keepalive, operations={})

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1' if args.http_keepalive else 'HTTP/1.0'
        def log_message(self, *_args):
            pass

        def do_GET(self):
            payload = json.dumps(dict(stats, weights=len(banks),
                allocated_mib=torch.cuda.memory_allocated() / 2**20,
                peak_mib=torch.cuda.max_memory_allocated() / 2**20)).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            nonlocal active_fingerprint
            received = time.monotonic()
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 128 * 2**20:
                self.send_error(413)
                return
            value = loads(self.rfile.read(length))
            key = self.headers.get("X-Request-ID")
            waiting = time.monotonic()
            with mutex:
                acquired = time.monotonic()
                try:
                    if key in replies and value["op"] not in ("weights", "release"):
                        payload = replies[key]
                        stats['reply_cache_hits'] += 1
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
                            # A task has one active model version; superseded versions waste capacity.
                            for old_key in list(banks):
                                if old_key[0] == identity and old_key != bank_key:
                                    del banks[old_key]
                                    fingerprints.pop(old_key, None)
                            banks[bank_key] = weights
                            fingerprints[bank_key] = hashlib.sha256(dumps(weights)).hexdigest()
                            banks.move_to_end(bank_key)
                            while len(banks) > args.max_weight_banks:
                                evicted, _ = banks.popitem(last=False)
                                fingerprints.pop(evicted, None)
                        elif op == "release":
                            for k in list(banks):
                                if k[0] == identity:
                                    del banks[k]
                                    fingerprints.pop(k, None)
                        elif op == "oracle":
                            sequences = value["sequences"]
                            with torch.no_grad():
                                scores = objective.query_oracle(sequences) if sequences else []
                            result = dict(scores=list(scores))
                            stats["oracle_calls"] += len(sequences)
                        elif op in ("encode", "encode_mean", "decode"):
                            if bank_key not in banks:
                                result = dict(missing_weights=True)
                            else:
                                banks.move_to_end(bank_key)
                                fingerprint = fingerprints[bank_key]
                                if not args.skip_identical_weights or active_fingerprint != fingerprint:
                                    objective.vae.load_state_dict(banks[bank_key], strict=True)
                                    active_fingerprint = fingerprint
                                    stats['weight_loads'] += 1
                                else:
                                    stats['weight_loads_skipped'] += 1
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
                                        elif op == "encode_mean":
                                            tokens = objective.dataobj.tokenize_sequence(value['sequences'])
                                            encoded = [objective.dataobj.encode(seq).unsqueeze(0) for seq in tokens]
                                            batch = objective.dataobj.collate_fn(encoded).cuda()
                                            mean, _ = objective.vae.encode(batch)
                                            result = dict(z=mean.detach().cpu().reshape(len(tokens), -1))
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
                finished = time.monotonic()
                operation = stats['operations'].setdefault(value.get('op', 'unknown'),
                    dict(calls=0, queue_seconds=0.0, execution_seconds=0.0))
                operation['calls'] += 1
                operation['queue_seconds'] += acquired - waiting
                operation['execution_seconds'] += finished - acquired
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('X-Queue-Seconds', str(acquired - waiting))
            self.send_header('X-Execution-Seconds', str(finished - acquired))
            self.send_header('X-Request-Seconds', str(finished - received))
            self.end_headers()
            self.wfile.write(payload)

    print(json.dumps(dict(status="ready", **stats)), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
