#!/usr/bin/env python3
"""Bounded, loopback-only GPU service with task state supplied per request."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pickle
import threading
import time
import traceback


class GPUService(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, args):
        from peptide_fsbo_cuda import CUDAFSBOBackend
        from peptide_transfer_policy import effective_recipe

        self.recipe_hash = hashlib.sha256(args.recipe.read_bytes()).hexdigest()
        self.backend = CUDAFSBOBackend(effective_recipe(args.recipe), args.memory_fraction)
        self.lock = threading.Lock()
        self.token = args.token_file.read_text().strip()
        self.gpu, self.cpu = os.environ['CUDA_VISIBLE_DEVICES'], args.cpu
        self.calls, self.errors, self.waiting, self.busy = 0, 0, 0, False
        self.compute_seconds = self.queue_seconds = 0.0
        self.execution_hashes = {str(Path(__file__).with_name(name)): hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('peptide_fsbo_cuda.py', 'peptide_fsbo_gpu_service.py')}
        super().__init__(('127.0.0.1', args.port), Handler)

    def health(self):
        return dict(pid=os.getpid(), gpu=self.gpu, cpu=self.cpu, calls=self.calls,
            errors=self.errors, waiting=self.waiting, busy=self.busy,
            compute_seconds=self.compute_seconds, queue_seconds=self.queue_seconds,
            recipe_sha256=self.recipe_hash, code_sha256=self.execution_hashes,
            peak_gpu_gib=self.backend.torch.cuda.max_memory_allocated() / 2**30)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, payload, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != '/health':
            self.respond(b'not found', 'text/plain', 404)
            return
        self.respond(json.dumps(self.server.health()).encode(), 'application/json')

    def do_POST(self):
        service = self.server
        if self.path != '/rpc' or self.headers.get('Authorization') != 'Bearer ' + service.token:
            self.respond(b'forbidden', 'text/plain', 403)
            return
        size = int(self.headers.get('Content-Length', '0'))
        if not 0 < size <= 64 * 2**20:
            self.respond(b'invalid size', 'text/plain', 413)
            return
        request = pickle.loads(self.rfile.read(size))
        queued = time.monotonic()
        service.waiting += 1
        with service.lock:
            service.waiting -= 1
            service.busy = True
            started = time.monotonic()
            wait_seconds = started - queued
            try:
                if request['recipe_sha256'] != service.recipe_hash:
                    raise ValueError('GPU service recipe identity mismatch')
                if request['op'] == 'fit':
                    result = service.backend.fit(request['x'], request['score'], request['seed'])
                    snapshot = service.backend.snapshot()
                elif request['op'] == 'sample':
                    service.backend.restore(request['snapshot'])
                    result = service.backend.sample(request['candidates'], request['count'], request['seed'])
                    snapshot = None
                else:
                    raise ValueError('Unsupported GPU operation')
                elapsed = time.monotonic() - started
                if isinstance(result, dict):
                    info = result if request['op'] == 'fit' else result['info']
                    info.update(service_gpu=service.gpu, service_pid=os.getpid(),
                                queue_seconds=wait_seconds, service_seconds=elapsed)
                reply = dict(result=result, snapshot=snapshot)
                service.calls += 1
                service.compute_seconds += elapsed
                service.queue_seconds += wait_seconds
            except Exception:
                service.errors += 1
                reply = dict(error=traceback.format_exc())
                service.backend.torch.cuda.empty_cache()
            finally:
                service.busy = False
        self.respond(pickle.dumps(reply, protocol=4), 'application/octet-stream')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--token-file', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--cpu', required=True, type=int)
    parser.add_argument('--memory-fraction', type=float, default=0.08)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    service = GPUService(args)
    print(json.dumps(dict(status='ready', **service.health())), flush=True)
    service.serve_forever()


if __name__ == '__main__':
    main()
