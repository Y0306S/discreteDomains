#!/usr/bin/env python3
"""Frozen query VAE inference service; no database calls or acquisition logic."""
import argparse
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

AUTHOR_SHA256 = '83a4942701bffaaf8311887e530e3157b6843134e52e9f53d59f957c24aebab1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--memory-fraction', type=float, default=0.02)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 0.05:
        parser.error('Shared query VAE must use at most 5% of GPU allocator memory')
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    checkpoint = args.checkpoint
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise ValueError('Query fine-tuned VAE checksum mismatch')
    sys.path.insert(0, str(args.native))
    from vae.model import VAEModule, encode_batch
    vae = VAEModule.load_from_checkpoint(str(checkpoint), map_location='cuda:0')
    vae.eval().requires_grad_(False).cuda()
    original_checkpoint = args.native / 'vae/CEB_64_author.ckpt'
    assert hashlib.sha256(original_checkpoint.read_bytes()).hexdigest() == AUTHOR_SHA256
    retrieval_vae = VAEModule.load_from_checkpoint(str(original_checkpoint), map_location='cuda:0')
    retrieval_vae.eval().requires_grad_(False).cuda()
    assert retrieval_vae.vocab == vae.vocab
    token = args.token_file.read_text().strip()
    mutex = threading.Lock()
    tasks = OrderedDict()
    stats = dict(domain='ceb_query_runtime', gpu=args.gpu, pid=os.getpid(), cpu=args.cpu,
        checkpoint_sha256=args.checkpoint_sha256, representation='finetuned_train80_VAE_posterior_mean_64',
        retrieval_checkpoint_sha256=AUTHOR_SHA256, encoder_device='cuda:0', float32_matmul_precision='highest',
        requests=0, errors=0, oracle_calls=0, started=time.time(), operations={}, acquisition_wrapper=None)

    def context(task):
        if task in tasks:
            tasks.move_to_end(task)
            return tasks[task]
        match = re.fullmatch(r'CEB_(\d+[A-Z])(\d+)', task)
        if not match:
            raise ValueError('Invalid query task ID')
        from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
        from codec.codec import AliasesCodec
        from oracle.oracle import plan_has_crossjoin
        sql = args.native / 'workload/ceb-3k' / match[1].lower() / ((match[1] + match[2]).lower() + '.sql')
        definition = get_workload_set('CEB_3K').extract_query(sql.read_text(), task)
        spec = WorkloadSpec.from_definition(definition, OracleCodec.Aliases)
        codec = AliasesCodec(spec.all_tables)
        def canonical(plan):
            tree = codec.decode(spec.query_tables, list(plan))
            result = codec.encode(tree)
            if not tree.equal(codec.decode(spec.query_tables, result.copy())):
                raise ValueError('Query canonicalization changed the executed plan')
            return result
        value = (canonical, lambda plan: plan_has_crossjoin(spec, list(plan)))
        tasks[task] = value
        if len(tasks) > 256:
            tasks.popitem(last=False)
        return value

    def operation(value):
        op = value['op']
        if op in ('encode_mean', 'encode_retrieval_mean'):
            plans = value['plans']
            if not 1 <= len(plans) <= 512:
                raise ValueError('Encoding batch must contain 1..512 plans')
            if value.get('task'):
                canonical, _ = context(value['task'])
                plans = [canonical(p) for p in plans]
            tokens = encode_batch(plans, vae.vocab).cuda()
            encoder = retrieval_vae if op == 'encode_retrieval_mean' else vae
            mean, _ = encoder.model.encode(tokens)
            return dict(z=mean.reshape(len(plans), 64).cpu().tolist())
        if op == 'decode':
            vectors = np.asarray(value['z'], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != 64 or not 1 <= len(vectors) <= 128 or not np.isfinite(vectors).all():
                raise ValueError('Decoding requires 1..128 finite 64-dimensional vectors')
            z = torch.tensor(vectors, device='cuda:0')
            samples = vae.sample(z)
            if not value.get('task'):
                return dict(plans=samples)
            _, crossjoin = context(value['task'])
            fallback = value['fallback']
            if crossjoin(fallback):
                raise ValueError('Fallback plan must not contain a cross join')
            output = []
            for index, sample in enumerate(samples):
                retries, used_fallback = 0, False
                while crossjoin(sample):
                    sample = vae.sample(z[index])[0]
                    retries += 1
                    if retries > 50:
                        sample, used_fallback = fallback, True
                        break
                output.append(dict(plan=list(sample), decode_retries=retries, fallback=used_fallback))
            return dict(rows=output)
        raise ValueError('Unknown query VAE operation')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *_args):
            pass
        def reply(self, status, result):
            payload = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            if self.path != '/health':
                self.reply(404, dict(error='Unknown endpoint'))
                return
            with mutex:
                result = dict(stats, allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    peak_mib=torch.cuda.max_memory_allocated() / 2**20, status='ready')
            self.reply(200, result)
        def do_POST(self):
            if self.headers.get('Authorization') != 'Bearer ' + token:
                self.close_connection = True
                self.reply(401, dict(error='Unauthorized'))
                return
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16 * 2**20:
                self.close_connection = True
                self.reply(413, dict(error='Invalid payload size'))
                return
            try:
                value = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError):
                self.reply(400, dict(error='Invalid JSON'))
                return
            waiting = time.monotonic()
            with mutex:
                acquired = time.monotonic()
                try:
                    with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                        torch.manual_seed(int(value.get('seed', 123)))
                        result = operation(value)
                    status = 200
                except Exception as exc:
                    stats['errors'] += 1
                    result, status = dict(error=type(exc).__name__ + ': ' + str(exc)), 400
                duration = time.monotonic() - acquired
                stats['requests'] += 1
                entry = stats['operations'].setdefault(value.get('op', 'unknown'), dict(calls=0, queue_seconds=0.0, execution_seconds=0.0))
                entry['calls'] += 1
                entry['queue_seconds'] += acquired - waiting
                entry['execution_seconds'] += duration
                result['timing'] = dict(queue_seconds=acquired - waiting, execution_seconds=duration)
            self.reply(status, result)

    print(json.dumps(dict(stats, status='ready')), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
