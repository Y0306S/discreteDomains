#!/usr/bin/env python3
"""CPU query FSBO readiness worker: Bao target warmup and bounded capacity probe.

No acquisition function, proposal loop, or Thompson sampler is provided.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    allocation = next(w for w in plan['workers'] if w['name'] == args.worker)
    os.sched_setaffinity(0, {allocation['cpus'][0]})
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, plan['fsbo_repo'])
    from fsbo_modules import DeepKernelGP
    from fsbo_target_refit import fit_target
    from query_vae_client import QueryVAEBackend
    for path, expected in plan['pinned_files'].items():
        if digest(path) != expected:
            raise ValueError('Pinned query FSBO asset changed: ' + path)
    if torch.cuda.is_available():
        raise ValueError('The FSBO pilot must use CPU; existing GPU workloads are preserved')
    model = DeepKernelGP(64, os.devnull, plan['seed'], hidden_size=[128, 128],
                         kernel='rbf', ard=False, checkpoint=str(Path(plan['weights']).parent),
                         epochs=10000, lr=0.001)
    weights = torch.load(plan['weights'], map_location='cpu')
    model.model.load_state_dict(weights['gp'], strict=True)
    model.likelihood.load_state_dict(weights['likelihood'], strict=True)
    model.feature_extractor.load_state_dict(weights['net'], strict=True)
    initial = copy.deepcopy(model.state_dict())
    tasks = json.loads(Path(plan['tasks_file']).read_text())
    task = tasks[allocation['task']]
    token = Path(plan['token_file']).read_text().strip()
    root = args.plan.parent
    mutex = threading.Lock()
    stats = dict(status='ready', method='fsbo', domain='ceb_query_runtime',
                 worker=args.worker, pid=os.getpid(), optimizer_cpu=allocation['cpus'][0],
                 postgres_cpus=allocation['cpus'], task=allocation['task'],
                 checkpoint_sha256=plan['pinned_files'][plan['weights']],
                 acquisition_wrapper=None, optimization_campaign_started=False,
                 fits=0, queries=0, errors=0, started=time.time())
    low = np.asarray(plan['lower'])
    width = np.asarray(plan['upper']) - low
    width = np.where(width > 0, width, 1.)
    vae = QueryVAEBackend(plan['vae_pool'], allocation['task'], task['fallback'], seed=plan['seed'])

    def fit():
        started = time.monotonic()
        z = np.asarray(vae.encode([row['plan'] for row in task['warmup']]))
        if z.shape != (len(task['warmup']), 64) or not np.isfinite(z).all():
            raise ValueError('Invalid query Bao features')
        x = torch.tensor((z-low)/width, dtype=torch.float32)
        score = -np.asarray([row['runtime'] for row in task['warmup']])
        scale = float(np.ptp(score)) or 1.
        y = torch.tensor((score-score.min())/scale, dtype=torch.float32)
        model.load_state_dict(initial, strict=True)
        info = fit_target(model, x, y, 10000, patience=16, min_delta=0.0001,
                          lr=0.001, seed=plan['seed'])
        with torch.no_grad():
            pred = model.likelihood(model.model(model.feature_extractor(x)))
            if not torch.isfinite(pred.mean).all() or not torch.isfinite(pred.variance).all():
                raise ValueError('Invalid query FSBO posterior after Bao warmup')
        snapshot = root / 'workers' / args.worker / 'target_warmup.pt'
        temp = snapshot.with_suffix('.tmp')
        torch.save(dict(state_dict=model.state_dict(), x=x, y=y, task=allocation['task'],
                        checkpoint_sha256=stats['checkpoint_sha256'], fit=info), str(temp))
        os.replace(str(temp), str(snapshot))
        stats['fits'] += 1
        stats['warmup_observations'] = len(y)
        stats['last_fit'] = {k:v for k,v in info.items() if k != 'trace'}
        return dict(info, elapsed_with_encoding=time.monotonic()-started,
                    imported_bao_observations=49, usable_distinct_uncensored=len(y),
                    source='query_Bao_initialization', posterior_finite=True)

    def query():
        command = [plan['query_python'], str(Path(__file__).with_name('query_fsbo_pilot_db.py'))]
        result = subprocess.run(command, input=json.dumps(dict(port=allocation['db_port'], sql=task['sql'])),
                                text=True, capture_output=True, timeout=90,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if result.returncode:
            raise RuntimeError('Query pilot helper failed: ' + result.stderr[-3000:])
        value = json.loads(result.stdout)
        stats['queries'] += len(value['results'])
        return value

    class Handler(BaseHTTPRequestHandler):
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
            # Avoid waiting for a fit when checking overall health.
            self.reply(200, dict(stats, rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

        def do_POST(self):
            if self.headers.get('Authorization') != 'Bearer ' + token:
                self.reply(401, dict(error='Unauthorized'))
                return
            if self.path != '/pilot':
                self.reply(404, dict(error='Only the bounded pilot endpoint is enabled'))
                return
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                self.reply(400, dict(error='Invalid request size'))
                return
            value = json.loads(self.rfile.read(length))
            with mutex:
                try:
                    mode = value['mode']
                    result = dict(worker=args.worker, task=allocation['task'])
                    if mode not in ['fit', 'query', 'fit_query']:
                        raise ValueError('Unknown bounded pilot mode')
                    if mode in ['fit', 'fit_query']:
                        result['fit'] = fit()
                    if mode in ['query', 'fit_query']:
                        result['query'] = query()
                    result['status'] = 'passed'
                    self.reply(200, result)
                except Exception as exc:
                    stats['errors'] += 1
                    self.reply(500, dict(error=type(exc).__name__ + ': ' + str(exc)))

    print(json.dumps(stats), flush=True)
    ThreadingHTTPServer(('127.0.0.1', allocation['service_port']), Handler).serve_forever()


if __name__ == '__main__':
    main()
