#!/usr/bin/env python3
"""Opt-in FP64 matrix benchmark; never writes the production model bank.

The CPU reference is the production bounded-memory SMAC likelihood. This
experiment is deliberately separate from the validated production fitter.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sqlite3
import struct
import subprocess
import sys
import time

import numpy as np


def send(stream, value):
    payload = pickle.dumps(value, protocol=4)
    stream.write(struct.pack('!Q', len(payload)))
    stream.write(payload)
    stream.flush()


def receive(stream):
    header = stream.read(8)
    if not header:
        raise EOFError('CUDA worker closed its output')
    size = struct.unpack('!Q', header)[0]
    payload = bytearray()
    while len(payload) < size:
        chunk = stream.read(size-len(payload))
        if not chunk:
            raise EOFError('Truncated CUDA worker reply')
        payload.extend(chunk)
    return pickle.loads(payload)


def worker():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.08)
    data = receive(sys.stdin.buffer)
    x = torch.tensor(data['x'], dtype=torch.float64, device='cuda')
    y = torch.tensor(data['y'], dtype=torch.float64, device='cuda')
    identity = torch.eye(len(x), dtype=torch.float64, device='cuda')
    while True:
        theta = receive(sys.stdin.buffer)
        if theta is None:
            return
        started = time.monotonic()
        params = torch.tensor(theta, dtype=torch.float64, device='cuda').exp()
        amplitude, scales, noise = params[0], params[1:-1], params[-1]
        scaled = x/scales
        radial = torch.cdist(scaled, scaled, compute_mode='donot_use_mm_for_euclid_dist') * np.sqrt(5.)
        exponential = (-radial).exp()
        signal = (1.+radial+radial.square()/3.)*exponential
        covariance = amplitude*signal+noise*identity
        factor, info = torch.linalg.cholesky_ex(covariance)
        if info.item():
            send(sys.stdout.buffer, dict(value=-np.inf, gradient=np.zeros_like(theta)))
            continue
        alpha = torch.cholesky_solve(y[:, None], factor)[:, 0]
        value = -.5*y.dot(alpha)-factor.diag().log().sum()-len(x)/2.*np.log(2.*np.pi)
        sensitivity = alpha[:, None]*alpha[None, :]-torch.cholesky_solve(identity, factor)
        gradient = torch.empty_like(params)
        gradient[0] = .5*amplitude*(sensitivity*signal).sum()
        gradient[-1] = .5*noise*sensitivity.diag().sum()
        sensitivity *= (5./3.)*amplitude*(1.+radial)*exponential
        sensitivity.fill_diagonal_(0.)
        edges = sensitivity.sum(0)+sensitivity.sum(1)
        gradient[1:-1] = (.5*edges.matmul(x.square())-(x*sensitivity.matmul(x)).sum(0))/scales.square()
        reply = dict(value=value.item(), gradient=gradient.cpu().numpy(),
                     peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                     seconds=time.monotonic()-started)
        send(sys.stdout.buffer, reply)


def benchmark(args):
    import peptide_transfer as transfer
    from peptide_rgpe_exact import kernel_parts, marginal_likelihood
    sys.path.insert(0, str(transfer.RGPE_REPO))
    os.sched_setaffinity(0, {args.cpu})
    with sqlite3.connect('file:'+str(args.bank)+'?mode=ro', uri=True, timeout=120) as db:
        task, payload, metadata = db.execute('SELECT task,payload,metadata FROM models ORDER BY rowid DESC LIMIT 1').fetchone()
    model = pickle.loads(payload)
    del payload
    kernel_parts(model.gp.kernel_, model.gp.X_train_.shape[1])
    data = dict(x=model.gp.X_train_, y=model.gp.y_train_)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.worker.log').open('ab') as log:
        child = subprocess.Popen([args.cuda_python, str(Path(__file__).resolve()), '--worker'],
                                 env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        try:
            send(child.stdin, data)
            cases = []
            # Use the real selected parameters plus deterministic perturbations.
            rng = np.random.RandomState(20261031)
            selected = model.gp.kernel_.theta
            for theta in [selected, selected+rng.uniform(-.05, .05, len(selected)),
                          selected+rng.uniform(-.2, .2, len(selected))]:
                started = time.monotonic()
                cpu_value, cpu_gradient = marginal_likelihood(model.gp, theta)
                cpu_seconds = time.monotonic()-started
                started = time.monotonic()
                send(child.stdin, theta)
                gpu = receive(child.stdout)
                gpu_wall = time.monotonic()-started
                value_ok = bool(np.allclose(cpu_value, gpu['value'], rtol=1e-8, atol=1e-7))
                gradient_ok = bool(np.allclose(cpu_gradient, gpu['gradient'], rtol=1e-6, atol=1e-7))
                cases.append(dict(cpu_seconds=cpu_seconds, gpu_wall_seconds=gpu_wall,
                    gpu_compute_seconds=gpu.get('seconds'), likelihood_abs_error=float(abs(cpu_value-gpu['value'])),
                    gradient_max_abs_error=float(np.max(np.abs(cpu_gradient-gpu['gradient']))),
                    likelihood_passed=value_ok, gradient_passed=gradient_ok,
                    peak_gpu_gib=gpu.get('peak_gpu_gib')))
                print(json.dumps(cases[-1]), flush=True)
            send(child.stdin, None)
            child.wait(timeout=30)
            assert child.returncode == 0
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
    report = dict(task=task, observations=len(data['x']), dimensions=data['x'].shape[1],
                  dtype='float64', gpu=args.gpu, cases=cases,
                  matrix_checks_passed=all(c['likelihood_passed'] and c['gradient_passed'] for c in cases),
                  production_enabled=False,
                  remaining_gate='Full ten-restart optimizer, posterior and serialization equivalence',
                  source_metadata=json.loads(metadata), completed_at=time.time())
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--cuda-python', default='/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python')
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        assert args.bank and args.output
        benchmark(args)
