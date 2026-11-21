#!/usr/bin/env python3
"""Print the serial test200 campaign plan; --execute is needed to launch it."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_query_ceb_baselines import load_split, TOTAL_BUDGET
from query_transfer_common import digest, validate_assets


def memory_headroom_gib():
    root = Path('/sys/fs/cgroup')
    limit = (root / 'memory.max').read_text().strip()
    if limit == 'max':
        raise RuntimeError('No finite cgroup memory limit found; review resource settings')
    current = int((root / 'memory.current').read_text())
    return (int(limit) - current) / 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--python', default='/root/data/cys/envs/bolt_clean_venv/bin/python')
    parser.add_argument('--cpus', default='44,45')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--min-memory-headroom-gib', type=float, default=32)
    parser.add_argument('--min-gpu-free-gib', type=float, default=8)
    parser.add_argument('--methods', nargs='+', choices=('random', 'tpe', 'lolbo'),
                        default=['random', 'tpe', 'lolbo'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    train, test = load_split()
    tasks = train if args.split == 'train' else test
    checkpoint = validate_assets(checkpoint=args.checkpoint)
    cpus = {int(value) for value in args.cpus.split(',')}
    if not cpus or not cpus.issubset(os.sched_getaffinity(0)):
        parser.error('Requested CPUs are not available')
    runner = Path(__file__).with_name('run_query_ceb_baselines.py')
    plan = {
        'tasks': len(tasks), 'split': args.split, 'methods': args.methods, 'seed': 0,
        'vae_checkpoint': str(checkpoint), 'vae_sha256': digest(checkpoint),
        'runs': len(tasks) * len(args.methods), 'budget_per_run': TOTAL_BUDGET,
        'charged_observations': len(tasks) * len(args.methods) * TOTAL_BUDGET,
        'proposals': len(tasks) * len(args.methods) * 1000,
        'initialization': '49 imported Bao rows per query and method',
        'batch_size': 10, 'update_e2e': False, 'concurrent_runs': 1,
        'cpus': sorted(cpus), 'lolbo_gpu': args.gpu,
        'random_tpe_device': 'cpu', 'memory_headroom_gib_now': memory_headroom_gib(),
        'required_memory_headroom_gib': args.min_memory_headroom_gib,
        'required_gpu_free_gib': args.min_gpu_free_gib,
        'output': str(args.output.resolve()), 'execute': args.execute,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    os.sched_setaffinity(0, cpus)
    os.nice(15)
    args.output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for method in args.methods:
            if memory_headroom_gib() < args.min_memory_headroom_gib:
                raise RuntimeError('Insufficient RAM headroom; campaign stopped before the next run')
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES=str(args.gpu) if method == 'lolbo' else '')
            if method == 'lolbo':
                free = subprocess.check_output([
                    'nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True)
                if float(free.strip()) / 1024 < args.min_gpu_free_gib:
                    raise RuntimeError('Insufficient GPU memory; campaign stopped before the next run')
            command = [args.python, str(runner), '--method', method, '--task-id', task,
                       '--split', args.split, '--checkpoint', str(checkpoint),
                       '--output', str(args.output.resolve()), '--device',
                       'cuda:0' if method == 'lolbo' else 'cpu']
            log = args.output / f'{task}_{method}.log'
            print(json.dumps({'task_id': task, 'method': method, 'log': str(log)}), flush=True)
            with log.open('a') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{task}/{method} failed; inspect {log} before resuming')


if __name__ == '__main__':
    main()
