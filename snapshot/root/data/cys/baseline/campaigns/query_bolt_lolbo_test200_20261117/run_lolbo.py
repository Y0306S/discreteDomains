#!/usr/bin/env python3
"""Durable Query LOLBO continuation from the frozen BOLT30 observations."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import types

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
TASKSET_LOLBO = Path('/root/data/cys/baseline/campaigns/taskset_bolt_semanticopt_test200_20261031/code/mindware/components/optimizers')
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
HOST = 'LLMOptimizer4'
WORKER_INDICES = tuple(range(24, 29))

sys.path.insert(0, str(SOURCE / 'batch_code'))
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TASKSET_LOLBO))

from query_protocol import atomic, digest
from query_data import canonicalize, definition, plan_key
from single_method import Journal, best, observed, seed_for
from query_vae_client import QueryVAEBackend
import lolbo_turbo_port as lolbo


def recipe():
    return json.loads((ROOT / 'recipe.json').read_text())


def worker_specs():
    dbroot = SOURCE / 'databases' / HOST
    scratchroot = SOURCE / 'scratch' / HOST
    return [dict(index=i, host=HOST, db_port=19720 + i - 22,
                 pgdata=str(dbroot / f'w{i}'),
                 scratch=str(scratchroot / f'worker{i}'),
                 postgres_cpus=list(range(68 + 4 * (i - 22), 71 + 4 * (i - 22))),
                 optimizer_cpu=71 + 4 * (i - 22)) for i in WORKER_INDICES]


def identity(task):
    cfg = recipe()
    return dict(recipe_sha256=digest(ROOT / 'recipe.json'), task=task,
                pool_sha256=cfg['pool_sha256'], vae_sha256=cfg['vae_sha256'],
                method='query_vae_lolbo_dkl_ts', rounds=80, batch_size=10)


def validate_pins():
    for path, expected in recipe()['pinned_files'].items():
        if digest(path) != expected:
            raise ValueError('Pinned LOLBO input changed: ' + path)


def load_pool(task):
    cfg = recipe()
    path = Path(cfg['pool_path'])
    if digest(path) != cfg['pool_sha256']:
        raise ValueError('Frozen BOLT pool changed')
    data = json.loads(path.read_text())
    rows = data['tasks'][task]
    if len(rows) != 30 or len({plan_key(r['plan']) for r in rows}) != 30:
        raise ValueError('Expected 30 distinct measured BOLT starts')
    for row in rows:
        canonicalize(task, row['plan'], strict=True)
    if best(rows) is None:
        raise ValueError('BOLT pool has no uncensored incumbent')
    return rows


def valid_fallback(task, rows):
    d, _, _, _ = definition(task)
    from workload.workloads import WorkloadSpec, OracleCodec
    from oracle.oracle import plan_has_crossjoin
    spec = WorkloadSpec.from_definition(d, OracleCodec.Aliases)
    valid = [r for r in rows if not r['censored'] and
             not plan_has_crossjoin(spec, list(r['plan']))]
    if not valid:
        raise ValueError('BOLT pool has no uncensored non-crossjoin VAE fallback')
    return min(valid, key=lambda r: r['runtime'])['plan']


def encode_rows(vae, rows):
    import numpy as np
    z = np.asarray(vae.encode([r['plan'] for r in rows]), dtype=np.float32)
    if z.shape != (len(rows), 64) or not np.isfinite(z).all():
        raise ValueError('VAE returned invalid posterior means')
    return z.tolist()


def fit_lolbo(model, mll, x, y, epochs):
    """Match the frozen LOLBO updater while reducing modern batched MLL output."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    model = model.train()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.01}], lr=0.01)
    batch_size = min(len(y), 128)
    loader = DataLoader(TensorDataset(x, y.squeeze(-1)), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for inputs, scores in loader:
            optimizer.zero_grad()
            loss = -mll(model(inputs), scores)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model.eval()


def make_initial(task, rows, vae):
    for row, z in zip(rows, encode_rows(vae, rows)):
        row['z'] = z
    return dict(rows=rows, rounds=[], turbo=vars(lolbo.TurboState(
        dim=64, batch_size=10, best_value=max(-r['runtime'] for r in rows if not r['censored']))))


def next_batch(task, state, vae, round_number):
    import numpy as np
    import torch

    rows = state['rows']
    observed_rows = []
    observed_keys = set()
    seen = {plan_key(r['plan']) for r in rows}
    for row in rows:
        key = plan_key(row['plan'])
        if not row['censored'] and key not in observed_keys:
            observed_rows.append(row)
            observed_keys.add(key)
    if not observed_rows:
        raise ValueError('LOLBO needs an uncensored incumbent')
    x = torch.tensor([r['z'] for r in observed_rows], dtype=torch.float32)
    y = torch.tensor([-r['runtime'] for r in observed_rows], dtype=torch.float32).view(-1, 1)
    turbo = lolbo.TurboState(**state['turbo'])
    device = torch.device('cpu')
    model, mll = lolbo.make_surrogate(x, device)
    seed = seed_for(task, 'lolbo', round_number, recipe()['seed'])
    torch.manual_seed(seed % (2**63 - 1))
    model = fit_lolbo(model, mll, x, y, 20 if round_number == 1 else 2)
    accepted, rejected = [], []
    for retry in range(20):
        torch.manual_seed((seed + retry) % (2**63 - 1))
        with torch.no_grad():
            z = lolbo.generate_batch(turbo, model, x, y, batch_size=10,
                                     n_candidates=2000, device=device, unit_cube=False)
        if z.shape != (10, 64) or not torch.isfinite(z).all():
            raise ValueError('LOLBO returned an invalid latent batch')
        decoded = vae.decode(z.tolist(), seed=(seed + retry) % (2**31 - 1))
        for latent, item in zip(z.tolist(), decoded):
            try:
                plan = canonicalize(task, item['plan'], strict=True)
                key = plan_key(plan)
                if item.get('fallback') or key in seen:
                    raise ValueError('decoder fallback or duplicate')
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                rejected.append(dict(retry=retry, error=str(exc)))
                continue
            accepted.append(dict(plan=plan, proposed_z=latent))
            seen.add(key)
            if len(accepted) == 10:
                return dict(items=accepted, rejected=rejected,
                            fit=dict(train_rows=len(observed_rows), epochs=20 if round_number == 1 else 2,
                                     seed=seed))
    raise RuntimeError(f'Could only decode {len(accepted)}/10 unique feasible plans')


def oracle(task):
    from backends import oracle as make_oracle
    return make_oracle(task, recipe()['timeout_seconds'])


def run_task(task, max_rounds=None):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    validate_pins()
    pool = load_pool(task)
    fallback = valid_fallback(task, pool)
    vae = QueryVAEBackend(recipe()['vae_pool'], task, fallback, seed=recipe()['seed'])
    offset = int(os.environ.get('QUERY_LOLBO_VAE_OFFSET', '0')) % len(vae.pool.endpoints)
    vae.pool.endpoints = vae.pool.endpoints[offset:] + vae.pool.endpoints[:offset]
    extra_endpoints = [url for url in os.environ.get('QUERY_LOLBO_VAE_ENDPOINTS', '').split(',') if url]
    preferred_endpoints = list(dict.fromkeys(extra_endpoints))
    fallback_endpoints = [url for url in vae.pool.endpoints
                          if url not in preferred_endpoints]
    vae.pool.endpoints = preferred_endpoints + fallback_endpoints
    vae.pool.counter = 0
    def decode_round_robin(self, vectors, seed=None):
        if seed is None:
            seed = self.seed + self.decode_index
            self.decode_index += 1
        endpoints = self.pool.endpoints
        preferred = endpoints[:len(preferred_endpoints)]
        fallback = endpoints[len(preferred_endpoints):]
        output = []
        for vector_index, vector in enumerate(vectors):
            failures = []
            start = self.pool.counter % max(1, len(fallback))
            ordered = preferred + [fallback[(start + i) % len(fallback)]
                                   for i in range(len(fallback))]
            for endpoint in ordered:
                try:
                    with urllib.request.urlopen(endpoint + '/health', timeout=2) as response:
                        health = json.load(response)
                    if health.get('status') != 'ready':
                        failures.append(endpoint + ': not ready')
                        continue
                except Exception as exc:
                    failures.append(endpoint + ': ' + type(exc).__name__)
                    continue
                payload = json.dumps(dict(z=[vector], task=self.task, fallback=self.fallback,
                    op='decode', seed=(seed + vector_index) % (2**31 - 1)), allow_nan=False).encode()
                request = urllib.request.Request(endpoint + '/inference', data=payload,
                    headers={'Content-Type': 'application/json',
                             'Authorization': 'Bearer ' + self.pool.token})
                with urllib.request.urlopen(request, timeout=self.pool.timeout) as response:
                    result = json.load(response)
                if 'error' in result:
                    raise RuntimeError(result['error'])
                output.extend(result['rows'])
                if endpoint in fallback:
                    self.pool.counter = (fallback.index(endpoint) + 1) % len(fallback)
                break
            else:
                raise RuntimeError('No responsive Query VAE endpoint: ' + '; '.join(failures))
        return output
    vae.decode = types.MethodType(decode_round_robin, vae)
    folder = ROOT / 'runs' / task
    folder.mkdir(parents=True, exist_ok=True)
    journal = Journal(folder / 'runtime.sqlite', identity(task))
    try:
        state = journal.get('lolbo_state')
        if state is None:
            state = make_initial(task, pool, vae)
            journal.put('lolbo_state', state)
        elif len(state['rows']) < 30:
            raise ValueError('Persisted LOLBO state lost BOLT initial rows')
        evaluate = oracle(task)
        target = min(80, max_rounds) if max_rounds is not None else 80
        while len(state['rounds']) < target:
            if (ROOT / 'PAUSE').exists():
                return
            number = len(state['rounds']) + 1
            batch = journal.once(f'proposal/{number}',
                dict(task=task, round=number, batch_size=10),
                lambda: next_batch(task, state, vae, number))
            if len(batch['items']) != 10:
                raise ValueError('LOLBO batch did not contain ten proposals')
            results = []
            for slot, item in enumerate(batch['items']):
                result = journal.once(f'measurement/{number}/{slot}',
                    dict(plan=item['plan']), lambda p=item['plan']: evaluate(p), measurement=True)
                row = observed(item['plan'], result, len(state['rows']) + len(results), 'lolbo')
                results.append(row)
                atomic(folder / 'progress.json', dict(task=task, completed_rounds=number - 1,
                    active_round=number, round_offset=len(results), rows=len(state['rows']) + len(results),
                    attempts=(number - 1) * 10 + len(results), best_runtime=best(state['rows'] + results),
                    at=time.time()))
            encoded = encode_rows(vae, results)
            for row, z in zip(results, encoded):
                row['z'] = z
            turbo = lolbo.TurboState(**state['turbo'])
            scores = [-r['runtime'] for r in results if not r['censored']]
            if not scores:
                scores = [turbo.best_value]
            turbo = lolbo.update_state(turbo, torch.tensor(scores, dtype=torch.float32).view(-1, 1))
            if turbo.restart_triggered:
                turbo = lolbo.TurboState(dim=64, batch_size=10,
                    best_value=max(-r['runtime'] for r in state['rows'] + results if not r['censored']))
            state['rows'].extend(results)
            state['rounds'].append(dict(round=number, fit=batch['fit'], rejected=batch['rejected'],
                results=[dict(plan=r['plan'],runtime=r['runtime'],censored=r['censored']) for r in results],
                best_runtime=best(state['rows'])))
            state['turbo'] = vars(turbo)
            journal.put('lolbo_state', state)
            atomic(folder / 'progress.json', dict(task=task, completed_rounds=number,
                rows=len(state['rows']), attempts=number * 10, best_runtime=best(state['rows']), at=time.time()))
        if max_rounds is None:
            atomic(folder / 'complete.json', dict(identity=identity(task), attempts=800,
                rows=state['rows'], rounds=state['rounds'], best_runtime=best(state['rows']),
                status='complete', at=time.time()))
    finally:
        journal.close()


def worker(index, task=None, max_rounds=None):
    spec = next(w for w in worker_specs() if w['index'] == index)
    os.sched_setaffinity(0, {spec['optimizer_cpu']})
    env = dict(os.environ, DB_HOST='127.0.0.1', DB_PORT=str(spec['db_port']),
        DB_NAME='imdb', DB_USER='imdb', DB_PASSWORD='imdb', CUDA_VISIBLE_DEVICES='',
        QUERY_LOLBO_VAE_OFFSET=str(index),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key, value in env.items():
        os.environ[key] = value
    tasks = [task] if task else recipe()['tasks']
    while True:
        if (ROOT / 'PAUSE').exists():
            return
        claimed = False
        for current in tasks:
            folder = ROOT / 'runs' / current
            if (folder / 'complete.json').exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'task.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if (folder / 'complete.json').exists():
                    continue
                claimed = True
                atomic(ROOT / 'services' / f'worker{index}.status.json', dict(index=index,
                    task=current, status='running', pid=os.getpid(), host=HOST, at=time.time()))
                try:
                    run_task(current, max_rounds=max_rounds)
                except BaseException as exc:
                    atomic(folder / 'error.json', dict(error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
                    (ROOT / 'PAUSE').touch()
                    raise
            if max_rounds is not None:
                return
            break
        if not claimed:
            if all((ROOT / 'runs' / current / 'complete.json').exists() for current in tasks):
                break
            time.sleep(2)
    atomic(ROOT / 'services' / f'worker{index}.status.json', dict(index=index,
        status='idle', pid=os.getpid(), host=HOST, at=time.time()))


def remote_command(spec, task=None, max_rounds=None):
    env = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='1', PYTHONDONTWRITEBYTECODE='1',
        QUERY_LOLBO_VAE_OFFSET=str(spec['index']),
        QUERY_LOLBO_VAE_ENDPOINTS=os.environ.get('QUERY_LOLBO_VAE_ENDPOINTS', ''))
    command = ['env'] + [f'{k}={v}' for k, v in env.items()] + [PYTHON, '-u', str(ROOT / 'run_lolbo.py'), 'worker', '--index', str(spec['index'])]
    if task:
        command += ['--task', task]
    if max_rounds is not None:
        command += ['--max-rounds', str(max_rounds)]
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
        '-o', 'ControlPath=none', '-o', 'ServerAliveInterval=20',
        '-o', 'ServerAliveCountMax=3', HOST, shlex.join(command)]


def run_remote(spec, task=None, max_rounds=None):
    log = (ROOT / 'logs' / f'worker{spec["index"]}.log').open('ab')
    return subprocess.Popen(remote_command(spec, task, max_rounds), stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True), log


def preflight():
    cfg = recipe()
    validate_pins()
    if len(cfg['tasks']) != 200 or cfg['rounds'] != 80 or cfg['batch_size'] != 10:
        raise ValueError('Unexpected test size or budget')
    audit = json.loads((SOURCE / 'evidence/bolt30_reuse_audit.json').read_text())
    if not audit.get('passed') or audit['measurements'] != 6000 or audit['tasks'] != 200:
        raise ValueError('BOLT warm-start reuse audit failed')
    if digest(SOURCE / 'pools/batch10_frozen_pool.json') != cfg['pool_sha256']:
        raise ValueError('BOLT pool hash mismatch')
    for spec in worker_specs():
        ready = SOURCE / 'operations/capacity' / f'w{spec["index"]}.ready.json'
        if not ready.exists():
            raise ValueError(f'Missing prepared LOLBO database {spec["index"]}')
        value = json.loads(ready.read_text())
        if not value.get('ready') or value.get('port') != spec['db_port']:
            raise ValueError('Invalid LOLBO database readiness record')
    return dict(passed=True, tasks=len(cfg['tasks']), warm_rows_per_task=30,
        rounds=80, batch_size=10, total_new_measurements=160000,
        worker_indices=list(WORKER_INDICES), pool_sha256=cfg['pool_sha256'], at=time.time())


def controller():
    lock = (ROOT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = preflight()
    atomic(ROOT / 'evidence/preflight.json', report)
    first = recipe()['tasks'][0]
    proc, log = run_remote(worker_specs()[0], first, 1)
    while proc.poll() is None:
        time.sleep(5)
    log.close()
    canary = ROOT / 'runs' / first / 'progress.json'
    if proc.returncode != 0 or not canary.exists():
        raise RuntimeError('The real ten-evaluation LOLBO canary failed')
    progress = json.loads(canary.read_text())
    if progress.get('completed_rounds') != 1 or progress.get('attempts') != 10:
        raise RuntimeError('The LOLBO canary did not commit a complete batch of ten')
    atomic(ROOT / 'evidence/canary.json', dict(passed=True, task=first,
        completed_rounds=1, measurements=10, best_runtime=progress.get('best_runtime'), at=time.time()))
    jobs = [run_remote(spec) for spec in worker_specs()]
    try:
        while any(p.poll() is None for p, _ in jobs):
            status()
            time.sleep(10)
        if any(p.returncode for p, _ in jobs):
            raise RuntimeError('A LOLBO worker failed; inspect worker logs and task error.json')
    finally:
        for _, stream in jobs:
            stream.close()
    atomic(ROOT / 'status.json', dict(status='complete', at=time.time()))


def status():
    tasks = recipe()['tasks']
    rounds = rows = complete = 0
    active = []
    for task in tasks:
        folder = ROOT / 'runs' / task
        progress = folder / 'progress.json'
        if progress.exists():
            data = json.loads(progress.read_text())
            rounds += data.get('completed_rounds', 0)
            rows += max(0, data.get('rows', 30) - 30)
            active.append(data)
        complete += (folder / 'complete.json').exists()
    worker_records = [json.loads(p.read_text()) for p in sorted((ROOT/'services').glob('worker*.status.json'))]
    active = any(w.get('status') == 'running' for w in worker_records)
    if (ROOT / 'PAUSE').exists():
        current = 'paused'
    elif complete == len(tasks):
        current = 'complete'
    else:
        current = 'running' if active else 'prepared'
    result = dict(status=current,
        tasks=len(tasks), complete_tasks=complete, rounds_completed=rounds,
        measured_lolbo_proposals=rows, target_rounds_per_task=80,
        batch_size=10, target_proposals=160000,
        workers=worker_records,
        at=time.time())
    atomic(ROOT / 'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['controller', 'worker', 'status', 'preflight'])
    parser.add_argument('--index', type=int)
    parser.add_argument('--task')
    parser.add_argument('--max-rounds', type=int)
    args = parser.parse_args()
    if args.command == 'controller':
        controller()
    elif args.command == 'worker':
        worker(args.index, args.task, args.max_rounds)
    elif args.command == 'preflight':
        print(json.dumps(preflight(), indent=2))
    else:
        print(json.dumps(status(), indent=2))


if __name__ == '__main__':
    main()
