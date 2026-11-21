#!/usr/bin/env python3
"""Add one bounded VAE replica and resume selected TPE workers on that replica."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0, str(TOOLS))
import launch_peptide_test200 as launch

BASE = launch.CAMPAIGN
PORT = 18874
VAE = Path('/root/data/cys/BOLT/optimization/peptides/uniref_vae/saved_models/'
           'dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118_pretrained_authors.pkl')
SERVER = Path('/root/data/logs/peptide_bo_launch_20261027_v5/peptide_vae_server.py')
LOCAL_INDICES = {0, 1, 2, 3, 4}
REMOTE_INDICES = {12, 13, 14, 15, 16, 17, 18, 19, 20, 23}
PAIRS = ((0, 5, 18875), (1, 8, 18876), (2, 11, 18877), (3, 23, 18878))


def replica_record(port, suffix=''):
    name = 'vae_gpu1' if port == PORT else 'vae_replica_%d' % port
    return BASE / 'workers' / (name + suffix + '.json')


def health(port):
    with urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=5) as response:
        return json.load(response)


def live(pid, token):
    try:
        command = Path('/proc/%d/cmdline' % pid).read_bytes()
        return token.encode() in command or (token == str(launch.WORKER)
            and str(launch.worker_script('tpe')).encode() in command)
    except OSError:
        return False


def start():
    record = BASE / 'workers' / 'vae_gpu1.json'
    if record.exists():
        saved = json.loads(record.read_text())
        if live(saved['pid'], str(SERVER)):
            print(json.dumps(dict(already_running=True, health=health(PORT))), flush=True)
            return
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.splitlines()}
    if free[1] < 8192:
        raise RuntimeError('GPU1 needs at least 8 GiB free before adding inference')
    # CPU39 previously held only idle queue supervisors; keep inference on its own core.
    moved = []
    for pid in (84691, 91173):
        if live(pid, 'peptide_') and os.sched_getaffinity(pid) == {39}:
            os.sched_setaffinity(pid, {40})
            moved.append(pid)
    command = [launch.PYTHON, str(SERVER), '--gpu', '1', '--cpu', '39',
               '--port', str(PORT), '--vae', str(VAE), '--gpu-memory-fraction', '0.05']
    log = BASE / 'workers' / 'vae_gpu1.log'
    with log.open('ab') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    launch.write(record, dict(pid=child.pid, gpu=1, cpu=39, port=PORT, command=command,
                             started=time.time(), moved_idle_supervisors=moved,
                             inference_only=True, gpu_memory_fraction=0.05))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError('New VAE service exited; inspect ' + str(log))
        try:
            value = health(PORT)
            if value['errors']:
                raise RuntimeError('New VAE service reported errors')
            print(json.dumps(dict(status='ready', pid=child.pid, health=value)), flush=True)
            return
        except OSError:
            time.sleep(2)
    raise RuntimeError('VAE startup did not finish within 180 seconds')


def validate(port=PORT):
    os.sched_setaffinity(0, {40})
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0, '/root/data/logs/peptide_bo_release_20261027_v5')
    sys.path.insert(0, '/root/data/cys/BOLT/optimization/peptides')
    import torch
    torch.set_num_threads(1)
    from mindware.domains.protein.peptide_rpc import dumps, loads

    weights = torch.load(VAE, map_location='cpu', weights_only=False)
    identity = 'vae_infrastructure_validation_%d' % port
    version = hashlib.sha256(dumps(weights)).hexdigest()

    def request(port, **value):
        value = dict(value, identity=identity)
        payload = dumps(value)
        req = urllib.request.Request('http://127.0.0.1:%d/rpc' % port, data=payload,
                                     headers={'X-Request-ID': hashlib.sha256(payload).hexdigest()})
        with urllib.request.urlopen(req, timeout=600) as response:
            result = loads(response.read())
        if result.get('error') or result.get('missing_weights'):
            raise RuntimeError(str(result))
        return result

    reference = json.loads((BASE / 'tpe' / 'recipe.json').read_text())['prepared_pool']
    pool = json.loads(Path(reference).read_text())
    # Probe only encode/decode. No oracle calls or campaign evaluations are made.
    if 'tasks' in pool:
        pool = pool['tasks']
    task = next(iter(pool.values()))
    sequences = task['sequences'][:2]
    ports = (18871, port)
    try:
        for port in ports:
            request(port, op='weights', version=version, weights=weights)
        values = [request(port, op='encode_mean', version=version, seed=123,
                          sequences=sequences)['z'] for port in ports]
        difference = float((values[0] - values[1]).abs().max())
        if difference > 1e-6 or not all(torch.isfinite(z).all() for z in values):
            raise RuntimeError('Frozen encoding changed across replicas: ' + str(difference))
        decoded = [request(port, op='decode', version=version, seed=456,
                           z=values[0])['sequences'] for port in ports]
        if decoded[0] != decoded[1] or len(decoded[1]) != len(sequences) or not all(decoded[1]):
            raise RuntimeError('Seeded decoding differed across replicas')
        result = dict(passed=True, encoding_max_difference=difference,
                      seeded_decode_identical=True, oracle_calls=0,
                      health=health(ports[1]), time=time.time())
        launch.write(replica_record(ports[1], '_validation'), result)
        print(json.dumps(result), flush=True)
    finally:
        for port in ports:
            request(port, op='release')


def tunnel(port=PORT):
    record = replica_record(port, '_tunnel')
    if record.exists():
        saved = json.loads(record.read_text())
        if live(saved['pid'], 'ssh'):
            print(json.dumps(dict(already_running=True, pid=saved['pid'])), flush=True)
            return
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
               '-o', 'ControlPath=none', '-o', 'ExitOnForwardFailure=yes',
               '-o', 'ServerAliveInterval=30', '-N', '-R',
               '127.0.0.1:%d:127.0.0.1:%d' % (port, port), 'LLMOptimizer4']
    log = record.with_suffix('.log')
    with log.open('ab') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)
    if child.poll() is not None:
        raise RuntimeError('Additional remote inference tunnel failed; inspect ' + str(log))
    probe = ('import json,urllib.request; '
             'print(json.dumps(json.load(urllib.request.urlopen('
             '"http://127.0.0.1:%d/health",timeout=5))))' % port)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'LLMOptimizer4', launch.PYTHON, '-'],
                            input=probe, text=True, capture_output=True, check=True, timeout=20)
    value = json.loads(result.stdout)
    if value['errors'] or value['started'] != health(port)['started']:
        raise RuntimeError('Remote tunnel did not reach the validated replica')
    launch.write(record, dict(pid=child.pid, port=port, command=command, started=time.time()))
    print(json.dumps(dict(pid=child.pid, remote_health=value)), flush=True)


def migrate(side):
    if launch.method_paused('tpe'):
        print(json.dumps(dict(side=side, skipped=True, reason='TPE is paused')), flush=True)
        return
    if not json.loads((BASE / 'workers' / 'vae_gpu1_validation.json').read_text())['passed']:
        raise RuntimeError('Replica validation must pass before changing worker routing')
    if health(PORT)['errors']:
        raise RuntimeError('Replica is not healthy')
    indices = LOCAL_INDICES if side == 'local' else REMOTE_INDICES
    path = BASE / 'workers' / (side + '.json')
    audit_path = BASE / 'workers' / ('vae_migration_' + side + '.json')
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    rows = json.loads(path.read_text())
    for row in rows:
        if row['method'] != 'tpe' or row['index'] not in indices:
            continue
        if row['server_port'] == PORT and live(row['pid'], str(launch.WORKER)):
            continue
        old = dict(row)
        if not live(old['pid'], str(launch.WORKER)):
            raise RuntimeError('Expected live worker is absent; will not silently replace it')
        os.kill(old['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 900
        while live(old['pid'], str(launch.WORKER)):
            if time.monotonic() >= deadline:
                raise RuntimeError('Worker did not finish its scored checkpoint; no forced kill')
            time.sleep(1)
        # Its manager exits only after the current child has committed and released its lock.
        command = list(old['command'])
        command[command.index('--server') + 1] = 'http://127.0.0.1:%d' % PORT
        if launch.method_paused('tpe'):
            return
        with Path(old['log']).open('ab') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, env=launch.ENV,
                                     start_new_session=True)
        row.update(pid=child.pid, server_port=PORT, command=command, started=time.time())
        audit.append(dict(previous=old, resumed=dict(row), checkpoint_preserved=True,
                          time=time.time()))
        launch.write(path, rows)
        launch.write(audit_path, audit)
        print(json.dumps(dict(side=side, cpu=row['cpu'], old_pid=old['pid'],
                              new_pid=row['pid'], server_port=PORT)), flush=True)


def resize_training():
    record = BASE / 'workers' / 'bolt_live_cpu_allocation.json'
    groups = {70984: {3, 4}, 70985: {6, 7}, 70986: {9, 10}, 70987: {21, 22}}
    retained = set().union(*groups.values())
    groups.update({70818: retained, 70821: retained})
    previous = []
    for pid, affinity in groups.items():
        if not live(pid, 'bolt_taskset_top10_full_sft/'):
            raise RuntimeError('Expected training process absent; refusing CPU reassignment')
        for thread in Path('/proc/%d/task' % pid).iterdir():
            tid = int(thread.name)
            try:
                old = sorted(os.sched_getaffinity(tid))
                os.sched_setaffinity(tid, affinity)
                previous.append(dict(pid=pid, tid=tid, previous=old, current=sorted(affinity)))
            except ProcessLookupError:
                pass
    if not record.exists():
        launch.write(record, dict(previous_threads=previous,
                                 training_cpus=sorted(retained), released_cpus=[5, 8, 11, 23],
                                 original_torch_thread_count=3, changed_torch_thread_count=False,
                                 training_recipe_unchanged=True, time=time.time()))


def start_pairs():
    resize_training()
    for gpu, cpu, port in PAIRS:
        record = replica_record(port)
        if record.exists() and live(json.loads(record.read_text())['pid'], str(SERVER)):
            continue
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
        free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.splitlines()}
        if free[gpu] < 8192:
            raise RuntimeError('Insufficient 8 GiB admission headroom on GPU%d' % gpu)
        command = [launch.PYTHON, str(SERVER), '--gpu', str(gpu), '--cpu', str(cpu),
                   '--port', str(port), '--vae', str(VAE), '--gpu-memory-fraction', '0.02']
        log = record.with_suffix('.log')
        with log.open('ab') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        launch.write(record, dict(pid=child.pid, gpu=gpu, cpu=cpu, port=port,
                                 command=command, started=time.time(), inference_only=True,
                                 gpu_memory_fraction=0.02))
        deadline = time.monotonic() + 180
        while True:
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('New VAE service failed to become ready: ' + str(log))
            try:
                value = health(port)
                if value['errors']:
                    raise RuntimeError('New VAE replica reported an error')
                print(json.dumps(dict(port=port, status='ready', health=value)), flush=True)
                break
            except OSError:
                time.sleep(2)


def prepare_pairs():
    plan_path = BASE / 'workers' / 'vae_pairs_plan.json'
    if plan_path.exists():
        for _, _, port in PAIRS:
            if health(port)['errors']:
                raise RuntimeError('Prepared replica is unhealthy')
        print(json.dumps(dict(already_prepared=True, plan=json.loads(plan_path.read_text()))), flush=True)
        return
    for _, _, port in PAIRS:
        validate(port)
        tunnel(port)
    all_rows = []
    for side in ('local', 'remote'):
        for row in json.loads((BASE / 'workers' / (side + '.json')).read_text()):
            all_rows.append(dict(row, side=side))
    plan = []
    for old_port, new_port in ((18871, 18875), (18874, 18876),
                               (18872, 18877), (18873, 18878)):
        rows = [r for r in all_rows if r['server_port'] == old_port]
        chosen = []
        for method in ('lolbo', 'tpe'):
            group = sorted([r for r in rows if r['method'] == method],
                           key=lambda r: (r['side'], r['cpu']))
            by_side = [[r for r in group if r['side'] == side] for side in ('local', 'remote')]
            ordered = []
            for i in range(max(map(len, by_side), default=0)):
                for side_rows in by_side:
                    if i < len(side_rows):
                        ordered.append(side_rows[i])
            chosen.extend(ordered[:len(group) // 2])
        for row in chosen:
            plan.append(dict(side=row['side'], method=row['method'], cpu=row['cpu'],
                             old_port=old_port, new_port=new_port))
    if not plan:
        raise RuntimeError('No worker assignments available for new replicas')
    routes_path = BASE / 'workers' / 'vae_routes.json'
    routes = json.loads(routes_path.read_text())
    for row in plan:
        routes['%s:%s:%d' % (row['side'], row['method'], row['cpu'])] = row['new_port']
    launch.write(plan_path, plan)
    launch.write(routes_path, routes)
    print(json.dumps(dict(planned_migrations=len(plan), plan=plan)), flush=True)


def migrate_pairs(side):
    plan = json.loads((BASE / 'workers' / 'vae_pairs_plan.json').read_text())
    path = BASE / 'workers' / (side + '.json')
    rows = json.loads(path.read_text())
    audit_path = BASE / 'workers' / ('vae_pairs_migration_' + side + '.json')
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    pending = []
    for change in plan:
        if change['side'] != side:
            continue
        if launch.method_paused(change['method']):
            continue
        row = next(r for r in rows if r['cpu'] == change['cpu'] and r['method'] == change['method'])
        if row['server_port'] == change['new_port'] and live(row['pid'], str(launch.WORKER)):
            continue
        if not live(row['pid'], str(launch.WORKER)):
            raise RuntimeError('Expected live worker is absent; will not silently replace it')
        if health(change['new_port'])['errors']:
            raise RuntimeError('Destination replica is unhealthy')
        pending.append((row, dict(row), change, time.monotonic() + 1200))
        os.kill(row['pid'], signal.SIGTERM)
    while pending:
        for entry in list(pending):
            row, old, change, deadline = entry
            if live(old['pid'], str(launch.WORKER)):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Worker did not reach a safe checkpoint; no forced kill')
                continue
            command = list(old['command'])
            command[command.index('--server') + 1] = 'http://127.0.0.1:%d' % change['new_port']
            if launch.method_paused(row['method']):
                pending.remove(entry)
                continue
            with Path(old['log']).open('ab') as stream:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, env=launch.ENV,
                                         start_new_session=True)
            row.update(pid=child.pid, server_port=change['new_port'], command=command,
                       started=time.time())
            audit.append(dict(previous=old, resumed=dict(row), checkpoint_preserved=True,
                              time=time.time()))
            launch.write(path, rows)
            launch.write(audit_path, audit)
            pending.remove(entry)
            print(json.dumps(dict(side=side, method=row['method'], cpu=row['cpu'],
                                  pid=row['pid'], port=change['new_port'])), flush=True)
        if pending:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'validate', 'tunnel',
                                         'migrate-local', 'migrate-remote', 'start-pairs',
                                         'prepare-pairs', 'pairs-local', 'pairs-remote'))
    args = parser.parse_args()
    if args.action == 'start':
        start()
    elif args.action == 'validate':
        validate()
    elif args.action == 'tunnel':
        tunnel()
    elif args.action == 'start-pairs':
        start_pairs()
    elif args.action == 'prepare-pairs':
        prepare_pairs()
    elif args.action.startswith('pairs-'):
        migrate_pairs(args.action.split('-', 1)[1])
    else:
        migrate(args.action.split('-', 1)[1])


if __name__ == '__main__':
    main()
