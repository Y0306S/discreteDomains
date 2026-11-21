"""Start prepared ML campaigns at a dependency barrier or through a shared pool."""

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from ml_fixed_campaign import claim, digest, load, process_alive, save, task_key
from ml_run_archive import open_text


def predecessor_readiness(root, host):
    protocol = load(root / 'protocol.json')
    reasons = []
    tasks = load(root / 'tasks_132.json')
    if digest(root / 'tasks_132.json') != protocol['manifest_sha256']:
        return ['Predecessor task manifest changed']
    if len(tasks) != protocol['task_count'] or len({task_key(t) for t in tasks}) != len(tasks):
        return ['Predecessor task coverage is invalid']
    budget = load(root / 'runtime_policy.json')['attempts_per_search']
    complete = 0
    for method in protocol['methods']:
        for task in tasks:
            key = task_key(task)
            progress = load(root / 'sessions' / method / key / 'progress.json', {})
            if not (progress.get('attempts') == budget and progress.get('budget_exhausted')
                    and not progress.get('pending') and progress.get('identity')
                    and progress.get('successful', 0) + progress.get('failed', 0) == budget):
                continue
            trajectory = Path(progress['trajectory'])
            if not trajectory.resolve().is_relative_to(root.resolve()):
                return ['Predecessor trajectory points outside its campaign']
            config = load(trajectory.parent / 'config.json', {})
            if (config.get('optimizer') != method or config.get('seed') != protocol['seed']
                    or config.get('include_algorithms') != [task['algorithm']]
                    or config.get('metric') != task['metric']
                    or config.get('task_id') != f"{task['dataset']}_{task['algorithm']}_{method}"):
                return ['Predecessor result identity mismatch: ' + key]
            with open_text(trajectory, newline='') as stream:
                rows = list(csv.DictReader(stream))
            if (len(rows) != budget or [int(r['iter_idx']) for r in rows] != list(range(budget))
                    or any(r.get('oracle_called') != 'True' for r in rows)
                    or sum(r.get('evaluation_status') == 'SUCCESS' for r in rows) != progress['successful']
                    or any(r.get('evaluation_status') not in ('SUCCESS', 'FAILED', 'TIMEOUT') for r in rows)):
                return ['Predecessor actual-evaluation ledger is invalid: ' + key]
            complete += 1
    total = len(tasks) * len(protocol['methods'])
    if complete != total:
        reasons.append(f"Waiting for {'/'.join(protocol['methods'])} budgets: {complete}/{total} complete")
    live = [s for p in (root / 'state').rglob('*.json')
            if (s := load(p)).get('host') == host and process_alive(s)]
    if live:
        reasons.append(f'{len(live)} predecessor task processes still active')
    lock = claim(root / 'locks' / f'{host}.lock')
    if lock is None:
        reasons.append('Predecessor scheduler still owns its host lock')
    else:
        lock.close()
    return reasons


def verify_campaign(root):
    checks = load(root / 'checksums.json')
    prepared = load(root / 'prepared.json')
    if digest(root / 'checksums.json') != prepared['checksums_sha256']:
        raise RuntimeError('Queued campaign checksums changed')
    for name, expected in checks.items():
        if digest(root / name) != expected:
            raise RuntimeError('Queued campaign file changed: ' + name)
    protocol = load(root / 'protocol.json')
    if protocol.get('adapter') and digest(Path(protocol['adapter']) / 'adapter_model.safetensors') != protocol['adapter_sha256']:
        raise RuntimeError('Queued adapter changed')
    for name, expected in protocol.get('external_checksums', {}).items():
        if digest(name) != expected:
            raise RuntimeError('Queued external asset changed: ' + name)
    for record in load(root / 'data_manifest.json').values():
        if digest(root / 'code' / record['snapshot']) != record['sha256']:
            raise RuntimeError('Queued dataset snapshot changed: ' + record['snapshot'])


def gpu_devices(slots):
    # A slot may expose an ordered pair for a policy and its frozen reference.
    return {int(device) for slot in slots for device in str(slot).split(',')}


def resource_readiness(root, host, shared=False):
    spec = load(root / 'protocol.json')['hosts'][host]
    if socket.gethostname() != spec['hostname']:
        return ['Connected to the wrong machine']
    pool = set(spec['cpus'])
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if quota != 'max' and int(quota) / int(period) < spec['cpu_quota']:
        return ['CPU quota is below the prepared allocation']
    cpuset = set()
    for part in Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text().strip().split(','):
        ends = list(map(int, part.split('-')))
        cpuset.update(range(ends[0], ends[-1] + 1))
    if not (pool | set(spec['scheduler_cpus'])) <= cpuset:
        return ['CPU allocation changed']
    reasons = []
    maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum != 'max' and int(maximum) < spec['memory_gib'] * 2**30:
        reasons.append('RAM quota is below the prepared allocation')
    if shared:
        from ml_campaign_pool import readiness
        return reasons + readiness(root, host)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            stat = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
            affinity = os.sched_getaffinity(int(proc.name))
            if stat[0] not in ('Z', 'X') and len(affinity) <= len(pool) and affinity & pool:
                reasons.append('Allocated CPU cores are occupied by PID ' + proc.name)
        except (OSError, ValueError, IndexError):
            continue
    from ml_gpu_health import gpu_memory
    try:
        gpus = gpu_memory(cache_seconds=15)
    except (OSError, ValueError, RuntimeError) as exc:
        return reasons + [str(exc)]
    reservations = spec.get('external_gpu_reservations_gib', {})
    for gpu, memory in gpus.items():
        allowance = reservations.get(str(gpu), 8 if gpu == 0 else 1)
        if gpu in gpu_devices(spec['gpu_slots']) and memory['used'] > allowance:
            reasons.append(f"GPU{gpu} has an existing allocation of {memory['used']:.2f} GiB")
    return reasons


def launch(root, host, protocol, predecessor):
    path = root / 'logs' / f'scheduler_{host}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'launching', 'updated_at': time.time(),
              'after_campaign': str(predecessor), 'log': str(path)}
    # Persist intent before spawning: restarting this watcher must not retry a launch.
    save(root / 'launch.json', record)
    os.sched_setaffinity(0, set(protocol['hosts'][host]['cpus'])
                         | set(protocol['hosts'][host]['scheduler_cpus']))
    with path.open('a') as stream:
        child = subprocess.Popen([protocol['scheduler_python'],
            str(root / 'code/examples/benchmark/ml_fixed_campaign.py'), 'run',
            '--campaign', str(root), '--host', host], cwd=root / 'code',
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record['scheduler_pid'] = child.pid
    save(root / 'launch.json', record)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        heartbeat = load(root / 'hosts' / f'{host}.json', {})
        if heartbeat.get('pid') == child.pid and heartbeat.get('updated_at', 0) >= record['updated_at']:
            record.update(status='launched', updated_at=time.time())
            break
        if child.poll() is not None:
            record.update(status='scheduler_failed', returncode=child.returncode, updated_at=time.time())
            break
        time.sleep(1)
    else:
        record.update(status='scheduler_startup_unverified', updated_at=time.time())
    save(root / 'launch.json', record)
    save(root / 'queue_status.json', record)


def run(root, host):
    protocol = load(root / 'protocol.json')
    assert socket.gethostname() == protocol['hosts'][host]['hostname']
    lock = claim(root / 'queue.lock')
    if lock is None:
        raise RuntimeError('A queue watcher already owns this campaign')
    if (root / 'launch.json').exists():
        print(json.dumps(load(root / 'launch.json')), flush=True)
        lock.close()
        return
    os.sched_setaffinity(0, protocol['hosts'][host]['scheduler_cpus'])
    predecessor = Path(protocol['after_campaign'])
    shared = bool(protocol.get('resource_pool'))
    while True:
        try:
            reasons = resource_readiness(root, host, shared=True) if shared else predecessor_readiness(predecessor, host)
            if not reasons:
                save(root / 'queue_status.json', {'status': 'verifying', 'pid': os.getpid(),
                    'updated_at': time.time(), 'after_campaign': str(predecessor),
                    'scheduling': 'shared_capacity' if shared else 'campaign_barrier'})
                verify_campaign(root)
                reasons = resource_readiness(root, host, shared=shared)
            save(root / 'queue_status.json', {'status': 'waiting' if reasons else 'launching',
                'pid': os.getpid(), 'updated_at': time.time(), 'after_campaign': str(predecessor),
                'reasons': reasons, 'poll_seconds': 5 if shared else 60,
                'scheduling': 'shared_capacity' if shared else 'campaign_barrier'})
            if not reasons:
                break
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            save(root / 'queue_status.json', {'status': 'waiting_for_inspection', 'pid': os.getpid(),
                 'updated_at': time.time(), 'reasons': [str(exc)], 'after_campaign': str(predecessor)})
        time.sleep(5 if shared else 60)
    launch(root, host, protocol, predecessor)
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    run(args.campaign, args.host)
