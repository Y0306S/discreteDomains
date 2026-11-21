#!/usr/bin/env python3
"""Return the four borrowed TaskSet CPU pairs to BOLT, retaining BO state."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import psutil

CAMPAIGN = Path('/root/data/logs/taskset_major_families_cpu_20260929')
NAMES = [f'LLMOptimizer2.w{i:02d}' for i in range(2, 6)]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    directory = CAMPAIGN / 'distributed'
    plan = read(directory / 'plan.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    all_states = [read(p) for p in (directory / 'workers').glob('*.json')]
    states = [next(s for s in all_states if s['name'] == name) for name in NAMES]
    order = {job['id']: i for i, job in enumerate(plan['jobs'])}
    report = {'at': time.time(), 'reason': 'Return borrowed cores 24-31 for authorized BOLT restart', 'workers': []}
    for state in states:
        process = psutil.Process(state['pid'])
        if state['boot_id'] != boot or abs(process.create_time() - state['started_at']) > 5:
            raise RuntimeError(f'Worker PID identity changed: {state["name"]}')
        if str(directory / 'runner.py') not in process.cmdline():
            raise RuntimeError(f'Unexpected worker process: {state["name"]}')
        job = state['job']
        earlier = [s['name'] for s in all_states if s['name'] not in NAMES and s.get('status') == 'running'
                   and s.get('job', {}).get('method') == job['method'] and s['job']['side'] == job['side']
                   and order[s['job']['id']] < order[job['id']]]
        if not earlier:
            raise RuntimeError(f'No remaining worker can reach released job: {job["id"]}')
        report['workers'].append({'state': state, 'remaining_workers_before_this_job': earlier})
    if not args.apply:
        print(json.dumps(report, indent=2))
        return
    backup = directory / 'plan.before_bolt_restart_20260930.json'
    if backup.exists():
        raise RuntimeError(f'Release already attempted: {backup}')
    write(backup, plan)
    for item in report['workers']:
        state = item['state']
        root = psutil.Process(state['pid'])
        root.send_signal(signal.SIGSTOP)
        children = {}
        while True:
            discovered = root.children(recursive=True)
            new = [p for p in discovered if p.pid not in children]
            if not new:
                break
            for child in new:
                try:
                    child.send_signal(signal.SIGSTOP)
                    children[child.pid] = child
                except psutil.NoSuchProcess:
                    pass
        children = list(children.values())
        base = Path(state['job']['output_root']) / state['job']['task_id'] / state['job']['method'] / 'seed1'
        item['before'] = {'observations': len(read(base / 'history.json')['observations']),
                          'pending': read(base / 'pending.json') if (base / 'pending.json').exists() else None,
                          'pending_elapsed_seconds': time.time() - (base / 'pending.json').stat().st_mtime
                          if (base / 'pending.json').exists() else None}
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        root.kill()
        _, alive = psutil.wait_procs([root, *children], timeout=5)
        if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
            raise RuntimeError(f'Processes survived release: {state["name"]}')
        after = read(base / 'history.json')
        if len(after['observations']) < item['before']['observations']:
            raise RuntimeError('Saved evaluation count decreased')
        item['after'] = {'observations': len(after['observations']),
                         'pending': read(base / 'pending.json') if (base / 'pending.json').exists() else None,
                         'files': {name: {'bytes': (base / name).stat().st_size,
                                          'sha256': hashlib.sha256((base / name).read_bytes()).hexdigest()}
                                   for name in ['history.json', 'optuna_study.pkl', 'pending.json'] if (base / name).exists()}}
        interrupted = {'at': time.time(), 'reason': report['reason'],
                       'worker': state['name'], 'pending_proposal_preserved': item['after']['pending'] is not None,
                       'pending_elapsed_seconds': item['before']['pending_elapsed_seconds'],
                       'saved_evaluations': item['after']['observations']}
        write(base / 'bolt_restart_interruption.json', interrupted)
        write(directory / 'workers' / (state['name'] + '.json'),
              {**state, 'status': 'released_to_bolt', 'updated_at': time.time(),
               'n_observations': item['after']['observations'], 'reason': report['reason']})
    machine = plan['hosts']['LLMOptimizer2']
    machine.setdefault('retired_workers', []).extend(
        {**w, 'reason': report['reason'], 'retired_at': time.time()} for w in machine['workers'] if w['name'] in NAMES)
    machine['workers'] = [w for w in machine['workers'] if w['name'] not in NAMES]
    machine['reservations'].append({'name': 'BOLT', 'cpus': list(range(24, 32))})
    machine['borrowed_reservations'] = []
    write(directory / 'plan.json', plan)
    locks = []
    for cpu in range(24, 32):
        lock = (CAMPAIGN / 'locks' / f'core.LLMOptimizer2.{cpu}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    report['verified_free_cpus'] = list(range(24, 32))
    write(directory / 'bolt_restart_release_20260930.json', report)
    print(json.dumps({'released': NAMES, 'verified_free_cpus': report['verified_free_cpus'],
                      'report': str(directory / 'bolt_restart_release_20260930.json')}))


if __name__ == '__main__':
    main()
