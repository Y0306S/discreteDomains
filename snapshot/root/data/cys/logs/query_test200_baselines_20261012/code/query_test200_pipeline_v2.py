"""Run the full fleet in strict LOLBO -> TPE -> Random Search order."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

import query_test200_fleet_v2 as fleet


ORDER = ['lolbo', 'tpe', 'random']


def finished(root, plan, method):
    for task in plan['tasks']:
        path = root / 'claims' / f'{method}.{task}.json'
        claim = fleet.read(path) if path.exists() else {}
        if (claim.get('status') != 'complete' or claim.get('method') != method
                or claim.get('task') != task):
            return False
    for worker in plan['workers']:
        path = root / 'workers' / f'{worker["name"]}.json'
        state = fleet.read(path) if path.exists() else {}
        if state.get('status') != 'complete' or state.get('method') != method:
            return False
        try:
            with fleet.lock(root / 'workers' / f'{worker["name"]}.lock'):
                pass
        except BlockingIOError:
            return False
    return True


def launch_host(root, host, names, smoke):
    command = [fleet.PYTHON, str(root / 'code/query_test200_fleet_v2.py'),
               '--campaign', str(root), 'launch', '--host', host, '--names', *names]
    if smoke:
        command.append('--smoke')
    command = ['env', 'PYTHONDONTWRITEBYTECODE=1', *command]
    if socket.gethostname() != fleet.HOSTS[host]:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                   host, shlex.join(command)]
    subprocess.run(command, check=True)


def snapshot(root, plan, method, phase, **extra):
    states = {}
    for worker in plan['workers']:
        path = root / 'workers' / f'{worker["name"]}.json'
        state = fleet.read(path) if path.exists() else {}
        states[worker['name']] = {key: state.get(key) for key in
            ('status', 'method', 'task', 'saved_proposals', 'updated_at', 'error')}
    counts = Counter()
    for path in (root / 'claims').glob(f'{method}.*.json'):
        counts[fleet.read(path)['status']] += 1
    fleet.write(root / 'pipeline.json', dict(updated_at=fleet.stamp(), pid=os.getpid(),
        host=socket.gethostname(), order=ORDER, active_method=method, phase=phase,
        claims=dict(counts), workers=states, **extra))


def ensure_smoke(root, plan, method):
    launched = set()
    while True:
        ready, verified = {}, set()
        for worker in plan['workers']:
            name = worker['name']
            try:
                fleet.validate_smoke(root, plan, worker)
                verified.add(name)
                continue
            except RuntimeError:
                pass
            path = root / 'workers' / f'{name}.json'
            state = fleet.read(path) if path.exists() else {}
            if name in launched:
                if state.get('status') == 'failed':
                    raise RuntimeError(f'Smoke failed on {name}: {state.get("error")}')
                continue
            try:
                with fleet.lock(root / 'workers' / f'{name}.lock'):
                    pass
            except BlockingIOError:
                continue
            ready.setdefault(worker['host'], []).append(name)
        snapshot(root, plan, method, 'smoke', verified_workers=len(verified))
        if len(verified) == len(plan['workers']):
            return
        for host, names in ready.items():
            launch_host(root, host, names, smoke=True)
            launched.update(names)
        time.sleep(15)


def run(root):
    if socket.gethostname() != fleet.HOSTS['LLMOptimizer2']:
        raise RuntimeError('The pipeline controller belongs on LLMOptimizer2')
    os.sched_setaffinity(0, {22})
    plan = fleet.read(root / 'plan.json')
    if plan['methods'] != ORDER:
        raise RuntimeError('Unexpected method order')
    with fleet.lock(root / 'pipeline.lock'):
        method = None
        try:
            for method in ORDER:
                done = root / f'stage.{method}.complete.json'
                if done.exists():
                    if fleet.read(done).get('tasks') != len(plan['tasks']):
                        raise RuntimeError('Invalid saved stage completion')
                    continue
                # The preceding stage must drain all worker locks before changing methods.
                if method != ORDER[0] and not (root / f'stage.{ORDER[ORDER.index(method)-1]}.complete.json').exists():
                    raise RuntimeError('A preceding stage has not completed')
                plan['active_method'] = method
                plan['workers'] = [dict(worker, method=method) for worker in plan['workers']]
                runtime = plan['method_runtime'][method]
                plan.update({key: runtime[key] for key in ('runner', 'batch_size', 'files_sha256')})
                fleet.write(root / 'plan.json', plan)
                states = [fleet.read(root / 'workers' / f'{w["name"]}.json') for w in plan['workers']]
                already_started = all(s.get('method') == method and s.get('mode') == 'production'
                                      for s in states)
                if not already_started:
                    ensure_smoke(root, plan, method)
                    snapshot(root, plan, method, 'launching')
                    for host in plan['hosts']:
                        names = [w['name'] for w in plan['workers'] if w['host'] == host]
                        launch_host(root, host, names, smoke=False)
                while not finished(root, plan, method):
                    snapshot(root, plan, method, 'production')
                    for worker in plan['workers']:
                        state = fleet.read(root / 'workers' / f'{worker["name"]}.json')
                        if state.get('status') == 'failed':
                            raise RuntimeError(f'Worker failed: {worker["name"]}: {state.get("error")}')
                    time.sleep(15)
                summaries = [fleet.validate_complete(root / 'outputs' / task / method / 'seed0',
                             plan, task, plan['budget'], method) for task in plan['tasks']]
                fleet.write(done, dict(method=method, completed_at=fleet.stamp(), tasks=len(summaries),
                    proposals=sum(s['proposals'] for s in summaries),
                    database_calls=sum(s['database_calls'] for s in summaries)))
            snapshot(root, plan, ORDER[-1], 'complete')
        except BaseException as exc:
            snapshot(root, plan, method, 'failed', error=str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    run(parser.parse_args().campaign.resolve())
