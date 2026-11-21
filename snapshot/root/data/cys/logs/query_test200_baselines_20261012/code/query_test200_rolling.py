"""Reuse each allocated QueryOpt slot in LOLBO, TPE, Random Search order."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

import query_test200_fleet_v3 as fleet

ORDER = ['lolbo', 'tpe', 'random']
SCRIPT = 'code/query_test200_rolling.py'


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, IndexError):
        return None


def verify(root):
    manifest = fleet.read(root / 'rolling_manifest.json')
    if fleet.digest(root / 'plan.json') != manifest['plan_sha256']:
        raise RuntimeError('Pinned campaign plan changed')
    for name, expected in manifest['files_sha256'].items():
        if fleet.digest(root / name) != expected:
            raise RuntimeError('Rolling dispatcher changed: ' + name)


def prepare(root):
    if (root / 'rolling_manifest.json').exists():
        verify(root)
        return
    plan = fleet.read(root / 'plan.json')
    if plan['methods'] != ORDER:
        raise RuntimeError('Unexpected method order')
    files = [SCRIPT, 'code/query_test200_fleet_v3.py']
    fleet.write(root / 'rolling_manifest.json', dict(created_at=fleet.stamp(),
        plan_sha256=fleet.digest(root / 'plan.json'), order=ORDER,
        files_sha256={name: fleet.digest(root / name) for name in files},
        scheduling='Each free slot advances when it has no available higher-priority task',
        active_legacy_workers='Wait for their worker lock; never interrupt them'))


def available(root, plan, worker):
    for task in plan['tasks']:
        path = root / 'claims' / f'{worker["method"]}.{task}.json'
        if not path.exists():
            return True
        claim = fleet.read(path)
        if claim['status'] != 'complete' and claim['worker'] == worker['name']:
            return True
    return False


def slot_free(root, name):
    try:
        with fleet.lock(root / 'workers' / f'{name}.lock'):
            return True
    except BlockingIOError:
        return False


def dispatch(root, name):
    verify(root)
    _, allocation = fleet.allocation(root, name)
    os.sched_setaffinity(0, {allocation['bo_cpu']})
    path = root / 'rolling' / f'{name}.json'
    state = dict(worker=name, host=allocation['host'], process=identity(os.getpid()),
                 started_at=fleet.stamp())

    def update(status, **extra):
        state.update(status=status, updated_at=fleet.stamp(), **extra)
        fleet.write(path, state)

    with fleet.lock(root / 'rolling' / f'{name}.lock'):
        try:
            while not slot_free(root, name):
                update('waiting_for_existing_worker')
                time.sleep(15)
            for method in ORDER:
                plan = fleet.method_plan(root, method)
                worker = next(w for w in plan['workers'] if w['name'] == name)
                if not available(root, plan, worker):
                    continue
                update('checking_database', method=method)
                fleet.db_start(root, name)
                try:
                    fleet.validate_smoke(root, plan, worker)
                except RuntimeError:
                    update('smoke', method=method)
                    fleet.run_worker(root, name, smoke=True, method=method)
                    fleet.validate_smoke(root, plan, worker)
                update('production', method=method)
                fleet.run_worker(root, name, method=method)
            update('complete')
        except BaseException as exc:
            update('failed', error=str(exc))
            raise


def start_worker(root, name):
    verify(root)
    _, worker = fleet.allocation(root, name)
    record_path = root / 'rolling' / f'{name}.process.json'
    if record_path.exists():
        old = fleet.read(record_path)
        if identity(old['pid']) == old:
            print(json.dumps(dict(worker=name, existing=old)), flush=True)
            return
    with fleet.lock(root / 'rolling' / f'{name}.launch.lock'):
        # Explicit affinity prevents inheriting the controller's single CPU.
        command = ['taskset', '-c', str(worker['bo_cpu']), fleet.PYTHON,
                   str(root / SCRIPT), '--campaign', str(root), 'worker', '--name', name]
        with (root / 'logs' / f'{name}.rolling.log').open('a') as stream:
            child = subprocess.Popen(command, env=fleet.environment(worker), cwd=root,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        record = identity(child.pid)
        if record is None:
            raise RuntimeError('Dispatcher failed at startup: ' + name)
        fleet.write(record_path, record)
        print(json.dumps(dict(worker=name, launched=record)), flush=True)


def snapshot(root, plan):
    counts = {}
    for method in ORDER:
        counts[method] = dict(Counter(fleet.read(path)['status']
            for path in (root / 'claims').glob(f'{method}.*.json')))
    workers = {}
    for worker in plan['workers']:
        name = worker['name']
        path = root / 'rolling' / f'{name}.json'
        state = fleet.read(path) if path.exists() else {'status': 'starting'}
        actual = root / 'workers' / f'{name}.json'
        if actual.exists():
            detail = fleet.read(actual)
            state['evaluation'] = {key: detail.get(key) for key in
                ('method', 'task', 'status', 'saved_proposals', 'updated_at', 'error')}
        workers[name] = state
    failed = [name for name, state in workers.items() if state['status'] == 'failed']
    complete = all(counts[m].get('complete', 0) == len(plan['tasks']) for m in ORDER)
    phase = 'complete' if complete else ('degraded' if failed else 'production')
    result = dict(updated_at=fleet.stamp(), pid=os.getpid(), host=socket.gethostname(),
                  order=ORDER, scheduling='per_worker', phase=phase, claims=counts,
                  workers=workers, failed_workers=failed)
    fleet.write(root / 'pipeline.json', result)
    return result


def run(root):
    verify(root)
    if socket.gethostname() != fleet.HOSTS['LLMOptimizer2']:
        raise RuntimeError('Controller belongs on LLMOptimizer2')
    os.sched_setaffinity(0, {22})
    plan = fleet.read(root / 'plan.json')
    with fleet.lock(root / 'pipeline.lock'):
        for worker in plan['workers']:
            command = [fleet.PYTHON, str(root / SCRIPT), '--campaign', str(root),
                       'start-worker', '--name', worker['name']]
            if socket.gethostname() != fleet.HOSTS[worker['host']]:
                command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           worker['host'], shlex.join(command)]
            subprocess.run(command, check=True, timeout=60)
        while True:
            report = snapshot(root, plan)
            for method in ORDER:
                done = root / f'stage.{method}.complete.json'
                if report['claims'][method].get('complete', 0) == len(plan['tasks']) and not done.exists():
                    runtime = fleet.method_plan(root, method)
                    summaries = [fleet.validate_complete(root / 'outputs' / task / method / 'seed0',
                        runtime, task, plan['budget'], method) for task in plan['tasks']]
                    fleet.write(done, dict(method=method, completed_at=fleet.stamp(),
                        tasks=len(summaries), proposals=sum(s['proposals'] for s in summaries),
                        database_calls=sum(s['database_calls'] for s in summaries)))
            if report['phase'] == 'complete':
                return
            for name, state in report['workers'].items():
                if state['status'] in ('failed', 'complete'):
                    continue
                worker = next(w for w in plan['workers'] if w['name'] == name)
                if worker['host'] == 'LLMOptimizer2' and state.get('process'):
                    if identity(state['process']['pid']) != state['process']:
                        state.update(status='failed', error='Dispatcher exited unexpectedly')
                        state.pop('evaluation', None)
                        fleet.write(root / 'rolling' / f'{name}.json', state)
            time.sleep(15)


def start(root):
    prepare(root)
    record_path = root / 'rolling_controller.json'
    if record_path.exists():
        old = fleet.read(record_path)
        if identity(old['pid']) == old:
            print(json.dumps(dict(existing=old)))
            return
    # Retire only the old global controller, leaving its detached evaluators alive.
    old_path = root / 'pipeline_launch.json'
    if old_path.exists():
        old = fleet.read(old_path)
        record = identity(old['pid'])
        if record:
            argv = Path(f'/proc/{old["pid"]}/cmdline').read_bytes().decode().split('\0')
            if str(root / 'code/query_test200_pipeline_v2.py') not in argv:
                raise RuntimeError('Unexpected old pipeline identity')
            fleet.write(root / 'rolling_previous_controller.json',
                        dict(launch=old, process=record, stopped_at=fleet.stamp()))
            os.kill(record['pid'], signal.SIGTERM)
            deadline = time.monotonic() + 15
            while identity(record['pid']) == record and time.monotonic() < deadline:
                time.sleep(.1)
            if identity(record['pid']) == record:
                raise RuntimeError('Old controller did not exit')
    with fleet.lock(root / 'pipeline.lock'):
        pass
    (root / 'rolling').mkdir(exist_ok=True)
    command = ['taskset', '-c', '22', fleet.PYTHON, str(root / SCRIPT),
               '--campaign', str(root), 'run']
    with (root / 'logs/rolling_controller.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=root,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    record = identity(child.pid)
    if record is None:
        raise RuntimeError('Controller failed at startup')
    fleet.write(record_path, record)
    print(json.dumps(dict(launched=record, order=ORDER, scheduling='per_worker')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('action', choices=['prepare', 'start', 'run', 'start-worker', 'worker'])
    parser.add_argument('--name')
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.action in ('worker', 'start-worker') and not args.name:
        parser.error('--name is required')
    if args.action == 'worker':
        dispatch(root, args.name)
    elif args.action == 'start-worker':
        start_worker(root, args.name)
    else:
        {'prepare': prepare, 'start': start, 'run': run}[args.action](root)
