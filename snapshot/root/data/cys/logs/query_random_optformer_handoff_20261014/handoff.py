"""Lend released OptFormer CPU cores to the existing QueryOpt random queue."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
QUERY = Path('/root/data/cys/logs/query_test200_baselines_20261012')
RUNTIME = Path('/root/data/cys/logs/post_bora_runtime_20261012')
OPT = Path('/root/data/cys/baseline/campaigns/ml_optformer_132_20261007')
BACKUP = Path('/root/data/cys/logs/semanticopt_recovery_20261007/database_backup.json')
HOST = 'LLMOptimizer4'
HOSTNAME = 'cs-fa498-8793f-server'
CONTROLLER_CPU = 39
MAX_WORKERS = 8
RUNTIME_RAM_GIB = 4
PREFIX = 'LLMOptimizer4.random'
sys.path.insert(0, str(QUERY / 'code'))
import query_test200_fleet_v3 as fleet
import query_test200_rolling as rolling

spec = importlib.util.spec_from_file_location('deployment', RUNTIME / 'deployment.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
from ml_run_archive import load_json
from ml_memory_admission import admission


def load(path, default=None):
    return d.load(path, default)


def write(path, value):
    fleet.write(path, value)


def require_host():
    if socket.gethostname() != HOSTNAME:
        raise RuntimeError('Run the handover on LLMOptimizer4')


def verify():
    require_host()
    rolling.verify(QUERY)
    manifest = load(ROOT / 'manifest.json')
    if not manifest or manifest['code_sha256'] != fleet.digest(Path(__file__)):
        raise RuntimeError('Handover code has changed since preparation')
    return manifest


def core_key(cpu):
    directory = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    return tuple((directory / key).read_text().strip()
                 for key in ('physical_package_id', 'core_id'))


def eligible_cpus():
    reserved = {core_key(cpu) for cpu in [*range(12), CONTROLLER_CPU]}
    cpus, seen = [], set(reserved)
    for cpu in load(ROOT / 'manifest.json')['original_optformer_cpus']:
        key = core_key(cpu)
        if key not in seen:
            cpus.append(cpu)
            seen.add(key)
    return cpus


def completed(progress):
    return (progress.get('attempts') == 100 and progress.get('budget_exhausted') is True
            and not progress.get('pending'))


def held_task_cpus(state, progress, alive):
    return set(state.get('cpus', [])) if alive or not completed(progress) else set()


def optformer_snapshot():
    held, alive, pending = set(), [], []
    for task in load(OPT / 'tasks_132.json'):
        key = '/'.join(task[k] for k in ('task_type', 'dataset', 'algorithm'))
        state = load_json(OPT / 'state/optformer' / (key + '.json'), {})
        progress = load_json(OPT / 'sessions/optformer' / key / 'progress.json', {})
        if state.get('host') != HOST:
            continue
        identity = {k: state[k] for k in ('pid', 'start_ticks', 'boot_id')}
        live = d.process(state['pid']) == identity
        held.update(held_task_cpus(state, progress, live))
        if live:
            alive.append(identity)
        if not completed(progress):
            pending.append({'task': key, 'attempts': progress.get('attempts', 0), 'alive': live})
    return {'held': held, 'live': alive, 'pending': pending}


def allocations():
    return [load(path) for path in sorted((ROOT / 'allocations').glob('*.json'))]


def physical_occupied():
    keys = set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        pid = int(path.name)
        if d.process(pid) is None:
            continue
        try:
            cpus = os.sched_getaffinity(pid)
        except ProcessLookupError:
            continue
        if len(cpus) <= 4:
            keys.update(core_key(cpu) for cpu in cpus)
    return keys


def choose_cpus(candidates, held, reserved, occupied_keys, key=core_key):
    excluded = {key(cpu) for cpu in held | reserved} | occupied_keys
    selected = []
    for cpu in candidates:
        physical = key(cpu)
        if physical not in excluded:
            selected.append(cpu)
            excluded.add(physical)
        if len(selected) == 4:
            return selected
    return None


def worker_config(name):
    config = load(ROOT / 'allocations' / (name + '.json'))
    if not config or config['name'] != name or not name.startswith(PREFIX):
        raise RuntimeError('Unknown handover worker')
    cpus = [config['bo_cpu'], *config['db_cpus']]
    if len(cpus) != 4 or len({core_key(cpu) for cpu in cpus}) != 4:
        raise RuntimeError('Worker does not have four distinct physical cores')
    return config


def bind_fleet():
    original_allocation = fleet.allocation
    original_command = fleet.run_command

    def allocation(root, name, method=None):
        if not name.startswith(PREFIX):
            return original_allocation(root, name, method)
        if Path(root).resolve() != QUERY or method not in (None, 'random'):
            raise RuntimeError('This handover runs only the existing random-search campaign')
        require_host()
        return fleet.method_plan(QUERY, 'random'), worker_config(name)

    def command(root, worker, *args, **kwargs):
        result = original_command(root, worker, *args, **kwargs)
        if worker['name'].startswith(PREFIX):
            result[result.index('--device') + 1] = 'cpu'
        return result

    fleet.allocation = allocation
    fleet.run_command = command


def prepare():
    require_host()
    rolling.verify(QUERY)
    if (ROOT / 'manifest.json').exists():
        return verify()
    for name in ('allocations', 'processes', 'logs', 'state', 'database_restore', 'locks'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    backup = load(BACKUP)
    archive = Path(backup['archive'])
    if archive.stat().st_size != backup['bytes'] or fleet.digest(archive) != backup['sha256']:
        raise RuntimeError('Database backup checksum mismatch')
    protocol = load(OPT / 'protocol.json')
    write(ROOT / 'original_optformer_protocol.json', protocol)
    manifest = dict(created_at=fleet.stamp(), code_sha256=fleet.digest(Path(__file__)),
        query_campaign=str(QUERY), query_plan_sha256=fleet.digest(QUERY / 'plan.json'),
        host=HOST, hostname=HOSTNAME, max_workers=MAX_WORKERS, poll_seconds=30,
        original_optformer_cpus=protocol['hosts'][HOST]['cpus'],
        database_backup=backup, device='cpu', runtime_ram_gib=RUNTIME_RAM_GIB,
        adaptation='CPU execution of the unchanged author VAE and random-search runner',
        output_and_claims='Use the existing test200 output directories and locked task queue')
    write(ROOT / 'manifest.json', manifest)
    return manifest


def scheduler_record():
    return load(RUNTIME / 'processes' / (HOST + '.optformer.scheduler.json'))


def stop_scheduler():
    record = scheduler_record()
    if record and d.process(record['pid']) == record:
        argv = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().decode().split('\0')
        if str(OPT) not in argv or 'run' not in argv:
            raise RuntimeError('Unexpected OptFormer scheduler process')
        d.stop(record)


def restart_scheduler():
    allocation = load(OPT / 'protocol.json')['hosts'][HOST]
    cpus = sorted(set(allocation['cpus']) | set(allocation['scheduler_cpus']))
    return d.spawn(HOST + '.optformer.scheduler',
        ['taskset', '-c', ','.join(map(str, cpus)), d.PYTHON,
         str(d.CODE / 'examples/benchmark/ml_fixed_campaign.py'), 'run',
         '--campaign', str(OPT), '--host', HOST, '--retry-incomplete'], d.environment())


def allocate_slot():
    verify()
    existing = allocations()
    if len(existing) >= MAX_WORKERS:
        return None
    reserved = {cpu for worker in existing for cpu in [worker['bo_cpu'], *worker['db_cpus']]}
    snap = optformer_snapshot()
    selected = choose_cpus(eligible_cpus(), snap['held'], reserved, physical_occupied())
    if selected is None:
        return None
    # Pause only the scheduler so its allocation cannot race this handover.
    stop_scheduler()
    try:
        snap = optformer_snapshot()
        selected = choose_cpus(eligible_cpus(), snap['held'], reserved, physical_occupied())
        if selected is None:
            return None
        index = len(existing)
        name = PREFIX + '%02d' % index
        port = 5470 + index
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', port))
        worker = dict(name=name, host=HOST, method='random', bo_cpu=selected[0],
            db_cpus=selected[1:], gpu='', port=port,
            pgdata=f'/dev/shm/query_random_optformer_20261014/w{index:02d}/pg_imdb_d',
            source_pgdata=load(BACKUP)['archive'], volatile_database=True,
            device='cpu', handover_deployment=str(ROOT))
        protocol = load(OPT / 'protocol.json')
        reserved.update(selected)
        protocol['hosts'][HOST]['cpus'] = [cpu for cpu in eligible_cpus() if cpu not in reserved]
        protocol['query_random_handover'] = dict(deployment=str(ROOT), excluded_cpus=sorted(reserved))
        d.write_protocol(OPT, protocol)
        write(ROOT / 'allocations' / (name + '.json'), worker)
        write(ROOT / 'state' / (name + '.json'), dict(status='allocated', worker=worker,
            time=time.time(), optformer_live_workers_preserved=snap['live']))
        if not all(d.process(p['pid']) == p for p in snap['live']):
            # Completion during handover is harmless; a surviving process is never signalled.
            write(ROOT / 'state' / (name + '.handover_observation.json'),
                  {'time': time.time(), 'workers_finished_during_handover': True})
        return worker
    finally:
        restart_scheduler()


def memory_spec():
    return load(OPT / 'protocol.json')['hosts'][HOST]


def reservation_size(worker):
    return RUNTIME_RAM_GIB + (0 if (ROOT / 'database_restore' / (worker['name'] + '.json')).exists()
                              else load(BACKUP)['bytes'] / 2**30 + 1)


def spawn_worker(worker, smoke_only=False):
    name = worker['name']
    record_path = ROOT / 'processes' / (name + '.json')
    previous = load(record_path)
    if previous and d.process(previous['pid']) == previous:
        return previous
    state = load(ROOT / 'state' / (name + '.json'), {})
    if state.get('status') == 'failed':
        raise RuntimeError('Inspect the failed worker before restarting: ' + name)
    size = reservation_size(worker)
    with admission(memory_spec()) as guard:
        if not guard.allowed(size):
            return None
        command = ['taskset', '-c', str(worker['bo_cpu']), fleet.PYTHON,
                   str(Path(__file__)), 'worker', '--name', name]
        if smoke_only:
            command.append('--smoke-only')
        env = fleet.environment(worker)
        with (ROOT / 'logs' / (name + '.log')).open('a') as stream:
            process = subprocess.Popen(command, env=env, cwd=QUERY, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = d.process(process.pid)
        if record is None:
            raise RuntimeError('Worker exited during startup')
        guard.reserve(dict(record, session=str(ROOT / 'state' / name)), size)
        write(record_path, record)
    return record


def restore_database(worker):
    marker = ROOT / 'database_restore' / (worker['name'] + '.json')
    target = Path(worker['pgdata'])
    backup = load(BACKUP)
    if marker.exists():
        record = load(marker)
        if record['archive_sha256'] != backup['sha256'] or not (target / 'PG_VERSION').exists():
            raise RuntimeError('Restored database is missing or has different provenance')
        return
    if target.exists():
        raise RuntimeError('Unverified database directory exists; inspect before retrying')
    available = os.statvfs('/dev/shm')
    if available.f_bavail * available.f_frsize < backup['bytes'] + 4 * 2**30:
        raise RuntimeError('Insufficient tmpfs space for a private database')
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '--numeric-owner', '--keep-old-files', '-xpf', backup['archive'],
                    '-C', str(target.parent)], check=True)
    control = subprocess.check_output([str(fleet.PG_BIN / 'pg_controldata'), str(target)],
                                     text=True, env=dict(os.environ, LC_ALL='C'))
    cluster_state = next(line.split(':', 1)[1].strip() for line in control.splitlines()
                         if line.startswith('Database cluster state:'))
    if cluster_state != 'shut down' or target.stat().st_uid != 10000:
        raise RuntimeError('Restored database was not shut down cleanly')
    write(marker, dict(time=time.time(), path=str(target), archive_sha256=backup['sha256'],
                       cluster_state=cluster_state))


def stop_database(worker):
    if fleet.postgres_pid(worker) is None:
        return
    subprocess.run(['taskset', '-c', ','.join(map(str, worker['db_cpus'])),
        'setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
        str(fleet.PG_BIN / 'pg_ctl'), '-D', worker['pgdata'], '-m', 'fast', '-w', '-t', '60',
        'stop'], cwd='/tmp', check=True)


def worker_main(name, smoke_only):
    verify()
    worker = worker_config(name)
    bind_fleet()
    os.sched_setaffinity(0, {worker['bo_cpu']})
    state_path = ROOT / 'state' / (name + '.json')

    def state(status, **extra):
        write(state_path, dict(status=status, worker=worker, time=time.time(), **extra))

    with fleet.lock(ROOT / 'locks' / (name + '.lock')):
        try:
            state('restoring_database')
            restore_database(worker)
            fleet.db_start(QUERY, name)
            with admission(memory_spec()) as guard:
                record = d.process(os.getpid())
                guard.reserve(dict(record, session=str(ROOT / 'state' / name)), RUNTIME_RAM_GIB)
            plan = fleet.method_plan(QUERY, 'random')
            try:
                fleet.validate_smoke(QUERY, plan, worker)
            except RuntimeError:
                state('smoke')
                fleet.run_worker(QUERY, name, smoke=True, method='random')
                fleet.validate_smoke(QUERY, plan, worker)
            if smoke_only:
                stop_database(worker)
                state('smoke_ready')
                return
            state('production')
            fleet.run_worker(QUERY, name, method='random')
            stop_database(worker)
            state('complete')
        except BaseException as exc:
            state('failed', error=str(exc))
            raise


def stage():
    prepare()
    with fleet.lock(ROOT / 'controller.lock'):
        existing = allocations()
        worker = existing[0] if existing else allocate_slot()
        if worker is None:
            raise RuntimeError('No four-core slot has been released yet')
        process = spawn_worker(worker, smoke_only=True)
        print(json.dumps({'worker': worker, 'process': process,
                          'status': 'smoke_started' if process else 'waiting_for_memory'}))


def status():
    rows = []
    for worker in allocations():
        name = worker['name']
        process = load(ROOT / 'processes' / (name + '.json'), {})
        state = load(ROOT / 'state' / (name + '.json'), {})
        progress = load(QUERY / 'workers' / (name + '.json'), {})
        rows.append(dict(name=name, alive=bool(process and d.process(process['pid']) == process),
            phase=state.get('status'), cpus=[worker['bo_cpu'], *worker['db_cpus']],
            task=progress.get('task'), proposals=progress.get('saved_proposals'),
            error=state.get('error')))
    return rows


def controller():
    verify()
    if not load(ROOT / 'enabled.json', {}).get('enabled'):
        raise RuntimeError('Production has not passed its smoke gate')
    os.sched_setaffinity(0, {CONTROLLER_CPU})
    with fleet.lock(ROOT / 'controller.lock'):
        while True:
            rows = status()
            for row in rows:
                if row['alive'] or row['phase'] in ('failed', 'complete'):
                    continue
                if row['phase'] in ('allocated', 'smoke_ready'):
                    spawn_worker(worker_config(row['name']))
                else:
                    path = ROOT / 'state' / (row['name'] + '.json')
                    previous = load(path)
                    write(path, dict(previous, status='failed',
                        error='Worker exited unexpectedly; inspect durable state before replay'))
            tasks = load(QUERY / 'plan.json')['tasks']
            complete = sum(load(QUERY / 'claims' / ('random.' + task + '.json'), {})
                           .get('status') == 'complete' for task in tasks)
            reason = None
            if complete < len(tasks) and len(allocations()) < MAX_WORKERS:
                extra = RUNTIME_RAM_GIB + load(BACKUP)['bytes'] / 2**30 + 1
                with admission(memory_spec()) as guard:
                    allowed = guard.allowed(extra)
                if allowed:
                    worker = allocate_slot()
                    if worker:
                        spawn_worker(worker)
                    else:
                        reason = 'waiting_for_four_free_physical_cores'
                else:
                    reason = 'waiting_for_ram'
            report = dict(time=time.time(), process=d.process(os.getpid()),
                status='complete' if complete == len(tasks) else 'running',
                random_completed_tasks=complete, workers=status(), admission_wait=reason,
                optformer_pending=optformer_snapshot()['pending'])
            write(ROOT / 'status.json', report)
            if complete == len(tasks) and not any(row['alive'] for row in report['workers']):
                return
            time.sleep(30)


def enable():
    verify()
    bind_fleet()
    first = allocations()[0]
    plan = fleet.method_plan(QUERY, 'random')
    evidence = fleet.validate_smoke(QUERY, plan, first)
    state = load(ROOT / 'state' / (first['name'] + '.json'))
    if state['status'] != 'smoke_ready':
        raise RuntimeError('Wait for the smoke worker to finish cleanly')
    write(ROOT / 'enabled.json', dict(enabled=True, time=time.time(),
        first_worker=first['name'], smoke_run_id=evidence['run_id'],
        proposals=evidence['summary']['proposals'], code_sha256=fleet.digest(Path(__file__))))
    return start()


def start():
    verify()
    path = ROOT / 'controller_process.json'
    old = load(path)
    if old and d.process(old['pid']) == old:
        print(json.dumps({'existing': old}))
        return old
    with fleet.lock(ROOT / 'controller.lock'):
        pass
    with (ROOT / 'logs/controller.log').open('a') as stream:
        child = subprocess.Popen(['taskset', '-c', str(CONTROLLER_CPU), fleet.PYTHON,
            str(Path(__file__)), 'controller'], env=d.environment(), cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = d.process(child.pid)
    if not record:
        raise RuntimeError('Controller failed to start')
    write(path, record)
    print(json.dumps({'started': record, 'poll_seconds': 30, 'max_workers': MAX_WORKERS}))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'stage', 'enable', 'start', 'controller', 'worker', 'status'])
    parser.add_argument('--name')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'worker':
        worker_main(args.name, args.smoke_only)
    elif args.action == 'status':
        print(json.dumps(status()))
    else:
        globals()[args.action]()
