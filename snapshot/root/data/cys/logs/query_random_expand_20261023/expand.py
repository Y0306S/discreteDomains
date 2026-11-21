"""Reuse completed ToSFiT capacity for the pinned test200 random campaign."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
HANDOFF = Path('/root/data/cys/logs/query_random_optformer_handoff_20261014')
STORAGE = Path('/root/data/cys/logs/query_random_storage_20261015/recover.py')
TOSFIT = Path('/root/data/cys/baseline/campaigns/ml_tosfit_132_20261014_long')
ACTORS = Path('/root/data/cys/logs/tosfit_length_20261014')
sys.path.insert(0, str(HANDOFF))
import handoff as h

fleet, d, QUERY = h.fleet, h.d, h.QUERY
HOSTS = d.HOSTS
HOST = next(key for key, value in HOSTS.items() if value == socket.gethostname())
SLOTS = {'LLMOptimizer2': [[0, 1, 2, 3], [4, 5, 6, 7], [16, 17, 18, 19]],
         'LLMOptimizer3': [[0, 1, 2, 28]],
         'LLMOptimizer4': [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]}
SOURCES = [Path(__file__), HANDOFF/'handoff.py', STORAGE,
           QUERY/'code/query_test200_fleet_v3.py', QUERY/'code/query_test200_rolling.py',
           QUERY/'plan.json', QUERY/'rolling_manifest.json']
load, write = h.load, h.write


def allocations():
    return [load(path) for path in sorted((ROOT/'allocations').glob(HOST + '.*.json'))]


def verify():
    manifest = load(ROOT/'manifest.json')
    for path, expected in manifest['sources'].items():
        if fleet.digest(Path(path)) != expected:
            raise RuntimeError('Pinned deployment source changed: ' + path)
    h.rolling.verify(QUERY)
    return manifest


def memory_spec():
    return dict(memory_gib=int(Path('/sys/fs/cgroup/memory.max').read_text()) / 2**30,
                memory_headroom_gib=32,
                memory_admission_registry=str(ROOT/'memory'/f'{HOST}.json'))


def bind():
    h.ROOT, h.HOST, h.HOSTNAME = ROOT, HOST, HOSTS[HOST]
    h.PREFIX = HOST + '.random_extra'
    h.verify, h.allocations, h.memory_spec = verify, allocations, memory_spec
    original_allocation, original_command = fleet.allocation, fleet.run_command

    def allocation(root, name, method=None):
        if not name.startswith(h.PREFIX):
            return original_allocation(root, name, method)
        if Path(root).resolve() != QUERY or method not in (None, 'random'):
            raise RuntimeError('Only the existing random campaign is authorized')
        return fleet.method_plan(QUERY, 'random'), h.worker_config(name)

    def command(root, worker, *args, **kwargs):
        argv = original_command(root, worker, *args, **kwargs)
        if worker['name'].startswith(h.PREFIX):
            argv[argv.index('--device') + 1] = worker['device']
        return argv

    fleet.allocation, fleet.run_command = allocation, command


def prepare():
    if (ROOT/'manifest.json').exists():
        return verify()
    h.rolling.verify(QUERY)
    for folder in ('allocations', 'processes', 'logs', 'state', 'database_restore',
                   'locks', 'storage', 'memory', 'retired'):
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    for host, slots in SLOTS.items():
        for index, cpus in enumerate(slots):
            name = f'{host}.random_extra{index:02d}'
            worker = dict(name=name, host=host, method='random', bo_cpu=cpus[0],
                db_cpus=cpus[1:], gpu='' if host == 'LLMOptimizer4' else str(index % 4),
                device='cpu' if host == 'LLMOptimizer4' else 'cuda:0', port=5480+index,
                pgdata=f'/dev/shm/query_random_expand_20261023/w{index:02d}/pg_imdb_d',
                source_pgdata=load(h.BACKUP)['archive'], volatile_database=True,
                handover_deployment=str(ROOT))
            write(ROOT/'allocations'/f'{name}.json', worker)
    write(ROOT/'manifest.json', dict(created_at=fleet.stamp(), slots=SLOTS,
        sources={str(p): fleet.digest(p) for p in SOURCES}, method='random',
        protocol='Unchanged test200, author VAE, seed0, 49 Bao starts + 1000 unique '
                 'runnable proposals, 100-second SQL timeout, SQLite artifacts',
        storage='Private tmpfs database; shared NVMe SQL spill tablespace',
        smoke='20 proposals with pause after 10 and checkpoint reload'))
    return verify()


def retire_actors():
    for task in load(TOSFIT/'tasks_132.json'):
        key = '/'.join(task[k] for k in ('task_type', 'dataset', 'algorithm'))
        progress = h.load_json(TOSFIT/'sessions/tosfit'/key/'progress.json')
        if not h.completed(progress):
            raise RuntimeError('ToSFiT has unfinished work: ' + key)
    retired = []
    for record_path in sorted((ACTORS/'processes').glob(HOST+'.tosfit*.json')):
        record = load(record_path)
        if not record or d.process(record['pid']) != record:
            continue
        argv = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
        if not any(arg.endswith(b'/ml_runtime_service.py') for arg in argv):
            raise RuntimeError('Unexpected process in actor record')
        config = load(ACTORS/'config'/record_path.name) if HOST == 'LLMOptimizer3' else load(
            Path('/root/data/cys/logs/tosfit_seventh_20261020/actor.json'))
        with urllib.request.urlopen(f'http://127.0.0.1:{config["port"]}/health', timeout=10) as response:
            health = json.load(response)['result']
        if health['busy'] or health['session'] is not None:
            raise RuntimeError('ToSFiT actor is not idle')
        write(ROOT/'retired'/record_path.name, dict(record=record, health=health,
              time=fleet.stamp(), reason='All 132 tasks complete; CPU capacity reassigned'))
        d.stop(record)
        retired.append(record_path.stem)
    return retired


def check_free(worker):
    wanted = {h.core_key(cpu) for cpu in [worker['bo_cpu'], *worker['db_cpus']]}
    if len(wanted) != 4:
        raise RuntimeError('CPU allocation repeats a physical core')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if d.process(int(path.name)) is None:
                continue
            cpus = os.sched_getaffinity(int(path.name))
            if len(cpus) <= 4 and wanted & {h.core_key(cpu) for cpu in cpus}:
                raise RuntimeError('Allocated cores still occupied by PID ' + path.name)
        except ProcessLookupError:
            pass
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', worker['port']))


def configure_storage(worker):
    spec = importlib.util.spec_from_file_location('spill_recovery', STORAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT/'storage'/worker['name']
    module.ROOT.mkdir(parents=True, exist_ok=True)
    module.HANDOFF = ROOT
    module.TEMP_ROOT = Path('/root/data/cys/queryopt_temp/query_random_expand_20261023')
    previous = h.allocations
    h.allocations = lambda: [worker]
    try:
        module.configure()
    finally:
        h.allocations = previous


def worker_main(name):
    verify()
    worker = h.worker_config(name)
    state_path = ROOT/'state'/f'{name}.json'
    def state(status, **extra):
        write(state_path, dict(status=status, worker=worker, time=fleet.stamp(), **extra))
    with fleet.lock(ROOT/'locks'/f'{name}.lock'):
        try:
            # Serialize only restores; each smoke and production runner is independent.
            with fleet.lock(ROOT/'locks'/f'{HOST}.restore.lock', blocking=True):
                state('restoring_database')
                h.restore_database(worker)
            fleet.db_start(QUERY, name)
            configure_storage(worker)
            with h.admission(memory_spec()) as guard:
                guard.reserve(dict(d.process(os.getpid()), session=str(state_path)), 8)
            state('smoke')
            fleet.run_worker(QUERY, name, smoke=True, method='random')
            fleet.validate_smoke(QUERY, fleet.method_plan(QUERY, 'random'), worker)
            state('production')
            fleet.run_worker(QUERY, name, method='random')
            h.stop_database(worker)
            state('complete')
        except BaseException as exc:
            state('failed', error=str(exc))
            raise


def launch():
    verify()
    with fleet.lock(ROOT/'locks'/f'{HOST}.launch.lock'):
        retired = retire_actors()
        backup = load(h.BACKUP)
        marker = ROOT/'database_restore'/f'{HOST}.archive_verified.json'
        if not marker.exists():
            archive = Path(backup['archive'])
            if archive.stat().st_size != backup['bytes'] or fleet.digest(archive) != backup['sha256']:
                raise RuntimeError('Database archive verification failed')
            write(marker, dict(backup, verified_at=fleet.stamp()))
        started = []
        for worker in allocations():
            name = worker['name']
            path = ROOT/'processes'/f'{name}.json'
            previous = load(path)
            if previous and d.process(previous['pid']) == previous:
                started.append(dict(worker=name, process=previous, existing=True))
                continue
            if load(ROOT/'state'/f'{name}.json', {}).get('status') == 'failed':
                raise RuntimeError('Review failed deployment before restart: ' + name)
            check_free(worker)
            with h.admission(memory_spec()) as guard:
                estimate = backup['bytes']/2**30 + 10
                if not guard.allowed(estimate):
                    raise RuntimeError('Insufficient cgroup RAM for ' + name)
                command = ['taskset', '-c', str(worker['bo_cpu']), fleet.PYTHON,
                           str(Path(__file__)), 'worker', '--name', name]
                with (ROOT/'logs'/f'{name}.log').open('a') as stream:
                    child = subprocess.Popen(command, env=fleet.environment(worker), cwd=QUERY,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                record = d.process(child.pid)
                if record is None:
                    raise RuntimeError('Worker exited during launch')
                write(path, record)
                guard.reserve(dict(record, session=str(ROOT/'state'/name)), estimate)
                started.append(dict(worker=name, process=record))
        report = dict(host=HOST, time=fleet.stamp(), retired_actors=retired, workers=started,
                      cpu_leases=[dict(worker=w['name'], cpus=[w['bo_cpu'], *w['db_cpus']])
                                  for w in allocations()])
        write(ROOT/f'{HOST}.launch.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'worker', 'status'])
    parser.add_argument('--name')
    args = parser.parse_args()
    bind()
    if args.action == 'worker':
        worker_main(args.name)
    elif args.action == 'status':
        print(json.dumps(h.status()))
    else:
        globals()[args.action]()
