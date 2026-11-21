"""Launch isolated author-VAE train800 workers on LLMOptimizer2 and 3."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time


PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
PG_BIN = Path('/root/data/cys/envs/pg16/bin')
QP_ROOT = Path('/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization')
SPLITS = Path('/root/data/cys/logs/bolt_resume')
HOSTS = {'LLMOptimizer2': 'cs-94bac-0693e-server', 'LLMOptimizer3': 'cs-0b2cd-18deb-server'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def lock(path, blocking=False):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def prepare(root):
    from query_transfer_common import validate_assets, AUTHOR_VAE_SHA256, OFFICIAL_BAO_SHA256
    from run_query_ceb_baselines import load_split
    validate_assets()
    train, test = load_split()
    if root.exists():
        raise RuntimeError('Campaign already exists')
    root.mkdir(parents=True)
    for name in ('code', 'workers', 'claims', 'logs', 'databases', 'outputs', 'smoke'):
        (root / name).mkdir()
    for name in ('run_query_ceb_baselines.py', 'query_transfer_common.py', Path(__file__).name):
        shutil.copy2(Path(__file__).with_name(name), root / 'code' / name)
    native = root / 'native'
    native.mkdir()
    for folder in ('codec', 'lolbo', 'oracle', 'vae', 'your_tasks', 'logger', 'utils', 'workload'):
        for source in (QP_ROOT / folder).rglob('*.py'):
            target = native / source.relative_to(QP_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(QP_ROOT / 'constants.py', native / 'constants.py')
    for item in (QP_ROOT / 'workload').iterdir():
        if item.name == '__pycache__' or item.suffix == '.py':
            continue
        if item.is_dir():
            shutil.copytree(item, native / 'workload' / item.name)
        else:
            shutil.copy2(item, native / 'workload' / item.name)
    (native / 'initialization_data').mkdir()
    for relative in ('vae/CEB_64_author.ckpt', 'initialization_data/ceb_3k_bao_initialization.csv'):
        shutil.copy2(QP_ROOT / relative, native / relative)
    shutil.copy2(SPLITS / 'train800_bsz10.txt', root / 'train800.txt')
    shutil.copy2(SPLITS / 'test200_holdout.txt', root / 'test200.txt')
    layouts = [('LLMOptimizer2', 24, [25, 26, 27], 3, 'pg_imdb', 5432),
               ('LLMOptimizer2', 28, [29, 30, 31], 3, 'pg_imdb_c3', 5433),
               ('LLMOptimizer2', 36, [37, 38, 39], 3, 'pg_imdb_b', 5434),
               ('LLMOptimizer3', 0, [1, 2, 3], 0, 'pg_imdb_d', 5435),
               ('LLMOptimizer3', 4, [5, 6, 7], 1, 'pg_imdb_e', 5436),
               ('LLMOptimizer3', 8, [9, 10, 11], 2, 'pg_imdb_f', 5437)]
    workers = [dict(name=f'{host}.w{i % 3:02}', host=host, bo_cpu=cpu,
                    db_cpus=dbcpus, gpu=gpu, pgdata=f'/root/data/cys/{data}', port=port)
               for i, (host, cpu, dbcpus, gpu, data, port) in enumerate(layouts)]
    random.Random(0).shuffle(train)
    assigned = {task: ('LLMOptimizer2' if i % 2 == 0 else 'LLMOptimizer3')
                for i, task in enumerate(train)}
    files = {str(p.relative_to(root)): digest(p) for folder in ('code', 'native')
             for p in sorted((root / folder).rglob('*')) if p.is_file()}
    write(root / 'plan.json', dict(created_at=stamp(), campaign=str(root), hosts=HOSTS,
          workers=workers, tasks=train, task_hosts=assigned, files_sha256=files,
          method='lolbo', variant='frozen_vae_latent_turbo', seed=0, split='train',
          train_sha256=digest(root / 'train800.txt'), test_sha256=digest(root / 'test200.txt'),
          vae_sha256=AUTHOR_VAE_SHA256, bao_sha256=OFFICIAL_BAO_SHA256,
          initial_observations=49, proposals=1000, budget=1049, batch_size=10,
          checkpoint=str(native / 'vae/CEB_64_author.ckpt'),
          allocation='One BO core and three dedicated PostgreSQL cores per worker',
          postgres_settings={'shared_buffers': '1GB', 'work_mem': '64MB',
              'max_parallel_workers_per_gather': '2', 'max_parallel_workers': '2'},
          completed_proposal_storage='Verified tar.gz archive; history and checkpoint retained'))
    print(json.dumps({'campaign': str(root), 'tasks': len(train), 'workers': workers}))


def allocation(root, name):
    plan = read(root / 'plan.json')
    worker = next(w for w in plan['workers'] if w['name'] == name)
    if socket.gethostname() != plan['hosts'][worker['host']]:
        raise RuntimeError('Wrong machine for worker')
    occupied = set()
    for w in plan['workers']:
        if w['host'] != worker['host']:
            continue
        for cpu in [w['bo_cpu']] + w['db_cpus']:
            topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            key = tuple((topology / f).read_text().strip() for f in ('physical_package_id', 'core_id'))
            if key in occupied:
                raise RuntimeError('Overlapping physical cores')
            occupied.add(key)
    reservations = plan.get('cpu_reservations', {}).get(worker['host'])
    reserved = 28 if reservations is None and worker['host'] == 'LLMOptimizer2' else 0
    for reservation in reservations or []:
        for cpu in reservation['cpus']:
            topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            key = tuple((topology / f).read_text().strip() for f in ('physical_package_id', 'core_id'))
            if key in occupied:
                raise RuntimeError('Worker or reservation overlaps physical cores')
            occupied.add(key)
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if quota != 'max' and len(occupied) + reserved > int(quota) / int(period):
        raise RuntimeError('CPU quota would be exceeded')
    return plan, worker


def environment(worker):
    return dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                CUDA_VISIBLE_DEVICES=str(worker['gpu']), PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
                DB_HOST='127.0.0.1', DB_PORT=str(worker['port']), DB_NAME='imdb', DB_USER='imdb')


def postgres_pid(worker):
    pidfile = Path(worker['pgdata']) / 'postmaster.pid'
    if not pidfile.exists():
        return None
    pid = int(pidfile.read_text().splitlines()[0])
    proc = Path(f'/proc/{pid}')
    if not proc.exists():
        return None
    argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
    if argv and b'postgres' in argv[0] and str(worker['pgdata']).encode() in argv:
        return pid
    return None


def db_start(root, name):
    plan, worker = allocation(root, name)
    statefile = root / 'databases' / f'{name}.json'
    with lock(root / 'databases' / f'{name}.lock'):
        pid = postgres_pid(worker)
        if pid is not None:
            if not statefile.exists() or read(statefile).get('pid') != pid:
                raise RuntimeError('Database is already running outside this campaign')
        else:
            settings = ['-p', str(worker['port'])]
            for key, value in plan['postgres_settings'].items():
                settings += ['-c', f'{key}={value}']
            settings += ['-c', 'dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:'
                         '/root/data/cys/envs/pg16/lib']
            command = ['taskset', '-c', ','.join(map(str, worker['db_cpus'])),
                       'setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
                       str(PG_BIN / 'pg_ctl'), '-D', worker['pgdata'], '-l',
                       str(Path(worker['pgdata']) / 'query_train800_author_20261002.log'),
                       '-o', shlex.join(settings), '-w', '-t', '90', 'start']
            subprocess.run(command, check=True, cwd='/tmp')
            pid = postgres_pid(worker)
            if pid is None:
                raise RuntimeError('PostgreSQL failed to remain alive')
        if os.sched_getaffinity(pid) != set(worker['db_cpus']):
            raise RuntimeError('Database CPU affinity differs from the plan')
        import psycopg
        with psycopg.connect(host='127.0.0.1', port=worker['port'], dbname='imdb',
                             user='imdb', connect_timeout=10, autocommit=True) as conn:
            conn.execute("LOAD 'pg_hint_plan'")
            conn.execute('SET pg_hint_plan.enable_hint = on')
            conn.execute('SET join_collapse_limit = 1')
            effective = {key: conn.execute('SELECT current_setting(%s)', (key,)).fetchone()[0]
                         for key in list(plan['postgres_settings']) +
                         ['server_version', 'pg_hint_plan.enable_hint', 'join_collapse_limit']}
            conn.execute('SELECT 1 FROM title LIMIT 1').fetchone()
        result = dict(worker, pid=pid, hostname=socket.gethostname(), checked_at=stamp(),
                      effective_settings=effective, status='ready')
        write(statefile, result)
        print(json.dumps(result), flush=True)


def run_command(root, worker, task, output, budget=1049, stop_after_batches=None):
    command = [PYTHON, str(root / 'code/run_query_ceb_baselines.py'), '--method', 'lolbo',
               '--task-id', task, '--split', 'train', '--output', str(output),
               '--qp-root', str(root / 'native'), '--checkpoint',
               str(root / 'native/vae/CEB_64_author.ckpt'), '--device', 'cuda:0',
               '--budget', str(budget)]
    if stop_after_batches:
        command += ['--stop-after-batches', str(stop_after_batches)]
    return command


def validate_complete(directory, plan, task, budget=1049):
    history = read(directory / 'history.json')
    meta, observations = history['meta_info'], history['observations']
    expected = dict(task_id=task, method='lolbo', split='train', seed=0, status='complete',
                    budget_total=budget, charged_observations=budget,
                    initial_observations=49, proposals=budget - 49,
                    vae_sha256=plan['vae_sha256'], bao_sha256=plan['bao_sha256'],
                    unresolved_database_calls=0)
    if any(meta.get(k) != v for k, v in expected.items()) or len(observations) != budget:
        raise RuntimeError(f'Incomplete or mismatched history: {task}')
    for index, row in enumerate(observations):
        if (row['extra_info']['charged_index'] != index or len(row['config']) != 64
                or not math.isfinite(row['objectives'][0]) or row['objectives'][0] <= 0):
            raise RuntimeError(f'Invalid recorded observation: {task}/{index}')
    return meta


def archive_proposals(directory, count):
    archive = directory / 'proposals.tar.gz'
    sources = [directory / f'proposal_{i:04d}.json' for i in range(count)]
    if not archive.exists():
        temporary = archive.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as tar:
            for path in sources:
                tar.add(path, arcname=path.name)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        temporary.replace(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        if tar.getnames() != [p.name for p in sources]:
            raise RuntimeError('Proposal archive is incomplete')
        for index, path in enumerate(sources):
            content = tar.extractfile(path.name).read()
            if json.loads(content)['proposal_index'] != index:
                raise RuntimeError('Proposal archive index mismatch')
            if path.exists() and content != path.read_bytes():
                raise RuntimeError('Proposal archive differs from saved results')
    write(directory / 'proposal_archive.json', dict(count=count, sha256=digest(archive),
                                                   verified_at=stamp()))
    for path in sources:
        path.unlink(missing_ok=True)


def claim_task(root, plan, worker):
    with lock(root / 'claims/.queue.lock', blocking=True):
        for task in plan['tasks']:
            if plan['task_hosts'][task] != worker['host']:
                continue
            path = root / 'claims' / f'{task}.json'
            if path.exists():
                claim = read(path)
                if claim['worker'] != worker['name'] or claim['status'] == 'complete':
                    continue
            claim = dict(task=task, worker=worker['name'], host=worker['host'],
                         status='running', updated_at=stamp())
            write(path, claim)
            return task
    return None


def run_worker(root, name, smoke=False):
    plan, worker = allocation(root, name)
    os.sched_setaffinity(0, {worker['bo_cpu']})
    os.nice(10)
    os.environ.update(environment(worker))
    state_path = root / 'workers' / f'{name}.json'
    process = None
    task = None
    state = dict(worker, pid=os.getpid(), started_at=stamp(), mode='smoke' if smoke else 'production')

    def interrupt(signum, frame):
        if process is not None and process.poll() is None:
            process.terminate()
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    with lock(root / 'workers' / f'{name}.lock'):
        try:
            db = read(root / 'databases' / f'{name}.json')
            if postgres_pid(worker) != db['pid']:
                raise RuntimeError('Database is not running on this worker host')
            for relative, expected in plan['files_sha256'].items():
                if digest(root / relative) != expected:
                    raise RuntimeError('Campaign snapshot changed: ' + relative)
            env = environment(worker)
            while True:
                if smoke:
                    task = 'CEB_6A1' if worker['name'].endswith('w00') else 'CEB_11A21'
                else:
                    task = claim_task(root, plan, worker)
                    if task is None:
                        write(state_path, dict(state, status='complete', updated_at=stamp()))
                        return
                output = root / ('smoke/' + name if smoke else 'outputs')
                directory = output / task / 'lolbo/seed0'
                budget = 69 if smoke else 1049
                if not smoke:
                    limits = Path('/sys/fs/cgroup')
                    memory_stats = dict(line.split() for line in
                                        (limits / 'memory.stat').read_text().splitlines())
                    headroom = (int((limits / 'memory.max').read_text()) -
                                int((limits / 'memory.current').read_text()) +
                                int(memory_stats.get('inactive_file', 0))) / 1024**3
                    storage = os.statvfs(root)
                    if headroom < 8 or storage.f_favail < 30000 or storage.f_bavail * storage.f_frsize < 10 * 1024**3:
                        raise RuntimeError('Insufficient RAM or shared-storage headroom for another task')
                phases = [1, None] if smoke else [None]
                for stop in phases:
                    log_path = root / 'logs' / f'{name}.{task}.{"smoke" if smoke else "run"}.log'
                    command = run_command(root, worker, task, output, budget, stop)
                    with log_path.open('a') as stream:
                        process = subprocess.Popen(command, env=env, cwd=root / 'native',
                                                   stdout=stream, stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            summary = read(directory / 'status.json') if (directory / 'status.json').exists() else {}
                            write(state_path, dict(state, task=task, child_pid=process.pid,
                                  status='running', updated_at=stamp(), log=str(log_path),
                                  saved_proposals=sum(1 for _ in directory.glob('proposal_[0-9][0-9][0-9][0-9].json')),
                                  summary=summary))
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                pass
                        if process.returncode:
                            raise RuntimeError(f'Query runner exited {process.returncode}; see {log_path}')
                    if stop:
                        partial = read(directory / 'status.json')
                        if partial['status'] != 'paused' or partial['proposals'] != 10:
                            raise RuntimeError('Smoke run did not pause at the requested checkpoint')
                summary = validate_complete(directory, plan, task, budget)
                archive_proposals(directory, budget - 49)
                if smoke:
                    write(root / 'smoke' / f'{name}.verified.json',
                          dict(verified_at=stamp(), worker=worker, summary=summary,
                               checkpoint_reload=True, archived_proposals=True))
                    write(state_path, dict(state, status='smoke_complete', updated_at=stamp()))
                    return
                write(root / 'claims' / f'{task}.json', dict(task=task, worker=name,
                      host=worker['host'], status='complete', updated_at=stamp(),
                      history_sha256=digest(directory / 'history.json')))
                print(json.dumps({'completed': task, 'worker': name, 'at': stamp()}), flush=True)
        except BaseException as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            write(state_path, dict(state, task=task, status='failed', updated_at=stamp(), error=str(exc)))
            raise


def launch(root, host, smoke=False, names=None):
    plan = read(root / 'plan.json')
    if socket.gethostname() != plan['hosts'][host]:
        raise RuntimeError('Wrong machine')
    workers = [worker for worker in plan['workers'] if worker['host'] == host]
    if names is not None:
        unknown = set(names) - {worker['name'] for worker in workers}
        if unknown or len(names) != len(set(names)):
            raise RuntimeError('Unknown or duplicate worker names')
        workers = [worker for worker in workers if worker['name'] in names]
    launched = []
    if not smoke:
        for worker in workers:
            if not (root / 'smoke' / f'{worker["name"]}.verified.json').exists():
                raise RuntimeError('Selected workers must pass smoke/reload checks before production')
    for worker in workers:
        allocation(root, worker['name'])
        with lock(root / 'workers' / f'{worker["name"]}.lock'):
            pass
    for worker in workers:
        log = root / 'logs' / f'{worker["name"]}.supervisor.log'
        with log.open('a') as stream:
            command = [PYTHON, str(root / 'code' / Path(__file__).name),
                       '--campaign', str(root), 'worker', '--name', worker['name']]
            if smoke:
                command.append('--smoke')
            process = subprocess.Popen(command, cwd=root, env=environment(worker),
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        launched.append(dict(worker=worker['name'], pid=process.pid, log=str(log)))
    launch_path = root / f'launch.{host}.{"smoke" if smoke else "production"}.json'
    previous = read(launch_path) if launch_path.exists() else {}
    combined = {worker['worker']: worker for worker in previous.get('workers', [])}
    combined.update({worker['worker']: worker for worker in launched})
    write(launch_path, dict(launched_at=previous.get('launched_at', stamp()),
                           updated_at=stamp(), workers=list(combined.values())))
    print(json.dumps(launched))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    db = sub.add_parser('db-start')
    db.add_argument('--name', required=True)
    worker = sub.add_parser('worker')
    worker.add_argument('--name', required=True)
    worker.add_argument('--smoke', action='store_true')
    launcher = sub.add_parser('launch')
    launcher.add_argument('--host', required=True)
    launcher.add_argument('--smoke', action='store_true')
    launcher.add_argument('--names', nargs='+', help='Launch only the named workers on this host.')
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'db-start':
        db_start(root, args.name)
    elif args.action == 'worker':
        run_worker(root, args.name, args.smoke)
    else:
        launch(root, args.host, args.smoke, args.names)


if __name__ == '__main__':
    main()
