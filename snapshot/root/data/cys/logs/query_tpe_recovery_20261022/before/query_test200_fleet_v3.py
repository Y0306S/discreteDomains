"""Launch isolated random/TPE/frozen-LOLBO workers on the QueryOpt test200 split."""

import argparse
import copy
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
import uuid
from run_artifact_store import DATABASE, JsonJournal


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
    predecessor = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
    prior = read(predecessor / 'plan.json')
    validate_assets(predecessor / 'native', Path(prior['checkpoint']))
    train, test = load_split(predecessor / 'train800.txt', predecessor / 'test200.txt')
    if (root / 'plan.json').exists():
        raise RuntimeError('Campaign already prepared')
    for name in ('code', 'workers', 'claims', 'logs', 'databases', 'outputs', 'smoke'):
        (root / name).mkdir(exist_ok=True)
    native = root / 'native'
    # Share the frozen native assets; hashes are verified at every worker launch.
    native.symlink_to(predecessor / 'native', target_is_directory=True)
    for name in ('train800.txt', 'test200.txt'):
        shutil.copy2(predecessor / name, root / name)
    methods = ('lolbo', 'tpe', 'random')
    workers = []
    for w in sorted(prior['workers'], key=lambda w: w['name']):
        workers.append(dict(w, method='lolbo'))
    random.Random(0).shuffle(test)
    files = {str(p.relative_to(root)): digest(p) for folder in ('code', 'native')
             for p in sorted((root / folder).rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    write(root / 'plan.json', dict(created_at=stamp(), campaign=str(root), hosts=HOSTS,
          workers=workers, tasks=test, methods=list(methods), files_sha256=files,
          predecessor=str(predecessor), total_jobs=600, seed=0, split='test',
          active_method='lolbo', scheduling='full_fleet_sequential_methods',
          smoke_task='CEB_10A32',
          cpu_reservations={
              'LLMOptimizer2': [dict(name='ML, inference and controllers',
                                     cpus=list(range(8)) + list(range(16, 24)))],
              'LLMOptimizer3': [dict(name='ML, inference and controllers',
                                     cpus=[0, 1, 2, 28, 29, 30, 31, 38])]},
          train_sha256=digest(root / 'train800.txt'), test_sha256=digest(root / 'test200.txt'),
          vae_sha256=AUTHOR_VAE_SHA256, bao_sha256=OFFICIAL_BAO_SHA256,
          initial_observations=49, proposals=1000, budget=1049, batch_size=10,
          random_tpe_timeout_seconds=100.0, random_tpe_latent_bound=4.0,
          variants={'random': 'uniform_frozen_vae_latent', 'tpe': 'optuna_tpe_frozen_vae_latent',
                    'lolbo': 'frozen_vae_latent_turbo'},
          cache_gp_policy='retain_existing_plan_duplicate_filter',
          cached_batch_counters='update_progress_and_trust_region_once_per_acquisition_batch',
          checkpoint=str(native / 'vae/CEB_64_author.ckpt'),
          allocation='One BO core and three dedicated PostgreSQL cores per worker',
          postgres_settings={'shared_buffers': '1GB', 'work_mem': '64MB',
              'max_parallel_workers_per_gather': '2', 'max_parallel_workers': '2'},
          completed_proposal_storage='One SQLite journal per method/task; history and latest checkpoint retained'))
    print(json.dumps({'campaign': str(root), 'tasks_per_method': len(test), 'workers': workers}))


def method_plan(root, method):
    """Resolve a method without modifying the plan used by live legacy workers."""
    plan = copy.deepcopy(read(root / 'plan.json'))
    if method not in plan['methods']:
        raise RuntimeError('Unknown method: ' + method)
    plan.update(plan['method_runtime'][method])
    plan['active_method'] = method
    plan['workers'] = [dict(worker, method=method) for worker in plan['workers']]
    manifest = read(root / 'rolling_manifest.json')
    if digest(root / 'plan.json') != manifest['plan_sha256']:
        raise RuntimeError('The pinned campaign plan changed')
    plan['files_sha256'].update(manifest['files_sha256'])
    return plan


def allocation(root, name, method=None):
    plan = method_plan(root, method) if method else read(root / 'plan.json')
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
        admission = plan.get('cpu_admission', {}).get(worker['host'], {})
        if admission.get('mode') != 'measured_headroom':
            raise RuntimeError('CPU quota would be exceeded')
        # Disjoint CPU affinities can exceed quota when database workers wait on I/O.
        # Admit added capacity only after observing room for its full four-core peak.
        def cpu_stats():
            return {key: int(value) for key, value in
                    (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}
        before, started = cpu_stats(), time.monotonic()
        time.sleep(10)
        after = cpu_stats()
        used = (after['usage_usec'] - before['usage_usec']) / 1e6 / (time.monotonic() - started)
        limit = int(quota) / int(period)
        if (after['nr_throttled'] != before['nr_throttled']
                or used + float(admission['headroom_cores']) > limit):
            raise RuntimeError('Measured CPU quota headroom is insufficient')
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
                prior = Path(plan['predecessor'])
                old = read(prior / 'databases' / f'{name}.json')
                state = read(prior / 'workers' / f'{name}.json')
                if old.get('pid') != pid or old.get('pgdata') != worker['pgdata']:
                    raise RuntimeError('Database ownership differs from the verified predecessor')
                if state.get('status') != 'complete':
                    raise RuntimeError('Predecessor worker has not completed')
                with lock(prior / 'workers' / f'{name}.lock'):
                    pass
                import psycopg
                with psycopg.connect(host='127.0.0.1', port=worker['port'], dbname='imdb',
                                     user='imdb', connect_timeout=10, autocommit=True) as conn:
                    clients = conn.execute("SELECT count(*) FROM pg_stat_activity WHERE "
                        "backend_type='client backend' AND pid<>pg_backend_pid()").fetchone()[0]
                    if clients:
                        raise RuntimeError('Predecessor database still has clients')
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
    plan = method_plan(root, worker['method'])
    runner = plan.get('runner', 'code/run_query_ceb_baselines.py')
    command = [PYTHON, str(root / runner), '--method', worker['method'],
               '--task-id', task, '--split', plan['split'], '--output', str(output),
               '--qp-root', str(root / 'native'), '--checkpoint',
               str(root / 'native/vae/CEB_64_author.ckpt'), '--device', 'cuda:0',
               '--budget', str(budget), '--batch-size', str(plan['batch_size']),
               '--seed', str(plan['seed']), '--timeout', str(plan['random_tpe_timeout_seconds']),
               '--latent-bound', str(plan['random_tpe_latent_bound'])]
    if stop_after_batches:
        command += ['--stop-after-batches', str(stop_after_batches)]
    if worker['method'] == 'random':
        command += ['--max-unique-draws-per-slot', str(plan['max_unique_draws_per_slot'])]
    previous = plan.get('resume_runner_sha256')
    if previous:
        command += ['--resume-runner-sha256', previous]
    return command


def validate_complete(directory, plan, task, budget=1049, method='lolbo'):
    history = read(directory / 'history.json')
    meta, observations = history['meta_info'], history['observations']
    expected = dict(task_id=task, method=method, split=plan['split'], seed=plan['seed'], status='complete',
                    budget_total=budget, charged_observations=budget,
                    initial_observations=49, proposals=budget - 49,
                    vae_sha256=plan['vae_sha256'], bao_sha256=plan['bao_sha256'],
                    unresolved_database_calls=0)
    if any(meta.get(k) != v for k, v in expected.items()) or len(observations) != budget:
        raise RuntimeError(f'Incomplete or mismatched history: {task}')
    if meta.get('batch_size') != plan['batch_size']:
        raise RuntimeError(f'Wrong batch size in completed history: {task}')
    if method == 'random':
        initial_keys = {row['extra_info'].get('executable_plan_key') for row in observations[:49]}
        keys = [row['extra_info'].get('executable_plan_key') for row in observations[49:]]
        if (None in initial_keys or None in keys or len(set(keys)) != budget - 49
                or initial_keys.intersection(keys) or meta.get('cache_hits') != 0
                or meta.get('unique_plan_policy') != 'exclude_bao_and_previous_recovered_queries'):
            raise RuntimeError(f'Random Search contains repeated or unidentified recovered plans: {task}')
    for index, row in enumerate(observations):
        if (row['extra_info']['charged_index'] != index or len(row['config']) != 64
                or not math.isfinite(row['objectives'][0]) or row['objectives'][0] <= 0):
            raise RuntimeError(f'Invalid recorded observation: {task}/{index}')
    return meta


def archive_proposals(directory, count):
    if (directory / DATABASE).exists():
        journal = JsonJournal(directory)
        expected = [f'proposal_{i:04d}.json' for i in range(count)]
        if journal.names('proposal_[0-9][0-9][0-9][0-9].json') != expected:
            raise RuntimeError('Proposal journal is incomplete')
        for index, name in enumerate(expected):
            if journal.read(name)['proposal_index'] != index:
                raise RuntimeError('Proposal journal index mismatch')
        if journal.names('proposal_*.pending.json'):
            raise RuntimeError('Pending database calls remain')
        write(directory / 'proposal_archive.json', dict(count=count, sha256=digest(directory / DATABASE),
              storage='sqlite_per_task_v1', verified_at=stamp()))
        return
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
            path = root / 'claims' / f'{worker["method"]}.{task}.json'
            if path.exists():
                claim = read(path)
                if claim['worker'] != worker['name'] or claim['status'] == 'complete':
                    continue
            claim = dict(task=task, worker=worker['name'], host=worker['host'],
                         method=worker['method'],
                         status='running', updated_at=stamp())
            write(path, claim)
            return task
    return None


def begin_smoke(root, plan, worker):
    run_id = uuid.uuid4().hex
    output = root / 'smoke' / worker['name'] / 'runs' / run_id
    output.mkdir(parents=True, exist_ok=False)
    verified = root / 'smoke' / f'{worker["name"]}.verified.json'
    if verified.exists():
        verified.replace(output / 'previous_verification.json')
    request = dict(run_id=run_id, started_at=stamp(), started_epoch=time.time(),
                   worker=worker, files_sha256=plan['files_sha256'], output=str(output))
    write(root / 'smoke' / f'{worker["name"]}.request.json', request)
    return request


def validate_smoke(root, plan, worker, not_before=None):
    prefix = root / 'smoke' / worker['name']
    request_path = prefix.with_name(prefix.name + '.request.json')
    verified_path = prefix.with_name(prefix.name + '.verified.json')
    if not request_path.exists() or not verified_path.exists():
        raise RuntimeError('A fresh smoke/reload verification is required')
    request, verified = read(request_path), read(verified_path)
    if (not request.get('run_id') or verified.get('run_id') != request['run_id']
            or request.get('worker') != worker or verified.get('worker') != worker
            or request.get('files_sha256') != plan['files_sha256']
            or verified.get('files_sha256') != plan['files_sha256']
            or verified.get('output') != request.get('output')
            or not isinstance(request.get('started_epoch'), (int, float))
            or not isinstance(verified.get('verified_epoch'), (int, float))
            or verified['verified_epoch'] < request['started_epoch']
            or (not_before is not None and request['started_epoch'] < not_before)):
        raise RuntimeError('Smoke verification is stale or belongs to different settings')
    stop_batches = 10 // plan['batch_size']
    expected_phases = [dict(stop_after_batches=stop_batches, returncode=0, status='paused', proposals=10),
                       dict(stop_after_batches=None, returncode=0, status='complete', proposals=20)]
    if (verified.get('phases') != expected_phases
            or verified.get('checkpoint_reload') is not True
            or verified.get('archived_proposals') is not True):
        raise RuntimeError('Smoke verification lacks successful pause/reload evidence')
    output = Path(request['output'])
    if output.resolve() != (prefix / 'runs' / request['run_id']).resolve():
        raise RuntimeError('Smoke output is outside its unique run directory')
    task = plan['smoke_task']
    summary = validate_complete(output / task / worker['method'] / 'seed0',
                                plan, task, 69, worker['method'])
    if summary != verified.get('summary'):
        raise RuntimeError('Smoke verification differs from the saved evaluation history')
    return verified


def run_worker(root, name, smoke=False, method=None):
    plan, worker = allocation(root, name, method)
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
            smoke_request = begin_smoke(root, plan, worker) if smoke else None
            if not smoke:
                validate_smoke(root, plan, worker)
            env = environment(worker)
            while True:
                if smoke:
                    task = plan['smoke_task']
                else:
                    task = claim_task(root, plan, worker)
                    if task is None:
                        write(state_path, dict(state, status='complete', updated_at=stamp()))
                        return
                output = Path(smoke_request['output']) if smoke else root / 'outputs'
                directory = output / task / worker['method'] / 'seed0'
                budget = 69 if smoke else plan['budget']
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
                phases = [10 // plan['batch_size'], None] if smoke else [None]
                phase_results = []
                for stop in phases:
                    log_path = root / 'logs' / f'{name}.{worker["method"]}.{task}.{"smoke" if smoke else "run"}.log'
                    command = run_command(root, worker, task, output, budget, stop)
                    with log_path.open('a') as stream:
                        process = subprocess.Popen(command, env=env, cwd=root / 'native',
                                                   stdout=stream, stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            summary = read(directory / 'status.json') if (directory / 'status.json').exists() else {}
                            write(state_path, dict(state, task=task, child_pid=process.pid,
                                  status='running', updated_at=stamp(), log=str(log_path),
                                  saved_proposals=len(JsonJournal(directory).names('proposal_[0-9][0-9][0-9][0-9].json')),
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
                    if smoke:
                        phase_summary = read(directory / 'status.json')
                        phase_results.append(dict(stop_after_batches=stop, returncode=process.returncode,
                                                  status=phase_summary['status'],
                                                  proposals=phase_summary['proposals']))
                summary = validate_complete(directory, plan, task, budget, worker['method'])
                archive_proposals(directory, budget - 49)
                if smoke:
                    write(root / 'smoke' / f'{name}.verified.json',
                          dict(verified_at=stamp(), worker=worker, summary=summary,
                               verified_epoch=time.time(), run_id=smoke_request['run_id'],
                               output=str(output), files_sha256=plan['files_sha256'],
                               phases=phase_results, checkpoint_reload=True, archived_proposals=True))
                    write(state_path, dict(state, status='smoke_complete', updated_at=stamp()))
                    return
                write(root / 'claims' / f'{worker["method"]}.{task}.json', dict(task=task, worker=name,
                      method=worker['method'],
                      host=worker['host'], status='complete', updated_at=stamp(),
                      history_sha256=digest(directory / 'history.json')))
                print(json.dumps({'completed': task, 'worker': name, 'at': stamp()}), flush=True)
        except BaseException as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            write(state_path, dict(state, task=task, status='failed', updated_at=stamp(), error=str(exc)))
            raise


def launch(root, host, smoke=False, names=None, method=None):
    plan = method_plan(root, method) if method else read(root / 'plan.json')
    if plan.get('cache_gp_policy') == 'pending_user_choice':
        raise RuntimeError('Campaign behavior has not been finalized')
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
            validate_smoke(root, plan, worker)
    for worker in workers:
        allocation(root, worker['name'], method)
        with lock(root / 'workers' / f'{worker["name"]}.lock'):
            pass
    for worker in workers:
        log = root / 'logs' / f'{worker["name"]}.supervisor.log'
        with log.open('a') as stream:
            command = [PYTHON, str(root / 'code' / Path(__file__).name),
                       '--campaign', str(root), 'worker', '--name', worker['name']]
            if method:
                command += ['--method', method]
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
    worker.add_argument('--method', choices=['lolbo', 'tpe', 'random'], required=True)
    launcher = sub.add_parser('launch')
    launcher.add_argument('--host', required=True)
    launcher.add_argument('--smoke', action='store_true')
    launcher.add_argument('--method', choices=['lolbo', 'tpe', 'random'], required=True)
    launcher.add_argument('--names', nargs='+', help='Launch only the named workers on this host.')
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'db-start':
        db_start(root, args.name)
    elif args.action == 'worker':
        run_worker(root, args.name, args.smoke, args.method)
    else:
        launch(root, args.host, args.smoke, args.names, args.method)


if __name__ == '__main__':
    main()
