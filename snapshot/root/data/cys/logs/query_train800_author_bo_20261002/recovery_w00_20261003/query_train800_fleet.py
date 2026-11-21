"""Restore w00 with the staged I/O recovery runner and the frozen task controller."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
NAME = 'LLMOptimizer3.w00'
TASK = 'CEB_11B187'
FIXED = CAMPAIGN / 'recovery_20261002/code/run_query_ceb_baselines.py'
ORIGINAL = CAMPAIGN / 'code/run_query_ceb_baselines.py'
DIRECTORY = CAMPAIGN / 'outputs' / TASK / 'lolbo/seed0'
ORIGINAL_SHA = 'a502931e374faab41661c0b678d627e4799a868e060603c7f2ce03631851d54a'
FIXED_SHA = '34f232633fed7afc802dc83ede0170de62673480012472047339eef9de84be5b'
ERROR = 'could not read file "base/pgsql_tmp/pgsql_tmp57480.0": Input/output error'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frozen():
    sys.path.insert(0, str(CAMPAIGN / 'code'))
    spec = importlib.util.spec_from_file_location('frozen_query_fleet', CAMPAIGN / 'code/query_train800_fleet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical(cpus):
    return {tuple((Path('/sys/devices/system/cpu') / f'cpu{cpu}/topology' / field).read_text().strip()
                  for field in ('physical_package_id', 'core_id')) for cpu in cpus}


def resources(module):
    plan, worker = module.allocation(CAMPAIGN, NAME)
    database_pid = module.postgres_pid(worker)
    if database_pid != read(CAMPAIGN / 'databases' / (NAME + '.json'))['pid']:
        raise RuntimeError('Original PostgreSQL identity changed')
    if os.sched_getaffinity(database_pid) != set(worker['db_cpus']):
        raise RuntimeError('Original database CPU allocation changed')
    processes = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] in ('Z', 'X'):
                continue
            processes[int(path.name)] = dict(parent=int(stat[1]), cpus=os.sched_getaffinity(int(path.name)))
        except OSError:
            continue
    database_family = {database_pid}
    while True:
        children = {pid for pid, p in processes.items() if p['parent'] in database_family}
        if children <= database_family:
            break
        database_family |= children
    intended = physical([worker['bo_cpu']] + worker['db_cpus'])
    for pid, process in processes.items():
        if pid in database_family or pid in (os.getpid(), os.getppid()):
            continue
        if len(process['cpus']) < os.cpu_count() and intended & physical(process['cpus']):
            raise RuntimeError('CPU allocation occupied by PID ' + str(pid))
    endpoint = f"127.0.0.1:{worker['port']}:imdb"
    lock_id = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    endpoint_lock = Path('/tmp') / ('ceb_baseline_db_' + lock_id + '.lock')
    if not endpoint_lock.is_file():
        raise RuntimeError('Original database endpoint lock is missing')
    with module.lock(endpoint_lock):
        pass
    usage = subprocess.run(['nvidia-smi', '-i', str(worker['gpu']), '--query-gpu=memory.used',
                            '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    if int(usage.stdout.strip()) > 512:
        raise RuntimeError('GPU 0 is already occupied')
    return plan, worker


def verify_release():
    release = read(ROOT / 'release.json')
    for path, expected in release['inputs'].items():
        if digest(path) != expected:
            raise RuntimeError('Recovery input changed: ' + path)
    return release


def environment(module, worker):
    env = module.environment(worker)
    cache = ROOT / 'runtime'
    overrides = dict(TMPDIR=str(cache / 'tmp'), TMP=str(cache / 'tmp'), TEMP=str(cache / 'tmp'),
                     XDG_CACHE_HOME=str(cache / 'cache'), TORCH_HOME=str(cache / 'torch'),
                     TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
                     CUDA_CACHE_PATH=str(cache / 'cuda'), HF_HOME=str(cache / 'huggingface'),
                     MPLCONFIGDIR=str(cache / 'matplotlib'), PYTHONPATH=str(CAMPAIGN / 'code'))
    env.update(overrides)
    return env


def command(module, root, worker, task, output, budget=1049, stop_after_batches=None):
    if worker['name'] != NAME or Path(root) != CAMPAIGN:
        raise ValueError('Recovery wrapper is restricted to LLMOptimizer3.w00')
    result = module.run_command(root, worker, task, output, budget, stop_after_batches)
    result[1] = str(FIXED)
    result += ['--resume-runner-sha256', ORIGINAL_SHA]
    return result


def stage():
    module = frozen()
    plan, worker = resources(module)
    if (ROOT / 'release.json').exists():
        print(json.dumps(verify_release(), indent=2))
        return
    if digest(ORIGINAL) != ORIGINAL_SHA or digest(FIXED) != FIXED_SHA:
        raise RuntimeError('Runner hash differs from the reviewed recovery release')
    with module.lock(CAMPAIGN / 'workers' / (NAME + '.lock')), module.lock(DIRECTORY / '.lock'):
        status = read(DIRECTORY / 'status.json')
        if (status['status'] != 'failed' or status['proposals'] != 605 or status['error'] != ERROR
                or status['runner_sha256'] != ORIGINAL_SHA or status['unresolved_database_calls'] != 1):
            raise RuntimeError('The failed task has changed since the audit')
        paths = sorted(DIRECTORY.glob('proposal_[0-9][0-9][0-9][0-9].json'))
        if len(paths) != 605 or [read(p)['proposal_index'] for p in paths] != list(range(605)):
            raise RuntimeError('Saved proposal count or numbering differs')
        pending = DIRECTORY / 'proposal_0605.pending.json'
        if list(DIRECTORY.glob('proposal_*.pending.json')) != [pending]:
            raise RuntimeError('Unexpected pending proposals')
        log = CAMPAIGN / 'logs' / (NAME + '.' + TASK + '.run.log')
        if 'psycopg.errors.IoError: ' + ERROR not in log.read_text():
            raise RuntimeError('No confirmed I/O error evidence')
        before = {str(path.relative_to(DIRECTORY)): digest(path) for path in paths}
        before.update({name: digest(DIRECTORY / name) for name in
                       ('checkpoint.pt', 'history.json', 'status.json', pending.name)})
        archive = ROOT / 'before_recovery.tar.gz'
        if archive.exists():
            raise RuntimeError('An unregistered backup already exists')
        with tarfile.open(archive, 'x:gz') as stream:
            for path in (DIRECTORY, log, CAMPAIGN / 'workers' / (NAME + '.json'),
                         CAMPAIGN / 'claims' / (TASK + '.json')):
                stream.add(path, arcname=str(path.relative_to(CAMPAIGN)))
        with tarfile.open(archive, 'r:gz') as stream:
            for relative, expected in before.items():
                name = str(DIRECTORY.relative_to(CAMPAIGN) / relative)
                if hashlib.sha256(stream.extractfile(name).read()).hexdigest() != expected:
                    raise RuntimeError('Backup verification failed: ' + relative)
        evidence = dict(worker=NAME, task=TASK, proposal_index=605,
                        error=ERROR, log=str(log), log_sha256=digest(log),
                        archive=str(archive), archive_sha256=digest(archive),
                        pending_sha256=digest(pending))
        ledger = DIRECTORY / 'proposal_0605.database_attempts.json'
        if ledger.exists():
            raise RuntimeError('An unreviewed database attempt ledger exists')
        module.write(ledger, [dict(proposal=read(pending), status='io_error', sqlstate='58030',
                                  error=ERROR, elapsed_seconds=None, started_at=None, finished_at=None,
                                  historical_recovery=True, evidence=evidence)])
        module.write(ROOT / 'before_hashes.json', before)
        for name in ('tmp', 'cache', 'torch', 'inductor', 'triton', 'cuda', 'huggingface', 'matplotlib'):
            (ROOT / 'runtime' / name).mkdir(parents=True, exist_ok=True)
        release = dict(worker=worker, task=TASK, staged_at=module.stamp(), evidence=evidence,
                       policy='Resume the exact saved proposal; preserve 605 results and original scientific settings',
                       inputs={str(Path(__file__).resolve()): digest(__file__), str(FIXED): FIXED_SHA,
                               str(ORIGINAL): ORIGINAL_SHA,
                               str(CAMPAIGN / 'code/query_train800_fleet.py'):
                                   plan['files_sha256']['code/query_train800_fleet.py']})
        module.write(ROOT / 'release.json', release)
    print(json.dumps(release, indent=2))


def work():
    verify_release()
    module = frozen()
    resources(module)
    original_command = module.run_command
    original_environment = module.environment

    class Originals:
        run_command = staticmethod(original_command)
        environment = staticmethod(original_environment)

    module.run_command = lambda *args, **kwargs: command(Originals, *args, **kwargs)
    module.environment = lambda worker: environment(Originals, worker)
    module.run_worker(CAMPAIGN, NAME)


def launch():
    release = verify_release()
    module = frozen()
    resources(module)
    with module.lock(CAMPAIGN / 'workers' / (NAME + '.lock')):
        pass
    launch_path = ROOT / 'launch.json'
    if launch_path.exists():
        raise RuntimeError('This recovery release already has a launch record')
    with (CAMPAIGN / 'logs' / (NAME + '.supervisor.log')).open('a') as stream:
        process = subprocess.Popen([module.PYTHON, str(Path(__file__).resolve()), 'worker', '--name', NAME],
                                   cwd=CAMPAIGN, env=environment(module, release['worker']),
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    record = dict(worker=NAME, pid=process.pid, launched_at=module.stamp(),
                  wrapper=str(Path(__file__).resolve()), release=str(ROOT / 'release.json'),
                  log=str(CAMPAIGN / 'logs' / (NAME + '.supervisor.log')))
    module.write(launch_path, record)
    path = CAMPAIGN / 'launch.LLMOptimizer3.production.json'
    previous = read(path)
    previous['workers'] = [record if row['worker'] == NAME else row for row in previous['workers']]
    previous['updated_at'] = module.stamp()
    module.write(path, previous)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'launch', 'worker'))
    parser.add_argument('--name', default=NAME, choices=(NAME,))
    args = parser.parse_args()
    {'stage': stage, 'launch': launch, 'worker': work}[args.action]()
