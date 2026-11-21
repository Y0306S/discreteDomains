"""Verify and requeue one failed FSBO task from its durable checkpoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

ROOT = Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1')
HERE = Path(__file__).resolve().parent
EXPANSION = Path('/root/data/logs/query_fsbo_local_expansion_20261107_v1')
EXPECTED = {
    'CEB_8A79': ('w01', 'XX000', 'Interrupted system call'),
    'CEB_8A139': ('w12', '53100', 'No space left on device'),
}


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def live(worker):
    try:
        fields = (Path('/proc') / str(worker['pid']) / 'stat').read_text().split()
        return fields[2] != 'Z' and fields[21] == str(worker['process_start_ticks'])
    except (FileNotFoundError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=EXPECTED, required=True)
    args = parser.parse_args()
    task = args.task
    worker_name, expected_sqlstate, error_text = EXPECTED[task]

    sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
    sys.path.insert(0, '/root/data/cys/logs/query_train800_author_fixed_20261004/code')
    from run_artifact_store import JsonJournal
    import query_fsbo_eval as evaluation

    recipe = evaluation.verify(ROOT)
    assert recipe['timeout_seconds'] == 50.0
    reservation = json.loads((EXPANSION / 'reservation.json').read_text())
    assert evaluation.digest(ROOT / 'recipe.json') == reservation['recipe_sha256']
    assert not (ROOT / 'STOP').exists() and not (EXPANSION / 'STOP').exists()

    ledger = json.loads((ROOT / 'current_workers.json').read_text())
    worker = next(w for w in ledger['workers'] if w['name'] == worker_name)
    assert worker['host'] == ('LLMOptimizer2' if worker_name == 'w01' else 'LLMOptimizer4')
    assert not live(worker), f'{worker_name} already has a live recorded worker process'
    process_start = (Path('/proc') / str(os.getpid()) / 'stat').read_text().split()[21]

    run = ROOT / 'runs' / task
    status_path = run / 'status.json'
    status = json.loads(status_path.read_text())
    assert status['status'] == 'failed' and status['unresolved_database_calls'] == 1
    journal = JsonJournal(run)
    pending_names = journal.names('proposal_*.pending.json')
    assert len(pending_names) == 1, pending_names
    pending_name = pending_names[0]
    proposal_index = int(pending_name.removeprefix('proposal_').removesuffix('.pending.json'))
    attempts_name = f'proposal_{proposal_index:04d}.database_attempts.json'
    result_name = f'proposal_{proposal_index:04d}.json'
    pending = journal.read(pending_name)
    attempts = journal.read(attempts_name)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert (attempt['status'] == 'error' and attempt.get('sqlstate') == expected_sqlstate
            and error_text in attempt.get('error', '') and attempt.get('proposal') == pending)
    assert pending['timeout'] == 50.0 and not journal.exists(result_name)

    evaluation.peptide_turbo(recipe)
    import torch
    checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu')
    assert checkpoint['meta']['task_id'] == task and checkpoint['total'] == 1049
    assert len(checkpoint['completed']) <= status['proposals']
    assert checkpoint['pending']
    assert any(item == pending for item in checkpoint['pending'])

    endpoint = f"127.0.0.1:{worker['db_port']}:imdb"
    lock_id = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    db_lock = (Path('/tmp') / f'ceb_baseline_db_{lock_id}.lock').open('a')
    task_lock = (ROOT / 'locks' / f'{task}.lock').open('a')
    fcntl.flock(db_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    env = dict(os.environ, PGPASSWORD='imdb', PGCONNECT_TIMEOUT='5',
               PGOPTIONS='-c statement_timeout=5000')
    probe_cmd = [
        '/root/data/cys/envs/pg16/bin/psql', '-h', '127.0.0.1', '-p', str(worker['db_port']),
        '-U', 'imdb', '-d', 'imdb', '-X', '-A', '-t', '-v', 'ON_ERROR_STOP=1', '-c',
        "SELECT count(*) FROM pg_stat_activity WHERE datname='imdb' AND pid<>pg_backend_pid(); "
        'SHOW temp_tablespaces; SELECT 1;'
    ]
    probe = subprocess.run(probe_cmd, env=env, text=True, capture_output=True, timeout=10, check=True)
    expected_tablespace = f"fsbo_tmp_{worker_name}"
    assert probe.stdout.splitlines() == ['0', expected_tablespace, '1'], probe.stdout

    memory_max = int(Path('/sys/fs/cgroup/memory.max').read_text())
    memory_current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    free_memory = memory_max - memory_current
    minimum_memory = 232 if worker['host'] == 'LLMOptimizer2' else 32
    assert free_memory >= minimum_memory * 2**30, free_memory / 2**30
    shm_free = os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize
    assert shm_free >= 4 * 2**30, shm_free / 2**30

    with __import__('sqlite3').connect(f'file:{run}/artifacts.sqlite3?mode=ro', uri=True) as db:
        committed = {
            name: digest for name, digest in db.execute('SELECT name,sha256 FROM artifacts')
            if name.startswith('proposal_') and name.endswith('.json')
            and not name.endswith(('.pending.json', '.database_attempts.json'))
        }
    assert len(committed) == status['proposals']
    evidence = HERE / f'{task}_proposal_{proposal_index:04d}_replay.json'
    assert not evidence.exists(), f'replay already authorized: {evidence}'
    record = dict(
        passed=True, authorized_at=time.time(), task=task, worker=worker_name,
        proposal_index=proposal_index, sqlstate=expected_sqlstate, original_error=attempt['error'],
        original_result_absent=True, active_database_sessions=0,
        temp_tablespaces=expected_tablespace, database_probe=probe.stdout.strip(),
        sql_timeout_seconds=50.0, replay_limit=1, prior_committed_result_hashes=committed,
        pending_proposal_sha256=hashlib.sha256(
            json.dumps(pending, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        authorization='User requested requeue; replay only the exact failed read-only proposal once.'
    )
    atomic(evidence, record)
    attempt['storage_recovery'] = dict(replay_authorized=True, replay_limit=1,
        active_database_sessions=0, original_result_unavailable=True, evidence_report=str(evidence))
    journal.write(attempts_name, attempts)
    status.update(status='paused', error=None, unresolved_database_calls=0,
                  recovery_evidence=str(evidence), replay_authorized=True)
    atomic(status_path, status)
    db_lock.close()
    task_lock.close()

    command = [
        'taskset', '-c', str(worker['cpus'][0]),
        '/root/data/cys/envs/bolt_clean_venv/bin/python', '-u', '-B',
        str(ROOT / 'recovery_20261108' / 'resilient_worker.py'),
        '--worker', worker_name, '--task', task
    ]
    log = HERE / f'{worker_name}_{task}_requeue.log'
    with log.open('a') as output:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
    worker.update(previous_pid=worker.get('pid'), pid=child.pid, command=command,
        actual_command=command, log=str(log), started_at=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
        recovery_evidence=str(evidence), task=task)
    ledger = json.loads((ROOT / 'current_workers.json').read_text())
    next(w for w in ledger['workers'] if w['name'] == worker_name).update(worker)
    atomic(ROOT / 'current_workers.json', ledger)
    atomic(HERE / f'{task}_requeue_launch.json', dict(
        at=time.time(), task=task, worker=worker_name, pid=child.pid, command=command,
        log=str(log), evidence=str(evidence), free_memory_gib=free_memory / 2**30,
        shm_free_gib=shm_free / 2**30, process_start_ticks=process_start))
    print(json.dumps(dict(status='requeued', task=task, worker=worker_name,
                          pid=child.pid, proposal=proposal_index,
                          sql_timeout_seconds=50.0, log=str(log))))


if __name__ == '__main__':
    main()
