"""Recover the explicitly recorded interrupted PostgreSQL temp-file operation."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1')
sys.path.insert(0, '/root/data/cys/logs/query_train800_author_fixed_20261004/code')
from run_artifact_store import JsonJournal

def atomic(path, value):
    temporary = path.with_name(path.name+'.recovery.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)

def main():
    task = 'CEB_2A593'
    run = CAMPAIGN/'runs'/task
    records = json.loads((ROOT/'production_processes.json').read_text())
    worker = next(w for w in records['workers'] if w['name'] == 'w02')
    proc = Path('/proc')/str(worker['pid'])
    if proc.exists():
        fields = (proc/'stat').read_text().split()
        assert fields[21] != worker['process_start_ticks'] or fields[2] == 'Z'
    reserve = json.loads((ROOT/'reservation.json').read_text())
    assert hashlib.sha256((CAMPAIGN/'recipe.json').read_bytes()).hexdigest() == reserve['recipe_sha256']
    assert not (ROOT/'STOP').exists() and not (CAMPAIGN/'STOP').exists()
    cg = Path('/sys/fs/cgroup')
    free = int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())
    assert free >= 240*2**30
    endpoint = f"127.0.0.1:{worker['db_port']}:imdb"
    db_lock = Path('/tmp')/('ceb_baseline_db_'+hashlib.sha256(endpoint.encode()).hexdigest()[:16]+'.lock')
    with db_lock.open('a') as db_handle, (CAMPAIGN/'locks'/(task+'.lock')).open('a') as task_handle:
        fcntl.flock(db_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(task_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, PGPASSWORD='imdb', PGCONNECT_TIMEOUT='5', PGOPTIONS='-c statement_timeout=5000')
        command = ['/root/data/cys/envs/pg16/bin/psql','-h','127.0.0.1','-p',str(worker['db_port']),
                   '-U','imdb','-d','imdb','-X','-A','-t','-F','|','-v','ON_ERROR_STOP=1']
        query = "SELECT count(*) FROM pg_stat_activity WHERE datname='imdb' AND pid<>pg_backend_pid(); SHOW temp_tablespaces; SELECT 1;"
        result = subprocess.run(command+['-c', query], env=env, text=True, capture_output=True, timeout=10, check=True)
        assert result.stdout.splitlines() == ['0','fsbo_tmp_w02','1'], result.stdout
        probe = "BEGIN; CREATE TEMP TABLE fsbo_recovery_probe AS SELECT generate_series(1,1000) AS n; SELECT sum(n) FROM fsbo_recovery_probe; ROLLBACK;"
        result = subprocess.run(command+['-c',probe], env=env, text=True, capture_output=True, timeout=10, check=True)
        assert '500500' in result.stdout
        log_path = Path(worker['pgdata'])/'fsbo_pilot_postgres.log'
        log = log_path.read_text()
        assert '[1778] ERROR:  could not open temporary file' in log and 'Interrupted system call' in log
        journal = JsonJournal(run)
        pending = journal.read('proposal_0286.pending.json')
        attempts = journal.read('proposal_0286.database_attempts.json')
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt['status'] == 'error' and attempt['sqlstate'] == 'XX000'
        assert attempt['proposal'] == pending and pending['timeout'] == 50
        assert 'could not open temporary file' in attempt['error'] and 'Interrupted system call' in attempt['error']
        assert attempt.get('finished_at') and not journal.exists('proposal_0286.json')
        status = json.loads((run/'status.json').read_text())
        assert status['status'] == 'failed' and status['proposals'] == 286
        hashes = {name:hashlib.sha256(json.dumps(journal.read(name),sort_keys=True).encode()).hexdigest()
                  for name in journal.names('proposal_*.json')
                  if not name.endswith(('.pending.json','.database_attempts.json'))}
        assert len(hashes) == 286
        report = ROOT/'w02_interrupted_tempfile_recovery.json'
        assert not report.exists(), 'Recovery already recorded'
        evidence = dict(at=time.time(), task=task, worker='w02', db_port=worker['db_port'],
                        proposal_index=286, original_error=attempt['error'], sqlstate='XX000',
                        reason='PostgreSQL explicitly returned an interrupted temporary-file error; no runtime result was produced. The original process exited and its database has no remaining sessions.',
                        replay_authorized=True, active_database_calls=0, original_result_unavailable=True,
                        database_ready=True, temp_tablespace_probe_passed=True, original_worker=worker,
                        committed_result_hashes=hashes,
                        checkpoint_sha256=hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest(),
                        recipe_sha256=reserve['recipe_sha256'], free_memory_gib=free/2**30)
        atomic(ROOT/'w02.status_before_recovery.json',status)
        atomic(ROOT/'w02.attempts_before_recovery.json',attempts)
        atomic(ROOT/'w02.history_before_recovery.json',json.loads((run/'history.json').read_text()))
        atomic(report,evidence)
        attempt['storage_recovery'] = dict(replay_authorized=True, active_database_calls=0,
                                          original_result_unavailable=True, evidence_report=str(report))
        journal.write('proposal_0286.database_attempts.json',attempts)
        status.update(status='paused', error=None, recovery_evidence=str(report))
        atomic(run/'status.json',status)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log = ROOT/'workers/w02'/('production.recovered_'+stamp+'.log')
    command = ['taskset','-c',str(worker['optimizer_cpu']),'/root/data/cys/envs/bolt_clean_venv/bin/python',
               '-u',str(ROOT/'run_worker.py'),'--worker','w02']
    with log.open('a') as out:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                   start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    updated = dict(worker, pid=process.pid, command=command, actual_command=command, log=str(log),
                   started_at=time.time(), process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                   recovery_evidence=str(report), previous_pid=worker['pid'], task=task)
    atomic(ROOT/'w02_recovery_process.json',updated)
    atomic(ROOT/'production_processes.before_w02_recovery.json',records)
    records['workers'] = [updated if w['name']=='w02' else w for w in records['workers']]
    records['w02_recovered_at'] = time.time()
    atomic(ROOT/'production_processes.json',records)
    ledger = json.loads((CAMPAIGN/'current_workers.json').read_text())
    atomic(ROOT/'current_workers.before_w02_recovery.json',ledger)
    assert next(w for w in ledger['workers'] if w['name']=='w02')['pid'] == worker['pid']
    ledger['workers'] = [updated if w['name']=='w02' else w for w in ledger['workers']]
    ledger['w02_recovered_at'] = time.time()
    atomic(CAMPAIGN/'current_workers.json',ledger)
    print(json.dumps(dict(pid=process.pid,worker='w02',task=task,log=str(log),recovery_evidence=str(report)),indent=2))

if __name__ == '__main__':
    main()
