"""Authorize replay only for a positively identified PostgreSQL ENOSPC failure."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import psycopg

ROOT = Path(__file__).resolve().parent
ORIGINAL_WORKERS = {'CEB_11A44':'w07', 'CEB_11A26':'w10', 'CEB_11A60':'w13',
                    'CEB_11A121':'w14', 'CEB_10A32':'w15',
                    'CEB_11A100':'w16', 'CEB_11A11':'w18'}
sys.path.insert(0, '/root/data/cys/logs/query_train800_author_fixed_20261004/code')
from run_artifact_store import JsonJournal


def recover(task, test_only=False):
    task_lock = (ROOT/'locks'/(task+'.lock')).open('a')
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = ROOT/'runs'/task
    journal = JsonJournal(run)
    pending_names = [name for name in journal.names('proposal_*.pending.json')]
    if len(pending_names) != 1:
        raise RuntimeError(f'{task}: expected exactly one pending query, found {pending_names}')
    pending_name = pending_names[0]
    number = pending_name.removeprefix('proposal_').removesuffix('.pending.json')
    attempts_name = f'proposal_{number}.database_attempts.json'
    attempts = journal.read(attempts_name)
    if len(attempts) != 1:
        raise RuntimeError(f'{task}: expected one original database attempt')
    attempt = attempts[0]
    pending = journal.read(pending_name)
    if (attempt['status'] != 'error' or attempt.get('sqlstate') != '53100'
            or attempt.get('proposal') != pending or pending.get('timeout') != 50.0):
        raise RuntimeError(f'{task}: not a recorded 50-second PostgreSQL ENOSPC failure')
    if test_only and task != 'CEB_11A44':
        raise RuntimeError('The one-worker recovery test is limited to CEB_11A44')
    worker_name = ORIGINAL_WORKERS[task]
    recipe = json.loads((ROOT/'recipe.json').read_text())
    worker = next(w for w in recipe['workers'] if w['name'] == worker_name)
    with psycopg.connect(host='127.0.0.1',port=worker['db_port'],dbname='imdb',user='imdb',
                         password='imdb',autocommit=True,connect_timeout=5) as connection:
        active = connection.execute("SELECT pid,state FROM pg_stat_activity WHERE datname='imdb' AND pid<>pg_backend_pid()").fetchall()
        if active:
            raise RuntimeError('Original query database still has sessions: '+str(active))
        tablespace = connection.execute('SHOW temp_tablespaces').fetchone()[0]
        if tablespace != 'fsbo_tmp_'+worker_name:
            raise RuntimeError('Disk-backed temporary tablespace is not active')
    evidence = dict(task=task, proposal_index=int(number), sqlstate='53100',
        original_error=attempt['error'], timeout_seconds=pending['timeout'],
        status='database_statement_failed_without_runtime_result',
        original_result_unavailable=True, active_database_calls=0,
        reason='PostgreSQL explicitly returned ENOSPC for the read-only plan; no measured runtime was returned or committed.',
        replay_authorized=True, test_only=test_only, checked_at=time.time(),
        worker=worker_name,db_port=worker['db_port'],temp_tablespaces=tablespace,
        committed_result_hashes={name:hashlib.sha256(json.dumps(journal.read(name),sort_keys=True).encode()).hexdigest()
            for name in journal.names('proposal_*.json')
            if not name.endswith(('.pending.json','.database_attempts.json'))})
    evidence_path = ROOT/'recovery'/f'{task}_proposal_{number}.json'
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads((run/'status.json').read_text())
    if status['status'] != 'failed':
        raise RuntimeError('Expected failed task before recovery')
    (evidence_path.parent/(task+'.status_before.json')).write_text(json.dumps(status,indent=2)+'\n')
    (evidence_path.parent/(task+'.attempts_before.json')).write_text(json.dumps(attempts,indent=2)+'\n')
    evidence_path.write_text(json.dumps(evidence, indent=2)+'\n')
    attempt['storage_recovery'] = dict(replay_authorized=True,
        active_database_calls=0, original_result_unavailable=True,
        evidence_report=str(evidence_path))
    journal.write(attempts_name, attempts)
    status.update(status='paused', error=None, recovery_evidence=str(evidence_path))
    temporary=run/'status.recovery.tmp'
    temporary.write_text(json.dumps(status,indent=2)+'\n')
    temporary.replace(run/'status.json')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(recover(args.task, args.test_only), indent=2))
