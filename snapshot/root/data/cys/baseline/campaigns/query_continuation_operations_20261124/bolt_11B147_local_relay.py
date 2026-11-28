#!/usr/bin/env python3
"""Run the audited CEB_11B147 continuation on LLMOptimizer2 via DB tunnel."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import psycopg2

OPS = Path(__file__).resolve().parent
ROOT = OPS.parent / 'query_bolt30_lolbo_total800_20261123'
TASK = 'CEB_11B147'
FOLDER = ROOT / 'runs' / TASK
REPORT = Path('/root/reports/bolt_11B147_lock_recovery_20261125')
TASK_PAUSE = ROOT / 'task_pauses' / f'{TASK}.json'
SLOT_PAUSE = ROOT / 'slot_pauses' / 'worker05.PAUSE'
STATUS = REPORT / 'local_relay_status.json'


def record(**fields):
    value = dict(task=TASK, pid=os.getpid(), at=time.time(), **fields)
    temp = STATUS.with_name(STATUS.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(STATUS)


def check_database():
    with psycopg2.connect(host='127.0.0.1', port=23435, dbname='imdb',
                          user='imdb', password='imdb', connect_timeout=10) as db:
        with db.cursor() as cursor:
            cursor.execute('SHOW data_directory')
            actual = cursor.fetchone()[0]
            if actual != '/dev/shm/query_expand_20261119/w35':
                raise RuntimeError(f'Unexpected PostgreSQL data directory: {actual}')
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND state='active'")
            active = cursor.fetchone()[0]
            if active:
                raise RuntimeError(f'Dedicated PostgreSQL instance has {active} other active sessions')


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    if not TASK_PAUSE.exists() or not SLOT_PAUSE.exists():
        raise RuntimeError('Task and remote slot must remain paused during local relay')
    with (FOLDER / 'task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_database()
        os.environ.update(DB_HOST='127.0.0.1', DB_PORT='23435', DB_NAME='imdb',
                          DB_USER='imdb', DB_PASSWORD='imdb', QUERY_LOLBO_VAE_OFFSET='5',
                          CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        os.sched_setaffinity(0, {64})
        TASK_PAUSE.write_text(json.dumps(dict(status='local_relay_running', task=TASK,
                                               host='LLMOptimizer2', db_host='LLMOptimizer3',
                                               at=time.time()), indent=2))
        record(status='running', port=23435, saved_rounds=32)
        try:
            sys.path.insert(0, str(ROOT))
            import run_lolbo
            from bolt_proposal_recovery import install
            install(run_lolbo, TASK, OPS)
            run_lolbo.run_task(TASK)
            complete = json.loads((FOLDER / 'complete.json').read_text())
            if complete.get('status') != 'complete' or len(complete['rows']) != 800:
                raise RuntimeError('Task returned without 800 completed evaluations')
            record(status='complete', rows=800, best_runtime=complete['best_runtime'])
            error = FOLDER / 'error.json'
            if error.exists():
                shutil.copy2(error, REPORT / 'historical_error.json')
                error.unlink()
            TASK_PAUSE.unlink(missing_ok=True)
        except BaseException as exc:
            record(status='failed', error=repr(exc), traceback=traceback.format_exc())
            TASK_PAUSE.write_text(json.dumps(dict(status='local_relay_failed', task=TASK,
                                                   reason=repr(exc), requires_audit_before_retry=True,
                                                   at=time.time()), indent=2))
            raise
        finally:
            # Keep the SQL slot productive; the task pause prevents unsafe automatic replay.
            SLOT_PAUSE.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
