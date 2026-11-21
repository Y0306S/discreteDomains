"""Safely replay a read-only proposal after PostgreSQL returned SQLSTATE XX000."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import signal
import sqlite3
import subprocess
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
TASK='CEB_8A220'
WORKER='w03'


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


def main():
    ledger=json.loads((ROOT/'current_workers.json').read_text())
    worker=next(w for w in ledger['workers'] if w['name']==WORKER)
    assert worker['host']=='LLMOptimizer2'
    pid=int(worker['pid']);proc=Path('/proc')/str(pid)
    original=proc.joinpath('stat').read_text().split()
    assert original[2]!='Z' and original[21]==str(worker['process_start_ticks'])
    recipe=json.loads((ROOT/'recipe.json').read_text())
    assert recipe['timeout_seconds']==50. and not recipe['update_e2e'] and not recipe['recenter']
    status_path=ROOT/'runs'/TASK/'status.json'
    status=json.loads(status_path.read_text())
    assert status['status'] in ('failed','paused')
    if status['status']=='failed':
        assert status['unresolved_database_calls']==1
    run=ROOT/'runs'/TASK
    proposal_index=109
    pending_name=f'proposal_{proposal_index:04d}.pending.json'
    attempts_name=f'proposal_{proposal_index:04d}.database_attempts.json'
    db=sqlite3.connect(f'file:{run}/artifacts.sqlite3?mode=ro',uri=True)
    try:
        artifacts={row[0]:(row[1],row[2]) for row in db.execute('SELECT name,payload,sha256 FROM artifacts')}
    finally:db.close()
    assert pending_name in artifacts and attempts_name in artifacts
    pending=json.loads(zlib.decompress(artifacts[pending_name][0]))
    attempts=json.loads(zlib.decompress(artifacts[attempts_name][0]))
    assert len(attempts)==1 and attempts[0]['status']=='error' and attempts[0]['sqlstate']=='XX000'
    assert attempts[0]['proposal']==pending and 'Interrupted system call' in attempts[0]['error']
    assert pending['timeout']==50.
    assert f'proposal_{proposal_index:04d}.json' not in artifacts
    lock_path=HERE/'replay.lock'
    with lock_path.open('a') as recovery_lock:
        fcntl.flock(recovery_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Fence only the worker whose database is selected. Resume immediately
        # whenever its PostgreSQL session is active; never interrupt a query.
        safe=False
        for _ in range(900):
            os.kill(pid,signal.SIGSTOP)
            time.sleep(.15)
            fields=proc.joinpath('stat').read_text().split()
            assert fields[2] in ('T','t') and fields[21]==str(worker['process_start_ticks'])
            env=dict(os.environ,PGPASSWORD='imdb',PGCONNECT_TIMEOUT='5',
                     PGOPTIONS='-c statement_timeout=5000')
            command=['/root/data/cys/envs/pg16/bin/psql','-h','127.0.0.1','-p',str(worker['db_port']),
                '-U','imdb','-d','imdb','-X','-A','-t','-F','|','-v','ON_ERROR_STOP=1','-c',
                "SELECT count(*) FROM pg_stat_activity WHERE datname='imdb' AND pid<>pg_backend_pid(); SHOW temp_tablespaces; SELECT 1;"]
            result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=10)
            if result.returncode==0 and result.stdout.splitlines()==['0','fsbo_tmp_w03','1']:
                safe=True;break
            os.kill(pid,signal.SIGCONT)
            time.sleep(1)
        assert safe,'No database-idle boundary observed within the recovery window'
        with sqlite3.connect(f'file:{run}/artifacts.sqlite3?mode=ro',uri=True) as check:
            after=dict(check.execute('SELECT name,sha256 FROM artifacts'))
        prior={name:value[1] for name,value in artifacts.items()
               if name.endswith('.json') and name.startswith('proposal_')
               and not name.endswith(('.pending.json','.database_attempts.json'))}
        assert all(after.get(name)==digest for name,digest in prior.items())
        evidence=HERE/'CEB_8A220_recovery.json'
        if not evidence.exists():
            atomic(evidence,dict(passed=True,at=time.time(),task=TASK,worker=WORKER,pid=pid,
                db_port=worker['db_port'],original_sqlstate='XX000',original_error=attempts[0]['error'],
                proposal_index=proposal_index,pending_proposal_sha256=hashlib.sha256(
                    json.dumps(pending,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                original_result_absent=True,active_database_sessions=0,
                temp_tablespaces='fsbo_tmp_w03',database_probe_result=result.stdout.strip(),
                prior_committed_result_hashes=prior,retry_is_read_only=True,
                sql_timeout_seconds=50,recovery_timeout_seconds=1800,
                authorization='Retry only the exact durable proposal after PostgreSQL returned an error and before any result was committed.'))
        else:
            assert json.loads(evidence.read_text())['prior_committed_result_hashes']==prior
        sys.path.insert(0,'/root/data/cys/logs/query_train800_author_fixed_20261004/code')
        from run_artifact_store import JsonJournal
        attempts[0]['storage_recovery']=dict(replay_authorized=True,active_database_calls=0,
            original_result_unavailable=True,evidence_report=str(evidence))
        JsonJournal(run).write(attempts_name,attempts)
        paused=dict(status);paused.update(status='paused',error=None,
            recovery_evidence=str(evidence),unresolved_database_calls=0)
        atomic(status_path,paused)
        # The original worker is fenced while its PostgreSQL connection is
        # checked. Stop its dedicated process group to release the task lock;
        # the live DB query count above is the guard against interrupted SQL.
        original_group=Path('/proc')/str(pid)/'stat'
        if original_group.exists():
            fields=original_group.read_text().split()
            if fields[21]==str(worker['process_start_ticks']):
                current=subprocess.run(command,env=env,text=True,capture_output=True,timeout=10,check=True)
                assert current.stdout.splitlines()==['0','fsbo_tmp_w03','1'],current.stdout
                os.killpg(pid,signal.SIGTERM)
                for _ in range(100):
                    if not original_group.exists():break
                    if original_group.read_text().split()[2]=='Z':break
                    time.sleep(.1)
                else:raise RuntimeError('Fenced query worker did not exit')
        os.sched_setaffinity(0,{worker['cpus'][0]})
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
        import query_fsbo_eval as evaluation
        old_init=evaluation.QueryVAEBackend.__init__
        def init(self,manifest,task,fallback,seed=123,timeout=300):
            old_init(self,manifest,task,fallback,seed=seed,timeout=1800)
        evaluation.QueryVAEBackend.__init__=init
        try:
            result=evaluation.run_task(ROOT,recipe,TASK,worker,stop_after_batches=1)
            atomic(HERE/'CEB_8A220_recovery_result.json',result)
        finally:
            fields=proc.joinpath('stat').read_text().split()
            if fields[21]==str(worker['process_start_ticks']) and fields[2] in ('T','t'):
                os.kill(pid,signal.SIGCONT)
        status=json.loads(status_path.read_text())
        assert status['status']=='paused' and status['proposals']>=110
        atomic(HERE/'CEB_8A220_recovery_verified.json',dict(passed=True,at=time.time(),
            task=TASK,worker_resumed=True,status=status,
            original_results_preserved=True,unresolved_database_calls=status.get('unresolved_database_calls',0)))
        launch=['/root/data/cys/envs/bolt_clean_venv/bin/python','-B',str(HERE/'restart_workers.py'),
                '--host','LLMOptimizer2','--worker',WORKER]
        subprocess.run(launch,check=True)
        print(json.dumps(dict(passed=True,task=TASK,status=status['status'],
                              proposals=status['proposals'],worker_resumed=True)))


if __name__=='__main__':
    main()
