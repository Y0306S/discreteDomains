"""Authorize one verified replay of CEB_8A235's unresolved read-only query."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

ROOT=Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1')
HERE=Path(__file__).resolve().parent
EXPANSION=Path('/root/data/logs/query_fsbo_local_expansion_20261107_v1')
TASK='CEB_8A235';WORKER='w02';PROPOSAL=820
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0,'/root/data/cys/logs/query_train800_author_fixed_20261004/code')
import query_fsbo_eval as evaluation
from run_artifact_store import JsonJournal,read_bytes

def sha(data):return hashlib.sha256(data).hexdigest()

def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    os.replace(temp,path)

def main():
    recipe=evaluation.verify(ROOT)
    assert recipe['timeout_seconds']==50.0
    reservation=json.loads((EXPANSION/'reservation.json').read_text())
    assert evaluation.digest(ROOT/'recipe.json')==reservation['recipe_sha256']
    assert not (ROOT/'STOP').exists() and not (EXPANSION/'STOP').exists()
    worker=next(w for w in json.loads((ROOT/'current_workers.json').read_text())['workers']
                if w['name']==WORKER)
    assert worker['host']=='LLMOptimizer2' and worker['db_port']==19702
    old=Path('/proc')/str(worker['pid'])
    assert not old.exists(), 'Recorded worker process is still present'
    endpoint=f"127.0.0.1:{worker['db_port']}:imdb"
    lock_id=hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    db_lock=(Path('/tmp')/f'ceb_baseline_db_{lock_id}.lock').open('a')
    fcntl.flock(db_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    task_lock=(ROOT/'locks'/f'{TASK}.lock').open('a')
    fcntl.flock(task_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PGPASSWORD='imdb',PGCONNECT_TIMEOUT='5',
             PGOPTIONS='-c statement_timeout=5000')
    command=['/root/data/cys/envs/pg16/bin/psql','-h','127.0.0.1','-p',str(worker['db_port']),
        '-U','imdb','-d','imdb','-X','-A','-t','-F','|','-v','ON_ERROR_STOP=1','-c',
        "SELECT count(*) FROM pg_stat_activity WHERE datname='imdb' AND pid<>pg_backend_pid(); SHOW temp_tablespaces; SELECT 1;"]
    probe=subprocess.run(command,env=env,text=True,capture_output=True,timeout=10,check=True)
    assert probe.stdout.splitlines()==['0','fsbo_tmp_w02','1'],probe.stdout
    status_path=ROOT/'runs'/TASK/'status.json';status=json.loads(status_path.read_text())
    assert status['status']=='failed' and status['proposals']==PROPOSAL
    assert status['unresolved_database_calls']==1
    run=ROOT/'runs'/TASK;journal=JsonJournal(run)
    pending_name=f'proposal_{PROPOSAL:04d}.pending.json'
    attempts_name=f'proposal_{PROPOSAL:04d}.database_attempts.json'
    pending=journal.read(pending_name);attempts=journal.read(attempts_name)
    assert len(attempts)==1 and attempts[0]['status']=='error'
    assert attempts[0]['sqlstate']=='XX000' and 'Interrupted system call' in attempts[0]['error']
    assert attempts[0]['proposal']==pending and pending['timeout']==50.0
    assert not journal.exists(f'proposal_{PROPOSAL:04d}.json')
    import torch
    evaluation.peptide_turbo(recipe)
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu')
    assert checkpoint['meta']['task_id']==TASK and checkpoint['total']==1049
    assert len(checkpoint['completed'])==PROPOSAL
    assert checkpoint['pending'] and checkpoint['pending'][0]==pending
    committed={name:sha(read_bytes(run,name)) for name in journal.names('proposal_*.json')
        if not name.endswith(('.pending.json','.database_attempts.json'))}
    assert len(committed)==PROPOSAL
    evidence=HERE/'CEB_8A235_proposal_0820_replay.json'
    if evidence.exists():raise RuntimeError('This one-time replay is already authorized')
    record=dict(passed=True,authorized_at=time.time(),task=TASK,worker=WORKER,
        proposal_index=PROPOSAL,sqlstate='XX000',original_error=attempts[0]['error'],
        original_result_absent=True,pending_proposal_sha256=sha(json.dumps(pending,sort_keys=True,separators=(',',':')).encode()),
        checkpoint_completed_count=len(checkpoint['completed']),checkpoint_pending_match=True,
        active_database_calls=0,temp_tablespaces='fsbo_tmp_w02',
        database_probe=probe.stdout.strip(),sql_timeout_seconds=50.0,
        replay_limit=1,prior_committed_result_hashes=committed,
        authorization='User requested requeue; retry only this exact proposal after a recorded PostgreSQL error with no result or active query.')
    atomic(evidence,record)
    attempts[0]['storage_recovery']=dict(replay_authorized=True,active_database_calls=0,
        original_result_unavailable=True,evidence_report=str(evidence),replay_limit=1)
    journal.write(attempts_name,attempts)
    status.update(status='paused',error=None,unresolved_database_calls=0,
        recovery_evidence=str(evidence),replay_authorized=True)
    atomic(status_path,status)
    task_lock.close();db_lock.close()
    available=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
    assert available >= 232*2**30, f'Reserved memory headroom unavailable: {available/2**30:.1f} GiB'
    command=['taskset','-c',str(worker['cpus'][0]),'/root/data/cys/envs/bolt_clean_venv/bin/python',
        '-u','-B',str(ROOT/'recovery_20261108/resilient_worker.py'),'--worker',WORKER,'--task',TASK]
    log=HERE/'w02_CEB_8A235_requeue.log'
    with log.open('a') as output:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    worker.update(previous_pid=worker['pid'],pid=child.pid,command=command,actual_command=command,
        log=str(log),started_at=time.time(),process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
        recovery_evidence=str(evidence),task=TASK)
    # Persist the updated worker ledger after process identity is recorded.
    ledger=json.loads((ROOT/'current_workers.json').read_text())
    next(w for w in ledger['workers'] if w['name']==WORKER).update(worker)
    atomic(ROOT/'current_workers.json',ledger)
    atomic(HERE/'requeue_launch.json',dict(at=time.time(),pid=child.pid,task=TASK,worker=WORKER,
        command=command,log=str(log),evidence=str(evidence),free_memory_gib=available/2**30))
    print(json.dumps(dict(status='requeued',task=TASK,worker=WORKER,pid=child.pid,
        proposal=PROPOSAL,sql_timeout_seconds=50,log=str(log))))

if __name__=='__main__':main()
