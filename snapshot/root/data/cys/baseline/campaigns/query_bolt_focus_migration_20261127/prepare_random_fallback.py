"""Back up the two paused journals and record the explicit policy amendment."""
import fcntl
import hashlib
import pickle
import sqlite3
from common import *

OUT=ROOT/'random_fallback_20261128'
TASKS={'CEB_2B326':(730,71),'CEB_2C202':(670,65)}

def measurements(db):
    rows=[]
    for key,value in db.execute("SELECT key,value FROM kv WHERE key LIKE 'measurement/%' ORDER BY key"):
        result=pickle.loads(value)
        if result['status']=='started':raise RuntimeError('Unresolved SQL '+key)
        rows.append((key,value))
    digest=hashlib.sha256()
    for key,value in rows:digest.update(key.encode()+b'\0'+value+b'\0')
    return dict(count=len(rows),sha256=digest.hexdigest())

def main():
    policy_path=ROOT/'random_feasible_fallback_20261128.json'
    assert not policy_path.exists(),'Policy already prepared; do not recreate'
    OUT.mkdir(exist_ok=True);locks=[];tasks={};audits={}
    for task,(expected_rows,expected_round) in TASKS.items():
        folder=BOLT/'runs'/task
        handle=(folder/'task.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        assert (BOLT/'task_pauses'/f'{task}.json').exists()
        pause=load(BOLT/'task_pauses'/f'{task}.json');progress=load(folder/'proposal_progress.json')
        assert progress['latent_batches']==load(BOLT/'recipe.json')['proposal_policy']['max_latent_batches']
        assert progress['round']==expected_round
        backup=OUT/task/'runtime.before.sqlite';backup.parent.mkdir(exist_ok=True)
        assert not backup.exists()
        with sqlite3.connect((folder/'runtime.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            state=pickle.loads(db.execute("SELECT value FROM kv WHERE key='lolbo_state'").fetchone()[0])
            assert len(state['rows'])==expected_rows and len(state['rounds'])+1==expected_round
            before=measurements(db)
            assert not db.execute("SELECT 1 FROM kv WHERE key LIKE ?",(f'measurement/{expected_round}/%',)).fetchone()
            proposal=pickle.loads(db.execute('SELECT value FROM kv WHERE key=?',(f'proposal/{expected_round}',)).fetchone()[0])
            assert proposal['status']=='started'
            with sqlite3.connect(backup) as dest:db.backup(dest)
        with sqlite3.connect(backup) as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        atomic(OUT/task/'original_pause.json',pause)
        atomic(OUT/task/'original_proposal_progress.json',progress)
        tasks[task]=dict(failed_round=expected_round,
            failed_proposal={k:progress[k] for k in ['round','latent_batches','latent_attempts','accepted','rejected','at']})
        audits[task]=dict(saved_rows=expected_rows,completed_measurements=before,
            backup=str(backup),backup_sha256=sha(backup))
    config=dict(version=1,authorization='User authorized random feasible fallback for these two tasks',
        scope='Only CEB_2B326 and CEB_2C202; replace their exhausted unmeasured batch, then latent-first with random top-up after bounded exhaustion',
        max_random_draws=10000,batch_size=10,total_evaluations_per_task=800,
        uniqueness='Against all prior observations and the entire current batch',
        feasibility='Strict native codec validation and no cross joins; ordinary measured SQL censoring remains unchanged',
        decoder_substitution_fallback='Still rejected',tasks=tasks,at=time.time())
    atomic(OUT/'baseline.json',dict(tasks=audits,recipe_sha256=sha(BOLT/'recipe.json'),
        runner_sha256=sha(BOLT/'run_lolbo.py'),other_pauses=[p.stem for p in (BOLT/'task_pauses').glob('*.json') if p.stem not in TASKS],at=time.time()))
    atomic(policy_path,config)
    print(json.dumps(dict(prepared_tasks=list(tasks),policy=str(policy_path),backed_up=True)))

if __name__=='__main__':main()
