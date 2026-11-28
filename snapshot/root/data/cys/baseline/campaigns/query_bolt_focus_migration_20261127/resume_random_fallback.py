"""Requeue only the two audited tasks after staging complete proposal batches."""
import fcntl
import pickle
import sqlite3
from common import *
from prepare_random_fallback import OUT,TASKS,measurements


def main():
    policy_path=ROOT/'random_feasible_fallback_20261128.json'
    staged=load(OUT/'staged.json');baseline=load(OUT/'baseline.json')
    assert staged['policy_sha256']==sha(policy_path) and staged['sql_calls']==0
    assert sha(BOLT/'recipe.json')==baseline['recipe_sha256']
    assert sha(BOLT/'run_lolbo.py')==baseline['runner_sha256']
    assert not (BOLT/'PAUSE').exists()
    locks=[];archives={}
    for task,(count,number) in TASKS.items():
        folder=BOLT/'runs'/task
        handle=(folder/'task.lock').open('a')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        pause=BOLT/'task_pauses'/f'{task}.json'
        archive=OUT/task/'pause.removed_on_resume.json'
        assert pause.exists() and not archive.exists()
        with sqlite3.connect((folder/'runtime.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            assert measurements(db)==baseline['tasks'][task]['completed_measurements']
            state=pickle.loads(db.execute("SELECT value FROM kv WHERE key='lolbo_state'").fetchone()[0])
            proposal=pickle.loads(db.execute('SELECT value FROM kv WHERE key=?',(f'proposal/{number}',)).fetchone()[0])
            assert len(state['rows'])==count
            assert proposal['status']=='complete'
            batch=proposal['result']
            assert len(batch['items'])==10 and batch['fit']['random_fallback_count']==10
            assert batch['fit']['proposal_policy_sha256']==sha(policy_path)
        archives[task]=(pause,archive)
    assert sorted(p.stem for p in (BOLT/'task_pauses').glob('*.json') if p.stem not in TASKS)==sorted(baseline['other_pauses'])
    atomic(OUT/'resume.intent.json',dict(tasks=list(TASKS),policy_sha256=sha(policy_path),at=time.time()))
    for pause,archive in archives.values():pause.rename(archive)
    remaining=sorted(p.stem for p in (BOLT/'task_pauses').glob('*.json'))
    assert remaining==sorted(baseline['other_pauses'])
    result=dict(tasks={task:dict(saved_rows=count,target=800,status='queued') for task,(count,_) in TASKS.items()},
        remaining_pauses=remaining,policy_sha256=sha(policy_path),at=time.time())
    atomic(OUT/'resumed.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
