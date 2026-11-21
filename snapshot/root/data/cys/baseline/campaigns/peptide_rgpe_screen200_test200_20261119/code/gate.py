"""Release the finite queue only after a real train800-bank canary commits."""
import json,os,pickle,socket,sqlite3,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
    assert socket.gethostname()=='cs-94bac-0693e-server'
    recipe=json.loads((ROOT/'rgpe/recipe.json').read_text())
    import sys
    sys.path.insert(0,str(ROOT/'code'))
    from worker import check_pins
    check_pins()
    task=recipe['allowed_tasks'][0]
    record=json.loads((ROOT/'evidence/canary.process.json').read_text())
    while True:
        proc=Path('/proc')/str(record['pid'])
        alive=False
        if proc.exists():
            fields=proc.joinpath('stat').read_text().split()
            alive=fields[21]==record['start_ticks'] and fields[2]!='Z'
        if not alive:break
        time.sleep(10)
    path=ROOT/'rgpe/runs'/task
    history=json.loads((path/'history.json').read_text())
    telemetry=ROOT/'telemetry'/task
    fit=json.loads((telemetry/'rgpe_fit.json').read_text())['info']
    screen=json.loads((telemetry/'rgpe_screen.json').read_text())['info']
    sample=json.loads((telemetry/'rgpe_sample.json').read_text())['info']
    # Read the saved pending slot without touching RNG or writing the database.
    with sqlite3.connect('file:'+str(path/'runtime.sqlite')+'?mode=ro',uri=True) as db:
        pending=pickle.loads(db.execute('SELECT value FROM meta WHERE key="pending"').fetchone()[0])
    checks=dict(unit_screen=json.loads((ROOT/'evidence/screen_validation.json').read_text())['passed'],
        real_posterior_parity=json.loads((ROOT/'evidence/real_validation.json').read_text())['passed'],
        real_batch50=history['proposals']==50 and history['events']==1,
        real_oracle=history['fresh_evaluations']>0,
        pending_committed=pending is None,
        full_source_bank=fit['source_models']==800 and len(fit['weights'])==801,
        shortlist200=screen['shortlisted']==200 and screen['candidates']==1500,
        full_joint_covariance=sample['candidates']==200 and sample['covariance_mode']=='full',
        frozen_vae=history['vae_policy']=='frozen_author' and history['e2e_updates']==0,
        heldout_split=len(recipe['test'])==200 and len(recipe['train'])==800 and not(set(recipe['test'])&set(recipe['train'])))
    report=dict(at=time.time(),passed=all(checks.values()),checks=checks,task=task,
                proposals=history['proposals'],fresh_evaluations=history['fresh_evaluations'])
    if not report['passed']:
        atomic(ROOT/'evidence/canary_failed.json',report);raise ValueError(report)
    # Queue release waits for the exact same pending checkpoint to be reopened
    # by its production worker. No reset and no second evaluation of the canary.
    atomic(ROOT/'evidence/canary_passed.json',report)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    cmd=[recipe['model_python'],'-u',str(ROOT/'code/benchmark.py')]
    with (ROOT/'evidence/benchmark.log').open('a') as f:
        child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(ROOT/'evidence/benchmark.process.json',dict(pid=child.pid,cmd=cmd,at=time.time()))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
