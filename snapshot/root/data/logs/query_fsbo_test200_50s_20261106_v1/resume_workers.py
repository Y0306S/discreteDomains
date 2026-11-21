"""Resume stopped workers after the successful, recorded storage pilot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
pilot=json.loads((ROOT/'storage_pilot_validation.json').read_text())
assert pilot['passed'] and pilot['committed_results_unchanged']
configuration=json.loads((ROOT/'storage_configuration.json').read_text())
assert len(configuration['workers'])==16
launch=json.loads((ROOT/'launch.json').read_text())
record_path=ROOT/'storage_resume.json'
assert not record_path.exists(), 'A resume was already launched; inspect its recorded PIDs first'
stopped={'w07','w10','w13','w14','w15','w16','w18'}
locks=[]
for worker in launch['workers']:
    if worker['name'] not in stopped:
        continue
    old=Path('/proc')/str(worker['pid'])
    assert not old.exists() or old.joinpath('stat').read_text().split()[2]=='Z'
    endpoint='127.0.0.1:%s:imdb'%worker['db_port']
    path=Path('/tmp')/('ceb_baseline_db_'+hashlib.sha256(endpoint.encode()).hexdigest()[:16]+'.lock')
    lock=path.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks.append(lock)
assert all(json.loads(path.read_text())['status']!='failed' for path in (ROOT/'runs').glob('*/status.json'))
for lock in locks:
    lock.close()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
         MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
current=[]
restarted=[]
for worker in launch['workers']:
    if worker['name'] not in stopped:
        current.append(worker)
        continue
    command=worker['actual_command']
    log=ROOT/'logs'/(worker['name']+'.storage_resume.log')
    with log.open('ab',buffering=0) as output:
        process=subprocess.Popen(command,cwd='/root/data/cys/llmoptimizer_benchmark',
            env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    entry=dict(worker,previous_pid=worker['pid'],pid=process.pid,started_at=time.time(),
               log=str(log),process_start_ticks=Path('/proc/%s/stat'%process.pid).read_text().split()[21])
    restarted.append(entry)
    current.append(entry)
    record_path.write_text(json.dumps(dict(at=time.time(),restarted=restarted),indent=2)+'\n')
(ROOT/'current_workers.json').write_text(json.dumps(dict(at=time.time(),workers=current),indent=2)+'\n')
print(json.dumps(dict(restarted=[dict(worker=w['name'],pid=w['pid']) for w in restarted])))
