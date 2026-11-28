"""Finite BOLT30-to-800 fleet; admission follows a committed ten-plan canary."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEM=ROOT.parent/'semanticopt_query_400_pilot800_20261123'
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
def load(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default
def atomic(p,v):
 p=Path(p);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(dict(at=time.time(),**v),indent=2)+'\n');q.replace(p)
def slot(number):
 cfg=load(ROOT/'recipe.json');spec=next(w for w in cfg['workers'] if w['slot']==number)
 os.sched_setaffinity(0,{spec['optimizer_cpu']})
 os.environ.update(DB_HOST='127.0.0.1',DB_PORT=str(spec['db_port']),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb',CUDA_VISIBLE_DEVICES='',QUERY_LOLBO_VAE_OFFSET=str(number),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 sys.path.insert(0,str(ROOT));import run_lolbo as r
 owner=(ROOT/'services'/f'slot{number}.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 statefile=ROOT/'services'/f'worker{number:02}.status.json'
 while not (ROOT/'PAUSE').exists():
  found=None
  for task in cfg['tasks']:
   folder=ROOT/'runs'/task;folder.mkdir(exist_ok=True)
   if (folder/'complete.json').exists():continue
   lease=(folder/'task.lock').open('a')
   try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lease.close();continue
   if (folder/'complete.json').exists():lease.close();continue
   found=(task,lease);break
  if found is None:atomic(statefile,dict(status='idle_complete',slot=number,host=spec['host']));return
  task,lease=found;atomic(statefile,dict(status='running',slot=number,host=spec['host'],task=task,port=spec['db_port'],pid=os.getpid()))
  try:r.run_task(task)
  except BaseException as e:
   error=dict(status='failed',slot=number,task=task,error=repr(e),traceback=traceback.format_exc());atomic(ROOT/'runs'/task/'error.json',error);atomic(statefile,error);(ROOT/'PAUSE').touch();raise
  finally:lease.close()
 atomic(statefile,dict(status='paused',slot=number))
def controller():
 owner=(ROOT/'control/controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time()
 while True:
  canary=load(ROOT/'evidence/canary.json')
  atomic(ROOT/'status.json',dict(status='canary_running',sql_workers=1,vae_replicas=8,warm_evaluations=6000,new_evaluation_target=154000,progress=load(ROOT/'runs/CEB_10A32/progress.json',{}),proposal_progress=load(ROOT/'runs/CEB_10A32/proposal_progress.json',{})))
  if canary:
   if not canary['passed']:raise RuntimeError('BOLT canary failed: '+str(canary))
   break
  if time.time()-started>1800:raise TimeoutError('BOLT canary exceeded thirty minutes')
  time.sleep(5)
 cfg=load(ROOT/'recipe.json');children=[]
 for spec in cfg['workers']:
  import shlex
  cmd=['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',spec['host'],shlex.join([PY,'-u',str(ROOT/'control/run.py'),'slot','--slot',str(spec['slot'])])]
  log=(ROOT/'logs'/f'slot{spec["slot"]:02}.log').open('ab',buffering=0);p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children.append((spec,p))
 atomic(ROOT/'control/workers.process.json',dict(workers=[dict(slot=s['slot'],host=s['host'],pid=p.pid) for s,p in children]))
 while True:
  summaries=[load(ROOT/'runs'/t/'progress.json',{}) for t in cfg['tasks']]
  n=sum(s.get('rows',30) for s in summaries);complete=sum((ROOT/'runs'/t/'complete.json').exists() for t in cfg['tasks'])
  failures=[dict(slot=s['slot'],exit_code=p.returncode) for s,p in children if p.poll() not in (None,0)]
  state='failed' if failures else 'complete' if complete==200 else 'paused' if (ROOT/'PAUSE').exists() else 'running'
  atomic(ROOT/'status.json',dict(status=state,completed_tasks=complete,total_tasks=200,evaluations=n,warm_evaluations=6000,new_evaluations=n-6000,new_evaluation_target=154000,live_worker_slots=sum(p.poll() is None for _,p in children),vae_replicas=8,canary_passed=True,failures=failures,elapsed_seconds=time.time()-started))
  if failures:(ROOT/'PAUSE').touch();raise RuntimeError('BOLT worker failed; inspect its task journal and log')
  if complete==200:return
  if not any(p.poll() is None for _,p in children):raise RuntimeError('No live BOLT workers but tasks remain')
  time.sleep(10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['controller','slot']);p.add_argument('--slot',type=int);a=p.parse_args()
 try:controller() if a.command=='controller' else slot(a.slot)
 except BaseException as e:
  if a.command=='controller':atomic(ROOT/'control/error.json',dict(error=repr(e),traceback=traceback.format_exc()));atomic(ROOT/'status.json',dict(status='failed',error=repr(e)))
  raise
