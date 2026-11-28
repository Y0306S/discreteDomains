"""Finite durable continuation: twenty private SQL slots, per-task leases, audited resume."""
import argparse,fcntl,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'

def load(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default

def atomic(p,v):
 p=Path(p);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(dict(at=time.time(),**v),indent=2)+'\n');q.replace(p)

def health(port):
 try:
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f'http://127.0.0.1:{port}/health',timeout=3) as r:return json.load(r)
 except Exception:return None

def process_alive(record):
 try:
  stat=Path(f'/proc/{record["pid"]}/stat').read_text().rsplit(') ',1)[1].split();return stat[0]!='Z' and stat[19]==record['start_ticks']
 except FileNotFoundError:return False

def slot(number):
 cfg=load(ROOT/'allocation.json');spec=next(w for w in cfg['workers'] if w['slot']==number);ext=load(ROOT/'extension.json')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 own=(ROOT/'leases'/f'slot{number}.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Pilot queries first; long historical queries first outside the pilot.
 source=Path(ext['source']);other=[t for t in ext['tasks'] if t not in ext['pilot_tasks']]
 def cost(task):
  s=load(source/'query/runs'/task/'status.json',{});return s.get('elapsed_seconds',0)
 tasks=ext['pilot_tasks']+sorted(other,key=cost,reverse=True)
 status=ROOT/'query'/f'worker{number:02}.status.json'
 while not (ROOT/'PAUSE').exists():
  picked=None
  for task in tasks:
   target=ext['pilot_budget'] if task in ext['pilot_tasks'] else ext['default_budget']
   state=load(ROOT/'query/runs'/task/'status.json',{})
   if state.get('status')=='complete' and state.get('evaluations')==target:continue
   lease=(ROOT/'leases'/(task+'.lock')).open('a')
   try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lease.close();continue
   state=load(ROOT/'query/runs'/task/'status.json',{})
   if state.get('status')=='complete' and state.get('evaluations')==target:lease.close();continue
   picked=(task,target,lease);break
  if picked is None:atomic(status,dict(status='idle_complete',slot=number,host=spec['host']));return
  task,target,lease=picked
  atomic(status,dict(status='running',slot=number,host=spec['host'],task=task,budget=target,port=spec['db_port']))
  logpath=ROOT/'logs'/f'{task}.log'
  with logpath.open('ab',buffering=0) as log:
   p=subprocess.Popen(['taskset','-c',str(spec['optimizer_cpu']),PY,'-u',str(ROOT/'code/evaluate.py'),'query',task,'--port',str(spec['db_port'])],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   code=p.wait()
  lease.close()
  state=load(ROOT/'query/runs'/task/'status.json',{})
  if code or state.get('status')!='complete' or state.get('evaluations')!=target:
   error=dict(status='failed',slot=number,task=task,exit_code=code,log=str(logpath))
   atomic(status,error);atomic(ROOT/'query'/f'worker{number:02}.error.json',error);(ROOT/'PAUSE').touch();raise RuntimeError(str(error))
 atomic(status,dict(status='paused',slot=number,host=spec['host']))

def controller():
 lock=(ROOT/'control/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 started=time.time();deadline=started+1800
 while True:
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=20) as pool:states=list(pool.map(health,range(23100,23120)))
  ready=sum(bool(s and s.get('ready')) for s in states)
  atomic(ROOT/'status.json',dict(status='loading_inference',ready_llm_replicas=ready,target_llm_replicas=20,retained_evaluations=40000))
  if ready==20:break
  if time.time()>deadline:raise TimeoutError('LLM pool loading exceeded 30 minutes; inspect replica logs')
  time.sleep(5)
 subprocess.run([sys.executable,str(ROOT/'control/start_services.py'),'LLMOptimizer2','--broker'],check=True,env=env,stdout=subprocess.DEVNULL)
 deadline=time.time()+120
 while not (health(23090) or {}).get('ready'):
  if time.time()>deadline:raise TimeoutError('Broker readiness')
  time.sleep(2)
 cfg=load(ROOT/'allocation.json');children=[]
 for spec in cfg['workers']:
  cmd=[PY,'-u',str(ROOT/'control/run.py'),'slot','--slot',str(spec['slot'])]
  if spec['host']!='LLMOptimizer2':
   import shlex
   cmd=['ssh','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',spec['host'],shlex.join(cmd)]
  log=(ROOT/'logs'/f'slot{spec["slot"]:02}.log').open('ab',buffering=0)
  p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);log.close();children.append((spec,p))
 atomic(ROOT/'control/workers.process.json',dict(workers=[dict(slot=s['slot'],host=s['host'],pid=p.pid) for s,p in children]))
 registered=set(range(12))
 while True:
  for index in range(12,20):
   if index in registered:continue
   candidate=health(23100+index)
   if candidate and candidate.get('ready'):
    body=json.dumps(dict(url=f'http://127.0.0.1:{23100+index}')).encode()
    request=urllib.request.Request('http://127.0.0.1:23090/add_replica',data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=15) as response:result=json.load(response)
    if result.get('added'):registered.add(index)
  summaries=[load(ROOT/'query/runs'/t/'status.json',{}) for t in load(ROOT/'extension.json')['tasks']]
  n=sum(s.get('evaluations',200) for s in summaries);complete=sum(s.get('status')=='complete' and s.get('evaluations')==s.get('budget') for s in summaries)
  failures=[dict(slot=s['slot'],exit_code=p.returncode) for s,p in children if p.poll() not in (None,0)]
  state='failed' if failures else 'complete' if complete==200 else 'paused' if (ROOT/'PAUSE').exists() else 'running'
  atomic(ROOT/'status.json',dict(status=state,completed_tasks=complete,total_tasks=200,evaluations=n,retained_evaluations=40000,new_evaluations=n-40000,new_evaluation_target=44000,live_worker_slots=sum(p.poll() is None for _,p in children),llm_replicas=len(registered),failures=failures,elapsed_seconds=time.time()-started))
  if failures:(ROOT/'PAUSE').touch();raise RuntimeError('SQL slot failed; see slot/task log')
  if complete==200:return
  if not any(p.poll() is None for _,p in children):raise RuntimeError('No active workers but campaign is incomplete')
  time.sleep(10)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['controller','slot']);p.add_argument('--slot',type=int);a=p.parse_args()
 try:controller() if a.command=='controller' else slot(a.slot)
 except BaseException as e:
  if a.command=='controller':atomic(ROOT/'control/error.json',dict(error=repr(e)));atomic(ROOT/'status.json',dict(status='failed',error=repr(e)))
  raise
