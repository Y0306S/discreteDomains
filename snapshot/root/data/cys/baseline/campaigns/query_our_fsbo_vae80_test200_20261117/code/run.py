import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from query_protocol import atomic,digest
from query_data import plan_key
from method import Journal,warm_start,optimize,best
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/root/data/cys/envs/bolt_clean_venv/bin/python'
def recipe():return json.loads((ROOT/'recipe.json').read_text())
def identity(stage,task):return dict(recipe_sha256=digest(ROOT/'recipe.json'),stage=stage,task=task)
def validate_completion(stage,task):
 p=ROOT/stage/'runs'/task/'complete.json'
 if not p.exists():return False
 r=json.loads(p.read_text());expected=50 if stage=='warm_start' else 80
 if any(r.get(k)!=v for k,v in identity(stage,task).items()) or r['attempts']!=expected:raise ValueError('Completion identity changed '+task)
 return True

def freeze():
 cfg=recipe();pools={}
 for t in cfg['tasks']:
  if not validate_completion('warm_start',t):raise ValueError('Incomplete pool '+t)
  rows=json.loads((ROOT/'warm_start/runs'/t/'complete.json').read_text())['rows']
  if len(rows)!=50 or len({plan_key(r['plan']) for r in rows})!=50 or best(rows) is None:raise ValueError('Invalid pool '+t)
  pools[t]=rows
 result=dict(recipe_sha256=digest(ROOT/'recipe.json'),tasks=pools,shared_for_later_RL=True)
 p=ROOT/'pools/frozen_pool.json'
 if p.exists() and json.loads(p.read_text())!=result:raise ValueError('Pool cannot be replaced')
 atomic(p,result);atomic(ROOT/'pools/identity.json',dict(sha256=digest(p)))

def run_task(stage,task):
 from backends import LLM,oracle,FSBO
 cfg=recipe();folder=ROOT/stage/'runs'/task;folder.mkdir(parents=True,exist_ok=True)
 journal=Journal(folder/'runtime.sqlite',identity(stage,task));fsbo=None
 try:
  llm=LLM(cfg,stage,'http://127.0.0.1:'+('21810' if stage=='warm_start' else '21811'))
  evaluate=oracle(task,cfg['timeout_seconds'])
  if stage=='warm_start':
   state=warm_start(journal,task,cfg,llm,evaluate)
   result=dict(identity(stage,task),attempts=50,generation_attempts=state['attempts'],rows=state['rows'],rejected=state['rejected'])
  else:
   path=ROOT/'pools/frozen_pool.json'
   if digest(path)!=json.loads((ROOT/'pools/identity.json').read_text())['sha256']:raise ValueError('Frozen pool changed')
   record=json.loads(path.read_text())
   if record['recipe_sha256']!=digest(ROOT/'recipe.json'):raise ValueError('Pool recipe changed')
   pool=record['tasks'][task];fsbo=FSBO(cfg,task,pool)
   state=optimize(journal,task,cfg,pool,llm,evaluate,fsbo)
   result=dict(identity(stage,task),attempts=len(state['events']),rows=state['rows'],events=state['events'],pool_sha256=digest(path),initial_best_runtime=best(pool))
  result.update(status='complete',best_runtime=best(result['rows']),at=time.time());atomic(folder/'complete.json',result)
  print(json.dumps(dict(stage=stage,task=task,status='complete',best_runtime=result['best_runtime'])),flush=True)
 finally:
  journal.close()
  if fsbo is not None:fsbo.close()

def worker(stage,index):
 os.sched_setaffinity(0,{24+index});cfg=recipe()
 for task in cfg['tasks']:
  if (ROOT/'PAUSE').exists():return
  folder=ROOT/stage/'runs'/task;folder.mkdir(parents=True,exist_ok=True)
  with (folder/'task.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if validate_completion(stage,task):continue
   atomic(ROOT/'services'/f'worker{index}.status.json',dict(stage=stage,task=task,pid=os.getpid(),status='running',at=time.time(),port=20402+index))
   try:run_task(stage,task)
   except BaseException as exc:
    atomic(folder/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));(ROOT/'PAUSE').touch();raise
 atomic(ROOT/'services'/f'worker{index}.status.json',dict(stage=stage,pid=os.getpid(),status='idle',at=time.time()))

def status():
 cfg=recipe();stages={}
 for stage in cfg['stages']:
  ps=list((ROOT/stage/'runs').glob('*/progress.json'));progress=[json.loads(p.read_text()) for p in ps]
  stages[stage]=dict(complete_tasks=sum(validate_completion(stage,t) for t in cfg['tasks']),started_tasks=len(ps),measurements=sum(r['rows']-(50 if stage=='orpo' else 0) for r in progress),generation_attempts_or_optimization_attempts=sum(r['attempts'] for r in progress))
 return dict(stages=stages,paused=(ROOT/'PAUSE').exists(),rl_enabled=False,at=time.time())

def controller():
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  preflight=json.loads((ROOT/'evidence/preflight.json').read_text())
  tests=json.loads((ROOT/'evidence/tests.json').read_text())
  if not preflight['passed'] or preflight['recipe_sha256']!=digest(ROOT/'recipe.json') or not tests['passed']:raise ValueError('Preflight/tests required')
  # Pin current implementation and all frozen model backend dependencies.
  pins=json.loads((ROOT/'evidence/code_pins.json').read_text())
  for p,h in pins.items():
   if digest(p)!=h:raise ValueError('Pinned code changed '+p)
  for stage in ['warm_start','orpo']:
   if stage=='orpo':freeze()
   if (ROOT/'PAUSE').exists():raise RuntimeError('Paused for audit')
   jobs=[]
   for i in range(8):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',DB_HOST='127.0.0.1',DB_PORT=str(20402+i),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb')
    log=(ROOT/'logs'/f'{stage}.worker{i}.log').open('ab')
    p=subprocess.Popen([PYTHON,'-u',str(ROOT/'code/run.py'),'worker','--stage',stage,'--index',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append((p,log));atomic(ROOT/'services'/f'worker{i}.process.json',dict(pid=p.pid,stage=stage,at=time.time()))
   while any(p.poll() is None for p,_ in jobs):
    s=status();s.update(status='paused_draining' if s['paused'] else 'running',stage=stage,worker_processes=[p.pid for p,_ in jobs if p.poll() is None]);atomic(ROOT/'status.json',s);time.sleep(10)
   for p,log in jobs:log.close()
   if any(p.returncode for p,_ in jobs) or (ROOT/'PAUSE').exists():
    atomic(ROOT/'status.json',dict(status(),status='paused',stage=stage));raise RuntimeError('Worker failure: inspect task error.json and logs')
   if not all(validate_completion(stage,t) for t in recipe()['tasks']):raise ValueError('Stage incomplete')
  atomic(ROOT/'status.json',dict(status(),status='complete'))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['controller','worker','status','freeze']);parser.add_argument('--stage',choices=['warm_start','orpo']);parser.add_argument('--index',type=int)
 args=parser.parse_args()
 if args.command=='controller':controller()
 elif args.command=='worker':worker(args.stage,args.index)
 elif args.command=='freeze':freeze()
 else:print(json.dumps(status(),indent=2))
