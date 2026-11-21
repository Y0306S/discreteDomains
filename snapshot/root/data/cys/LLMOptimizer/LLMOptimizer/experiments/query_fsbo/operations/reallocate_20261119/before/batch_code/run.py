import argparse,fcntl,json,os,subprocess,sys,time,traceback,shlex
from pathlib import Path
from query_protocol import atomic,digest
from query_data import plan_key
from method import Journal,optimize,best,validate_pool,SafePause
from warm import warm_start
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/root/data/cys/envs/bolt_clean_venv/bin/python'
def recipe():return json.loads((ROOT/'batch_recipe.json').read_text())
def resources():return json.loads((ROOT/'resources.json').read_text())
def allocation(index):return next(w for w in resources()['workers'] if w['index']==index)
def identity(stage,task):
 result=dict(recipe_sha256=digest(ROOT/'batch_recipe.json'),policy_sha256=digest(ROOT/'batch_policy.json'),stage=stage,task=task)
 if stage=='orpo_batch10':result['pool_sha256']=digest(ROOT/'pools/batch10_frozen_pool.json')
 return result
def validate_completion(stage,task):
 p=ROOT/stage/'runs'/task/'complete.json'
 if not p.exists():return False
 r=json.loads(p.read_text());expected=30 if stage=='warm30' else 77
 if any(r.get(k)!=v for k,v in identity(stage,task).items()) or r['attempts']!=expected:raise ValueError('Completion identity changed '+task)
 return True

def freeze():
 cfg=recipe();pools={}
 for t in cfg['tasks']:
  if not validate_completion('warm30',t):raise ValueError('Incomplete pool '+t)
  rows=json.loads((ROOT/'warm30/runs'/t/'complete.json').read_text())['rows']
  validate_pool(t,rows,30)
  pools[t]=rows
 result=dict(recipe_sha256=digest(ROOT/'batch_recipe.json'),tasks=pools,shared_for_later_RL=True)
 p=ROOT/'pools/batch10_frozen_pool.json'
 if p.exists() and json.loads(p.read_text())!=result:raise ValueError('Pool cannot be replaced')
 atomic(p,result);atomic(ROOT/'pools/batch10_identity.json',dict(sha256=digest(p)))

def run_task(stage,task):
 from backends import LLM,oracle,FSBO
 cfg=recipe();policy=json.loads((ROOT/'batch_policy.json').read_text());folder=ROOT/stage/'runs'/task;folder.mkdir(parents=True,exist_ok=True)
 journal=Journal(folder/'runtime.sqlite',identity(stage,task));fsbo=None
 try:
  llm=LLM(cfg,'warm_start' if stage=='warm30' else 'orpo','http://127.0.0.1:'+('21810' if stage=='warm30' else '21811'))
  evaluate=oracle(task,cfg['timeout_seconds'])
  if stage=='warm30':
   state=warm_start(journal,task,cfg,llm,evaluate,paused=lambda:(ROOT/'PAUSE').exists())
   result=dict(identity(stage,task),attempts=30,generation_attempts=state['attempts'],rows=state['rows'],rejected=state['rejected'])
  else:
   path=ROOT/'pools/batch10_frozen_pool.json'
   if digest(path)!=json.loads((ROOT/'pools/batch10_identity.json').read_text())['sha256']:raise ValueError('Frozen pool changed')
   record=json.loads(path.read_text())
   if record['recipe_sha256']!=digest(ROOT/'batch_recipe.json'):raise ValueError('Pool recipe changed')
   pool=record['tasks'][task];fsbo=FSBO(cfg,task,pool)
   from retrieval import retrieve
   state=optimize(journal,task,cfg,policy,pool,llm,evaluate,fsbo,retrieve,paused=lambda:(ROOT/'PAUSE').exists())
   result=dict(identity(stage,task),attempts=len(state['events']),rows=state['rows'],events=state['events'],pool_sha256=digest(path),initial_best_runtime=best(pool))
  result.update(status='complete',best_runtime=best(result['rows']),at=time.time());atomic(folder/'complete.json',result)
  print(json.dumps(dict(stage=stage,task=task,status='complete',best_runtime=result['best_runtime'])),flush=True)
 finally:
  journal.close()
  if fsbo is not None:fsbo.close()

def worker(stage,index):
 spec=allocation(index);os.sched_setaffinity(0,{spec['optimizer_cpu']});cfg=recipe()
 # A controller transport failure leaves remote workers alive only until the
 # shared PAUSE is observed at their next durable boundary.
 atomic(ROOT/'services'/f'batch_worker{index}.process.json',dict(pid=os.getpid(),host=spec['host'],stage=stage,launch_id=os.environ.get('QUERY_WORKER_LAUNCH_ID'),start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],at=time.time()))
 for task in cfg['tasks']:
  if (ROOT/'PAUSE').exists():return
  folder=ROOT/stage/'runs'/task;folder.mkdir(parents=True,exist_ok=True)
  with (folder/'task.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if validate_completion(stage,task):continue
   atomic(ROOT/'services'/f'batch_worker{index}.status.json',dict(stage=stage,task=task,pid=os.getpid(),host=spec['host'],status='running',at=time.time(),port=spec['db_port'],optimizer_cpu=spec['optimizer_cpu'],postgres_cpus=spec['postgres_cpus']))
   try:run_task(stage,task)
   except SafePause:
    atomic(ROOT/'services'/f'batch_worker{index}.status.json',dict(stage=stage,task=task,pid=os.getpid(),host=spec['host'],status='paused_at_durable_boundary',at=time.time()));return
   except BaseException as exc:
    atomic(folder/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));(ROOT/'PAUSE').touch();raise
 atomic(ROOT/'services'/f'batch_worker{index}.status.json',dict(stage=stage,pid=os.getpid(),host=spec['host'],status='idle',at=time.time()))

def worker_command(stage,spec,launch_id='manual'):
 env=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',DB_HOST='127.0.0.1',DB_PORT=str(spec['db_port']),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb')
 env['QUERY_WORKER_LAUNCH_ID']=launch_id
 cmd=['env']+[k+'='+v for k,v in env.items()]+[PYTHON,'-u',str(ROOT/'batch_code/run.py'),'worker','--stage',stage,'--index',str(spec['index'])]
 if spec['host']!='LLMOptimizer2':return ['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',spec['host'],shlex.join(cmd)]
 return cmd

def status():
 cfg=recipe();stages={}
 for stage in cfg['stages']:
  ps=list((ROOT/stage/'runs').glob('*/progress.json'));progress=[json.loads(p.read_text()) for p in ps]
  stages[stage]=dict(complete_tasks=sum(validate_completion(stage,t) for t in cfg['tasks']),started_tasks=len(ps),measurements=sum(r['rows']-(30 if stage=='orpo_batch10' else 0) for r in progress),generation_attempts_or_optimization_attempts=sum(r['attempts'] for r in progress))
 return dict(stages=stages,total_rounds=80,batch_size=10,evaluations_per_task=800,total_tasks=200,measurement_target=160000,counted_measurements=sum(v['measurements'] for v in stages.values()),paused=(ROOT/'PAUSE').exists(),rl_enabled=False,at=time.time())

def controller():
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  preflight=json.loads((ROOT/'evidence/batch_preflight.json').read_text())
  tests=json.loads((ROOT/'evidence/batch_tests.json').read_text())
  if not preflight['passed'] or preflight['recipe_sha256']!=digest(ROOT/'batch_recipe.json') or preflight['policy_sha256']!=digest(ROOT/'batch_policy.json') or not tests['passed']:raise ValueError('Preflight/tests required')
  if 21 not in preflight['batches'] or preflight.get('fsbo_demonstrated_plans',0)<1:raise ValueError('Native FSBO batch decoding preflight required')
  context=json.loads((ROOT/'evidence/batch_context_smoke.json').read_text())
  if not context['passed'] or context['input_tokens']<28000:raise ValueError('Long-context preflight required')
  # Pin current implementation and all frozen model backend dependencies.
  pins=json.loads((ROOT/'evidence/batch_code_pins.json').read_text())
  for p,h in pins.items():
   if digest(p)!=h:raise ValueError('Pinned code changed '+p)
  from resources import health,inference_stage
  for stage in ['warm30','orpo_batch10']:
   if stage=='orpo_batch10':freeze()
   if (ROOT/'PAUSE').exists():raise RuntimeError('Paused for audit')
   if all(validate_completion(stage,t) for t in recipe()['tasks']):continue
   atomic(ROOT/'status.json',dict(status(),status='loading_inference',stage=stage))
   inference_stage(stage);health(require_workers=False)
   jobs=[]
   for spec in resources()['workers']:
    i=spec['index']
    log=(ROOT/'logs'/f'batch10.{stage}.worker{i}.log').open('ab')
    launch_id=f'{os.getpid()}:{stage}:{i}:{time.time_ns()}'
    p=subprocess.Popen(worker_command(stage,spec,launch_id),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append((p,log))
    if spec['host']!='LLMOptimizer2':
     # Avoid sshd MaxStartups admission failures from fifteen simultaneous
     # unauthenticated connections. The shared acknowledgement proves startup,
     # rather than assuming a fixed delay was long enough.
     deadline=time.monotonic()+90
     while True:
      record=ROOT/'services'/f'batch_worker{i}.process.json'
      if record.exists() and json.loads(record.read_text()).get('launch_id')==launch_id:break
      if p.poll() is not None or time.monotonic()>deadline:
       atomic(ROOT/'services/resource_error.json',dict(error='Remote worker startup failed',worker=i,returncode=p.poll(),at=time.time()));(ROOT/'PAUSE').touch();break
      time.sleep(.2)
     if (ROOT/'PAUSE').exists():break
   while any(p.poll() is None for p,_ in jobs):
    if any(p.poll() not in (None,0) for p,_ in jobs):(ROOT/'PAUSE').touch()
    try:health(require_workers=False,deep=False)
    except Exception as exc:
     atomic(ROOT/'services/resource_error.json',dict(error=repr(exc),at=time.time()));(ROOT/'PAUSE').touch()
    s=status();s.update(status='paused_draining' if s['paused'] else 'running',stage=stage,worker_launchers=[p.pid for p,_ in jobs if p.poll() is None],worker_count=sum(p.poll() is None for p,_ in jobs),worker_hosts={h:sum(p.poll() is None for spec,(p,_) in zip(resources()['workers'],jobs) if spec['host']==h) for h in ['LLMOptimizer2','LLMOptimizer4']},resources_sha256=digest(ROOT/'resources.json'));atomic(ROOT/'status.json',s);time.sleep(10)
   for p,log in jobs:log.close()
   if any(p.returncode for p,_ in jobs) or (ROOT/'PAUSE').exists():
    atomic(ROOT/'status.json',dict(status(),status='paused',stage=stage));raise RuntimeError('Worker failure: inspect task error.json and logs')
   if not all(validate_completion(stage,t) for t in recipe()['tasks']):raise ValueError('Stage incomplete')
  atomic(ROOT/'status.json',dict(status(),status='complete'))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['controller','worker','status','freeze']);parser.add_argument('--stage',choices=['warm30','orpo_batch10']);parser.add_argument('--index',type=int)
 args=parser.parse_args()
 if args.command=='controller':controller()
 elif args.command=='worker':worker(args.stage,args.index)
 elif args.command=='freeze':freeze()
 else:print(json.dumps(status(),indent=2))
