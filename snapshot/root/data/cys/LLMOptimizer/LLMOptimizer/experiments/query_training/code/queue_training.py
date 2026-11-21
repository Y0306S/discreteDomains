"""Audited query ORPO -> fresh-policy four-phase GRPO, durable and resumable."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from query_protocol import atomic,digest
from resources import reserve,cpu_clear,gpu_free,host_free
ROOT=Path(__file__).resolve().parents[1];CPU='/root/data/cys/envs/bolt_clean_venv/bin/python';GPU='/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
def load(p):return json.loads(Path(p).read_text())
def env(gpu=''):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4' if str(gpu) else '1',MKL_NUM_THREADS='4' if str(gpu) else '1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DISABLE_VERSION_CHECK='1',WANDB_MODE='disabled',HF_HOME=str(ROOT/'hf_cache'),HF_DATASETS_CACHE=str(ROOT/'hf_cache/datasets'),TMPDIR=str(ROOT/'scratch'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'scratch/torchinductor'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=os.pathsep.join([str(ROOT/'code'),str(ROOT/'runtime_deps'),str(ROOT/'query_deps')]))
def publish(**v):atomic(ROOT/'services/queue_status.json',dict(pid=os.getpid(),at=time.time(),**v))
def run(name,command,gpu='',gate=None):
 if gate and Path(gate).exists() and load(gate).get('passed'):return
 with (ROOT/'logs'/f'{name}.log').open('ab') as log:p=subprocess.Popen(command,env=env(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 atomic(ROOT/'services'/f'{name}_process.json',dict(pid=p.pid,start_ticks=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19],command=command,at=time.time()))
 try:
  while p.poll() is None:publish(status='running',stage=name,child_pid=p.pid);time.sleep(10)
 except BaseException:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  raise
 if p.returncode:raise RuntimeError(name+' failed with '+str(p.returncode)+'; see logs/'+name+'.log')
 if gate and not load(gate).get('passed'):raise RuntimeError('Gate failed: '+name)
def wait_gpu():
 while True:
  free=gpu_free();host=host_free()
  if free[0]>=54 and cpu_clear([30,31,32,33,34,35]) and host['disk_free_gib']>=35:return
  publish(status='queued',stage='waiting_for_GPU0_CPU_handoff',gpu_free_gib=free,**host);time.sleep(20)
def freeze():
 settings=load(ROOT/'settings.json');orpo=load(ROOT/'orpo_status.json')
 if orpo['status']!='completed' or digest(ROOT/'models/orpo/adapter_model.safetensors')!=orpo['adapter_sha256']:raise ValueError('ORPO parent missing or corrupt')
 settings.update(status='ready_to_launch',parent_adapter_sha256=orpo['adapter_sha256']);atomic(ROOT/'settings.json',settings)
 atomic(ROOT/'protocol.json',dict(domain='query',source_train_tasks=800,grpo_tasks=32,groups=384,cycles=1,phases=settings['phases'],training=settings['training'],parent_sha256=orpo['adapter_sha256'],censoring='Timeout bounds receive fixed non-improvement reward -0.5, never delta best or a fabricated measured loss',sampling=settings['sampling'],query_timeout_seconds=settings['query_timeout_seconds'],postgres_parallel_workers=2))
 gates={}
 for name,p in [('pairs',ROOT/'reports/independent_audit.json'),('tests',ROOT/'evidence/offline_tests.json'),('ORPO_tokens',ROOT/'orpo_preflight.json'),('GP_export',ROOT/'evidence/source_gp_export.json'),('retrieval',ROOT/'evidence/retrieval_preflight.json'),('RL_tokens',ROOT/'evidence/rl_token_preflight.json'),('CPU_oracle',ROOT/'evidence/cpu_oracle_check.json'),('exact_parent_backward',ROOT/'evidence/rl_parent_check.json'),('GRPO_scope',ROOT/'evidence/grpo_scope_check.json')]:
  if not load(p).get('passed'):raise ValueError('Missing gate: '+name)
  gates[name]=dict(status='passed',evidence=str(p),sha256=digest(p))
 atomic(ROOT/'launch_gates.json',dict(checks=gates));checks={}
 for directory in ['code','assets','query_deps']:
  for p in (ROOT/directory).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.json','.npz']:checks[str(p.relative_to(ROOT))]=digest(p)
 for name in ['settings.json','protocol.json','orpo_config.json','data/orpo_train.jsonl','data/dataset_info.json']:checks[name]=digest(ROOT/name)
 atomic(ROOT/'checksums.json',checks)
def main():
 lock=(ROOT/'services/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,{32,33,34,35})
 def stop(*_):raise KeyboardInterrupt('Queue interrupted')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  reserve()
  while not (ROOT/'manifest.json').exists():publish(status='preparing',stage='waiting_for_data_export');time.sleep(10)
  run('data_audit',[CPU,'-u',str(ROOT/'code/audit_query_data.py')],gate=ROOT/'reports/independent_audit.json')
  run('training_tests',[CPU,'-m','unittest','discover','-s',str(ROOT/'code'),'-p','test_query_training.py'])
  atomic(ROOT/'evidence/offline_tests.json',dict(passed=True,at=time.time(),tests=5))
  run('orpo_preflight',[GPU,'-u',str(ROOT/'code/ml_orpo_train.py'),'prepare','--campaign',str(ROOT)],gate=ROOT/'orpo_preflight.json')
  wait_gpu()
  run('orpo',[GPU,'-u',str(ROOT/'code/ml_orpo_train.py'),'train','--campaign',str(ROOT)],gpu=0)
  # GP preparation can run independently while the ORPO trainer is active.
  run('gp_export',[load(ROOT/'settings.json')['retrieval']['python'],'-u',str(ROOT/'code/export_source_gp.py')],gate=ROOT/'evidence/source_gp_export.json')
  run('retrieval',[CPU,'-u',str(ROOT/'code/prepare_retrieval.py')],gate=ROOT/'evidence/retrieval_preflight.json')
  run('rl_tokens',[GPU,'-u',str(ROOT/'code/validate_training.py'),'tokens'],gate=ROOT/'evidence/rl_token_preflight.json')
  run('cpu_oracle',[CPU,'-u',str(ROOT/'code/validate_training.py'),'oracle'],gate=ROOT/'evidence/cpu_oracle_check.json')
  wait_gpu();run('rl_parent',[GPU,'-u',str(ROOT/'code/validate_training.py'),'parent'],gpu=0,gate=ROOT/'evidence/rl_parent_check.json');freeze()
  run('rl_pipeline',[CPU,'-u',str(ROOT/'code/run_pipeline.py'),'--execute','--gpu','0','--workers','2','--port','19895','--min-free-gib','54'],gpu=0)
  if not (ROOT/'cycles/round1/complete.json').exists():raise ValueError('RL did not produce a complete cycle')
  publish(status='complete',ORPO=True,RL_cycles=1,cycle_checkpoint=str(ROOT/'cycles/round1'))
 except BaseException as e:publish(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'error',error=repr(e));raise
if __name__=='__main__':main()
