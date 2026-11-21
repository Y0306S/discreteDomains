"""Launch fresh-base one-GPU three-epoch BOLT full SFT with bounded storage and GPU admission."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from train import ROOT,load,atomic,sha,checkpoint_files,audit
QUERY=ROOT.parent/'query_our_orpo_rl800_20261112';sys.path.insert(0,str(QUERY/'code'))
from resources import reserve,cpu_clear,gpu_free,host_free
GPU='/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
PREVIOUS=ROOT.parent/'peptide_warmstart_full800_20261111'
def env(gpu=''):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DISABLE_VERSION_CHECK='1',WANDB_MODE='disabled',HF_HOME=str(ROOT/'hf_cache'),HF_DATASETS_CACHE=str(ROOT/'hf_cache/datasets'),TMPDIR=str(ROOT/'scratch'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'scratch/torchinductor'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(PREVIOUS/'scratch/torch_extensions'),PATH=str(PREVIOUS/'runtime_deps/bin')+os.pathsep+os.environ['PATH'],PYTHONPATH=os.pathsep.join([str(ROOT/'code'),str(ROOT/'runtime_deps'),str(PREVIOUS/'runtime_deps')]))
def publish(**v):atomic(ROOT/'status.json',dict(pid=os.getpid(),at=time.time(),**v))
def identity(final=False):
 names=[ROOT/'code/train.py',ROOT/'code/queue_training.py',ROOT/'code/trusted_rng.py',ROOT/'data/train.jsonl',ROOT/'data/provenance.jsonl',ROOT/'data/manifest.json',ROOT/'data/dataset_info.json',ROOT/'evidence/data_audit.json']
 base=Path(load(ROOT/'sft_config.json')['model_name_or_path']);names += [base/n for n in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'] if (base/n).exists()]
 if final:names+=[ROOT/'sft_config.json',ROOT/'deepspeed_config.json',ROOT/'evidence/preflight.json']
 atomic(ROOT/'identity.json',dict(files={str(p):sha(p) for p in names},at=time.time(),stage='training' if final else 'preparation'))
def run(name,command,gpu=''):
 with (ROOT/'logs'/f'{name}.log').open('ab') as log:p=subprocess.Popen(command,env=env(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 atomic(ROOT/'services'/f'{name}_process.json',dict(pid=p.pid,start_ticks=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19],command=command,at=time.time()))
 try:
  while p.poll() is None:publish(status='training' if name=='sft' else 'preparing',stage=name,child_pid=p.pid);time.sleep(10)
 except BaseException:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  raise
 if p.returncode:raise RuntimeError(name+' failed '+str(p.returncode)+'; see logs/'+name+'.log')
def storage_required():
 # Free bytes already exclude a retained checkpoint. Credit at most one
 # verified complete checkpoint against the two-checkpoint save budget.
 retained=0.
 for path in (ROOT/'model').glob('checkpoint-*'):
  try:
   checkpoint_files(path,int(path.name.split('-')[-1]))
   retained=max(retained,sum(p.stat().st_size for p in path.rglob('*') if p.is_file())/2**30)
  except (OSError,ValueError,KeyError):pass
 return (2*119+18+16+55)-min(retained,119.)

def main():
 lock=(ROOT/'services/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def stop(*_):raise KeyboardInterrupt('SFT queue interrupted')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  while not (ROOT/'evidence/data_audit.json').exists():publish(status='preparing',stage='waiting_for_data_audit');time.sleep(10)
  identity()
  gate=load(ROOT/'evidence/preflight.json') if (ROOT/'evidence/preflight.json').exists() else {}
  if (gate.get('passed') and gate.get('config_sha256')==sha(ROOT/'sft_config.json')
      and gate.get('deepspeed_sha256')==sha(ROOT/'deepspeed_config.json')):
   audit()
   atomic(ROOT/'evidence/preflight_reuse.json',dict(at=time.time(),passed=True,reason='Identical data, tokenizer, preprocessing recipe and optimizer configuration; prior complete token/CPUAdam preflight retained'))
  else:run('preflight',[GPU,'-u',str(ROOT/'code/train.py'),'prepare'])
  identity(final=True)
  # Two overlapping 119-GiB resumable checkpoints + 18-GiB final export,
  # 16 GiB PostgreSQL temp allowance and >=55 GiB unrelated-work margin.
  atomic(ROOT/'evidence/storage_budget.json',dict(checkpoint_gib=119,peak_overlapping_checkpoints=2,final_export_gib=18,postgres_temp_allowance_gib=16,unrelated_margin_gib=55,minimum_free_gib=327,save_total_limit=1))
  reserve(sft=True)
  while True:
   free=gpu_free();host=host_free();ready=free[3]>=36 and host['ram_free_gib']>=220 and host['disk_free_gib']>=storage_required() and host['free_inodes']>=5000 and cpu_clear([42,43,44,45,46,47])
   if ready:break
   publish(status='queued',stage='waiting_for_SFT_resources',gpu_free_gib=free,**host);time.sleep(20)
  command=['taskset','-c','42-47',GPU,'-m','torch.distributed.run','--nproc_per_node=1','--master_addr=127.0.0.1','--master_port=19897']
  if os.environ.get('SFT_NATIVE_DEBUG')=='1':
   command+=['--no-python','gdb','-batch','--return-child-result','-ex','set pagination off','-ex','set debuginfod enabled off','-ex','set print thread-events off','-ex','handle SIGALRM nostop noprint pass','-ex','handle SIGPIPE nostop noprint pass','-ex','run','-ex','thread apply all bt 20','-ex','bt full','--args',GPU,'-u']
  command += [str(ROOT/'code/train.py'),'train']
  run('sft',command,'3')
  result=load(ROOT/'training_status.json')
  if result['status']!='completed':raise ValueError('Missing completed full-SFT evidence')
  publish(status='completed',stage='finished',optimizer_steps=result['step'],epochs=3,model_dir=result['model_dir'],final_weights_evidence=result['final_weights_evidence'])
 finally:reserve(sft=True,release=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:publish(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'error',error=repr(e));raise
