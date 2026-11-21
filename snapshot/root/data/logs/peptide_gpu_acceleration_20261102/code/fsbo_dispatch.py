"""Peptide FSBO slots; retain workers across dispatcher replacement."""
import fcntl,json,os,subprocess,time,signal,sys
from pathlib import Path
sys.path.insert(0,'/root/data/cys/baseline/campaigns/ml_ckpt32_fsbo_20261103/services')
from resource_admission import AdoptedProcess,resources,can_admit,admission_lock
ROOT=Path('/root/data/logs/peptide_gpu_acceleration_20261102')
C=Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo')
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
CONFIG='/root/data/logs/peptide_transfer_test200_20261028_v1/fsbo/gpu_runtime.json'
CPUS=[64,65,66,67,69];PORTS=[18871,18872,18884,18881,18883]
HOST='LLMOptimizer2'
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def cpu_usage():
 d=dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
 return int(d['usage_usec']),time.monotonic()
def free_cpu(cpu):
 siblings=set(map(int,Path('/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list'%cpu).read_text().strip().split(',')))
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   if (p/'stat').read_text().rsplit(') ',1)[1].startswith('Z'):continue
   affinity=os.sched_getaffinity(int(p.name))
   if len(affinity)<=32 and siblings.intersection(affinity):return False
  except OSError:pass
 return True
def main():
 (C/'dispatch/logs').mkdir(parents=True,exist_ok=True)
 lock=(C/'dispatch/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.sched_setaffinity(0,{68});tasks=json.loads((C/'recipe.json').read_text())['allowed_tasks'];active={};stopping=False
 status_path=C/'dispatch/status.json'
 if status_path.exists():
  for record in json.loads(status_path.read_text()).get('active',[]):
   child=AdoptedProcess(record['pid'],[str(ROOT/'code/fsbo_launch.py'),'--task '+record['task']],allow_exited=True)
   active[record['cpu']]=dict(record,child=child)
   print(json.dumps(dict(event='adopted',task=record['task'],pid=record['pid'])),flush=True)
 def stop(*a):
  nonlocal stopping
  stopping=True
  # Drain current tasks; a control-process signal does not stop evaluations.
 signal.signal(signal.SIGTERM,stop)
 signal.signal(signal.SIGINT,stop)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
 previous=cpu_usage()
 while True:
  completed=[t for t in tasks if (C/'runs'/(t+'.tar.gz')).exists()]
  for cpu,r in list(active.items()):
   rc=r['child'].poll()
   if rc is None:continue
   if rc or r['task'] not in completed:
    write(C/'dispatch/pause.json',dict(task=r['task'],returncode=rc,log=r['log'],at=time.time()))
   del active[cpu]
  paused=(C/'dispatch/pause.json').exists()
  current=cpu_usage();used=(current[0]-previous[0])/max(1e6,(current[1]-previous[1])*1e6);previous=current
  with admission_lock(HOST):
   report=resources(HOST)
   if not paused and not stopping:
    for cpu,port in zip(CPUS,PORTS):
     if cpu in active or not free_cpu(cpu):continue
     if used>=36 or not can_admit(report,'peptide'):break
     task=next((t for t in tasks if t not in completed and t not in [r['task'] for r in active.values()]),None)
     if task is None:break
     command=['nice','-n','10','taskset','-c',str(cpu),PY,str(ROOT/'code/fsbo_launch.py'),'--gpu-config',CONFIG,'run-task','--campaign',str(C),'--task',task,'--cpu',str(cpu),'--server',f'http://127.0.0.1:{port}']
     log=C/'dispatch/logs'/(task+'.log')
     with log.open('ab') as s:child=subprocess.Popen(command,stdout=s,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
     active[cpu]=dict(task=task,pid=child.pid,child=child,log=str(log),cpu=cpu,started=time.time())
     report['reserved_memory_gib']+=report['worker_reservation_gib']['peptide'];used+=1
     print(json.dumps(dict(event='launched',task=task,pid=child.pid,cpu=cpu)),flush=True)
  write(status_path,dict(at=time.time(),pid=os.getpid(),status='paused' if paused else 'draining' if stopping else 'complete' if len(completed)==len(tasks) else 'running',completed=len(completed),total=len(tasks),batch_size=50,maximum_workers=len(CPUS),resource_admission=report,active=[{k:v for k,v in r.items() if k!='child'} for r in active.values()]))
  if len(completed)==len(tasks) or stopping and not active:break
  time.sleep(5)
if __name__=='__main__':main()
