import os,subprocess,json,time
from pathlib import Path
p=Path(__file__).resolve().parent
jobs=[]
for k,name in enumerate(['original80','retrained80','original788']):
 for shard in range(2):
  env=dict(os.environ,PROBE_MODEL=name,PROBE_SHARD=str(shard),PROBE_CPU=str(34+2*k+shard),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
  log=open(p/('shard_'+name+'_'+str(shard)+'.log'),'w')
  child=subprocess.Popen(['/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python','-u',str(p/'target_probe_shard.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
  jobs.append((name,shard,child,log))
(p/'shard_processes.json').write_text(json.dumps([dict(model=n,shard=s,pid=c.pid,cpu=34+i) for i,(n,s,c,l) in enumerate(jobs)],indent=2))
status=[]
for n,s,c,l in jobs:
 code=c.wait();l.close();status.append(dict(model=n,shard=s,exit_code=code))
(p/'shard_exit_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
