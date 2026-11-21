import os,time,json,sys,resource,statistics
from pathlib import Path
import torch
import numpy as np
r=Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110')
recipe=json.loads((r/'recipe.json').read_text());settings=recipe['peptide']
os.sched_setaffinity(0,{11});torch.set_num_threads(1);torch.set_num_interop_threads(1)
sys.path.insert(0,settings['cpu_upstream'])
from apex_oracle.APEX_predict import apex_wrapper,APEX_models
pool=json.loads(Path(settings['pool']).read_text());seqs=pool[settings['smoke_task']]['sequences'][:50]
import random
sys.path.insert(0,str(r/'code'))
from random_fallback import peptide_proposal
from protocol import peptide_measure
reference=pool[settings['smoke_task']]['reference'];rng=random.Random(42);seqs=[]
while len(seqs)<50:
 c=peptide_proposal(reference,rng)['sequence']
 if c not in seqs and peptide_measure(c,reference)['feasible']:seqs.append(c)
# Reuse initial training observations, with variable lengths; no test oracle slots.
with torch.inference_mode():
 apex_wrapper(seqs[:1]);apex_wrapper(seqs[:8])
 sequential=[];batched=[]
 for repeat in range(2):
  start=time.perf_counter();single=np.concatenate([apex_wrapper([s]) for s in seqs]);sequential.append(time.perf_counter()-start)
  start=time.perf_counter();batch=apex_wrapper(seqs);batched.append(time.perf_counter()-start)
rel=np.abs(single-batch)/np.maximum(np.abs(single),1e-12)
report=dict(at=time.time(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
 interop_threads=torch.get_num_interop_threads(),ensemble_models=len(APEX_models),sequences=len(seqs),lengths=sorted(set(map(len,seqs))),
 sequential_seconds=sequential,batch_seconds=batched,speedup=statistics.median(sequential)/statistics.median(batched),
 max_absolute_difference=float(np.max(np.abs(single-batch))),max_relative_difference=float(rel.max()),
 objective_max_absolute_difference=float(np.max(np.abs(single[:,0]-batch[:,0]))),
 agrees_rtol_1e_5_atol_1e_4=bool(np.allclose(single,batch,rtol=1e-5,atol=1e-4)),
 peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
(r/'changes/peptide_cpu_batch50/benchmark_mixed.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
