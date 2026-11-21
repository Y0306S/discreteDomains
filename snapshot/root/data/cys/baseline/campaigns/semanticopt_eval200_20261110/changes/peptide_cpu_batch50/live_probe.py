import os,time,sys,json
from pathlib import Path
# Initialize libraries before adding the campaign's queue.py directory.
import torch
r=Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110');sys.path.insert(0,str(r/'code'))
from common import verify,atomic,canonical
from evaluate import peptide_setup,connect,get,put
from peptide_batch import score_and_commit
from random_fallback import choose_fallback
os.sched_setaffinity(0,{11})
recipe=verify();settings=recipe['peptide'];task=settings['smoke_task']
context,initial,validate,oracle=peptide_setup(task,settings)
seen={canonical(i['configuration']) for i in initial};excluded=set(seen);items=[]
for i in range(50):
 c,provenance=choose_fallback('peptide',context,validate,excluded,[],seed=42,number=i,policy=recipe['fallback'])
 items.append(dict(answer=dict(next_configuration=c),provenance=provenance));excluded.add(canonical(c))
e=r/'changes/peptide_cpu_batch50';db=connect(e/'live_training_probe')
start=time.perf_counter()
results=score_and_commit(db,get,put,batch=dict(start=0,items=items,policy_sha256=recipe['batch_policy_sha256']),start=0,total=50,
 validate=validate,oracle=oracle.batch,seen=seen,runtime_sha=recipe['runtime_policy_sha256'],policy_sha=recipe['peptide_oracle_policy_sha256'])
report=dict(passed=len(results)==50 and get(db,'oracle_batch')['status']=='committed',at=time.time(),training_task=task,
 results=len(results),cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
 seconds=time.perf_counter()-start,oracle_seconds=results[0]['oracle_batch_seconds'],
 peptide_oracle_policy_sha256=recipe['peptide_oracle_policy_sha256'])
atomic(e/'live_probe.json',report);print(json.dumps(report,indent=2));db.close()
