"""Read-only native trust-region-state replay from saved acquisition batches."""
import ast,hashlib,json,math,tarfile,types,sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np,torch
from Levenshtein import distance
OUT=Path('/root/reports/peptide_train800_trust_region_20261121')
p=Path('/root/data/cys/BOLT/optimization/peptides/lolbo/utils/bo_utils/turbo.py')
tree=ast.parse(p.read_text());wanted={'TurboState','update_tr_length','update_state_constrained','update_state','update_state_unconstrained'}
module=types.ModuleType('radius_native_replay');sys.modules[module.__name__]=module
module.__dict__.update(torch=torch,math=math,dataclass=dataclass)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted],type_ignores=[]),str(p),'exec'),module.__dict__)
with tarfile.open('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo/runs/pep_999.tar.gz') as t:frozen=json.load(t.extractfile('history.json'))
latest=json.loads(Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/evaluation/adapted_online/pep_999/result.json').read_text())
previous=json.loads(Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118/evaluation/adapted_online/pep_999/result.json').read_text())
report={}
for name,history in [('frozen',frozen),('online_rollback',latest),('online_previous',previous)]:
 rows=history['observations'];initial=[r for r in rows if r['phase']=='initialization'];best=max(r['score'] for r in initial if r['feasible']);ref=history['reference']
 bestrow=max((r for r in initial if r['feasible']),key=lambda r:r['score'])
 state=module.TurboState(dim=256,batch_size=50,best_value=best,best_constraint_values=torch.tensor([.75-(1-distance(bestrow['sequence'],ref)/len(ref))],dtype=torch.float32))
 groups=defaultdict(list)
 for row in rows:
  if row['phase']=='acquisition' and row['round']<=80:groups[row['round']].append(row)
 trace=[]
 for rd,batch in sorted(groups.items()):
  assert len(batch)==50,(name,rd,len(batch))
  c=np.asarray([[.75-(1-distance(r['sequence'],ref)/len(ref))] for r in batch],dtype=np.float32)
  before=state.length;old=state.best_value
  module.update_state(state,torch.tensor([r['score'] for r in batch]).reshape(-1,1),torch.as_tensor(c))
  trace.append(dict(round=rd,proposal_length=before,next_length=state.length,incumbent_success=state.best_value>old,best_mic=-state.best_value,failure_counter=state.failure_counter))
  assert not state.restart_triggered,'Unexpected restart needs full recenter replay'
 report[name]=dict(trace=trace,contractions=[r for r in trace if r['next_length']<r['proposal_length']],successes=[r for r in trace if r['incumbent_success']],note='Reconstructed from actual saved acquisition batches with native state-update code; no original per-round radius log. No restart in comparison prefix.')
snapshot=Path('/root/data/logs/peptide_bo_train800_20261027_v5b/upstream_cpu/lolbo/utils/bo_utils/turbo.py')
def rules(path):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'update_state','update_state_constrained','update_tr_length'}}
assert rules(snapshot)==rules(p)
report['frozen_native_snapshot_rule_match']=True
report['frozen_native_snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
report['native_state_code_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'radius_replay.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:{a:v[a] for a in ['contractions','successes']} for k,v in report.items() if isinstance(v,dict)},indent=2))
