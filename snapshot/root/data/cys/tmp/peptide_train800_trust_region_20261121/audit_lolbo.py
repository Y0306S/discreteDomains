"""Read-only comparison of LOLBO feasibility, separating acquisition/recenter."""
import csv,hashlib,json,tarfile
from pathlib import Path
from Levenshtein import distance
OUT=Path('/root/reports/peptide_train800_trust_region_20261121')
BASE=Path('/root/data/logs/peptide_test200_baselines_20261027_v2/lolbo')
cfg=json.loads(Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/config.json').read_text())
prior=json.loads((OUT/'ten_task_feasibility_audit.json').read_text())
known={r['task']:r for r in csv.DictReader(Path('/root/reports/peptide_convergence_20261101/task_summary.csv').open()) if r['method']=='LOLBO'}
rows=[]
for task in cfg['tasks']:
 path=BASE/'runs'/(task+'.tar.gz')
 with tarfile.open(path) as tar:
  payload=tar.extractfile('history.json').read();manifest=json.load(tar.extractfile('manifest.json'))
 digest=hashlib.sha256(payload).hexdigest();assert digest==manifest['members']['history.json']==known[task]['history_sha256']
 h=json.loads(payload);ref=h['reference'];observations=h['observations']
 with tarfile.open('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo/runs/'+task+'.tar.gz') as tar:f=json.load(tar.extractfile('history.json'))
 init=lambda x:[(r['sequence'],r['score']) for r in x['observations'] if r['phase']=='initialization']
 assert init(h)==init(f)
 assert all(r['feasible']==(distance(r['sequence'],ref)<=len(ref)//4) for r in observations)
 assert all(r['sequence'] and len(r['sequence'])<=30 and r['sequence'].count('C')<=1 and not set(r['sequence'])-set('ACDEFGHIKLMNPQRSTVWY') for r in observations)
 for scope in ['first80rounds_or_run_end','full_completed_run']:
  obs=[r for r in observations if r['phase']!='initialization' and (scope=='full_completed_run' or r['round']<=80)]
  phases={}
  for phase in ['acquisition','recenter','combined']:
   a=[r for r in obs if (phase=='combined' or r['phase']==phase) and not r['cached']]
   n=len(a);feas=sum(r['feasible'] for r in a)
   phases[phase]=dict(fresh=n,feasible=feas,feasible_percent=100*feas/n if n else None)
  initialbest=min(r['mic'] for r in observations if r['phase']=='initialization')
  best=min([initialbest]+[r['mic'] for r in obs if r['feasible']])
  rows.append(dict(task=task,scope=scope,phases=phases,acquisition_proposals=sum(r['phase']=='acquisition' for r in obs),acquisition_rounds=max(r['round'] for r in obs if r['phase']=='acquisition'),initial_best_mic=initialbest,best_mic=best,history_sha256=digest))
 for r in rows[-2:]:
  if r['scope']=='full_completed_run':
   assert r['phases']['acquisition']['fresh']==h['acquisition_fresh'] and r['phases']['recenter']['fresh']==h['recenter_fresh'] and r['phases']['combined']['fresh']==h['fresh_evaluations']
summary={}
for scope in ['first80rounds_or_run_end','full_completed_run']:
 summary[scope]={}
 for phase in ['acquisition','recenter','combined']:
  a=[r['phases'][phase] for r in rows if r['scope']==scope];n=sum(r['fresh'] for r in a);v=sum(r['feasible'] for r in a)
  summary[scope][phase]=dict(fresh=n,feasible=v,feasible_percent=100*v/n)
result=dict(status='complete',tasks=len(cfg['tasks']),summary=summary,rows=rows,checks=['history hashes match archive manifest and previous full200audit','exact reference edit feasibility and structural rules recomputed for every recorded sequence','initial100sequences/scores match frozen FSBO','full-run fresh counters reconciled'],limitations=['First80round prefix means at most4000acquisition proposals; some tasks stop earlier at the shared5000fresh-call limit. This is not equal-cost convergence evidence.','Acquisition and recenter rates must be reported separately.','Recorded sequences are postprocessed and scored; counts do not describe all raw decoder tokens.'],new_oracle_calls=0)
(OUT/'lolbo_feasibility_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
for r in rows:
 if r['scope']=='first80rounds_or_run_end':print(r['task'],'proposals',r['acquisition_proposals'],'rates',[(p,round(a['feasible_percent'],3),a['feasible'],a['fresh']) for p,a in r['phases'].items()],'MIC',r['best_mic'])
# Human follow-up table, preserving phase denominators and support.
lines=[]
for r in rows:
 if r['scope']!='first80rounds_or_run_end':continue
 t=r['task'];a=r['phases'];lines.append(f"|{t}|{r['acquisition_proposals']}|{a['acquisition']['feasible']}/{a['acquisition']['fresh']} ({a['acquisition']['feasible_percent']:.2f}%)|{a['recenter']['feasible']}/{a['recenter']['fresh']} ({a['recenter']['feasible_percent']:.2f}%)|{a['combined']['feasible_percent']:.2f}%|")
report='''# LOLBO feasibility on the same ten peptides

Fresh acquisition and recenter predictions are counted separately. Initialization and cache hits are excluded. The prefix includes at most the first80acquisition rounds (4000proposals); tasks can terminate earlier because acquisition and recenter share a5000fresh-call cap. These rates do not constitute an equal-cost optimization-quality comparison.

|Peptide|Acquisition proposals represented|Acquisition feasible/fresh|Recenter feasible/fresh|Combined feasible rate|
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'\n\nAggregated counts and complete-run results: lolbo_feasibility_audit.json. All recorded sequence constraints were recomputed exactly, initialization equivalence checked, and history SHA256 verified against manifests and the prior200-task archive audit. No new inference, oracle predictions or training were performed.\n'
(OUT/'LOLBO_FEASIBILITY.md').write_text(report)
