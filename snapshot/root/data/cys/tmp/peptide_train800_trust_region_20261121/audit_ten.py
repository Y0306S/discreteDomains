import json,tarfile,statistics
from pathlib import Path
from Levenshtein import distance
out=Path('/root/reports/peptide_train800_trust_region_20261121')
base=Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119')
old=Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
tasks=json.loads((base/'config.json').read_text())['tasks'];result=[]
for task in tasks:
 histories={}
 with tarfile.open('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo/runs/'+task+'.tar.gz') as t:histories['frozen']=json.load(t.extractfile('history.json'))
 histories['online_rollback']=json.loads((base/'evaluation/adapted_online'/task/'result.json').read_text())
 histories['online_previous']=json.loads((old/'evaluation/adapted_online'/task/'result.json').read_text())
 for name,h in histories.items():
  ref=h['reference'];rows=[r for r in h['observations'] if r['phase']=='acquisition' and r['round']<=80 and not r['cached']]
  assert all(r['feasible']==(distance(r['sequence'],ref)<=len(ref)//4) for r in rows)
  feasible=sum(r['feasible'] for r in rows);phases=[]
  for lo,hi in [(1,20),(21,40),(41,64),(65,80)]:
   a=[r for r in rows if lo<=r['round']<=hi];n=sum(r['feasible'] for r in a)
   phases.append(dict(rounds=[lo,hi],fresh=len(a),feasible=n,rate=n/len(a) if a else None))
  allrows=[r for r in h['observations'] if r['phase']=='initialization' or r['round']<=80]
  result.append(dict(task=task,method=name,fresh_acquisition=len(rows),feasible_acquisition=feasible,rate=feasible/len(rows),periods=phases,recenter_fresh=sum(r['phase']=='recenter' and not r['cached'] for r in allrows),recenter_fresh_feasible=sum(r['phase']=='recenter' and not r['cached'] and r['feasible'] for r in allrows)))
(out/'ten_task_feasibility_audit.json').write_text(json.dumps(dict(status='complete',cap='first80 acquisition rounds /4000 acquisition proposals; fresh acquisition denominator, initialization+cache+recenter excluded; exact reference edit flags rechecked',rows=result),indent=2))
print('task frozen% previous_online% rollback% fresh_valid/fresh')
for task in tasks:
 a={r['method']:r for r in result if r['task']==task};r=a['online_rollback']
 print(task,*[round(100*a[m]['rate'],3) for m in ['frozen','online_previous','online_rollback']],f"{r['feasible_acquisition']}/{r['fresh_acquisition']}")
for method in histories:
 a=[r for r in result if r['method']==method];f=sum(r['feasible_acquisition'] for r in a);n=sum(r['fresh_acquisition'] for r in a)
 print(method,'all10',f,n,round(100*f/n,3),'median%',round(statistics.median(r['rate'] for r in a)*100,3),'recenter fresh feasible',sum(r['recenter_fresh_feasible'] for r in a),'/',sum(r['recenter_fresh'] for r in a))
a=[r for r in result if r['method']=='online_rollback' and r['task']!='pep_999'];f=sum(r['feasible_acquisition'] for r in a);n=sum(r['fresh_acquisition'] for r in a);print('other9 rollback',f,n,f/n)
