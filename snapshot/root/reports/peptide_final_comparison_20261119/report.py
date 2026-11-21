"""Read-only completed peptide comparison, preserving unequal endpoint budgets."""
import csv, hashlib, json, math, sqlite3, tarfile, statistics
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
from Levenshtein import distance
OUT=Path(__file__).resolve().parent
ON=Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
SEM=Path('/root/data/cys/baseline/campaigns/semanticopt_peptide_test10_5000_20261117')
FR=Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo/runs')
POOL=json.loads(Path('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json').read_text())
TASKS=json.loads((ON/'config.json').read_text())['tasks']
assert TASKS==json.loads((SEM/'extension.json').read_text())['tasks']
LABELS=['Original frozen FSBO','Online VAE + FSBO','SemanticOPT with fallback']

def write_csv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def curve(rows,task):
 init=[r for r in rows if r['phase']=='initialization'];pool=POOL[task]
 assert [r['sequence'] for r in init]==pool['sequences']
 assert [r['score'] for r in init]==pool['scores']
 best=min(r['mic'] for r in init);fresh=0;proposals=0;out=[(0,best,0.,0)]
 batches=[];last=None;seen=set(pool['sequences'])
 for r in rows[len(init):]:
  assert math.isfinite(r['mic']) and r['mic']==-r['score']
  assert r['cached']==(r['sequence'] in seen);seen.add(r['sequence'])
  assert r['feasible']==(1-distance(r['sequence'],pool['reference'])/len(pool['reference'])>=.75)
  key=(r['phase'],r['elapsed'])
  if last!=key:batches.append([])
  batches[-1].append(r);last=key
 for batch in batches:
  fresh+=sum(not r['cached'] for r in batch)
  if batch[0]['phase']=='acquisition':proposals+=len(batch)
  best=min([best]+[r['mic'] for r in batch if r['feasible']])
  out.append((fresh,best,batch[-1].get('wall_elapsed',batch[-1]['elapsed']),proposals))
 return out

def at(c,budget):
 assert c[-1][0]>=budget
 return min(y for x,y,t,p in c if x<=budget)

curves={};endpoints=[];runtimes=[];results={};online=[]
sem_launch=json.loads(Path('/root/reports/peptide_stop4000_20261118/before/semanticopt/processes.json').read_text())['at']
imports={t['task']:t['source_prefix_sha256'] for t in json.loads((SEM/'evidence/import_audit.json').read_text())['tasks']}
for task in TASKS:
 with tarfile.open(FR/(task+'.tar.gz')) as archive:
  raw=archive.extractfile('history.json').read();manifest=json.load(archive.extractfile('manifest.json'))
 assert hashlib.sha256(raw).hexdigest()==manifest['members']['history.json']
 f=json.loads(raw);o=json.loads((ON/'evaluation/adapted_online'/task/'result.json').read_text())
 assert f['status']==o['status']=='complete'
 assert o['proposals']<=4000 and o['fresh_evaluations']<=4000
 assert o['proposals']==4000 or o['fresh_evaluations']==4000
 online.append(o)
 for label,d in [(LABELS[0],f),(LABELS[1],o)]:
  c=curve(d['observations'],task);assert c[-1][0]==d['fresh_evaluations'] and c[-1][3]==d['proposals']
  assert c[-1][1]==d['best_feasible_mic']
  curves[task,label]=c;results[task,label]=d
 with sqlite3.connect(f'file:{SEM}/peptide/runs/{task}/observations.sqlite?mode=ro',uri=True) as db:
  raw=db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
  init=json.loads(db.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0])
 assert len(raw)==4000 and [n for n,v in raw]==list(range(4000))
 assert hashlib.sha256(json.dumps(raw[:1000],separators=(',',':')).encode()).hexdigest()==imports[task]
 assert [r['configuration']['sequence'] for r in init]==POOL[task]['sequences']
 assert [r['objective'] for r in init]==[-y for y in POOL[task]['scores']]
 records=[json.loads(v) for n,v in raw];seen=set(POOL[task]['sequences']);best=min(r['objective'] for r in init);c=[(0,best,0.,0)]
 for i,r in enumerate(records,1):
  seq=r['configuration']['sequence'];assert seq not in seen;seen.add(seq)
  assert r['evaluation']==i and r['status']=='success' and r['feasible'] and math.isfinite(r['objective'])
  assert 1-distance(seq,POOL[task]['reference'])/len(POOL[task]['reference'])>=.75
  best=min(best,r['objective'])
  if i%50==0:c.append((i,best,r['at'],i))
 curves[task,LABELS[2]]=c
 s=json.loads((SEM/'peptide/runs'/task/'status.json').read_text());assert s['status']=='complete' and s['evaluations']==4000
 results[task,LABELS[2]]=dict(best_feasible_mic=best,fresh_evaluations=4000,proposals=4000,random_fallback=sum(r['proposal_source']=='random_fallback' for r in records))
 endpoints.append(dict(task=task,initial_best_mic=min(-y for y in POOL[task]['scores']),original_final_mic=f['best_feasible_mic'],online_final_mic=o['best_feasible_mic'],semanticopt_final_mic=best,original_proposals=f['proposals'],original_fresh=f['fresh_evaluations'],online_proposals=o['proposals'],online_fresh=o['fresh_evaluations'],semanticopt_fresh=4000,original_task_minutes=f['seconds']/60,online_task_minutes=o['wall_seconds']/60,semanticopt_continuation_elapsed_hours=(records[-1]['at']-sem_launch)/3600))
 runtimes.extend([dict(task=task,method=LABELS[0],seconds=f['seconds'],scope='Original full saved endpoint; excludes offline training'),dict(task=task,method=LABELS[1],seconds=o['wall_seconds'],scope='Accumulated evaluation wall across restarts; includes online VAE/source refresh/scratch FSBO; excludes initial offline training'),dict(task=task,method=LABELS[2],seconds=records[-1]['at']-sem_launch,scope='Elapsed continuation1000-to4000, including service queue and restart gap; tasks concurrent; excludes earlier imported1000 and offline model training',last_process_seconds=s['seconds_this_process'])])
 print(task,'audited',flush=True)

matched=[]
for budget in [1000,2000,2500,3000,3500]:
 for task in TASKS:
  matched.append(dict(task=task,fresh_budget=budget,original_mic=at(curves[task,LABELS[0]],budget),online_mic=at(curves[task,LABELS[1]],budget),semanticopt_mic=at(curves[task,LABELS[2]],budget)))
# Also compare each task at the actual fresh budget achieved by the online run.
for task in TASKS:
 budget=results[task,LABELS[1]]['fresh_evaluations']
 matched.append(dict(task=task,fresh_budget=budget,comparison='same_as_online_final_fresh',original_mic=at(curves[task,LABELS[0]],budget),online_mic=at(curves[task,LABELS[1]],budget),semanticopt_mic=at(curves[task,LABELS[2]],budget)))

metrics={}
for label,field in zip(LABELS,['original_final_mic','online_final_mic','semanticopt_final_mic']):
 vals=[r[field] for r in endpoints]
 metrics[label]=dict(mean_final_mic=statistics.mean(vals),median_final_mic=statistics.median(vals),mean_task_improvement_percent=statistics.mean(100*(r['initial_best_mic']-r[field])/r['initial_best_mic'] for r in endpoints),wins=sum(r[field]==min(r[k] for k in ['original_final_mic','online_final_mic','semanticopt_final_mic']) for r in endpoints))
paired_summary={}
for tag,selection in [('3500_fresh',[r for r in matched if r['fresh_budget']==3500 and 'comparison' not in r]),('online_actual_fresh',[r for r in matched if r.get('comparison')=='same_as_online_final_fresh'])]:
 paired_summary[tag]=dict(tasks=len(selection),mean_mic={m:statistics.mean(r[k] for r in selection) for m,k in zip(LABELS,['original_mic','online_mic','semanticopt_mic'])},online_vs_original_wins=sum(r['online_mic']<r['original_mic'] for r in selection),online_vs_original_ties=sum(r['online_mic']==r['original_mic'] for r in selection),online_vs_semanticopt_wins=sum(r['online_mic']<r['semanticopt_mic'] for r in selection))
fsum=sum(r['original_task_minutes'] for r in endpoints)*60;osum=sum(r['online_task_minutes'] for r in endpoints)*60
source_seconds=sum(d['timing']['source_meta_retraining'] for d in online)
on_sum=json.loads((ON/'summary.json').read_text())
initial_pipeline=json.loads(Path('/root/reports/peptide_stop4000_20261118/before/online/pipeline_process.json').read_text())['started_at']
on_end=json.loads((ON/'pipeline_status.json').read_text())['updated_at']
sem_end=json.loads((SEM/'status.json').read_text())['at']
summary=dict(checked_at=datetime.now(timezone.utc).isoformat(),status='all30_method_task_results_complete',endpoint_metrics=metrics,endpoint_online_vs_original_wins=sum(r['online_final_mic']<r['original_final_mic'] for r in endpoints),endpoint_online_vs_semanticopt_wins=sum(r['online_final_mic']<r['semanticopt_final_mic'] for r in endpoints),matched=paired_summary,runtime=dict(original_mean_task_minutes=fsum/600,original_sum_task_hours=fsum/3600,online_mean_task_hours=osum/36000,online_sum_task_hours=osum/3600,online_to_original_sum_ratio=osum/fsum,online_initial_offline_hours=on_sum['offline_total_seconds']/3600,online_campaign_start_to_report_hours=(on_end-initial_pipeline)/3600,semanticopt_continuation_start_to_report_hours=(sem_end-sem_launch)/3600,semanticopt_mean_task_continuation_hours=statistics.mean(r['semanticopt_continuation_elapsed_hours'] for r in endpoints),source_reencoding_and_scratch_fsbo_fraction_of_online_wall=source_seconds/osum),semanticopt_random_fallback_fraction=sum(results[t,LABELS[2]]['random_fallback'] for t in TASKS)/40000,audits='All initial100 sequences and measured scores identical; frozen archive payload hashes and SemanticOPT imported1000 hashes verified; unique/cache accounting and75% similarity constraints checked; batch-causal curves; endpoints and counts validated.',limitations=['Endpoint budgets unequal: original stops at5000 proposal or fresh cap; online at4000 proposal or fresh cap; SemanticOPT exactly4000 fresh. See matched budgets to assess quality independently of this discrepancy.','SemanticOPT first1000 observations imported from an earlier campaign.26.6h continuation runtime is not the runtime for all4000; last process timer is only about7.9h and is not total.','Frozen and online task wall sums are aggregate task hours, not elapsed campaign duration. SemanticOPT tasks run concurrently and its task hours must not be interpreted as CPU/GPU compute.','Historical hardware/software, inference concurrency, shared resources and pause/restart histories differ; runtimes are operational measurements, not a controlled speed benchmark.','One seed per method/task. Online changes initial VAE and FSBO training population as well as online updates; this does not isolate the VAE contribution.'])
write_csv('final_scores_and_times.csv',endpoints);write_csv('matched_fresh_budgets.csv',matched);write_csv('runtime_details.csv',runtimes)
write_csv('convergence.csv',[dict(task=t,method=m,fresh_evaluations=x,best_mic=y,recorded_time=z,acquisition_proposals=p) for (t,m),c in curves.items() for x,y,z,p in c])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
content=['# Completed comparison: ten peptide tasks','','All30 method/task results complete. Best feasible predicted MIC in µM; lower is better. Endpoints reuse exact100 starting measurements.','','| Task | Original frozen FSBO | Online VAE + FSBO | SemanticOPT | Frozen runtime(min) | Online runtime(h) | SemanticOPT continuation(h) |','|---|---:|---:|---:|---:|---:|---:|']
for r in endpoints:content.append(f"| {r['task']} | {r['original_final_mic']:.3f} | {r['online_final_mic']:.3f} | {r['semanticopt_final_mic']:.3f} | {r['original_task_minutes']:.1f} | {r['online_task_minutes']/60:.2f} | {r['semanticopt_continuation_elapsed_hours']:.2f} |")
content+=['','Mean endpoint MIC: '+', '.join(f'{m}: {v["mean_final_mic"]:.4f}' for m,v in metrics.items())+'.','','Budget note: original endpoint uses older5000 cap; online uses4000 proposals or fresh evaluations (3693–4000 fresh); SemanticOPT4000 distinct fresh. These endpoints are not equal-budget.','','Matched at3500 fresh evaluations per task: '+json.dumps(paired_summary['3500_fresh'])+'.','','Matched at each online task actual final fresh count: '+json.dumps(paired_summary['online_actual_fresh'])+'.','','Original mean task runtime: '+f'{fsum/600:.2f}min; online {osum/36000:.2f}h; aggregate task sums {fsum/3600:.2f}h and {osum/3600:.2f}h, respectively. Online offline preparation adds {on_sum["offline_total_seconds"]/3600:.2f}h. Online campaign start through final report took {(on_end-initial_pipeline)/3600:.2f}h with four concurrent task lanes and restarts.','','SemanticOPT continuation from imported1000 to4000 took '+f'{(sem_end-sem_launch)/3600:.2f}h elapsed for all ten concurrent tasks; this excludes the original1000 generation and model training. The7.9h last-process timer covers only the post-restart segment. Full4000 runtime cannot be reconstructed as one clean active-duration timer from current evidence.','','Source re-encoding/full-source scratch FSBO accounts for '+f'{100*source_seconds/osum:.1f}% of accumulated online evaluation wall time. SemanticOPT proposals include {100*summary["semanticopt_random_fallback_fraction"]:.1f}% labeled random fallback.','','Audits: '+summary['audits'],'']+['- '+x for x in summary['limitations']]
(OUT/'REPORT.md').write_text('\n'.join(content)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=['#1f77b4','#d62728','#9467bd'];fig,axes=plt.subplots(2,5,figsize=(18,7))
for task,ax in zip(TASKS,axes.flat):
 for label,color in zip(LABELS,colors):
  c=curves[task,label];ax.step([x for x,y,t,p in c],[y for x,y,t,p in c],where='post',label=label,color=color,lw=1.6)
 ax.axvline(3500,color='grey',ls=':',lw=.8);ax.set(title=task,xlabel='Fresh evaluations',ylabel='Best predicted MIC (µM) ↓',xlim=(0,5000));ax.grid(alpha=.2)
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=3);fig.tight_layout(rect=(0,0,1,.94))
for ext in ['png','pdf']:fig.savefig(OUT/f'completed_convergence.{ext}',dpi=170,bbox_inches='tight')
print(json.dumps(summary,indent=2),flush=True)
