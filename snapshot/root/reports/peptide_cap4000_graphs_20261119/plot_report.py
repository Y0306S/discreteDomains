"""Apply the same retrospective 4,000 cap; export measured scores and runtime estimates."""
import csv,json,sqlite3,statistics,math
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/root/reports/peptide_final_comparison_20261119')
OUT=Path(__file__).resolve().parent
ON=Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
SEM=Path('/root/data/cys/baseline/campaigns/semanticopt_peptide_test10_5000_20261117')
LABELS=['Original frozen FSBO','Online VAE + FSBO','SemanticOPT with fallback']
COLORS=dict(zip(LABELS,['#1f77b4','#d62728','#9467bd']))
rows=list(csv.DictReader((BASE/'final_scores_and_times.csv').open()));TASKS=[r['task'] for r in rows]
curves=defaultdict(list)
for r in csv.DictReader((BASE/'convergence.csv').open()):
 curves[r['task'],r['method']].append(dict(fresh=int(r['fresh_evaluations']),mic=float(r['best_mic']),time=float(r['recorded_time']),proposals=int(r['acquisition_proposals'])))
capped={}
for key,c in curves.items():
 keep=[]
 for v in c:
  if v['fresh']>4000 or v['proposals']>4000:break
  keep.append(v)
  if v['fresh']==4000 or v['proposals']==4000:break
 assert keep
 capped[key]=keep
 assert all(v['fresh']<=4000 and v['proposals']<=4000 for v in keep)
# Shared budget evaluation reveals only fully completed batches.
def value(c,x):
 return min(v['mic'] for v in c if v['fresh']<=x)

def write_csv(name,data):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)

telemetry=defaultdict(lambda:dict(requests=0,service_seconds=0.,queue_seconds=0.,errors=0))
with sqlite3.connect(f'file:{SEM}/peptide/services/requests.sqlite?mode=ro',uri=True) as db:
 for body,response,error in db.execute('select body,response,error from requests'):
  task=Path(json.loads(body)['task']).name;d=telemetry[task]
  if response:
   r=json.loads(response);d['requests']+=1;d['service_seconds']+=r['finished_at']-r['started_at'];d['queue_seconds']+=r.get('queue_wait_seconds',0.)
  if error:d['errors']+=1
assert set(telemetry)==set(TASKS) and sum(x['errors'] for x in telemetry.values())==0
per_task=[];timings=[]
for original in rows:
 task=original['task'];rec=dict(task=task,initial_mic=float(original['initial_best_mic']))
 for i,m in enumerate(LABELS):
  end=capped[task,m][-1];prefix=['frozen','online','semanticopt'][i]
  rec.update({prefix+'_mic':end['mic'],prefix+'_fresh':end['fresh'],prefix+'_proposals':end['proposals']})
  if i<2:
   wall=end['time'];est=wall*4000/end['fresh']
   timings.append(dict(task=task,method=m,observed_seconds=wall,observed_fresh=end['fresh'],estimated_4000_seconds=est,estimate_scope='Recorded capped task wall ×4000/actual fresh; no new trial was run; excludes initial offline preparation'))
 with sqlite3.connect(f'file:{SEM}/peptide/runs/{task}/observations.sqlite?mode=ro',uri=True) as db:
  measurements=[json.loads(v) for v, in db.execute('select value from records where n>=1000 order by n')]
 assert len(measurements)==3000
 batches={r['oracle_batch_start']:r['oracle_batch_seconds'] for r in measurements}
 oracle=sum(batches.values());data=telemetry[task]
 service=data['service_seconds']+oracle
 single_est=service*4000/3000
 loaded_est=float(original['semanticopt_continuation_elapsed_hours'])*3600*4000/3000
 timings.append(dict(task=task,method=LABELS[2],observed_seconds=float(original['semanticopt_continuation_elapsed_hours'])*3600,observed_fresh=3000,estimated_4000_seconds=single_est,estimate_scope='Sum inference generation+oracle service for continuation 3,000 ×4000/3000; no queue wait, loading or client overhead; estimate of one isolated task, unbenchmarked',estimated_4000_with_campaign_queue_seconds=loaded_est,generation_seconds=data['service_seconds'],queue_wait_seconds=data['queue_seconds'],oracle_seconds=oracle,inference_requests=data['requests']))
 per_task.append(rec)

method_summary={}
for i,m in enumerate(LABELS):
 prefix=['frozen','online','semanticopt'][i];times=[r['estimated_4000_seconds']/3600 for r in timings if r['method']==m]
 method_summary[m]=dict(mean_final_mic=statistics.mean(r[prefix+'_mic'] for r in per_task),estimated_4000_hours_mean=statistics.mean(times),estimated_4000_hours_min=min(times),estimated_4000_hours_max=max(times),actual_fresh_min=min(r[prefix+'_fresh'] for r in per_task),actual_fresh_max=max(r[prefix+'_fresh'] for r in per_task))
sem_queue=[r['estimated_4000_with_campaign_queue_seconds']/3600 for r in timings if r['method']==LABELS[2]]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),cap=4000,batch_size=50,max_rounds=80,retrospective_frozen_cap=True,method_summary=method_summary,semanticopt_estimated_hours_under_existing_queue_mean=statistics.mean(sem_queue),frozen_vs_online_wins=sum(r['online_mic']<r['frozen_mic'] for r in per_task),semanticopt_wins=sum(r['semanticopt_mic']<min(r['frozen_mic'],r['online_mic']) for r in per_task),smoothing='Same causal trailing mean over 200 evaluation positions for every method; smoothing applies to displayed aggregate curves only; reported MIC values and per-task plots remain exact.',score_caveat='All methods capped at 4,000 acquisition proposals or 4,000 fresh evaluations, whichever first. Frozen history uses the last fully measured batch within both caps; crossing batches excluded, so two tasks stop at 3,990 fresh. Stops after 80 rounds can leave fewer than 4,000 fresh. No fabricated extra observations.',runtime_caveat='All 4,000-fresh runtimes are linear estimates using actual recorded timings, not fresh isolated benchmarks. Original/online historical hardware and contention differ; SemanticOPT estimates use 3,000 continuation measurements and the continuation request telemetry. Offline training/loading excluded.',offline_online_hours=json.loads((ON/'summary.json').read_text())['offline_total_seconds']/3600)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
write_csv('capped_scores.csv',per_task);write_csv('estimated_runtime_per_peptide.csv',timings)
write_csv('capped_convergence.csv',[dict(task=t,method=m,**v) for (t,m),c in capped.items() for v in c])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)

# Aggregate mean MIC, with exactly the same display smoothing for all methods.
grid=np.arange(4001);fig,axes=plt.subplots(1,2,figsize=(12.2,4.3))
exports=[]
for m in LABELS:
 a=np.array([[value(capped[t,m],x) for x in grid] for t in TASKS]);mean=a.mean(0)
 # Time-causal200-position window, using only current and earlier measurements.
 cs=np.r_[0.,np.cumsum(mean)];starts=np.maximum(0,np.arange(len(mean))-199);smooth=(cs[np.arange(len(mean))+1]-cs[starts])/(np.arange(len(mean))+1-starts)
 axes[0].step(grid,mean,where='post',lw=1.6,color=COLORS[m],label=m)
 axes[1].plot(grid,smooth,lw=2,color=COLORS[m],label=m)
 for x in range(0,4001,25):exports.append(dict(method=m,fresh_budget=x,raw_mean_mic=float(mean[x]),display_smoothed_mean_mic=float(smooth[x]),tasks_with_actual_fresh_support=sum(capped[t,m][-1]['fresh']>=x for t in TASKS)))
for ax,title in zip(axes,['Measured incumbent means','Same trailing 200-evaluation smoothing']):
 ax.set(title=title,xlabel='Fresh evaluation budget',ylabel='Mean best predicted MIC (µM) ↓',xlim=(0,4000));ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=9)
fig.text(.5,.005,'Ten fixed peptide tasks · Stopped incumbents retained through the cap; no additional evaluations implied.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.90));save(fig,'mean_convergence_4000');write_csv('aggregate_curves.csv',exports)

fig,axes=plt.subplots(2,5,figsize=(18,7))
for task,ax in zip(TASKS,axes.flat):
 for m in LABELS:
  c=capped[task,m];ax.step([v['fresh'] for v in c],[v['mic'] for v in c],where='post',label=m,color=COLORS[m],lw=1.7)
  ax.scatter(c[-1]['fresh'],c[-1]['mic'],s=15,color=COLORS[m],zorder=3)
 ax.set(title=task,xlabel='Fresh evaluations',ylabel='Best predicted MIC (µM) ↓',xlim=(0,4000));ax.grid(alpha=.2)
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=3)
fig.tight_layout(rect=(0,0,1,.93));save(fig,'per_task_convergence_4000')

fig,ax=plt.subplots(figsize=(13,4.5));x=np.arange(10);width=.25
for i,m in enumerate(LABELS):
 key=['frozen_mic','online_mic','semanticopt_mic'][i];ax.bar(x+(i-1)*width,[r[key] for r in per_task],width,label=m,color=COLORS[m])
ax.set(xticks=x,xticklabels=TASKS,ylabel='Final best predicted MIC (µM) ↓',title='Final MIC under the same 4,000 cap');ax.legend(ncol=3,fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'final_mic_4000')

fig,ax=plt.subplots(figsize=(10,4.3));labels=LABELS+['SemanticOPT: existing ten-task queue'];means=[];lo=[];hi=[]
for m in LABELS:
 d=method_summary[m];means.append(d['estimated_4000_hours_mean']);lo.append(means[-1]-d['estimated_4000_hours_min']);hi.append(d['estimated_4000_hours_max']-means[-1])
means.append(statistics.mean(sem_queue));lo.append(means[-1]-min(sem_queue));hi.append(max(sem_queue)-means[-1])
for i,(lab,val) in enumerate(zip(labels,means)):
 ax.barh(i,val,color=COLORS[LABELS[min(i,2)]],alpha=.9 if i<3 else .45,hatch=None if i<3 else '//',xerr=np.array([[lo[i]],[hi[i]]]),capsize=4)
 ax.text(val+hi[i]+.45,i,f'{val:.2f} h',va='center')
ax.set(yticks=np.arange(4),yticklabels=labels,xlabel='Estimated hours per peptide for 4,000 fresh evaluations',title='Runtime estimates from saved measurements (bars: mean; whiskers: task range)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlim(0,max(means)+3)
fig.text(.5,.005,'SemanticOPT solid bar excludes queue wait; hatched bar retains observed queue conditions. Initial offline training excluded.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1));save(fig,'runtime_per_peptide_4000')

report=['# Peptide comparison under a 4,000 cap','','The user’s “400 cap” is interpreted as 4,000, consistent with the preceding 80 rounds × 50 and explicit request for 4,000-evaluation timing. Original FSBO is truncated retrospectively; no rerun or modification of production results.','','| Method | Mean endpoint MIC (µM) | Estimated hours / 4,000 fresh evaluations | Range across 10 tasks |','|---|---:|---:|---:|']
for m,d in method_summary.items():report.append(f"| {m} | {d['mean_final_mic']:.3f} | {d['estimated_4000_hours_mean']:.2f} | {d['estimated_4000_hours_min']:.2f}–{d['estimated_4000_hours_max']:.2f} |")
report+=['','SemanticOPT with the observed ten-task queue: '+f'{statistics.mean(sem_queue):.2f} h per task estimated. The solid-bar estimate removes queue wait using broker request start/finish timestamps; it is not a new isolated-task benchmark. Both use ×4000/3000 continuation scaling. Input prompts and source mixture may differ during the first 1,000, so this is a planning estimate.','','Frozen/online times use measured capped task wall ×4000/actual fresh. Offline preparation for online VAE/FSBO was '+f'{summary["offline_online_hours"]:.2f}h shared across the cohort; it is excluded from per-task estimates.','',''+summary['score_caveat'],'',summary['smoothing'],'',summary['runtime_caveat'],'','| Task | Frozen MIC | Online MIC | SemanticOPT MIC | Frozen fresh | Online fresh | SemanticOPT fresh |','|---|---:|---:|---:|---:|---:|---:|']
for r in per_task:report.append(f"| {r['task']} | {r['frozen_mic']:.3f} | {r['online_mic']:.3f} | {r['semanticopt_mic']:.3f} | {r['frozen_fresh']} | {r['online_fresh']} | {r['semanticopt_fresh']} |")
report+=['','[Mean convergence](mean_convergence_4000.png) · [Per-task convergence](per_task_convergence_4000.png) · [Final MIC](final_mic_4000.png) · [Runtime](runtime_per_peptide_4000.png)','','SemanticOPT retains its labeled random-fallback policy (78.3% of scored proposals in this cohort). All methods use the identical 100 imported starting measurements, excluded from evaluation counts. One seed per task/method; no claim of statistical superiority.','']
(OUT/'REPORT.md').write_text('\n'.join(report))
print(json.dumps(summary,indent=2),flush=True)
