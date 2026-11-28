"""Read-only analysis of fully completed SemanticOPT 800-proposal pilots."""
import csv
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/root/data/cys/baseline/campaigns/semanticopt_query_400_pilot800_20261123')
OUT=Path(__file__).resolve().parent
ext=json.loads((ROOT/'extension.json').read_text())
summaries=[];curves=[];rows_csv=[];snapshot={}
for task in ext['pilot_tasks']:
 folder=ROOT/'query/runs'/task
 status=json.loads((folder/'status.json').read_text())
 if status.get('status')!='complete' or status.get('evaluations')!=800:continue
 with sqlite3.connect((folder/'observations.sqlite').as_uri()+'?mode=ro',uri=True,timeout=10) as db:
  db.execute('BEGIN')
  initial=json.loads(db.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0])
  raw=db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
  assert len(raw)==800 and [n for n,_ in raw]==list(range(800))
  records=[json.loads(v) for _,v in raw]
 assert len(initial)==49
 successful=lambda r:not r.get('censored',False) and r.get('status')=='success'
 initial_best=min(float(r['objective']) for r in initial if successful(r))
 best=np.minimum.accumulate([initial_best]+[float(r['objective']) if successful(r) else np.inf for r in records])
 assert np.isfinite(best).all() and (np.diff(best)<=0).all()
 improvement=100*(best[400]-best[800])/best[400]
 changes=(np.flatnonzero(np.diff(best)<0)+1).tolist()
 summary=dict(task=task,best_at_initialization_seconds=initial_best,best_at_200_seconds=float(best[200]),best_at_400_seconds=float(best[400]),best_at_800_seconds=float(best[800]),runtime_reduction_400_to_800_percent=float(improvement),last_improvement_evaluation=changes[-1] if changes else 0,improvement_evaluations=changes,unchanged_evaluations_after_last_improvement=800-(changes[-1] if changes else 0),post400_measured_runtime_sum_seconds=sum(float(r['objective']) for r in records[400:]),prior400_measured_runtime_sum_seconds=sum(float(r['objective']) for r in records[:400]))
 summaries.append(summary);curves.append((task,best));snapshot[task]=dict(initial=initial,records=records)
 for n,y in enumerate(best):rows_csv.append(dict(task=task,proposal_evaluations=n,batch_rounds=n/10,best_runtime_seconds=float(y),runtime_relative_to_evaluation400=float(y/best[400])))
assert summaries,'No completed 800-evaluation pilot queries'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors=plt.get_cmap('tab10').colors
for metric,stem,ylabel in [('relative','semanticopt_completed800_plateau','Best runtime / best runtime at evaluation 400'),('seconds','semanticopt_completed800_runtime_seconds','Best measured query runtime (seconds)')]:
 fig,ax=plt.subplots(figsize=(10.4,5.5))
 ax.axvspan(400,800,color='#F0B46A',alpha=.13,zorder=0)
 ax.axvline(400,color='#8B633A',ls='--',lw=1.25,label='400 evaluations (40 batches)')
 for i,(task,best) in enumerate(curves):
  y=best/best[400] if metric=='relative' else best
  ax.step(np.arange(801),y,where='post',color=colors[i%len(colors)],lw=1.65,label=task)
  ax.scatter([400,800],[y[400],y[800]],color=colors[i%len(colors)],s=35,zorder=3)
 if metric=='relative':
  aggregate=np.sum([best for _,best in curves],axis=0)
  aggregate=aggregate/aggregate[400]
  ax.step(np.arange(801),aggregate,where='post',color='#111111',lw=3,label=f'Sum of {len(curves)} incumbents')
  ax.scatter([400,800],[aggregate[400],aggregate[800]],color='#111111',s=45,zorder=4)
 if len(curves)==1:
  summary=summaries[0];last=summary['last_improvement_evaluation'];y=curves[0][1];y=y/y[400] if metric=='relative' else y
  ax.annotate(f"Last improvement: evaluation {last}\nNo further gain through 800",xy=(last,y[last]),xytext=(420,y[400]+.025*(max(y)-min(y))),fontsize=10,arrowprops=dict(arrowstyle='->',color='#555555'),va='bottom')
  ax.text(.97,.91,f"400 → 800: {summary['runtime_reduction_400_to_800_percent']:.2f}% lower runtime",ha='right',transform=ax.transAxes,fontsize=12,weight='bold',color='#195D74')
 ax.set(xlim=(0,800),xlabel='Proposal evaluations (10 per batch; 49 Bao starts excluded)',ylabel=ylabel,title=f'SemanticOPT convergence · completed 800-evaluation pilots (n={len(curves)})')
 ax.set_xticks(np.arange(0,801,100));ax.grid(axis='y',alpha=.2)
 low=min((best/best[400] if metric=='relative' else best).min() for _,best in curves);high=max((best/best[400] if metric=='relative' else best).max() for _,best in curves);pad=max((high-low)*.18,.005);ax.set_ylim(low-pad,high+pad)
 ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1.02,1),fontsize=9)
 fig.text(.42,.02,'Only fully completed 800-evaluation pilots; running and 400-budget queries excluded.\nIncumbents include the saved Bao initialization; timeouts count as evaluations but do not improve the incumbent.',ha='center',fontsize=8.2,color='#555555')
 fig.tight_layout(rect=(0,.09,.78,1))
 for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{suffix}',dpi=220,bbox_inches='tight',facecolor='white')
 plt.close(fig)
with (OUT/'convergence.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows_csv[0]));writer.writeheader();writer.writerows(rows_csv)
(OUT/'summary.json').write_text(json.dumps(dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),completed_800_queries=len(summaries),summaries=summaries),indent=2)+'\n')
(OUT/'observations_snapshot.json').write_text(json.dumps(snapshot,separators=(',',':'))+'\n')
sum400=sum(s['best_at_400_seconds'] for s in summaries)
sum800=sum(s['best_at_800_seconds'] for s in summaries)
improved=sum(s['runtime_reduction_400_to_800_percent']>0 for s in summaries)
notes=f'''# SemanticOPT improvement after 400 evaluations

Only pilot queries with a completed, saved 800-proposal trajectory are included. Queries completed at their 400-evaluation target cannot answer whether there is improvement after 400 and are excluded. Running pilots are also excluded, so the cohort is fixed over the entire plot.

The horizontal axis counts proposal evaluations, excluding 49 saved Bao initialization measurements; 400 proposals are 40 batches of ten, and 800 are 80 batches. Evaluation zero is the best uncensored successful Bao initialization result. The incumbent can improve only on uncensored successful measurements. Timeouts remain charged evaluations. No smoothing, extrapolation, padding, or statistical uncertainty bands are used.

The primary chart normalizes each query's best runtime by its own value at evaluation 400; a value below one shows improvement beyond that budget. Its black line shows the sum of the included queries' incumbent runtimes, divided by that sum at evaluation 400. This normalization is specific to the plateau question and differs from the earlier multi-method min–max score and normalized summed-runtime figures. The second chart exports the individual trajectories in seconds.

This snapshot contains {len(summaries)} completed 800-proposal pilots. {improved} improved between 400 and 800 evaluations. Their summed best runtime fell from {sum400:.6f} to {sum800:.6f} seconds, a {100*(sum400-sum800)/sum400:.2f}% reduction. The trajectories are descriptive evidence for this cohort, not an estimate of how all ten pilots or 200 held-out queries behave. The completed cohort may favor faster/easier queries.

Outputs: semanticopt_completed800_plateau.png/.pdf/.svg, semanticopt_completed800_runtime_seconds.png/.pdf/.svg, convergence.csv, summary.json, observations_snapshot.json.

Reproduce: python3 build_plot.py. All campaign inputs are read-only.
'''
(OUT/'README.md').write_text(notes)
print(json.dumps(summaries,indent=2))
