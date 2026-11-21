"""Fixed-cohort mean incumbent MIC, preserving the previous report's budget policy."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
SOURCE=Path('/root/reports/peptide_cap4000_graphs_20261119/capped_convergence.csv')
METHODS=['Original frozen FSBO','Online VAE + FSBO','SemanticOPT with fallback']
COLORS=['#1f77b4','#d62728','#9467bd']
curves=defaultdict(list)
for r in csv.DictReader(SOURCE.open()):
    curves[r['task'],r['method']].append((int(float(r['fresh'])),float(r['mic'])))
tasks=sorted({k[0] for k in curves})
assert len(tasks)==10
grid=np.arange(4001)
all_rows=[]
data={}
for method in METHODS:
    vals=np.array([[min(y for n,y in curves[task,method] if n<=x) for x in grid] for task in tasks])
    mean=vals.mean(axis=0)
    prefix=np.r_[0,np.cumsum(mean)]
    start=np.maximum(0,grid-199)
    smooth=(prefix[grid+1]-prefix[start])/(grid+1-start)
    data[method]=(mean,smooth)
    for x in range(0,4001,50):
        all_rows.append(dict(fresh_evaluation_budget=x,method=method,mean_best_mic=float(mean[x]),
                             same_200_evaluation_trailing_mean=float(smooth[x]),
                             tasks_with_actual_fresh_support=sum(curves[task,method][-1][0]>=x for task in tasks)))
with (OUT/'mean_mic_by_stage.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
for smoothed in (False,True):
    fig,ax=plt.subplots(figsize=(10.5,5.6))
    for method,color in zip(METHODS,COLORS):
        mean,smooth=data[method]
        displayed=smooth if smoothed else mean
        label=f'{method} (final mean: {mean[-1]:.2f} µM)'
        ax.plot(grid,displayed,color=color,label=label,lw=2)
        if not smoothed:
            sample=np.arange(0,4001,500)
            ax.scatter(sample,mean[sample],color=color,s=19,zorder=3)
    ax.set(title='Mean peptide convergence across the same ten tasks',xlabel='Fresh evaluation budget',
           ylabel='Mean best predicted MIC (µM) ↓',xlim=(0,4000),xticks=range(0,4001,500))
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=1,fontsize=10,frameon=False)
    subtitle='Same causal trailing 200-evaluation smoothing for all methods.' if smoothed else 'Exact cohort mean at each measured stage; dots mark 500-evaluation budgets.'
    fig.text(.5,.04,subtitle,ha='center',fontsize=9)
    fig.text(.5,.005,'Identical 100-point initialization excluded. Stopped incumbents retained through 4,000; no additional measurements implied.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.85))
    name='mean_mic_by_stage_smoothed' if smoothed else 'mean_mic_by_stage'
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

report=OUT/'REPORT.md'
text=report.read_text()
addition=['## Mean convergence requested by the user','',
          'The requested primary graph is the cohort mean best MIC at each evaluation stage, using the same ten tasks and established4000 cap. No new500-round experiment was requested. The CSV provides every50-evaluation budget; dots and the table below show500-evaluation checkpoints. Stopped incumbents remain in the fixed cohort, without implying new measurements.','',
          '[Exact mean convergence](mean_mic_by_stage.png) · [Same200-evaluation smoothing](mean_mic_by_stage_smoothed.png) · [Vector PDF](mean_mic_by_stage.pdf)','',
          '| Fresh budget | Frozen mean MIC | Online mean MIC | SemanticOPT mean MIC |','|---|---:|---:|---:|']
for x in range(0,4001,500):
    addition.append(f'| {x} | '+ ' | '.join(f'{data[m][0][x]:.3f}' for m in METHODS)+' |')
text=text.split('\n## Mean convergence requested by the user')[0]
report.write_text(text+'\n'+'\n'.join(addition)+'\n')
print('\n'.join(addition[-11:]),flush=True)
