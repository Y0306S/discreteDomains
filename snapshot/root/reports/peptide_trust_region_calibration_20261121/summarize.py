import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent


def main():
    d=json.loads((OUT/'results.json').read_text())
    assert d['status']=='complete' and len(d['rows'])==23*8
    rows=[]
    for cohort in ['source_development','pep999_diagnostic']:
        models=sorted({r['model'] for r in d['rows'] if r['cohort']==cohort})
        for model in models:
            for radius in d['radii']:
                tasks=[r for r in d['rows'] if r['cohort']==cohort and r['model']==model and r['radius']==radius]
                for edits in [1,2,3,100]:
                    n=sum(r['draws'] for r in tasks)
                    c=[r['counts'][str(edits)] for r in tasks]
                    row=dict(cohort=cohort,model=model,radius=radius,local_edits=edits,tasks=len(tasks),draws=n,
                        valid_percent=100*sum(x['valid_draws'] for x in c)/n,
                        novel_valid_percent=100*sum(x['novel_valid_draws'] for x in c)/n,
                        mean_unique_novel_valid_per192=float(np.mean([x['unique_novel_valid'] for x in c])),
                        minimum_unique_novel_valid_per192=min(x['unique_novel_valid'] for x in c),
                        tasks_with_any_novel_valid=sum(x['unique_novel_valid']>0 for x in c))
                    rows.append(row)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(oracle_calls=0,training_performed=False,production_changes=False,summaries=rows,development_frontier={})
    for edits in [1,2]:
        candidates=[r for r in rows if r['cohort']=='source_development' and r['model']=='adapted_before_online' and r['local_edits']==edits]
        result['development_frontier'][str(edits)]=sorted(candidates,key=lambda r:r['mean_unique_novel_valid_per192'],reverse=True)[:3]
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['development_frontier'],indent=2))
    plot(rows,d)


def plot(rows,d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for edits,color in [(1,'#0072B2'),(2,'#009E73'),(3,'#E69F00')]:
        rs=[r for r in rows if r['cohort']=='source_development' and r['model']=='adapted_before_online' and r['local_edits']==edits]
        label=f'At most {edits} local edits'
        axes[0].plot([r['radius'] for r in rs],[r['valid_percent'] for r in rs],'-o',color=color,label=label,ms=3)
        axes[1].plot([r['radius'] for r in rs],[r['mean_unique_novel_valid_per192'] for r in rs],'-o',color=color,label=label,ms=3)
    for model,color,label in [('adapted_before_online','#777777','Before online'),('pep999_after_round20','#CC79A7','After round 20'),('pep999_after_round40','#56B4E9','After round 40')]:
        rs=[r for r in rows if r['cohort']=='pep999_diagnostic' and r['model']==model and r['local_edits']==2]
        axes[2].plot([r['radius'] for r in rs],[r['mean_unique_novel_valid_per192'] for r in rs],'-o',color=color,label=label,ms=3)
    axes[0].set(title='Source development: accepted decodes',ylabel='Pass reference and local edit limits (%)',ylim=(0,105))
    axes[1].set(title='Source development: useful novelty',ylabel='Mean unique unseen valid sequences / 192 draws')
    axes[2].set(title='pep_999 diagnostic: two local edits',ylabel='Unique unseen valid sequences / 192 draws')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([.025,.05,.1,.2,.4,.8],[str(v) for v in [.025,.05,.1,.2,.4,.8]])
        ax.set_xlabel('Latent trust-region length');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.text(.5,.015,'Ten source development tasks; 3 paired seeds × 64 decodes per task/radius. Exact reference constraint always applies. No MIC calls.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/('calibration.'+ext),dpi=180)


if __name__=='__main__':
    main()
