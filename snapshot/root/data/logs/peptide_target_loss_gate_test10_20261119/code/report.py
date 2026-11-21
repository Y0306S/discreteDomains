"""Paired MIC/retention report at the unchanged proposal4000 cap."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT,read,write

PREVIOUS_CURVES=ROOT/'comparison_inputs/capped_convergence.csv'
METHOD='Online VAE + FSBO + target rollback'
COLORS={'Original frozen FSBO':'#1f77b4','Online VAE + FSBO':'#d62728',
        'SemanticOPT with fallback':'#9467bd',METHOD:'#009E73'}


def csv_write(name, rows):
    with (ROOT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    cfg=read(ROOT/'config.json');tasks=cfg['tasks']
    old=defaultdict(list)
    for row in csv.DictReader(PREVIOUS_CURVES.open()):
        if row['task'] in tasks:
            old[row['method'],row['task']].append({k:float(row[k]) for k in ('fresh','mic','time','proposals')})
    records=[];new=defaultdict(list)
    for task in tasks:
        result=read(ROOT/'evaluation/adapted_online'/task/'result.json')
        assert result['status']=='complete' and result['proposals']<=4000 and result['fresh_evaluations']<=4000
        assert result['proposals']==4000 or result['fresh_evaluations']==4000
        best=result['initialization_best_mic'];fresh=0;proposals=0
        new[METHOD,task].append(dict(fresh=0.,mic=best,time=0.,proposals=0.))
        groups=[];group=[];key=None
        for row in result['observations']:
            if row['phase']=='initialization':continue
            current=(row['phase'],row['elapsed'])
            if key is not None and current!=key:groups.append(group);group=[]
            key=current;group.append(row)
        if group:groups.append(group)
        for group in groups:
            fresh+=sum(not r['cached'] for r in group)
            proposals+=sum(r['phase']=='acquisition' for r in group)
            best=min([best]+[r['mic'] for r in group if r['feasible']])
            new[METHOD,task].append(dict(fresh=float(fresh),mic=best,
                time=group[-1].get('wall_elapsed',group[-1]['elapsed']),proposals=float(proposals)))
        assert fresh==result['fresh_evaluations']
        events=result['adaptation_events']
        for event in events:
            if event['accepted']:
                after=event['target_loss_after']['mean_target_loss']
                assert after<=event['target_loss_before']['mean_target_loss']+1e-12
                assert after<=event['target_loss_initial_baseline']['mean_target_loss']+1e-12
            else:assert event['vae_change_l2']==0
        records.append(dict(task=task,final_mic=best,initial_mic=result['initialization_best_mic'],
            proposals=proposals,fresh_evaluations=fresh,wall_seconds=result['wall_seconds'],
            update_attempts=len(events),updates_accepted=sum(e['accepted'] for e in events),
            updates_rolled_back=sum(not e['accepted'] for e in events)))
    curves={**old,**new};methods=[m for m in COLORS if all((m,t) in curves for t in tasks)]
    assert len(methods)==4
    export=[];summary=[]
    x=np.arange(0,4001)
    fig,ax=plt.subplots(figsize=(11,6.5))
    for method in methods:
        values=[]
        for task in tasks:
            c=curves[method,task]
            assert all(v['proposals']<=4000 for v in c)
            arr=np.asarray([[v['fresh'],v['mic']] for v in c])
            y=arr[np.maximum(0,np.searchsorted(arr[:,0],x,side='right')-1),1]
            values.append(y)
            export.extend(dict(method=method,task=task,**r) for r in c)
        mean=np.asarray(values).mean(0)
        # Use the same causal trailing200 smoothing for every method.
        cumulative=np.r_[0,np.cumsum(mean)];start=np.maximum(0,np.arange(len(mean))-199)
        smooth=(cumulative[1:]-cumulative[start])/(np.arange(len(mean))-start+1)
        ax.plot(x,smooth,label=method,color=COLORS[method],linewidth=2.4)
        summary.append(dict(method=method,mean_final_mic=float(np.mean([curves[method,t][-1]['mic'] for t in tasks]))))
    ax.set(xlabel='Fresh evaluations (shared initialization100 excluded)',ylabel='Mean best feasible MIC (µM, lower is better)',
        title='Same 10 peptides · target-loss rollback comparison',xlim=(0,4000))
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.text(.5,.01,'Same seed1, initialization100 and proposal4000 cap. All methods use trailing200 smoothing.\nFinished incumbents carry forward after proposal exhaustion; actual fresh support is exported.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ('png','pdf','svg'):fig.savefig(ROOT/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    paired=[]
    for method in methods[:-1]:
        differences=[curves[method,t][-1]['mic']-curves[METHOD,t][-1]['mic'] for t in tasks]
        paired.append(dict(reference=method,wins=sum(d>1e-8 for d in differences),
            losses=sum(d < -1e-8 for d in differences),ties=sum(abs(d)<=1e-8 for d in differences),
            mean_mic_reduction=float(np.mean(differences))))
    csv_write('task_results.csv',records);csv_write('capped_convergence.csv',export);csv_write('paired_comparison.csv',paired)
    write(ROOT/'summary.json',dict(status='complete',tasks=tasks,methods=summary,paired=paired,
        execution=read(ROOT/'execution_policy.json'),all_accepted_target_losses_nonworsening=True,
        all_rejected_weight_deltas_zero=True,single_seed=True,offline_training_reused=True))
    table='\n'.join(f"| {r['method']} | {r['mean_final_mic']:.4f} |" for r in summary)
    (ROOT/'RESULTS.md').write_text(f'''# Target-loss rollback comparison

Same ten held-out peptide tasks, seed1, initialization100, batch50 and proposal/fresh caps4000.

| Method | Mean final MIC (µM, lower is better) |
| --- | ---: |
{table}

![Convergence](comparison.png)

Earlier comparison trajectories come from {PREVIOUS_CURVES} and use the same proposal cap.
This run reuses the previous offline VAE/FSBO assets. Accepted updates pass the
new fixed-target no-worsening gate and original source gate; rejected updates
restore the VAE exactly. See task_results.csv for per-task updates, MIC and runtime.
Runtime is descriptive: previous experiment used A800, current experiment uses
A16 with shared workloads. Ten tasks and one optimizer seed do not establish
general significance. No future test observations enter validation.
''')
    write(ROOT/'report_status.json',dict(status='complete',tasks=len(tasks)))


if __name__=='__main__':main()
