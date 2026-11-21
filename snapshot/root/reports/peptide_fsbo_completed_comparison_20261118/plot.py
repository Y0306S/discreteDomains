"""Completed original FSBO versus the completed first adaptation pilot."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
PILOT=Path('/root/data/logs/peptide_vae_fsbo_adaptation_test10_20261117')
BASE=Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo/runs')
TASKS=json.loads((PILOT/'config.json').read_text())['tasks']
GRID=np.arange(0,5001,25)


def curve(d):
    initial=[r for r in d['observations'] if r['phase']=='initialization']
    start=min(r['mic'] for r in initial)
    best=start;fresh=0;points=[(0,best)];group=[];key=None
    def commit(batch):
        nonlocal fresh,best
        fresh+=sum(not r['cached'] for r in batch)
        best=min([best]+[r['mic'] for r in batch if r['feasible']])
        points.append((fresh,best))
    for row in d['observations']:
        if row['phase']=='initialization':continue
        current=(row['phase'],row['round'],row['elapsed'])
        if key is not None and current!=key:
            commit(group);group=[]
        key=current;group.append(row)
    if group:commit(group)
    assert fresh==d['fresh_evaluations']
    assert np.isclose(best,d['best_feasible_mic'])
    arr=np.asarray(points)
    ix=np.maximum(0,np.searchsorted(arr[:,0],GRID,side='right')-1)
    return start,arr[ix,1]


def main():
    rows=[];curves={'Original FSBO':[],'Earlier adapted FSBO':[]};raw={k:[] for k in curves}
    source_hashes={}
    for task in TASKS:
        with tarfile.open(BASE/(task+'.tar.gz')) as archive:
            payload=archive.extractfile('history.json').read()
            manifest=json.load(archive.extractfile('manifest.json'))
        assert hashlib.sha256(payload).hexdigest()==manifest['members']['history.json']
        original=json.loads(payload)
        newpath=PILOT/'evaluation/adapted_online'/task/'result.json'
        adapted=json.loads(newpath.read_text())
        source_hashes[task]={'original_history':hashlib.sha256(payload).hexdigest(),
                             'adapted_result':hashlib.sha256(newpath.read_bytes()).hexdigest()}
        assert original['status']==adapted['status']=='complete'
        initial_old=[(o['sequence'],o['score']) for o in original['observations'] if o['phase']=='initialization']
        initial_new=[(o['sequence'],o['score']) for o in adapted['observations'] if o['phase']=='initialization']
        assert initial_old==initial_new
        for label,d in [('Original FSBO',original),('Earlier adapted FSBO',adapted)]:
            start,best=curve(d)
            gain=1-best/start
            curves[label].append(gain);raw[label].append(best)
            new=[o for o in d['observations'] if o['phase']!='initialization']
            rows.append(dict(task=task,method=label,initial_best_mic=start,final_best_mic=d['best_feasible_mic'],
                improvement_percent=100*(1-d['best_feasible_mic']/start),fresh_evaluations=d['fresh_evaluations'],
                acquisition_proposals=d['proposals'],new_feasible_proposals=sum(o['feasible'] for o in new),
                joint_updates=d['e2e_updates']))
    with (OUT/'task_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    colors={'Original FSBO':'#1768AC','Earlier adapted FSBO':'#D95D39'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,5.4),gridspec_kw={'width_ratios':[1.05,1.2]})
    rng=np.random.RandomState(20261118);draws=rng.randint(0,10,(5000,10))
    exports=[]
    for label in curves:
        arr=np.asarray(curves[label]);mean=arr.mean(0)
        lo,hi=np.percentile(arr[draws].mean(1),[2.5,97.5],axis=0)
        axes[0].step(GRID,mean*100,where='post',label=label,color=colors[label],lw=2.4)
        axes[0].fill_between(GRID,lo*100,hi*100,step='post',color=colors[label],alpha=.13)
        for i,budget in enumerate(GRID):
            exports.append(dict(method=label,fresh_evaluations=int(budget),mean_improvement_percent=mean[i]*100,
                bootstrap95_low=lo[i]*100,bootstrap95_high=hi[i]*100,
                mean_best_mic=float(np.mean(raw[label],axis=0)[i]),
                tasks_reaching_budget=sum(r['fresh_evaluations']>=budget for r in rows if r['method']==label)))
    axes[0].set(title='Convergence across the same 10 tasks',xlabel='Fresh evaluations after 100 shared initial measurements',
        ylabel='Mean improvement from initialization (%)',xlim=(0,5000),ylim=(-2,70))
    axes[0].legend(loc='upper left',frameon=False,fontsize=10)
    axes[0].grid(alpha=.2)
    y=np.arange(10)
    old=np.array([r['final_best_mic'] for r in rows if r['method']=='Original FSBO'])
    new=np.array([r['final_best_mic'] for r in rows if r['method']=='Earlier adapted FSBO'])
    axes[1].hlines(y,old,new,color='#B5BBC2',lw=2)
    axes[1].scatter(old,y,c=colors['Original FSBO'],s=48,label='Original FSBO',zorder=3)
    axes[1].scatter(new,y,c=colors['Earlier adapted FSBO'],s=48,marker='s',label='Earlier adapted FSBO',zorder=3)
    axes[1].set(yticks=y,yticklabels=TASKS,title='Final best feasible MIC by task',
        xlabel='MIC (µM, log scale; lower is better)',xscale='log',xlim=(2,180))
    axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.2,which='both')
    fig.suptitle('Completed comparison: original vs earlier adapted FSBO',fontsize=15,fontweight='bold',y=.98)
    fig.text(.055,.045,'Earlier adaptation: train800 VAE + target-only online updates. The new full-source scratch-retraining run is not included.',fontsize=9,color='#444444')
    fig.text(.055,.012,'Bands: 95% task-bootstrap intervals, one seed. Incumbents carry forward after proposal exhaustion; 2/10 original tasks reach 5,000 fresh evaluations.',fontsize=8.5,color='#555555')
    fig.tight_layout(rect=[0,.08,1,.93])
    for extension in ['png','pdf','svg']:fig.savefig(OUT/('completed_fsbo_comparison.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)
    with (OUT/'convergence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(exports[0]));w.writeheader();w.writerows(exports)
    summary=dict(tasks=TASKS,original_mean_final_mic=float(old.mean()),earlier_adapted_mean_final_mic=float(new.mean()),
        original_mean_task_improvement_percent=float(np.mean(curves['Original FSBO'],axis=0)[-1]*100),
        earlier_adapted_mean_task_improvement_percent=float(np.mean(curves['Earlier adapted FSBO'],axis=0)[-1]*100),
        adapted_wins=int((new<old).sum()),adapted_losses=int((new>old).sum()),
        adapted_fresh_evaluations=sum(r['fresh_evaluations'] for r in rows if r['method']=='Earlier adapted FSBO'),
        adapted_new_feasible_proposals=sum(r['new_feasible_proposals'] for r in rows if r['method']=='Earlier adapted FSBO'),
        source_hashes=source_hashes)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
