"""Historical timing audit and measured paired pilot reports (never fill missing runs)."""
import argparse
import csv
import json
import tarfile
from collections import defaultdict
from common import *
import numpy as np


def csv_write(path,rows):
    if not rows:
        return
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def history(path):
    with tarfile.open(path) as t:
        raw=t.extractfile('history.json').read()
        manifest=json.load(t.extractfile('manifest.json'))
    expected=manifest.get('members',{}).get('history.json')
    assert expected==hashlib.sha256(raw).hexdigest()
    d=json.loads(raw)
    assert d['status']=='complete'
    return d


def historical():
    cfg=prepare()
    with open('/root/reports/peptide_convergence_20261101/fsbo_local_task_summary.csv') as f:
        allrows=list(csv.DictReader(f))
    chosen=[r for r in allrows if r['task'] in cfg['tasks']]
    assert len(chosen)==10
    rows=[]
    for r in chosen:
        rows.append(dict(task=r['task'],method='historical_frozen_FSBO',
            seconds=float(r['elapsed_seconds']),fresh_evaluations=int(r['fresh_evaluations']),
            updates=0,initial_best_mic=float(r['initial_best_mic']),final_best_mic=float(r['final_best_mic'])))
    for task in cfg['tasks']:
        path=Path('/root/data/logs/peptide_test200_baselines_20261027_v2/lolbo/runs')/(task+'.tar.gz')
        d=history(path)
        assert d['reference']==read(SOURCE/'recipe.json')['references'][task]
        rows.append(dict(task=task,method='historical_native_LOLBO',seconds=d['seconds'],
            fresh_evaluations=d['fresh_evaluations'],updates=d['e2e_updates'],
            initial_best_mic=min(r['mic'] for r in d['observations'] if r['phase']=='initialization'),
            final_best_mic=d['best_feasible_mic']))
    csv_write(ROOT/'historical_task_timings.csv',rows)
    old=read(OLD/'fsbo/status.json')
    values=np.asarray([float(r['elapsed_seconds']) for r in allrows])
    summary=dict(fsbo_full_meta_training_seconds=old['elapsed_seconds'],fsbo_full_meta_steps=old['requested_steps'],
        fsbo_full_meta_best_step=old['best_step'],fsbo_tasks=len(allrows),
        fsbo_mean_task_seconds=float(values.mean()),fsbo_median_task_seconds=float(np.median(values)),
        fsbo_selected10_mean_seconds=float(np.mean([r['seconds'] for r in rows if r['method']=='historical_frozen_FSBO'])),
        lolbo_selected10_mean_seconds=float(np.mean([r['seconds'] for r in rows if r['method']=='historical_native_LOLBO'])),
        lolbo_selected10_mean_updates=float(np.mean([r['updates'] for r in rows if r['method']=='historical_native_LOLBO'])),
        warning='Historical methods differ in surrogate, device placement, dates and contention. Their runtime ratio does not isolate VAE retraining cost.',
        current_results_available=False)
    write(ROOT/'historical_summary.json',summary)
    print(json.dumps(summary),flush=True)


def curve(result,budget):
    # Reveal batches together, including infeasible and recenter oracle calls.
    rows=result['observations']
    best=min(r['mic'] for r in rows if r['phase']=='initialization')
    fresh=0
    values=[(0,best,0.)]
    group=[];key=None
    for row in rows:
        if row['phase']=='initialization':
            continue
        current=(row['phase'],row['elapsed'])
        if key is not None and current!=key:
            fresh+=sum(not r['cached'] for r in group)
            best=min([best]+[r['mic'] for r in group if r['feasible']])
            values.append((fresh,best,group[-1].get('wall_elapsed',group[-1]['elapsed'])))
            group=[]
        key=current;group.append(row)
    if group:
        fresh+=sum(not r['cached'] for r in group)
        best=min([best]+[r['mic'] for r in group if r['feasible']])
        values.append((fresh,best,group[-1].get('wall_elapsed',group[-1]['elapsed'])))
    assert fresh==result['fresh_evaluations']
    arr=np.asarray(values)
    indices=np.maximum(0,np.searchsorted(arr[:,0],budget,side='right')-1)
    return arr[indices,1],arr


def report():
    cfg=prepare()
    results={}
    for task in cfg['tasks']:
        for arm in cfg['arms']:
            path=ROOT/'evaluation'/arm/task/'result.json'
            if path.exists():
                d=read(path)
                assert d['status']=='complete'
                results[task,arm]=d
    if len(results)!=30:
        write(ROOT/'report_status.json',dict(status='awaiting_all_paired_results',completed=len(results),required=30))
        return
    from Levenshtein import distance
    pool=read('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json')
    rows=[]
    budget=np.arange(0,5001,25)
    curves=defaultdict(list)
    time_curves=defaultdict(list)
    max_common_time=min(r['wall_seconds'] for r in results.values())
    time_budget=np.linspace(0,max_common_time,201)
    for (task,arm),r in results.items():
        initial=r['observations'][:100]
        assert [x['sequence'] for x in initial]==pool[task]['sequences']
        assert [x['score'] for x in initial]==pool[task]['scores']
        cache=set(pool[task]['sequences'])
        for obs in r['observations'][100:]:
            assert obs['cached']==(obs['sequence'] in cache)
            cache.add(obs['sequence'])
            assert obs['feasible']==(1-distance(obs['sequence'],r['reference'])/len(r['reference'])>=.75)
            assert np.isfinite(obs['mic']) and obs['mic']==-obs['score']
        assert r['proposals']<=5000 and r['fresh_evaluations']<=5000
        assert r['proposals']==5000 or r['fresh_evaluations']==5000
        ys,actual=curve(r,budget)
        curves[arm].append(ys)
        time_curves[arm].append(actual[np.maximum(0,np.searchsorted(actual[:,2],time_budget,side='right')-1),1])
        final=r['best_feasible_mic'];start=r['initialization_best_mic']
        rows.append(dict(task=task,arm=arm,initial_best_mic=start,best_mic_at_1000=float(ys[40]),
            final_best_mic=final,improvement_percent=100*(start-final)/start,
            wall_seconds=r['wall_seconds'],active_seconds=r['seconds'],fresh_evaluations=r['fresh_evaluations'],
            acquisition_proposals=r['proposals'],joint_updates=r['e2e_updates'],
            **{k+'_seconds':v for k,v in r['timing'].items()}))
    csv_write(ROOT/'paired_results.csv',rows)
    offline={}
    for name,path in [('vae',ROOT/'vae/status.json'),('embedding_refresh',ROOT/'corpus/status.json'),('fsbo_meta',ROOT/'fsbo/status.json')]:
        offline[name]=read(path)['elapsed_seconds']
    offline_total=sum(offline.values())
    stage_wall={p.stem:read(p)['elapsed_seconds'] for p in (ROOT/'timings').glob('*.json')}
    index={(r['task'],r['arm']):r for r in rows}
    comparisons=[]
    for reference,variant in [('author_frozen','adapted_frozen'),('adapted_frozen','adapted_online'),('author_frozen','adapted_online')]:
        a=np.asarray([index[t,reference]['final_best_mic'] for t in cfg['tasks']])
        b=np.asarray([index[t,variant]['final_best_mic'] for t in cfg['tasks']])
        ta=np.asarray([index[t,reference]['wall_seconds'] for t in cfg['tasks']])
        tb=np.asarray([index[t,variant]['wall_seconds'] for t in cfg['tasks']])
        rng=np.random.RandomState(20261117)
        draws=rng.randint(0,10,size=(5000,10))
        delta=(a-b)[draws].mean(1)
        ratios=tb/ta
        comparisons.append(dict(reference=reference,variant=variant,
            wins=int((b<a).sum()),losses=int((b>a).sum()),ties=int((b==a).sum()),
            mean_paired_mic_reduction=float((a-b).mean()),
            paired_task_bootstrap95_mic_reduction=np.percentile(delta,[2.5,97.5]).tolist(),
            mean_paired_runtime_ratio=float(ratios.mean()),median_paired_runtime_ratio=float(np.median(ratios)),
            ratio_of_total_online_seconds=float(tb.sum()/ta.sum()),
            ratio_including_offline_cost_for_this10=float((tb.sum()+offline_total)/ta.sum()) if reference=='author_frozen' else None))
    summary=dict(status='complete',tasks=cfg['tasks'],offline_seconds=offline,offline_total_seconds=offline_total,
        stage_process_wall_seconds=stage_wall,
        offline_amortized_per_task_at10=offline_total/10,offline_amortized_per_task_at200=offline_total/200,
        comparisons=comparisons,limitations=[cfg['limitation'],
        'All new arms use the same PyTorch 2.5.1 / GPyTorch 1.15.2 GPU runner. Old FSBO used older GPyTorch; historical runtime is supplemental.',
        'GPU hardware is shared with existing work; matched task/arm hardware and recorded utilization do not eliminate temporal contention.',
        'Online joint optimization uses exact FSBO marginal likelihood plus native constraint predictive likelihood; it is a new FSBO adaptation variant.',
        'Completed incumbents carry forward after proposal exhaustion; paired_results.csv records actual fresh-call support for every task.',
        'Offline VAE sees all 800 source tasks, including FSBO internal validation tasks. The 10 held-out evaluation tasks remain excluded.'])
    write(ROOT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    labels=dict(author_frozen='Author VAE, frozen',adapted_frozen='Train800 VAE, frozen',adapted_online='Train800 VAE + online updates')
    export=[]
    for arm in cfg['arms']:
        arr=np.asarray(curves[arm]);mean=arr.mean(0)
        rng=np.random.RandomState(20261117)
        lo,hi=np.percentile(arr[rng.randint(0,10,(5000,10))].mean(1),[2.5,97.5],axis=0)
        axes[0].plot(budget,mean,label=labels[arm]);axes[0].fill_between(budget,lo,hi,alpha=.13)
        axes[1].plot(time_budget/60,np.mean(time_curves[arm],axis=0),label=labels[arm])
        for x,y,l,h in zip(budget,mean,lo,hi):
            export.append(dict(arm=arm,fresh_evaluations=int(x),mean_best_mic=float(y),bootstrap95_low=float(l),bootstrap95_high=float(h),
                tasks_reaching_budget=sum(results[t,arm]['fresh_evaluations']>=x for t in cfg['tasks'])))
    axes[0].set(xlabel='Fresh logical APEX evaluations',ylabel='Mean best feasible MIC (µM, lower is better)')
    axes[1].set(xlabel='Wall time per task (minutes)',ylabel='Mean best feasible MIC (µM)')
    means=[np.mean([index[t,a]['wall_seconds'] for t in cfg['tasks']])/60 for a in cfg['arms']]
    axes[2].bar(range(3),means)
    axes[2].set(xticks=range(3),xticklabels=['Author frozen','Adapted frozen','Adapted online'],ylabel='Mean wall minutes per task')
    axes[2].tick_params(axis='x',rotation=15)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ['png','pdf','svg']:
        fig.savefig(ROOT/('comparison.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)
    csv_write(ROOT/'convergence.csv',export)
    lines=['# Peptide VAE / FSBO adaptation pilot','',
        'Completed 10 held-out tasks × three matched arms. Lower MIC is better.','',
        '| Arm | Mean final MIC (µM) | Mean online minutes | Mean updates |',
        '|---|---:|---:|---:|']
    for arm in cfg['arms']:
        group=[index[t,arm] for t in cfg['tasks']]
        lines.append('| %s | %.4f | %.2f | %.2f |'%(labels[arm],np.mean([r['final_best_mic'] for r in group]),np.mean([r['wall_seconds'] for r in group])/60,np.mean([r['joint_updates'] for r in group])))
    lines+=['','Offline VAE + embedding refresh + FSBO meta-training: %.2f hours.'%(offline_total/3600),'',
        'See summary.json for paired quality differences, bootstrap intervals, runtime ratios and amortized offline costs.','',
        'These are 10 tasks and one optimizer seed. A better VAE or improved search is an experimental outcome, not an assumption.','',*['- '+x for x in summary['limitations']]]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(ROOT/'report_status.json',dict(status='complete',completed=30,required=30))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['historical','report'])
    args=parser.parse_args()
    historical() if args.stage=='historical' else report()
