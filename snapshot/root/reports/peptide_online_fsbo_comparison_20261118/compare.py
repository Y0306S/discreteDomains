"""Read-only, batch-causal snapshot of the ten held-out peptide comparisons."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tarfile
import time
from datetime import datetime, timezone

sys.path.insert(0, '/root/data/cys/BOLT/optimization/peptides')
import torch
from Levenshtein import distance

OUT = Path(__file__).resolve().parent
ONLINE = Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
FROZEN = Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo/runs')
SEM = Path('/root/data/cys/baseline/campaigns/semanticopt_peptide_test10_5000_20261117')
TASKS = json.loads((ONLINE/'config.json').read_text())['tasks']
POOL = json.loads(Path('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json').read_text())

def dump_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

def make_curve(rows, task):
    initial = [r for r in rows if r['phase']=='initialization']
    assert [r['sequence'] for r in initial] == POOL[task]['sequences']
    assert [r['score'] for r in initial] == POOL[task]['scores']
    best = min(r['mic'] for r in initial)
    fresh = 0
    values = [(0, best)]
    seen = set(r['sequence'] for r in initial)
    batch = []
    last = None
    feasible_fresh = 0
    def commit():
        nonlocal best, fresh
        fresh += sum(not r['cached'] for r in batch)
        best = min([best]+[r['mic'] for r in batch if r['feasible']])
        values.append((fresh, best))
    for r in rows[len(initial):]:
        assert math.isfinite(r['mic']) and r['mic'] == -r['score']
        assert r['cached'] == (r['sequence'] in seen)
        seen.add(r['sequence'])
        assert r['feasible'] == (1-distance(r['sequence'],POOL[task]['reference'])/len(POOL[task]['reference']) >= .75)
        feasible_fresh += bool(not r['cached'] and r['feasible'])
        key = (r['phase'], r['elapsed'])
        if last is not None and key != last:
            commit()
            batch = []
        batch.append(r)
        last = key
    if batch:
        commit()
    return values, fresh, feasible_fresh

def at(curve, budget):
    assert budget <= curve[-1][0]
    return min(y for x,y in curve if x <= budget)

def main():
    started = time.time()
    curves = {}
    progress = []
    online_states = {}
    for task in TASKS:
        with tarfile.open(FROZEN/(task+'.tar.gz')) as archive:
            raw = archive.extractfile('history.json').read()
            manifest = json.load(archive.extractfile('manifest.json'))
        assert hashlib.sha256(raw).hexdigest() == manifest['members']['history.json']
        frozen = json.loads(raw)
        assert frozen['status']=='complete' and frozen['vae_policy']=='frozen_author'
        curve, n, nf = make_curve(frozen['observations'],task)
        assert n == frozen['fresh_evaluations']
        curves[task,'Frozen VAE + FSBO'] = curve
        progress.append(dict(task=task,method='Frozen VAE + FSBO',status='complete',fresh_evaluations=n,proposals=frozen['proposals'],best_mic=curve[-1][1],fresh_feasible=nf,wall_seconds=frozen['seconds']))

        dbpath = SEM/'peptide/runs'/task/'observations.sqlite'
        with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
            db.execute('BEGIN')
            raw = db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
            initial = json.loads(db.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0])
        assert [r['configuration']['sequence'] for r in initial] == POOL[task]['sequences']
        assert [r['objective'] for r in initial] == [-y for y in POOL[task]['scores']]
        assert [n for n,v in raw] == list(range(len(raw)))
        imports = json.loads((SEM/'evidence/import_audit.json').read_text())
        expected = next(r['source_prefix_sha256'] for r in imports['tasks'] if r['task']==task)
        assert hashlib.sha256(json.dumps(raw[:1000],separators=(',',':')).encode()).hexdigest()==expected
        records = [json.loads(v) for n,v in raw]
        best = min(r['objective'] for r in initial)
        curve = [(0,best)]
        seen = set(POOL[task]['sequences'])
        for i,r in enumerate(records,1):
            seq = r['configuration']['sequence']
            assert seq not in seen and r['evaluation']==i and r['status']=='success' and r['feasible']
            seen.add(seq)
            assert math.isfinite(r['objective'])
            assert 1-distance(seq,POOL[task]['reference'])/len(POOL[task]['reference']) >= .75
            best = min(best,r['objective'])
            if i%50==0:
                curve.append((i,best))
        assert len(records)%50==0
        curves[task,'SemanticOPT (with fallback)'] = curve
        progress.append(dict(task=task,method='SemanticOPT (with fallback)',status='complete' if len(records)==5000 else 'running',fresh_evaluations=len(records),proposals=len(records),best_mic=best,random_fallback=sum(r['proposal_source']=='random_fallback' for r in records)))

        folder = ONLINE/'evaluation/adapted_online'/task
        result = folder/'result.json'
        checkpoint = folder/'checkpoint.pt'
        if not checkpoint.exists() and not result.exists():
            progress.append(dict(task=task,method='Online VAE + retrained FSBO',status='pending'))
            continue
        if result.exists():
            d = json.loads(result.read_text())
            state = dict(rows=d['observations'],progress=d,timing=d['timing'],wall_seconds=d['wall_seconds'],adaptation_events=d['adaptation_events'])
            status = 'complete'
            source = result
        else:
            state = torch.load(checkpoint,map_location='cpu',weights_only=False)['state']
            status = 'running'
            source = checkpoint
        online_states[task] = state
        curve,n,nf = make_curve(state['rows'],task)
        assert n==state['progress']['fresh_evaluations']
        curves[task,'Online VAE + retrained FSBO'] = curve
        progress.append(dict(task=task,method='Online VAE + retrained FSBO',status=status,fresh_evaluations=n,proposals=state['progress']['proposals'],rounds=state['progress']['rounds'],best_mic=curve[-1][1],fresh_feasible=nf,wall_seconds=state['wall_seconds'],online_updates=len(state['adaptation_events']),accepted_updates=sum(e['accepted'] for e in state['adaptation_events']),snapshot_mtime=source.stat().st_mtime,**{k+'_seconds':v for k,v in state['timing'].items()}))
        print(task, status, n, flush=True)

    methods = ['Frozen VAE + FSBO','Online VAE + retrained FSBO','SemanticOPT (with fallback)']
    comparisons = []
    for task in TASKS:
        for budget in [1000,2000,2500]:
            row = dict(task=task,fresh_budget=budget,initial_best_mic=min(-y for y in POOL[task]['scores']))
            for method in methods:
                c = curves.get((task,method))
                if c and c[-1][0]>=budget:
                    row[method] = at(c,budget)
            comparisons.append(row)
    paired = [r for r in comparisons if r['fresh_budget']==2500 and all(m in r for m in methods)]
    summary = dict(snapshot_started_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),snapshot_finished_utc=datetime.now(timezone.utc).isoformat(),tasks=TASKS,online_started=len(online_states),online_completed=sum(r['method']==methods[1] and r['status']=='complete' for r in progress),paired_budget=2500,paired_tasks=[r['task'] for r in paired],mean_best_mic={m:sum(r[m] for r in paired)/len(paired) for m in methods},mean_task_improvement_percent={m:sum(100*(r['initial_best_mic']-r[m])/r['initial_best_mic'] for r in paired)/len(paired) for m in methods},online_vs_frozen_wins=sum(r[methods[1]]<r[methods[0]] for r in paired),online_vs_semantic_wins=sum(r[methods[1]]<r[methods[2]] for r in paired),audits='Passed: exact identical 100 initial sequences and MIC values, frozen archive hashes, SemanticOPT imported prefix hashes, constraints, cache/unique sequence accounting, finite scores, batch-causal curves.',limitations=['Only started online tasks contribute to the three-way mean; pending tasks are never flattened or imputed.','Fresh evaluations exclude the 100 initial observations, count infeasible proposals and recenter calls, and exclude cache hits. Batches are revealed only when fully within budget.','SemanticOPT includes labeled random fallback; this is a hybrid campaign result.','One seed per task/method; historical baseline hardware and software differ.','The full revised method changes offline VAE training and FSBO training as well as online updates; this does not isolate their individual effects.'])
    dump_csv('progress.csv',progress)
    dump_csv('matched_budgets.csv',comparisons)
    dump_csv('convergence.csv',[dict(task=t,method=m,fresh_evaluations=x,best_mic=y) for (t,m),c in curves.items() for x,y in c])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {methods[0]:'#1f77b4',methods[1]:'#d62728',methods[2]:'#9467bd'}
    fig,axes = plt.subplots(2,5,figsize=(18,7))
    for ax,task in zip(axes.flat,TASKS):
        for method in methods:
            c = curves.get((task,method))
            if c:
                ax.step([x for x,y in c],[y for x,y in c],where='post',label=method,color=colors[method],linewidth=1.5)
        if task not in online_states:
            ax.text(.97,.92,'Online pending',transform=ax.transAxes,ha='right',fontsize=9,color=colors[methods[1]])
        ax.axvline(2500,color='grey',linestyle=':',linewidth=.8)
        ax.set(title=task,xlabel='Fresh evaluations',ylabel='Best predicted MIC (µM) ↓',xlim=(0,5000))
        ax.grid(alpha=.2)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'comparison.{ext}',dpi=170,bbox_inches='tight')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
