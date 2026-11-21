#!/usr/bin/env python3
"""Read completed peptide archives; validate and plot without loading training state."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import itertools
import json
from pathlib import Path
import tarfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOTS = {
    'FSBO': Path('/root/data/logs/peptide_transfer_test200_20261028_v1/fsbo'),
    'LOLBO': Path('/root/data/logs/peptide_test200_baselines_20261027_v2/lolbo'),
    'TPE': Path('/root/data/logs/peptide_test200_baselines_20261027_v2/tpe'),
}
POOL = Path('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json')
GRID = np.arange(0, 5001, 25)
COLORS = {'FSBO': '#1764ab', 'LOLBO': '#da7624', 'TPE': '#2b8b57'}


def load_archive(item):
    method, task, path, recipe_hash, pool = item
    hashes, documents = {}, {}
    # Stream once: do not unpickle model/checkpoint or SQLite contents.
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            assert member.isfile(), (path, member.name)
            digest = hashlib.sha256()
            wanted = member.name in ('history.json', 'progress.json', 'manifest.json')
            chunks = []
            with archive.extractfile(member) as stream:
                while chunk := stream.read(2**20):
                    digest.update(chunk)
                    if wanted:
                        chunks.append(chunk)
            hashes[member.name] = digest.hexdigest()
            if wanted:
                documents[member.name] = json.loads(b''.join(chunks))
    manifest = documents['manifest.json']
    for name, expected in manifest['members'].items():
        assert hashes[name] == expected, (path, 'checksum', name)
    h = documents['history.json']
    assert h['status'] == 'complete', path
    assert h['identity'] == manifest['identity'], path
    assert h['identity']['task'] == task, path
    assert h['identity']['recipe_sha256'] == recipe_hash, path
    assert h['reference'] == pool['reference'], path
    assert h['initialization'] == 100 and h['proposal_budget'] == 5000 and h['fresh_budget'] == 5000, path
    rows = h['observations']
    initial = [r for r in rows if r['phase'] == 'initialization']
    post = [r for r in rows if r['phase'] != 'initialization']
    assert len(initial) == 100, path
    assert [r['sequence'] for r in initial] == pool['sequences'], (path, 'starting sequences')
    assert np.allclose([r['score'] for r in initial], pool['scores'], rtol=0, atol=1e-6), (path, 'starting scores')
    assert all(r['phase'] in ('initialization', 'acquisition', 'recenter') for r in rows), path
    mic_values = np.array([r['mic'] for r in rows])
    score_values = np.array([r['score'] for r in rows])
    assert np.isfinite(mic_values).all() and np.allclose(score_values, -mic_values), path
    fresh = sum(not r['cached'] for r in post)
    acquisition = sum(r['phase'] == 'acquisition' for r in post)
    recenter_fresh = sum(not r['cached'] and r['phase'] == 'recenter' for r in post)
    assert fresh == h['fresh_evaluations'] <= 5000, path
    assert acquisition == h['proposals'] <= 5000, path
    assert recenter_fresh == h['recenter_fresh'], path
    assert len(post) - fresh == h['cache_hits'], path
    assert fresh == 5000 or acquisition == 5000, (path, 'budget not exhausted')
    best = min(r['mic'] for r in initial if r['feasible'])
    assert best > 0, path
    initial_best = best
    xs, ys = [0], [best]
    used = 0
    # Scores from a parallel batch become available together, at the batch end.
    for _, group in itertools.groupby(post, key=lambda r: (r['round'], r['phase'], r['elapsed'])):
        batch = list(group)
        used += sum(not r['cached'] for r in batch)
        best = min([best] + [r['mic'] for r in batch if r['feasible']])
        if used == xs[-1]:
            ys[-1] = best
        else:
            xs.append(used)
            ys.append(best)
    assert used == fresh and np.isclose(best, h['best_feasible_mic'], rtol=0, atol=1e-6), path
    indices = np.searchsorted(xs, GRID, side='right') - 1
    curve = np.asarray(ys)[indices]
    assert np.all(np.diff(curve) <= 1e-10), path
    assert curve[0] == initial_best, (path, 'unexpected cached improvement at budget zero')
    return method, task, curve, dict(method=method, task=task, initial_best_mic=initial_best,
        final_best_mic=best, relative_improvement_percent=100*(1-best/initial_best),
        fresh_evaluations=fresh, acquisition_proposals=acquisition,
        recenter_fresh=recenter_fresh, cache_hits=h['cache_hits'],
        elapsed_seconds=h['seconds'], archive=str(path), history_sha256=hashes['history.json'])


def main():
    pool = json.loads(POOL.read_text())
    tasks = sorted(pool)
    assert len(tasks) == 200
    jobs = []
    for method, root in ROOTS.items():
        recipe_bytes = (root/'recipe.json').read_bytes()
        recipe = json.loads(recipe_bytes)
        assert sorted(recipe['test']) == tasks
        assert recipe['seed'] == 1 and recipe['similarity'] == 0.75
        assert recipe['oracle'] == 'negative_APEX_bacteria_0'
        assert {p.name[:-7] for p in (root/'runs').glob('*.tar.gz')} == set(tasks)
        for task in tasks:
            jobs.append((method, task, root/'runs'/f'{task}.tar.gz', hashlib.sha256(recipe_bytes).hexdigest(), pool[task]))
    records, curves = [], {m:{} for m in ROOTS}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i, (method, task, curve, record) in enumerate(executor.map(load_archive, jobs), 1):
            curves[method][task] = curve
            records.append(record)
            if i % 25 == 0:
                print(f'Validated {i}/600 archives', flush=True)
    matrices = {m:np.stack([curves[m][t] for t in tasks]) for m in ROOTS}
    starts = matrices['FSBO'][:,0]
    assert all(np.array_equal(a[:,0], starts) for a in matrices.values())
    # Same bootstrap task weights across methods and x coordinates; uncertainty across
    # tasks only, because there is one optimizer seed per task.
    rng = np.random.default_rng(20261101)
    weights = rng.multinomial(200, np.full(200,1/200), size=2000)/200
    stats = {}
    rows = []
    for method, a in matrices.items():
        improvement = 100*(1-a/starts[:,None])
        stats[method] = {}
        for metric, values in [('mic',a), ('improvement_percent', improvement)]:
            mean = values.mean(axis=0)
            lo, hi = np.quantile(weights @ values, [.025,.975], axis=0)
            stats[method][metric] = (mean,lo,hi)
        for j,x in enumerate(GRID):
            rows.append(dict(method=method,fresh_evaluations=int(x),tasks=200,
                **{f'{metric}_{field}':float(stats[method][metric][k][j])
                   for metric in stats[method] for k,field in enumerate(['mean','ci95_low','ci95_high'])},
                tasks_with_budget_remaining=sum(r['fresh_evaluations']>=x for r in records if r['method']==method)))
    def write_csv(name, values):
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write_csv('task_summary.csv',records)
    write_csv('convergence.csv',rows)
    np.savez_compressed(OUT/'per_task_curves.npz', tasks=np.array(tasks),fresh_evaluations=GRID,**matrices)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titlesize':13,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4))
    for ax,metric in zip(axes,['mic','improvement_percent']):
        for method in ROOTS:
            mean,lo,hi=stats[method][metric]
            ax.fill_between(GRID,lo,hi,color=COLORS[method],alpha=.13,linewidth=0)
            ax.plot(GRID,mean,color=COLORS[method],lw=2.3,label=method)
        ax.set_xlim(0,5000);ax.set_xticks(np.arange(0,5001,1000));ax.grid(alpha=.18)
        ax.set_xlabel('Fresh oracle evaluations after initialization')
    axes[0].set_ylabel('Mean best feasible MIC');axes[0].set_title('Best feasible objective · lower is better')
    axes[1].set_ylabel('Mean improvement from initial best (%)');axes[1].set_title('Improvement per task · higher is better')
    axes[1].set_ylim(bottom=0);axes[1].legend(frameon=False,loc='lower right')
    fig.suptitle('Peptide optimization · 200 held-out tasks',fontsize=17,y=.98)
    fig.text(.5,.905,'FSBO, LOLBO and TPE · identical 100-peptide initialization · similarity ≥ 0.75',ha='center',fontsize=11,color='#444444')
    fig.text(.5,.045,'Shading: 95% bootstrap CI across tasks (one seed per task). Includes LOLBO recentering calls.\nCompleted runs retain their final best after their proposal budget is exhausted; no extra evaluations are implied.',ha='center',fontsize=9,color='#555555',linespacing=1.5)
    fig.subplots_adjust(left=.075,right=.975,bottom=.21,top=.80,wspace=.29)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'peptide_convergence.{ext}',dpi=200,facecolor='white')
    # A standalone primary panel is convenient for presentations.
    fig2,ax=plt.subplots(figsize=(8,5.4))
    for method in ROOTS:
        mean,lo,hi=stats[method]['mic'];ax.fill_between(GRID,lo,hi,color=COLORS[method],alpha=.13,linewidth=0);ax.plot(GRID,mean,color=COLORS[method],lw=2.3,label=method)
    ax.set(xlim=(0,5000),xlabel='Fresh oracle evaluations after initialization',ylabel='Mean best feasible MIC (lower is better)',title='Peptide optimization · 200 held-out tasks')
    ax.grid(alpha=.18);ax.legend(frameon=False);fig2.text(.5,.035,'Identical 100-peptide initialization · 95% task-bootstrap CI\nIncludes recentering; terminal best carried forward after proposal-budget exhaustion.',ha='center',fontsize=9,color='#555555')
    fig2.subplots_adjust(bottom=.21,top=.89,left=.13,right=.97)
    for ext in ['png','pdf','svg']:fig2.savefig(OUT/f'peptide_convergence_mic.{ext}',dpi=200,facecolor='white')
    summary={}
    for m in ROOTS:
        rs=[r for r in records if r['method']==m]
        summary[m]=dict(tasks=200,initial_mean_best_mic=float(starts.mean()),
            final_mean_best_mic=float(matrices[m][:,-1].mean()),
            mean_relative_improvement_percent=float(stats[m]['improvement_percent'][0][-1]),
            final_mean_mic_ci95=[float(stats[m]['mic'][i][-1]) for i in (1,2)],
            fresh_evaluations_range=[min(r['fresh_evaluations'] for r in rs),max(r['fresh_evaluations'] for r in rs)],
            mean_fresh_evaluations=float(np.mean([r['fresh_evaluations'] for r in rs])),
            improved_tasks=sum(r['final_best_mic']<r['initial_best_mic'] for r in rs))
    audit=dict(completed_archives_validated=600,methods=summary,
        checks=['all archived member SHA256 checksums','complete status and frozen recipe identity','same held-out task IDs',
            'same initial sequences and revalidated scores','finite scores and MIC sign','fresh/cache/recenter/proposal accounting',
            'terminal budget exhaustion','recorded final incumbent','nonincreasing feasible incumbent curves'],
        x_axis='Post-initialization fresh logical oracle evaluations, including recentering. Cache hits have zero cost.',
        batch_policy='Whole logged batches revealed at their endpoint; no arbitrary within-batch ordering.',
        terminal_policy='Last measured incumbent carried forward to budget 5000 after either production budget is exhausted.',
        uncertainty='Pointwise percentile bootstrap 95% CI; 2000 paired task resamples; seed20261101. One optimizer seed/task.',
        excluded={'RGPE':'test200 optimization incomplete','BOLT':'SFT completion alone is not completed peptide optimization'},
        source_roots={m:str(p) for m,p in ROOTS.items()},initialization=str(POOL))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    text='# Peptide convergence: completed test200 baselines\n\n'
    text+='Read-only extraction from 600 completed archives: 200 each for FSBO, LOLBO and TPE. All archive member checksums and recorded recipe identities passed. No model state was loaded and no oracle evaluations or training were run.\n\n'
    text+='The objective is best feasible APEX bacteria-0 MIC (negative of the maximized score); lower is better. All three methods use the same 200 held-out references, same revalidated 100 starting sequences and scores per task, similarity threshold 0.75 and seed1.\n\n'
    text+='The x-axis counts fresh post-initialization logical oracle evaluations, including LOLBO recentering; cached duplicate scores cost zero. Each logged batch is revealed at its endpoint. Production stops at either 5000 acquisition proposals or 5000 shared fresh evaluations, so runs ending earlier in fresh-evaluation count retain their final measured incumbent through x=5000. This does not imply new measurements or simulate continuation. Exact terminal costs and support counts are exported.\n\n'
    text+='The right panel averages 100 × (1 − current best MIC / initial best MIC) across tasks. Shading is a pointwise 95% percentile bootstrap CI over 200 tasks, with 2000 paired resamples. There is one optimizer seed per task; these bands do not estimate across-seed variation. RGPE remains incomplete and BOLT SFT is not peptide optimization, so neither is plotted.\n\n'
    text+='| Method | Tasks | Final mean best MIC | Mean per-task improvement |\n|---|---:|---:|---:|\n'
    for m,s in summary.items():text+=f"| {m} | 200 | {s['final_mean_best_mic']:.4f} | {s['mean_relative_improvement_percent']:.2f}% |\n"
    text+='\nFiles: `peptide_convergence.{png,pdf,svg}`, `peptide_convergence_mic.{png,pdf,svg}`, `convergence.csv`, `task_summary.csv`, `per_task_curves.npz`, `audit.json`. Reproduce with `python3 plot_convergence.py`.\n'
    (OUT/'README.md').write_text(text)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
