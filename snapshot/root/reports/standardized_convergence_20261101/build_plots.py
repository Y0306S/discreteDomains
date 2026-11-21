"""Shared presentation, interpolation and raw-incumbent ranks for ML and TaskSet."""
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

OUT = Path(__file__).resolve().parent
DL = Path('/root/reports/taskset_convergence_20261101')
ML = Path('/root/reports/ml_convergence_main_methods_20261101')
RUNTIME = Path('/root/reports/task_runtime_analysis_20261026/ml_task_current2400.csv')
REFERENCE = Path('/root/data/cys/llmoptimizer_benchmark/examples/benchmark/plot_mltest20_algo40_avg_rank.py')
STYLE = {
    'tpe': dict(label='TPE', color='#0072B2', transfer=False),
    'lolbo': dict(label='LOLBO', color='#E69F00', transfer=False),
    'rgpe': dict(label='RGPE', color='#555555', transfer=True),
    'fsbo_minmax': dict(label='FSBO', color='#009E73', transfer=True),
    'bolt': dict(label='BOLT', color='#7040B6', transfer=True),
    'semanticopt': dict(label='SemanticOPT', color='#00A6BF', transfer=True),
    'orpo_fsbo': dict(label='ORPO + FSBO', color='#D43D62', transfer=True),
    'ckpt32_rgpe': dict(label='ckpt32 LLM + RGPE', color='#D55E00', transfer=True),
    'bora': dict(label='BORA', color='#CC79A7', transfer=True),
    'tosfit': dict(label='ToSFiT', color='#9C7425', transfer=True),
}
DL_METHODS = ['tpe', 'lolbo', 'rgpe', 'fsbo_minmax', 'orpo_fsbo', 'bolt', 'semanticopt']
ML_METHODS = ['ckpt32_rgpe', 'rgpe', 'lolbo', 'tpe', 'bora', 'tosfit', 'fsbo_minmax', 'bolt', 'semanticopt']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_task_ranks(values):
    """Descending raw incumbent ranks, averaged ties at reference absolute tolerance."""
    methods, tasks, rounds = values.shape
    result = np.empty(values.shape, dtype=float)
    for task in range(tasks):
        for t in range(rounds):
            scores = values[:, task, t]
            assert not np.isnan(scores).any() and not np.isposinf(scores).any()
            order = np.argsort(-scores, kind='stable')
            i = 0
            while i < methods:
                j = i
                while j+1 < methods and (scores[order[j+1]] == scores[order[i]] or
                        abs(scores[order[j+1]]-scores[order[i]]) < 1e-12):
                    j += 1
                result[order[i:j+1], task, t] = (i+1+j+1)/2
                i = j+1
    np.testing.assert_allclose(result.sum(axis=0), methods*(methods+1)/2, rtol=0, atol=1e-12)
    # Independent reference function, extracted without importing its plot stack.
    tree = ast.parse(REFERENCE.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('average_rank', 'task_ranks')]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(REFERENCE), 'exec'), namespace)
    for task in range(tasks):
        for t in range(rounds):
            scores = values[:, task, t]
            expected = namespace['task_ranks']({str(i):float(v) for i,v in enumerate(scores)})
            # The original reference does not tie -inf; failed-before-success
            # incumbents must share the worst ranks, as in the audited ML scorer.
            failures = np.flatnonzero(np.isneginf(scores))
            if failures.size:
                for i in failures:expected[str(i)] = (methods-failures.size+1+methods)/2
            np.testing.assert_allclose(result[:,task,t], [expected[str(i)] for i in range(methods)], atol=0, rtol=0)
    return result


def interpolate(values):
    x = np.arange(1, len(values)+1)
    # Every actual round is included exactly in the rendered grid.
    dense = np.linspace(1, len(values), (len(values)-1)*16+1)
    smoother = PchipInterpolator(x, values, extrapolate=False)
    y = smoother(dense)
    np.testing.assert_allclose(y[::16], values, rtol=0, atol=1e-12)
    lower = np.minimum(values[:-1], values[1:])
    upper = np.maximum(values[:-1], values[1:])
    for i in range(len(values)-1):
        segment = y[i*16:(i+1)*16+1]
        assert segment.min() >= lower[i]-1e-12 and segment.max() <= upper[i]+1e-12
    if np.all(np.diff(values) >= -1e-12):assert np.all(np.diff(y) >= -1e-12)
    return dense, y


def draw(domain, methods, scores, ranks, rounds, score_ylim, filename, counts, rank=False, panels=False):
    title_domain = 'TaskSet' if domain == 'taskset' else 'ML'
    keys = ranks if rank else scores
    fields = ['all', 'classification', 'regression'] if panels else ['all']
    if panels:
        fig = plt.figure(figsize=(13.5, 10.5), facecolor='white')
        grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=.36, wspace=.20)
        axes = [fig.add_subplot(grid[0,:]), fig.add_subplot(grid[1,0]), fig.add_subplot(grid[1,1])]
    else:
        fig, ax = plt.subplots(figsize=(12.8, 7.5), facecolor='white')
        axes = [ax]
    for ax, field in zip(axes, fields):
        for method in methods:
            values = keys[method][field][:rounds]
            x, y = interpolate(values)
            style = STYLE[method]
            ax.plot(x, y, color=style['color'], linestyle='-' if style['transfer'] else ':',
                    linewidth=2.5, label=f"{style['label']}  {values[-1]:.3f}")
        ax.set(xlim=(1, rounds), ylim=(len(methods)+.2, .8) if rank else score_ylim,
               xlabel='Evaluation attempts per task',
               ylabel='Average rank (lower is better)' if rank else 'Mean normalized best-so-far score')
        if rank:ax.set_yticks(np.arange(1, len(methods)+1))
        ax.set_xticks([v for v in (1,10,20,40,60,80,100) if v <= rounds])
        ax.grid(axis='y', alpha=.18)
        if panels:ax.set_title(f"{dict(all='All',classification='Classification ·',regression='Regression ·')[field]} {counts[field]} tasks")
        assert len(ax.lines) == len(methods)
        assert tuple(ax.get_ylim()) == ((len(methods)+.2,.8) if rank else score_ylim)
    if panels:
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.047),
                   ncol=3, frameon=False, fontsize=10, title=f"{'Average rank' if rank else 'Score'} at attempt {rounds}")
    else:
        axes[0].legend(loc='lower right', ncol=3, frameon=False, fontsize=9.5,
                       title=f"{'Average rank' if rank else 'Score'} at attempt {rounds}", title_fontsize=10)
    fig.suptitle(f"{title_domain} {'average rank' if rank else 'convergence'}", fontsize=20, weight='bold', y=.97)
    subtitle = 'Same 200 held-out tasks · seed 1 · 80 observed attempts' if domain == 'taskset' else f"Same {counts['all']} tasks · {counts['classification']} classification + {counts['regression']} regression · seed 1"
    fig.text(.5, .925 if panels else .905, subtitle, ha='center', color='#555555', fontsize=11)
    note = ('Ranks computed per task among displayed methods; ties share rank. ' if rank else
            'Equal task weights; observed scores preserved. ')
    note += 'Shape-preserving interpolation between rounds.\nSolid: transfer learning / pretrained knowledge · dotted: no cross-task transfer.'
    fig.text(.5, .014 if panels else .035, note, ha='center', fontsize=9.5, color='#555555')
    fig.subplots_adjust(left=.08 if panels else .10, right=.975, top=.87 if panels else .85,
                        bottom=.20 if panels else .18)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{filename}.{ext}', dpi=200)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    dl_manifest = json.loads((DL/'manifest.json').read_text())
    ml_audit = json.loads((ML/'audit.json').read_text())
    assert sha(DL/'common80_per_task.npz') == dl_manifest['output_sha256']['common80_per_task.npz']
    assert sha(ML/'per_task_curves.npz') == ml_audit['output_sha256']['per_task_curves.npz']
    with np.load(DL/'common80_per_task.npz', allow_pickle=False) as data:
        dl_tasks = data['task_ids'].copy()
        dl_scores = np.asarray([data[m] for m in DL_METHODS])
    with np.load(ML/'per_task_curves.npz', allow_pickle=False) as data:
        ml_tasks = data['task_ids'].copy()
        source_methods = list(data['methods'])
        order = [source_methods.index('bolt_ml' if m == 'bolt' else m) for m in ML_METHODS]
        ml_scores = data['normalized'][order].copy()
        ml_measured = data['measured'][order].copy()
    assert dl_scores.shape == (7,200,80) and ml_scores.shape == (9,132,100)
    assert len(set(dl_tasks)) == 200 and len(set(ml_tasks)) == 132
    assert np.isfinite(dl_scores).all() and np.isfinite(ml_scores).all()
    assert np.all(np.diff(dl_scores,axis=2) >= -1e-12) and np.all(np.diff(ml_scores,axis=2) >= -1e-12)
    ml_full_scores = ml_scores.copy()
    with RUNTIME.open() as stream:runtime_rows = list(csv.DictReader(stream))
    assert len({r['task_id'] for r in runtime_rows}) == len(runtime_rows)
    runtime_rows = [r for r in runtime_rows if r['task_id'] in set(ml_tasks)]
    assert len(runtime_rows) == 132
    assert all(np.isfinite(float(r['median_method_eval_mean_s'])) for r in runtime_rows)
    runtime_rows.sort(key=lambda r:(-float(r['median_method_eval_mean_s']),r['task_id']))
    runtime_ranking = [dict(runtime_rank=i+1,task_id=r['task_id'],
        excluded=i<12,median_method_eval_mean_s=float(r['median_method_eval_mean_s']),
        methods=int(r['methods']),completed_runs=int(r['completed_runs']),
        observations=int(r['observations'])) for i,r in enumerate(runtime_rows)]
    excluded = {r['task_id'] for r in runtime_ranking[:12]}
    keep = np.asarray([t not in excluded for t in ml_tasks])
    assert len(excluded) == 12 and keep.sum() == 120
    ml_tasks,ml_scores,ml_measured = ml_tasks[keep],ml_scores[:,keep],ml_measured[:,keep]
    assert ml_scores.shape == ml_measured.shape == (9,120,100)
    write_csv(OUT/'ml_task_runtime_ranking.csv',runtime_ranking)
    write_csv(OUT/'ml_excluded_12_slowest_tasks.csv',runtime_ranking[:12])
    write_csv(OUT/'ml_retained_120_tasks.csv',[dict(task_id=t) for t in ml_tasks])
    dl_rank_values = per_task_ranks(dl_scores)
    ml_rank_values = per_task_ranks(np.maximum.accumulate(ml_measured, axis=2))
    classified = np.asarray([t.startswith('cls/') for t in ml_tasks])
    assert classified.sum() == 79 and (~classified).sum() == 41
    input_checks = []
    outputs = []
    for domain, methods, scores, ranks, task_ids in (
            ('taskset', DL_METHODS, dl_scores, dl_rank_values, dl_tasks),
            ('ml', ML_METHODS, ml_scores, ml_rank_values, ml_tasks)):
        fields = dict(all=np.ones(len(task_ids), dtype=bool))
        if domain == 'ml':fields.update(classification=classified, regression=~classified)
        score_curves = {m:{f:scores[i,mask].mean(axis=0) for f,mask in fields.items()} for i,m in enumerate(methods)}
        rank_curves = {m:{f:ranks[i,mask].mean(axis=0) for f,mask in fields.items()} for i,m in enumerate(methods)}
        # Reproduce source mean curves independently of interpolation.
        root = DL if domain == 'taskset' else ML
        with (root/'convergence.csv').open() as stream:prior=list(csv.DictReader(stream))
        for m in methods:
            source_key = 'bolt_ml' if domain == 'ml' and m == 'bolt' else m
            selected = sorted((r for r in prior if r['method'] == source_key),
                              key=lambda r:int(r['round' if domain == 'taskset' else 'attempt']))[:scores.shape[2]]
            old = [float(r['mean_best_normalized_score' if domain == 'taskset' else 'mean_score']) for r in selected]
            source_mean = ml_full_scores[methods.index(m)].mean(axis=0) if domain == 'ml' else score_curves[m]['all']
            np.testing.assert_allclose(source_mean, old, rtol=0, atol=1e-12)
            input_checks.append(dict(domain=domain,method=m,points=len(old)))
        # Independent scalar accumulation verifies every filtered aggregate.
        for i,m in enumerate(methods):
            for field,mask in fields.items():
                for values,curves in ((scores,score_curves),(ranks,rank_curves)):
                    expected=np.asarray([sum(float(values[i,j,t]) for j in np.flatnonzero(mask))/int(mask.sum()) for t in range(values.shape[2])])
                    np.testing.assert_allclose(curves[m][field],expected,rtol=0,atol=1e-12)
        counts={field:int(mask.sum()) for field,mask in fields.items()}
        aggregate = []
        for m in methods:
            for t in range(scores.shape[2]):
                aggregate.append(dict(method=m,label=STYLE[m]['label'],attempt=t+1,tasks=len(task_ids),
                    **{f'mean_score_{f}':score_curves[m][f][t] for f in fields},
                    **{f'average_rank_{f}':rank_curves[m][f][t] for f in fields}))
        write_csv(OUT/f'{domain}_convergence_and_rank.csv', aggregate)
        write_csv(OUT/f'{domain}_final_values.csv', [r for r in aggregate if r['attempt'] == scores.shape[2]])
        np.savez_compressed(OUT/f'{domain}_per_task_ranks.npz',methods=np.asarray(methods),task_ids=task_ids,ranks=ranks)
        for rounds in ([80] if domain == 'taskset' else [100,80]):
            for rank in (False, True):
                stem=f"{domain}_{'average_rank' if rank else 'convergence'}_round{rounds}_smoothed"
                draw(domain,methods,score_curves,rank_curves,rounds,(.6,.9) if domain == 'taskset' else (.5,1.),stem,counts,rank=rank)
                outputs.append(stem)
        if domain == 'ml':
            for rank in (False,True):
                stem=f"ml_{'average_rank' if rank else 'convergence'}_by_task_type_smoothed"
                draw(domain,methods,score_curves,rank_curves,100,(.5,1.),stem,counts,rank=rank,panels=True)
                outputs.append(stem)
    write_csv(OUT/'method_styles.csv', [dict(method=m,**v,line_style='solid' if v['transfer'] else 'dotted') for m,v in STYLE.items()])
    manifest = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        taskset_snapshot=dl_manifest['generated_at'],ml_snapshot=ml_audit['generated_at'],
        methods=dict(taskset=DL_METHODS,ml=ML_METHODS),styles=STYLE,
        task_counts=dict(taskset=200,ml=120,ml_classification=79,ml_regression=41),
        ml_cohort_selection=dict(original_tasks=132,retained_tasks=120,excluded_tasks=runtime_ranking[:12],
            criterion='Descending median across method-level mean configuration evaluation seconds, using completed runs from uniformly 2400-second campaigns in the audited 2026-10-26 runtime snapshot; task_id breaks ties.',
            same_cohort_for_every_method=True,normalization_scales_unchanged=True,
            selection_independent_of_performance=True),
        limits=dict(taskset=[.6,.9],ml=[.5,1]),
        smoothing='Identical PCHIP shape-preserving cubic interpolation at 16 subdivisions per interval. Passes through every observed round; no averaging, no endpoint changes, no overshoot within intervals.',
        rank='Rank raw incumbent performance per task and attempt among displayed methods only; average tied ranks using absolute tolerance <1e-12; failed-before-success incumbents share the worst ranks; equal task weighting.',
        classification='Solid for cross-task transfer or pretrained knowledge (RGPE, FSBO, BORA, ToSFiT, BOLT, SemanticOPT, LLM hybrids); dotted for TPE and the deployed target-only LOLBO wrapper.',
        validation=dict(source_mean_points_reproduced=sum(r['points'] for r in input_checks),
            all_per_task_ranks_match_reference=True,rank_sums_correct=True,
            interpolation_replays_measured_points=True,no_interpolation_overshoot=True,
            displayed_method_sets_preserved=True,no_production_changes=True),
        caveats=['Rank is relative to each displayed method set; ML has nine, TaskSet seven, so rank magnitudes are not directly comparable.',
            'Interpolation is a visual guide between integer attempts, not additional evaluations.',
            'ML retains audited fixed 15-method pooled normalization; ranks use raw performance to avoid clipping-induced ties.',
            'TaskSet includes failures and 11 previously audited replacement measurements; ML retains historical lineage, timeout and ToSFiT schedule caveats. Single seed.'],
        input_sha256={str(p):sha(p) for p in [DL/'manifest.json',DL/'common80_per_task.npz',DL/'convergence.csv',ML/'audit.json',ML/'per_task_curves.npz',ML/'convergence.csv',REFERENCE,RUNTIME]},
        script_sha256=sha(Path(__file__)),figures=outputs)
    manifest['output_sha256']={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf','.svg','.csv','.npz')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Standardized ML and TaskSet convergence and average rank

TaskSet uses the latest audited 200-task, round-80 snapshot. Random Search and FSBO (original) remain excluded; FSBO is the minmax variant. Its score axis is **0.6–0.9**. ML uses **120 tasks (79 classification + 41 regression)** and nine methods, including **RGPE**, and retains **0.5–1.0**, because several scores exceed 0.9. Both use seed 1. ML is exported at rounds 100 and 80, with additional classification/regression panels.

The 12 slowest ML tasks are excluded identically from every method. Tasks are ordered by the median across method-level mean configuration evaluation seconds, descending, using completed runs from campaigns uniformly recording the 2400-second timeout in the audited 2026-10-26 runtime snapshot. Task ID breaks ties. This measures evaluator cost, excluding optimizer overhead, and selects independently of performance. Seven or eight completed methods contribute per task (OptFormer lacks one completion). The original 132-task scores are validated before selecting the 120-task cohort. Existing per-task normalization scales remain fixed. See ml_excluded_12_slowest_tasks.csv, ml_task_runtime_ranking.csv and ml_retained_120_tasks.csv for the complete audit. This changes the reporting cohort only; production runs are unchanged.

All figures use identical **PCHIP shape-preserving interpolation**: 16 subdivisions between observed attempts. It passes through every measured point, keeps endpoints and monotonicity of score curves, and creates no interval overshoot. This supplies smooth joins without changing empirical results. Interpolated positions are visual guides, not additional measured evaluations. Underlying unsmoothed values remain in the CSV files.

Colours are shared across domains and metrics. **Solid** means cross-task transfer or pretrained knowledge: RGPE, FSBO, BORA, ToSFiT, BOLT, SemanticOPT and the LLM hybrids. **Dotted** means no cross-task transfer: TPE and the deployed target-only LOLBO wrapper. See method_styles.csv for the palette.

Average rank is computed independently per task and integer attempt from raw incumbent performance, then averaged with equal task weights. Rank 1 is best; ties share the average rank with the historical absolute tolerance <1e-12. Failure before any success shares the worst rank. ML ranks use raw balanced accuracy / negative MSE, not normalized clipped scores. TaskSet ranks use its native normalized validation score. Ranks are recomputed among the displayed methods only (nine ML, seven TaskSet); ranks from different method sets are not directly comparable. Smoothing is applied after computing integer-round ranks, never before ranking.

Validation reproduces 1,460 original source mean values before filtering, independently recomputes all filtered score and rank aggregates, checks every retained per-task rank against the historical reference (with the established failure-tie correction), verifies rank sums, and verifies exact interpolation at every observed round and no interval overshoot. PNG, PDF and SVG are provided for all eight figures. The manifest records source and output hashes. Original snapshots and production runs are unchanged.

Original limitations apply: TaskSet retains 11 audited replacement measurements and recorded failures. ML retains its fixed audited 15-method normalization scales, historical ckpt32 lineage, differing timeout policies and mixed ToSFiT schedule. These are single-seed descriptive comparisons. See source report READMEs for details.

Reproduce: `python3 build_plots.py`.
''')
    print(json.dumps(dict(output=str(OUT),figures=outputs,validation=manifest['validation']),indent=2))


if __name__ == '__main__':main()
