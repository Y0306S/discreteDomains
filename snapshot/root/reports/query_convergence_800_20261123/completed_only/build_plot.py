"""Two completed-method convergence metrics; all inputs are read-only."""
import csv
import hashlib
import json
import pickle
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
SNAPSHOT = OUT.parent / 'observed_total_evaluation_curves.npz'
HYBRID = Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
BASELINE_AUDIT = Path('/root/reports/query_convergence_fsbo_tpe_lolbo_20261109/audit.json')
PAPER = Path('/root/21295_Scaling_Multi_Task_Bayes.pdf')
METHODS = ['LLM + FSBO', 'FSBO', 'LOLBO', 'TPE']
COLORS = ['#B33D84', '#009E73', '#E69F00', '#0072B2']
START = {'LLM + FSBO': 30, 'FSBO': 49, 'LOLBO': 49, 'TPE': 49}
sys.path.insert(0, str(HYBRID / 'batch_code'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline(source):
    path = Path(source['source'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
    doc = json.loads(raw)
    assert doc['task_id'] == source['task'] and len(doc['observations']) == 1049
    rows = [r['extra_info'] for r in doc['observations'][:800]]
    return source['method'], source['task'], rows


def hybrid(task):
    path = HYBRID / 'orpo_batch10/runs' / task / 'runtime.sqlite'
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        raw = db.execute("SELECT value FROM kv WHERE key='optimization_state'").fetchone()[0]
    state = pickle.loads(raw)
    assert len(state['rows']) == 800 and len(state['events']) == 77
    return 'LLM + FSBO', task, state['rows']


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, stem):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    with np.load(SNAPSHOT, allow_pickle=False) as data:
        tasks = data['tasks'].tolist()
        curves = {m: data[m].copy() for m in METHODS}
    assert len(tasks) == 200
    assert json.loads((HYBRID / 'status.json').read_text())['status'] == 'complete'
    sources = json.loads(BASELINE_AUDIT.read_text())['sources']
    histories = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for m, task, rows in executor.map(baseline, sources):
            histories[m, task] = rows
        for m, task, rows in executor.map(hybrid, tasks):
            histories[m, task] = rows
    assert len(histories) == 800
    pools = {task: [] for task in tasks}
    for m in METHODS:
        for i, task in enumerate(tasks):
            best, measured = np.inf, []
            for row in histories[m, task]:
                value = float(row['runtime'])
                assert value > 0 and np.isfinite(value)
                if not row['censored']:
                    best = min(best, value)
                    pools[task].append(value)
                measured.append(best)
            # Validate the raw history against the saved measured convergence prefix.
            np.testing.assert_array_equal(measured[START[m]-1:], curves[m][i, START[m]:])
        assert np.isfinite(curves[m][:, START[m]:]).all()
        assert np.all(np.diff(curves[m][:, START[m]:], axis=1) <= 0)
    bounds = []
    normalized = {m: np.full_like(curves[m], np.nan) for m in METHODS}
    for i, task in enumerate(tasks):
        pool = np.sort(pools[task])
        drop = int(np.floor(.2 * len(pool)))
        lo, hi = float(pool[0]), float(pool[len(pool)-drop-1])
        assert hi > lo
        for m in METHODS:
            normalized[m][i] = np.clip((hi-curves[m][i])/(hi-lo), 0, 1)
        bounds.append(dict(task=task, pooled_uncensored_observations=len(pool),
                           slowest_dropped_for_bounds=drop, best_seconds=lo,
                           worst_retained_seconds=hi))
    reference_seconds = float(curves['FSBO'][:, 49].sum())
    assert reference_seconds > 0
    values = {m: dict(score=normalized[m][:, START[m]:].mean(axis=0),
                      summed=curves[m][:, START[m]:].sum(axis=0)/reference_seconds)
              for m in METHODS}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    def axis(ax, metric):
        for m, color in zip(METHODS, COLORS):
            x = np.arange(START[m], 801)
            y = values[m][metric]
            ax.step(x, y, where='post', lw=2.3, color=color, label=m,
                    linestyle='--' if m in ('LOLBO', 'TPE') else '-')
            ax.plot(800, y[-1], 'o', color=color, ms=4)
        ax.axvline(400, color='#999999', ls=':', lw=1)
        ax.set(xlim=(0, 800), xlabel='Total counted evaluations per query (initialization included)')
        if metric == 'score':
            ax.set(title='Mean normalized score · higher is better',
                   ylabel='Mean per-query min–max score', ylim=(.8, 1.005))
        else:
            ax.set(title='Normalized summed runtime · lower is better',
                   ylabel='Summed best runtime / summed Bao49 best', ylim=(.4, 1.025))
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=10)

    footer = ('200 held-out queries · Four completed methods · 800 total evaluation slots per query\n'
              'LLM + FSBO: 30 BOLT starts; baselines: 49 Bao starts. Cached baseline proposals count as slots; historical execution conditions differ.')
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))
    axis(axes[0], 'score')
    axis(axes[1], 'summed')
    fig.suptitle('Query convergence · completed methods only', fontsize=17, weight='bold')
    fig.text(.5, .013, footer, ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout(rect=(0, .09, 1, .93))
    save(fig, 'query_completed_800')
    for metric, stem in [('score', 'query_normalized_score_800'), ('summed', 'query_normalized_summed_runtime_800')]:
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        axis(ax, metric)
        fig.text(.5, .014, footer, ha='center', fontsize=7.8, color='#555555')
        fig.tight_layout(rect=(0, .1, 1, 1))
        save(fig, stem)
    rows = []
    summary = {}
    for m in METHODS:
        summary[m] = dict(normalized_score_at_800=float(values[m]['score'][-1]),
                          normalized_summed_runtime_at_800=float(values[m]['summed'][-1]),
                          summed_runtime_seconds_at_800=float(curves[m][:, 800].sum()))
        for n in range(START[m], 801):
            j = n-START[m]
            rows.append(dict(method=m, total_evaluations=n, queries=200,
                             mean_normalized_score=float(values[m]['score'][j]),
                             normalized_summed_runtime=float(values[m]['summed'][j]),
                             summed_best_runtime_seconds=float(curves[m][:, n].sum())))
    write_csv('convergence.csv', rows)
    write_csv('normalization_bounds.csv', bounds)
    notes = [
        'Only LLM + FSBO, FSBO, LOLBO and TPE are included. All200queries in each method have a saved trajectory through800total evaluation slots. SemanticOPT and ongoingBOLTcontinuation are excluded.',
        'The x-axis includes initialization:30BOLTmeasurements for the hybrid,49Baomeasurements for standalone baselines. Curves start after initialization and end at800. Baseline cache hits remain charged slots, so this is not the paper\'s exact fresh-oracle-call axis.',
        'Mean normalized score: for each query pool uncensored runtimes from these four methods\' first800total slots, discard the slowest floor(20%) only to choose the upper bound h, set l to the pooled minimum, then average clip((h-best_runtime)/(h-l),0,1) equally across200queries. The bounds are retrospective and fixed over each curve. Display score axis is0.8–1.0; exported values preserve[0,1].',
        'BOLT paper Scaling Multi-Task Bayesian Optimization, Figure1, PDFpage7: curves sum best query runtimes across held-out test tasks and label that summed objective normalized. The caption/text does not specify the normalization denominator. We follow the summed aggregation and explicitly choose a single shared reference, sum of best runtimes from the49Bao starts. We do not claim that this unspecified denominator matches the paper\'s original plotting code.',
        f'Normalized summed runtime R(n)=sum_q best_runtime_q(n) / {reference_seconds:.12g} seconds. Sum first, then divide by the same constant for every method. This is neither a geometric mean nor a mean per-query runtime ratio, and is not1-minus-runtime.',
        'Uncensored successful runtimes alone improve incumbents; timeouts remain evaluation slots. Each curve uses all200queries at every displayed point. No extrapolation, smoothing, ongoing tasks or seed uncertainty bands.',
        'Raw histories were verified against the prior saved curve snapshot; all600baseline source file hashes were verified. Bounds use only the four displayed methods and the requested800-slot horizon, so scores differ from the previous four-method report that includedSemanticOPT.',
        'Initialization pools, optimization seeds, hosts, dates, timeout policies and storage differ across method families. These are descriptive historical comparisons. Production files and running tasks were not modified.',
    ]
    audit = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 methods=METHODS, tasks=tasks, total_evaluation_horizon=800,
                 shared_summed_bao49_reference_seconds=reference_seconds, summary=summary, notes=notes,
                 paper=str(PAPER), paper_sha256=sha(PAPER), paper_figure=1, paper_pdf_page=7,
                 snapshot=str(SNAPSHOT), snapshot_sha256=sha(SNAPSHOT),
                 baseline_audit=str(BASELINE_AUDIT), builder_sha256=sha(Path(__file__)))
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (OUT / 'README.md').write_text('# Completed query methods:800evaluations\n\n'
        +'\n\n'.join(notes)+'\n\nOutputs:query_completed_800(two panels),query_normalized_score_800,'
        'query_normalized_summed_runtime_800, each PNG/PDF/SVG; convergence.csv;normalization_bounds.csv;audit.json.\n'
        '\nReproduce:python3 build_plot.py.\n')
    print(json.dumps(dict(shared_summed_bao49_reference_seconds=reference_seconds, summary=summary), indent=2))


if __name__ == '__main__':
    main()
