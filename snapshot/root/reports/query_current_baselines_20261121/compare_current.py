"""Read-only descriptive comparison of current query hybrid and audited baselines."""
import csv
import hashlib
import json
import time
import sqlite3
import pickle
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import hybrid_reader as reader

OUT = Path(__file__).resolve().parent
CACHE = Path('/root/reports/query_convergence_four_methods_20261117/per_task_curves.npz')
BASELINES = ['FSBO', 'LOLBO', 'TPE', 'SemanticOPT']
METHODS = ['Our ORPO + FSBO'] + BASELINES
COLORS = ['#CC79A7', '#009E73', '#E69F00', '#0072B2', '#00A6BF']


def main():
    begin = time.time()
    states = {}
    for path in sorted((reader.N / 'orpo_batch10/runs').glob('*/runtime.sqlite')):
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=120) as db:
            row = db.execute("SELECT value FROM kv WHERE key='optimization_state'").fetchone()
        if row is None:
            continue
        state = pickle.loads(row[0])
        assert len(state['rows']) == 30 + 10 * len(state['events'])
        states[path.parent.name] = state
    print('Read query journals:', len(states), flush=True)
    z = np.load(CACHE)
    idx = {str(t): i for i, t in enumerate(z['tasks'])}
    result = dict(snapshot_started=begin, snapshot_finished=time.time(),
                  campaign_status=json.loads((reader.N / 'status.json').read_text()),
                  available_task_journals=len(states), cache_sha256=reader.sha(CACHE),
                  baseline_audit='/root/reports/query_convergence_four_methods_20261117/audit.json',
                  equal_proposal_budgets={}, equal_total_evaluations={}, completed_cohort_full_baselines={})
    for budget in [200, 500, 770]:
        tasks = sorted(t for t, s in states.items() if len(s['rows']) >= 30 + budget)
        if not tasks:
            continue
        methods = METHODS if budget == 200 else METHODS[:-1]
        curves = {m: [] for m in methods}
        rows = []
        for t in tasks:
            st = states[t]
            hybrid = reader.curve(st['rows'][:30 + budget])[29:]
            assert len(hybrid) == budget + 1 and np.isfinite(hybrid).all()
            anchor = float(z['FSBO'][idx[t], 0])
            curves[methods[0]].append(hybrid / anchor)
            row = dict(task=t, optimization_proposals=budget, our_best_runtime=float(hybrid[-1]),
                       bolt30_best_runtime=float(hybrid[0]), bao49_best_runtime=anchor)
            for m in methods[1:]:
                curve = z[m][idx[t], :budget + 1]
                curves[m].append(curve / anchor)
                row[m + '_runtime'] = float(curve[-1])
                row['our_over_' + m] = float(hybrid[-1] / curve[-1])
                row['our_over_' + m + '_equal_total'] = float(hybrid[-1] / z[m][idx[t], budget - 19])
            rows.append(row)
        with (OUT / f'per_query_{budget}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        runtime = np.array([[r['our_best_runtime']] + [r[m + '_runtime'] for m in methods[1:]] for r in rows])
        ranks = rankdata(runtime, axis=1, method='average').mean(axis=0)
        summary = dict(queries=len(tasks), proposals=budget, tasks=tasks,
                       pairwise={m: reader.comparison(rows, 'our_over_' + m) for m in methods[1:]},
                       average_rank=dict(zip(methods, ranks.tolist())),
                       geomean_runtime_relative_to_bao={m: float(np.exp(np.log(v).mean(axis=0)[-1])) for m, v in curves.items()})
        result['equal_proposal_budgets'][str(budget)] = summary
        result['equal_total_evaluations'][str(budget)] = {m: reader.comparison(rows, 'our_over_' + m + '_equal_total') for m in methods[1:]}
        if budget == 770:
            fullrows = []
            for r in rows:
                fullrows.append(dict(task=r['task'], **{'our_over_' + m: r['our_best_runtime'] / float(z[m][idx[r['task']], 1000]) for m in ['FSBO', 'LOLBO', 'TPE']}))
            result['completed_cohort_full_baselines'] = {m: reader.comparison(fullrows, 'our_over_' + m) for m in ['FSBO', 'LOLBO', 'TPE']}
        np.savez(OUT / f'curves_{budget}.npz', tasks=np.array(tasks), **{m: np.exp(np.log(v).mean(axis=0)) for m, v in curves.items()})
        print(json.dumps(dict(budget=budget, queries=len(tasks), pairwise=summary['pairwise'], ranks=summary['average_rank'])), flush=True)
    result['limitations'] = [
        'Current partial cohort is determined by scheduling and completion; not the full 200-task result.',
        'Hybrid starts with 30 BOLT observations; baselines with 49 Bao observations. Equal-proposal comparison has 19 more baseline observations.',
        'Equal-total sensitivity compares hybrid budget B with baseline B-19 new proposals.',
        'Seeds, dates, hardware, storage and timeout policies differ; descriptive historical comparison.',
        'Task-bootstrap intervals quantify task variation in this cohort, not independent training seeds.',
        'SemanticOPT ends at 200 measured proposals; no extrapolation.',
        'Our current query campaign uses ORPO + FSBO, with RL disabled.',
        'Queries minimize measured uncensored best-so-far runtime; percentages use geometric means of per-query ratios.'
    ]
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    plot(result)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline_curves = np.load(CACHE)
    baseline_index = {str(t): i for i, t in enumerate(baseline_curves['tasks'])}
    budgets = [b for b in [200, 770] if str(b) in result['equal_proposal_budgets']]
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.5 * len(budgets), 5), squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        z = np.load(OUT / f'curves_{budget}.npz')
        for m, color in zip(METHODS, COLORS):
            if m in z:
                ax.plot(np.arange(budget + 1), z[m], color=color, label=m, lw=2)
            elif m == 'SemanticOPT':
                selected = [baseline_index[str(t)] for t in z['tasks']]
                values = baseline_curves[m][selected, :201] / baseline_curves['FSBO'][selected, 0, None]
                mean = np.exp(np.log(values).mean(axis=0))
                ax.plot(np.arange(201), mean, color=color, label=m + ' (ends at 200)', lw=2)
                ax.plot(200, mean[-1], 'o', color=color, ms=4)
        n = len(z['tasks'])
        ax.set(xlabel='New proposals per query (initialization excluded)',
               ylabel='Geometric mean runtime / Bao49 best',
               title=f'{n} matched queries · {budget} new proposals', xlim=(0, budget))
        ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=9)
    fig.text(.5, .015, 'Lower is better. BOLT30 vs Bao49; historical conditions differ. Each panel uses its own cohort. SemanticOPT ends at 200 proposals.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / ('comparison.' + ext), dpi=190)


if __name__ == '__main__':
    main()
