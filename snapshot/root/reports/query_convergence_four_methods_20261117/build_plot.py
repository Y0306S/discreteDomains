"""Read-only query convergence refresh; never extend SemanticOPT past its budget."""
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
OLD = Path('/root/reports/query_convergence_fsbo_tpe_lolbo_20261109')
SEM = Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110')
METHODS = ['FSBO', 'LOLBO', 'TPE', 'SemanticOPT']
COLORS = ['#009E73', '#E69F00', '#0072B2', '#00A6BF']
BUDGETS = dict(FSBO=1000, LOLBO=1000, TPE=1000, SemanticOPT=200)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def groups(plan):
    assert len(plan) % 5 == 0
    return sorted(tuple(plan[i:i+5]) for i in range(0, len(plan), 5))


def read_baseline(source):
    path = Path(source['source'])
    raw = path.read_bytes()
    assert sha(raw) == source['sha256'], ('Historical source changed', path)
    doc = json.loads(raw)
    rows = doc['observations']
    status = json.loads((path.parent / 'status.json').read_text())
    assert doc['task_id'] == source['task']
    assert status['status'] == 'complete' and status['proposals'] == 1000
    assert status['unresolved_database_calls'] == 0 and len(rows) == 1049
    initial, curve, pool = [], [], []
    best = np.inf
    for i, row in enumerate(rows):
        e = row['extra_info']
        value = float(e['runtime'])
        assert np.isfinite(value) and value > 0
        if i < 49:
            assert e['source'] == 'bao_import'
            initial.append(dict(runtime=value, censored=bool(e['censored']),
                                plan_id=str(e['plan_id']), groups=groups(e['plan'])))
        else:
            assert e['proposal_index'] == i - 49 and e['charged_index'] == i
        if not e['censored']:
            best = min(best, value)
            pool.append(value)
        if i >= 48:
            curve.append(best)
    assert np.isclose(best, status['best_uncensored_runtime'], rtol=1e-7)
    return dict(method=source['method'], task=source['task'], initial=initial,
                curve=np.asarray(curve), pool=pool, source=source,
                seed=status['seed'])


def read_semantic(task):
    folder = SEM / 'query/runs' / task
    path = folder / 'observations.sqlite'
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        initial_raw = db.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0]
        initial = json.loads(initial_raw)
        records = db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
        pending = json.loads(db.execute("SELECT value FROM state WHERE key='pending'").fetchone()[0])
    finally:
        db.close()
    status = json.loads((folder / 'status.json').read_text())
    assert status['status'] == 'complete' and status['evaluations'] == status['budget'] == 200
    assert status['batch_size'] == 10 and status['completed_rounds'] == 20
    assert len(initial) == 49 and len(records) == 200 and pending is None
    best = np.inf
    pool, curve, sources, outcomes = [], [], Counter(), Counter()
    for row in initial:
        value = float(row['objective'])
        assert np.isfinite(value) and value > 0
        if not row['censored'] and row['status'] == 'success':
            best = min(best, value)
            pool.append(value)
    assert np.isfinite(best)
    curve.append(best)
    digest = hashlib.sha256(initial_raw.encode())
    for expected, (n, raw) in enumerate(records):
        assert n == expected
        digest.update(raw.encode())
        row = json.loads(raw)
        assert row['evaluation'] == n + 1
        value = float(row['objective'])
        assert np.isfinite(value) and value > 0
        outcomes[row['status']] += 1
        sources[row.get('proposal_source', 'historical_unlabelled')] += 1
        if not row['censored'] and row['status'] == 'success':
            best = min(best, value)
            pool.append(value)
        curve.append(best)
    assert sources['random_fallback'] == status['random_fallback_evaluations']
    return dict(method='SemanticOPT', task=task, initial=initial, curve=np.asarray(curve),
                pool=pool, seed=42, sources=dict(sources), outcomes=dict(outcomes),
                source=dict(method='SemanticOPT', task=task, source=str(path),
                            logical_snapshot_sha256=digest.hexdigest()))


def draw_comparison(values, ylabel, stem, footer, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7),
                             gridspec_kw={'width_ratios': [1, 1.25]})
    for method, color in zip(METHODS, COLORS):
        y = values[method]
        style = ':' if method in ('TPE', 'LOLBO') else '-'
        for ax, end in zip(axes, [200, BUDGETS[method]]):
            ax.plot(np.arange(end + 1), y[:end + 1], label=method, color=color,
                    lw=2.5, linestyle=style)
        axes[1].plot(BUDGETS[method], y[-1], 'o', color=color, ms=5)
    for ax, limit, title in zip(axes, [200, 1000],
                               ['Equal budget: 200 proposals', 'All observed proposals']):
        ax.set(xlim=(0, limit), xlabel='New proposals per query (49 Bao starts excluded)',
               title=title, ylabel=ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=10)
    axes[1].axvline(200, color='#777777', alpha=.35, lw=1)
    fig.suptitle('Query convergence · 200 held-out CEB queries', fontsize=17, weight='bold')
    fig.text(.5, .025, footer, ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .09, 1, .94))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    audit_old = json.loads((OLD / 'audit.json').read_text())
    tasks = sorted({s['task'] for s in audit_old['sources']})
    assert len(tasks) == 200
    complete = json.loads((SEM / 'query/evaluation_complete.json').read_text())
    assert complete['status'] == 'complete' and complete['tasks'] == 200
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(read_baseline, audit_old['sources']))
        print('Verified 600 complete baseline histories and original hashes.', flush=True)
        records += list(executor.map(read_semantic, tasks))
    index = {(r['method'], r['task']): r for r in records}
    for task in tasks:
        anchor = index['FSBO', task]['initial']
        for method in ['TPE', 'LOLBO']:
            other = index[method, task]['initial']
            assert [(r['runtime'], r['censored'], r['plan_id']) for r in anchor] == [
                (r['runtime'], r['censored'], r['plan_id']) for r in other]
        initial = index['SemanticOPT', task]['initial']
        for a, b in zip(anchor, initial):
            assert a['runtime'] == b['objective'] and a['censored'] == b['censored'], task
            # Independent join groups may be serialized in another order.
            assert a['groups'] == groups(b['configuration']['plan']), (task, 'Bao plan groups differ')
    curves = {m: np.stack([index[m, t]['curve'] for t in tasks]) for m in METHODS}
    for m in METHODS:
        assert curves[m].shape == (200, BUDGETS[m] + 1)
        assert np.isfinite(curves[m]).all() and np.all(np.diff(curves[m], axis=1) <= 0)
        np.testing.assert_array_equal(curves[m][:, 0], curves['FSBO'][:, 0])
    print('Verified all 800 tasks, causal incumbents, and identical Bao initialization measurements/plans.', flush=True)
    normalized = {m: np.empty_like(curves[m]) for m in METHODS}
    bounds = []
    for j, task in enumerate(tasks):
        pool = np.sort(np.asarray([v for m in METHODS for v in index[m, task]['pool']]))
        dropped = int(np.floor(.2 * len(pool)))
        lo, hi = float(pool[0]), float(pool[len(pool) - dropped - 1])
        for m in METHODS:
            if hi <= lo:
                assert np.all(curves[m][j] == lo)
                normalized[m][j] = 1
            else:
                normalized[m][j] = np.clip((hi - curves[m][j]) / (hi - lo), 0, 1)
        bounds.append(dict(task=task, pool_size=len(pool), dropped_slowest=dropped,
                           best_seconds=lo, worst_retained_seconds=hi))
    means = {m: normalized[m].mean(axis=0) for m in METHODS}
    summed = {m: 1 - curves[m].sum(axis=0) / curves[m][:, 0].sum() for m in METHODS}
    geometric = {m: np.exp(np.log(curves[m] / curves[m][:, :1]).mean(axis=0)) for m in METHODS}
    ranks = rankdata(np.stack([curves[m][:, :201] for m in METHODS]), axis=0).mean(axis=1)
    np.testing.assert_allclose(ranks.sum(axis=0), 10, rtol=0, atol=1e-12)
    for m in METHODS:
        assert np.all(np.diff(means[m]) >= -1e-12)
        assert np.all(np.diff(summed[m]) >= -1e-12)
    fallback = sum(index['SemanticOPT', t]['sources'].get('random_fallback', 0) for t in tasks)
    footer = ('SemanticOPT: 10 × 20 rounds; ends at proposal 200. Baselines: 1,000 proposals each.\n'
              f'SemanticOPT includes {fallback:,}/40,000 constrained random fallback proposals. '
              'SQL caps, seeds, hosts and run dates differ.')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    draw_comparison(means, 'Mean normalized score (higher is better)', 'query_convergence',
                    footer + '\nFixed per-query min–max bounds pool all four observed runs; slowest 20% excluded only from bounds. Score axis: 0.8–1.0.', (.8, 1.005))
    draw_comparison(summed, '1 − summed best runtime / summed Bao best\n(higher is better)',
                    'query_summed_runtime_score', footer)
    draw_comparison(geometric, 'Best runtime / initial Bao best\n(geometric mean; lower is better)',
                    'query_relative_runtime', footer)
    dump_csv('normalization_bounds.csv', bounds)
    dump_csv('convergence.csv', [dict(method=m, proposals=n, tasks=200,
        mean_normalized_score=float(means[m][n]), summed_runtime_score=float(summed[m][n]),
        geomean_runtime_ratio=float(geometric[m][n]),
        mean_rank_at_equal_budget=float(ranks[i, n]) if n <= 200 else '')
        for i, m in enumerate(METHODS) for n in range(BUDGETS[m] + 1)])
    dump_csv('per_task_comparison_at_200.csv', [dict(task=t, method=m,
        initial_best_seconds=float(curves[m][j, 0]), best_seconds=float(curves[m][j, 200]),
        runtime_ratio=float(curves[m][j, 200] / curves[m][j, 0]),
        normalized_score=float(normalized[m][j, 200])) for j, t in enumerate(tasks) for m in METHODS])
    np.savez_compressed(OUT / 'per_task_curves.npz', tasks=np.asarray(tasks),
                        **{m: curves[m] for m in METHODS},
                        **{m + '_normalized': normalized[m] for m in METHODS})
    summary = {m: dict(complete_queries=200, completed_proposals_per_query=BUDGETS[m],
        at_200=dict(normalized_score=float(means[m][200]),
                    geomean_runtime_ratio=float(geometric[m][200]),
                    geomean_speedup=float(1 / geometric[m][200]),
                    summed_runtime_reduction_percent=float(100 * summed[m][200]),
                    mean_rank=float(ranks[i, 200])),
        final=dict(normalized_score=float(means[m][-1]),
                   geomean_runtime_ratio=float(geometric[m][-1]))) for i, m in enumerate(METHODS)}
    notes = [
        'All 200 fixed held-out queries are included. No task filtering or extrapolation.',
        'All four methods share exactly 49 Bao initialization runtime/censoring measurements; source plan IDs agree for baselines and join groups agree with SemanticOPT.',
        'Only uncensored successful measured runtimes improve incumbents. Charged baseline cache hits remain proposal slots.',
        'Min–max bounds use full observed uncensored pools across all four methods, dropping the slowest floor(20%) for bounds only, and clipping scores to [0,1]. Bounds are retrospective and method-dependent; these scores differ from the previous three-method normalization.',
        'Summed-runtime and geometric-relative-runtime views do not use pooled min–max bounds.',
        'Individual points are observed proposal prefixes; lines connect observations without smoothing or resampling.',
        'The displayed normalized score axis is zoomed to 0.8–1.0; exported scores retain the full [0,1] scale.',
        'SemanticOPT batch size is 10; within-batch ordering is recorded evaluation order, not feedback between its ten proposals.',
        f'SemanticOPT includes {fallback} constrained random fallback proposals out of 40000; not a pure LLM-only trajectory.',
        'SQL timeout: FSBO 50s, SemanticOPT 50s, TPE 100s, LOLBO adaptive. Seeds: FSBO123, SemanticOPT42, TPE/LOLBO0.',
        'One run per method; seeds, hosts, dates and storage differ. No seed-replication uncertainty is estimated.',
        'Production histories were opened read-only and were not modified.',
    ]
    audit = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        summary=summary, semanticopt_random_fallback=fallback, notes=notes,
        initialization_verified_equal=True, sources=[r['source'] for r in records],
        completion_marker=complete, builder_sha256=sha(Path(__file__).read_bytes()))
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# Query convergence: FSBO, LOLBO, TPE, SemanticOPT', '', audit['generated_at'], '',
        'Main plot: `query_convergence.png` (also PDF and SVG). Left: equal 200-proposal budget; right: all observed proposals. SemanticOPT ends at 200.', '',
        '| Method | Complete queries | Proposals/query | Score at 200 ↑ | Runtime ratio at 200 ↓ | Speedup at 200 ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        s = summary[m]['at_200']
        lines.append(f"| {m} | 200/200 | {BUDGETS[m]} | {s['normalized_score']:.4f} | {s['geomean_runtime_ratio']:.4f} | {s['geomean_speedup']:.3f}× |")
    lines += ['', *['- ' + note for note in notes], '',
        'Additional plots: `query_summed_runtime_score` and `query_relative_runtime` (PNG/PDF/SVG).',
        'Exports: convergence.csv, per_task_comparison_at_200.csv, per_task_curves.npz, normalization_bounds.csv, audit.json.',
        'Reproduce: `python3 build_plot.py`. Only this report directory is written.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(summary=summary, random_fallback=fallback), indent=2), flush=True)


if __name__ == '__main__':
    main()
