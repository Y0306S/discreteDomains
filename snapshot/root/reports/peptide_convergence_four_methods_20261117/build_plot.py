"""Read-only refresh of four peptide methods with explicitly different budgets."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import sqlite3
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
OLD = Path('/root/reports/peptide_convergence_20261101')
SEM = Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110')
POOL = Path('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json')
METHODS = ['FSBO', 'LOLBO', 'TPE', 'SemanticOPT']
COLORS = dict(zip(METHODS, ['#009E73', '#E69F00', '#0072B2', '#00A6BF']))
LABELS = {'FSBO': 'FSBO (frozen VAE)', 'LOLBO': 'LOLBO', 'TPE': 'TPE (global)', 'SemanticOPT': 'SemanticOPT'}
SPEC = importlib.util.spec_from_file_location('distance_audit', '/root/reports/peptide_tpe_diagnosis_20261101/audit.py')
DIST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIST)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_baseline(job):
    method, task, source, expected_hash, pool = job
    path = Path(source)
    cache = OUT / 'cache' / f'{method}_{task}.npz'
    stamp = [path.stat().st_size, path.stat().st_mtime_ns]
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            record = json.loads(str(d['record']))
            if (record['history_sha256'] == expected_hash and record['archive_stat'] == stamp
                    and 'proposal_curve' in d and d['proposal_curve'].shape == (5001,)):
                return method, task, d['curve'].copy(), d['proposal_curve'].copy(), d['pool'].copy(), record
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.name == 'history.json':
                raw = archive.extractfile(member).read()
                break
        else:
            raise ValueError(f'No history: {path}')
    assert sha(raw) == expected_hash, path
    h = json.loads(raw)
    assert h['identity']['task'] == task and h['status'] == 'complete'
    assert h['reference'] == pool['reference']
    assert (h['initialization'], h['proposal_budget'], h['fresh_budget']) == (100, 5000, 5000)
    rows = h['observations']
    initial, post = rows[:100], rows[100:]
    assert all(r['phase'] == 'initialization' for r in initial)
    assert [r['sequence'] for r in initial] == pool['sequences']
    np.testing.assert_allclose([r['score'] for r in initial], pool['scores'], rtol=0, atol=1e-6)
    assert all(np.isfinite(r['mic']) and r['mic'] > 0 and np.isclose(r['score'], -r['mic']) for r in rows)
    fresh = sum(not r['cached'] for r in post)
    assert fresh == h['fresh_evaluations'] <= 5000
    assert sum(r['phase'] == 'acquisition' for r in post) == h['proposals'] <= 5000
    assert h['proposals'] == 5000 or fresh == 5000
    assert len(post) - fresh == h['cache_hits']
    best = min(r['mic'] for r in initial if r['feasible'])
    xs, ys, used = [0], [best], 0
    px, py, proposals_used = [0], [best], 0
    for _, batch in itertools.groupby(post, key=lambda r: (r['round'], r['phase'], r['elapsed'])):
        batch = list(batch)
        used += sum(not r['cached'] for r in batch)
        proposals_used += sum(r['phase'] == 'acquisition' for r in batch)
        best = min([best] + [r['mic'] for r in batch if r['feasible']])
        if used == xs[-1]:
            ys[-1] = best
        else:
            xs.append(used)
            ys.append(best)
        if proposals_used == px[-1]:
            py[-1] = best
        else:
            px.append(proposals_used)
            py.append(best)
    assert used == fresh and np.isclose(best, h['best_feasible_mic'], rtol=0, atol=1e-6)
    # At no extra cost, a completed task keeps its actual measured incumbent.
    curve = np.asarray(ys)[np.searchsorted(xs, np.arange(5001), side='right') - 1]
    proposal_curve = np.asarray(py)[np.searchsorted(px, np.arange(5001), side='right') - 1]
    assert proposals_used == h['proposals'] and proposals_used >= 1000
    valid_pool = np.asarray([r['mic'] for r in rows if r['feasible']])
    assert np.all(np.diff(curve) <= 0)
    record = dict(method=method, task=task, source=str(path), history_sha256=expected_hash,
        archive_stat=stamp, fresh_evaluations=fresh, acquisition_proposals=h['proposals'],
        cache_hits=h['cache_hits'], feasible_fresh=sum(r['feasible'] and not r['cached'] for r in post),
        final_best_mic=best, initial_best_mic=float(curve[0]))
    np.savez_compressed(cache, curve=curve, proposal_curve=proposal_curve, pool=valid_pool, record=json.dumps(record))
    return method, task, curve, proposal_curve, valid_pool, record


def read_semantic(task, pool):
    path = SEM / 'peptide/runs' / task / 'observations.sqlite'
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        c.execute('BEGIN')
        raw_initial = c.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0]
        initial = json.loads(raw_initial)
        records = c.execute('SELECT n,value FROM records ORDER BY n').fetchall()
        pending_row = c.execute("SELECT value FROM state WHERE key='pending'").fetchone()
        pending = json.loads(pending_row[0]) if pending_row is not None else None
    finally:
        c.close()
    status = json.loads((path.parent / 'status.json').read_text())
    assert status['status'] == 'complete' and status['evaluations'] == status['budget'] == 1000
    assert status['batch_size'] == 50 and status['completed_rounds'] == 20
    assert len(initial) == 100 and len(records) == 1000 and pending is None
    assert [r['configuration']['sequence'] for r in initial] == pool['sequences']
    np.testing.assert_allclose([r['objective'] for r in initial], -np.asarray(pool['scores']), rtol=0, atol=1e-6)
    distance = DIST.distances(pool['reference'])
    observed = set()
    values, sources, parsed = [], Counter(), []
    digest = hashlib.sha256(raw_initial.encode())
    for n, raw in records:
        r = json.loads(raw)
        assert n == len(parsed) and r['evaluation'] == n + 1
        parsed.append(r)
        sources[r.get('proposal_source', 'historical_unlabelled')] += 1
        digest.update(raw.encode())
    for r in initial + parsed:
        seq = r['configuration']['sequence']
        value = float(r['objective'])
        assert r['status'] == 'success' and not r['censored'] and r['feasible']
        assert np.isfinite(value) and value > 0
        assert 1 <= len(seq) <= 30 and set(seq) <= set('ACDEFGHIKLMNPQRSTVWY') and seq.count('C') <= 1
        assert 1 - distance(seq) / len(pool['reference']) >= .75
        assert seq not in observed, (task, 'duplicate observation')
        observed.add(seq)
        values.append(value)
    assert sources['random_fallback'] == status['random_fallback_evaluations']
    best = min(values[:100])
    curve = [best]
    # All 50 scores from a batch become visible at the round endpoint.
    for n in range(0, 1000, 50):
        curve.extend([best] * 49)
        best = min(best, *values[100 + n:150 + n])
        curve.append(best)
    record = dict(method='SemanticOPT', task=task, source=str(path),
        logical_snapshot_sha256=digest.hexdigest(), fresh_evaluations=1000,
        acquisition_proposals=1000, cache_hits=0, feasible_fresh=1000,
        initial_best_mic=curve[0], final_best_mic=best, proposal_sources=dict(sources))
    return 'SemanticOPT', task, np.asarray(curve), np.asarray(curve), np.asarray(values), record


def draw(values, ylabel, stem, footer, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={'width_ratios': [1, 1.25]})
    for m in METHODS:
        style = ':' if m in ('TPE', 'LOLBO') else '-'
        for ax, end in zip(axes, [1000, len(values[m]) - 1]):
            x = np.arange(end + 1)
            ax.plot(x, values[m][:end + 1], color=COLORS[m], label=LABELS[m], lw=2.4,
                    linestyle=style, drawstyle='steps-post')
        axes[1].plot(len(values[m]) - 1, values[m][-1], 'o', color=COLORS[m], ms=4)
    for ax, limit, title in zip(axes, [1000, 5000], ['Equal budget: 1,000 evaluations', 'Full completed baseline runs']):
        ax.set(xlim=(0, limit), xlabel='Fresh oracle evaluations (100 starts excluded)', ylabel=ylabel, title=title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=10)
    axes[1].axvline(1000, color='#777777', alpha=.35, lw=1)
    fig.suptitle('Peptide convergence · 200 held-out tasks', fontsize=17, weight='bold')
    fig.text(.5, .025, footer, ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .12, 1, .94))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    pool = json.loads(POOL.read_text())
    tasks = sorted(pool)
    assert len(tasks) == 200
    jobs = []
    with (OLD / 'fsbo_local_task_summary.csv').open() as f:
        rows = [('FSBO', r) for r in csv.DictReader(f)]
    with (OLD / 'task_summary.csv').open() as f:
        rows += [(r['method'], r) for r in csv.DictReader(f) if r['method'] in ('LOLBO', 'TPE')]
    for method, r in rows:
        jobs.append((method, r['task'], r['archive'], r['history_sha256'], pool[r['task']]))
    assert len(jobs) == 600
    extracted = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, r in enumerate(executor.map(read_baseline, jobs), 1):
            extracted.append(r)
            if i % 25 == 0:
                print(f'Validated {i}/600 baseline histories against the prior full archive audit.', flush=True)
    for i, task in enumerate(tasks, 1):
        extracted.append(read_semantic(task, pool[task]))
        if i % 50 == 0:
            print(f'Validated {i}/200 SemanticOPT checkpoints.', flush=True)
    index = {(m, t): (curve, p, record) for m, t, curve, proposal_curve, p, record in extracted}
    proposal_curves = {m: np.stack([next(r[3] for r in extracted if r[0] == m and r[1] == t) for t in tasks]) for m in METHODS}
    curves = {m: np.stack([index[m, t][0] for t in tasks]) for m in METHODS}
    with np.load(OLD / 'per_task_curves_updated_fsbo.npz', allow_pickle=False) as old:
        assert old['tasks'].tolist() == tasks
        for method, key in [('FSBO', 'FSBO-local'), ('LOLBO', 'LOLBO'), ('TPE', 'TPE')]:
            np.testing.assert_array_equal(curves[method][:, ::25], old[key])
    for m in METHODS:
        np.testing.assert_array_equal(curves[m][:, 0], curves['FSBO'][:, 0])
        assert np.all(np.diff(curves[m], axis=1) <= 0) and np.isfinite(curves[m]).all()
    normalized = {m: np.empty_like(curves[m]) for m in METHODS}
    scales = []
    for j, task in enumerate(tasks):
        p = np.sort(np.concatenate([index[m, task][1] for m in METHODS]))
        dropped = int(np.floor(.2 * len(p)))
        lo, hi = float(p[0]), float(p[len(p) - dropped - 1])
        for m in METHODS:
            if hi <= lo:
                assert np.all(curves[m][j] == lo)
                normalized[m][j] = 1
            else:
                normalized[m][j] = np.clip((hi - curves[m][j]) / (hi - lo), 0, 1)
        scales.append(dict(task=task, pool_size=len(p), slowest_dropped=dropped,
                           best_mic=lo, worst_retained_mic=hi))
    means = {m: normalized[m].mean(axis=0) for m in METHODS}
    geometric = {m: np.exp(np.log(curves[m] / curves[m][:, :1]).mean(axis=0)) for m in METHODS}
    improvement = {m: (100 * (1 - curves[m] / curves[m][:, :1])).mean(axis=0) for m in METHODS}
    rawmeans = {m: curves[m].mean(axis=0) for m in METHODS}
    records = [r[-1] for r in extracted]
    fallback = sum(r.get('proposal_sources', {}).get('random_fallback', 0) for r in records)
    footer = ('SemanticOPT: 50 × 20 rounds; stops at 1,000. All methods share 100 measured starts per task.\n'
              'Baselines: up to 5,000 fresh calls or acquisition proposals; completed tasks retain their measured final incumbent.\n'
              f'SemanticOPT includes {fallback:,}/200,000 random fallback proposals. TPE is the global variant; batches revealed at endpoints.')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    draw(means, 'Mean normalized score (higher is better)', 'peptide_convergence',
         footer + '\nPer-task min–max bounds pool feasible measurements from all four runs; slowest 20% excluded only from bounds.', (0, 1.025))
    draw(geometric, 'Best MIC / initial best MIC\n(geometric mean; lower is better)', 'peptide_relative_mic', footer)
    draw(improvement, 'Mean improvement from initial best MIC (%)', 'peptide_improvement', footer)
    draw(rawmeans, 'Mean best feasible MIC (lower is better)', 'peptide_mean_mic', footer)
    np.savez_compressed(OUT / 'per_task_curves.npz', tasks=np.asarray(tasks),
                        **curves, **{m + '_normalized': normalized[m] for m in METHODS},
                        **{m + '_by_proposal': proposal_curves[m] for m in METHODS})
    write_csv('normalization_bounds.csv', scales)
    write_csv('convergence.csv', [dict(method=m, fresh_evaluations=n, tasks=200,
        mean_normalized_score=float(means[m][n]), mean_best_mic=float(rawmeans[m][n]),
        geomean_mic_ratio=float(geometric[m][n]), mean_improvement_percent=float(improvement[m][n]),
        tasks_with_observed_budget=sum(index[m, t][2]['fresh_evaluations'] >= n for t in tasks))
        for m in METHODS for n in range(len(means[m]))])
    write_csv('per_task_comparison_at_1000.csv', [dict(task=t, method=m,
        initial_best_mic=float(curves[m][j, 0]), best_mic_at_1000=float(curves[m][j, 1000]),
        final_best_mic=float(curves[m][j, -1]), actual_fresh_evaluations=index[m, t][2]['fresh_evaluations'])
        for j, t in enumerate(tasks) for m in METHODS])
    summary = {m: dict(tasks=200, at_1000=dict(normalized_score=float(means[m][1000]),
        mean_best_mic=float(rawmeans[m][1000]), geomean_mic_ratio=float(geometric[m][1000]),
        mean_improvement_percent=float(improvement[m][1000])),
        final=dict(mean_best_mic=float(rawmeans[m][-1]), mean_improvement_percent=float(improvement[m][-1]),
                   normalized_score=float(means[m][-1])),
        fresh_evaluations_range=[min(index[m, t][2]['fresh_evaluations'] for t in tasks),
                                 max(index[m, t][2]['fresh_evaluations'] for t in tasks)]) for m in METHODS}
    notes = [
        'All 200 held-out tasks; identical 100 starting peptides and measured scores checked across methods.',
        'FSBO uses the latest completed frozen-author-VAE local/recenter run (November 2 campaign). LOLBO uses completed native V5. TPE uses completed global independent latent-space TPE; prepared TPE-TR has not launched.',
        'X counts fresh logical APEX calls, including infeasible outcomes and recentering. Cached duplicates cost zero. Batch outcomes appear together at their endpoints.',
        'All baselines have at least 1000 fresh calls per task, so the left panel uses a common fully observed budget.',
        'Completed baseline tasks retain their last measured incumbent after exhausting either 5000 acquisition proposals or 5000 fresh calls; no continuation is simulated. CSV exports actual task support.',
        'SemanticOPT ends at 1000; no extension to 5000.',
        'Only feasible measurements update incumbents or enter normalization pools. Min–max bounds use pooled measurements from all four full trajectories, dropping slowest floor(20%) for bounds only. Scores are retrospective and method-dependent.',
        f'SemanticOPT includes {fallback}/200000 constrained random fallback proposals.',
        'Global TPE has very low similarity feasibility (171/888995 fresh measurements in the prior audit); its nearly flat curve describes this configuration.',
        'Baseline history hashes match the previous full archive-member audit; regenerated 25-call checkpoints match the previously validated curves exactly. SemanticOPT constraints and uniqueness were checked for all observations.',
        'One optimizer seed per task: baselines1, SemanticOPT42. Different devices/dates and numerical execution settings; no replicated-seed uncertainty estimated.',
        'Production data opened read-only; no inference, oracle calls or training performed.',
    ]
    audit = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), summary=summary,
        semanticopt_random_fallback=fallback, notes=notes, initialization_verified_equal=True,
        sources=records, pool_source=str(POOL), pool_sha256=sha(POOL.read_bytes()), builder_sha256=sha(Path(__file__).read_bytes()))
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# Peptide convergence: four methods', '', audit['generated_at'], '',
        '| Method | Tasks | Score at 1000 ↑ | Mean MIC at 1000 ↓ | Mean improvement at 1000 ↑ |',
        '|---|---:|---:|---:|---:|']
    for m in METHODS:
        s = summary[m]['at_1000']
        lines.append(f"| {LABELS[m]} | 200 | {s['normalized_score']:.4f} | {s['mean_best_mic']:.4f} | {s['mean_improvement_percent']:.2f}% |")
    lines += ['', *['- ' + note for note in notes], '',
        'Plots: peptide_convergence, peptide_relative_mic, peptide_improvement, peptide_mean_mic (PNG/PDF/SVG).',
        'Reproduce: `python3 build_plot.py`. Cached validated extractions avoid rereading unchanged baseline archives.',
        'Exports: convergence.csv, per_task_comparison_at_1000.csv, per_task_curves.npz, normalization_bounds.csv, audit.json.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(summary=summary, random_fallback=fallback), indent=2), flush=True)


if __name__ == '__main__':
    main()
