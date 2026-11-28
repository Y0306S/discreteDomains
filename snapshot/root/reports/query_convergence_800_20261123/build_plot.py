"""Read-only convergence snapshot. No extrapolation or production writes."""
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
BASE = Path('/root/data/cys/baseline/campaigns')
HYB = BASE / 'query_our_fsbo_vae80_test200_20261117'
SEM = BASE / 'semanticopt_query_400_pilot800_20261123'
BOLT = BASE / 'query_bolt30_lolbo_total800_20261123'
CACHE = Path('/root/reports/query_convergence_four_methods_20261117/per_task_curves.npz')
sys.path.insert(0, str(HYB / 'batch_code'))
METHODS = ['LLM + FSBO', 'FSBO', 'LOLBO', 'TPE', 'SemanticOPT', 'BOLT continuation']
COLORS = ['#B33D84', '#009E73', '#E69F00', '#0072B2', '#00A6BF', '#6A51A3']
START = {m: (30 if m in ('LLM + FSBO', 'BOLT continuation') else 49) for m in METHODS}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def journal(path, key):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as db:
        db.execute('BEGIN')
        row = db.execute('SELECT value FROM kv WHERE key=?', (key,)).fetchone()
    return pickle.loads(row[0]) if row else None


def incumbent(rows):
    best = np.inf
    values = []
    for row in rows:
        value = float(row['runtime'])
        assert np.isfinite(value) and value > 0
        if not row['censored']:
            best = min(best, value)
        values.append(best)
    return np.asarray(values)


def read_task(task):
    state = journal(HYB / 'orpo_batch10/runs' / task / 'runtime.sqlite', 'optimization_state')
    rows = state['rows']
    assert len(rows) == 800 and len(state['events']) == 77
    assert [(r['plan'], r['runtime'], r['censored']) for r in rows[:30]] == [
        (r['plan'], r['runtime'], r['censored']) for r in WARM[task]]
    result = {'LLM + FSBO': incumbent(rows)[29:]}
    path = SEM / 'query/runs' / task / 'observations.sqlite'
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as db:
        db.execute('BEGIN')
        initial = json.loads(db.execute("SELECT value FROM state WHERE key='initial'").fetchone()[0])
        records = db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
    assert len(initial) == 49
    best = min(float(r['objective']) for r in initial if not r['censored'] and r['status'] == 'success')
    curve = [best]
    for expected, (n, raw) in enumerate(records):
        assert n == expected
        row = json.loads(raw)
        assert row['evaluation'] == n + 1
        if not row['censored'] and row['status'] == 'success':
            best = min(best, float(row['objective']))
        curve.append(best)
    result['SemanticOPT'] = np.asarray(curve)
    path = BOLT / 'runs' / task / 'runtime.sqlite'
    state = journal(path, 'lolbo_state') if path.exists() else None
    bolt_rows = state['rows'] if state else WARM[task]
    assert [(r['plan'], r['runtime'], r['censored']) for r in bolt_rows[:30]] == [
        (r['plan'], r['runtime'], r['censored']) for r in rows[:30]]
    result['BOLT continuation'] = incumbent(bolt_rows)[29:]
    return task, result


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, stem):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def aggregate(cohort, method, total_axis=True):
    n = min(len(DATA[t][method]) for t in cohort)
    start = START[method] if total_axis else 0
    n = min(n, 801 - start)
    values = np.stack([DATA[t][method][:n] for t in cohort])
    anchors = np.asarray([ANCHOR[t] for t in cohort])[:, None]
    ratio = np.exp(np.log(values / anchors).mean(axis=0))
    return np.arange(start, start + n), ratio, values.sum(axis=0)


def draw(cohorts, stem, total_axis=True, raw=False):
    fig, axes = plt.subplots(1, len(cohorts), figsize=(7 * len(cohorts), 5.3), squeeze=False)
    for ax, (label, cohort) in zip(axes[0], cohorts.items()):
        for method, color in zip(METHODS, COLORS):
            x, ratio, summed = aggregate(cohort, method, total_axis)
            y = summed if raw else ratio
            end = int(x[-1])
            ax.step(x, y, where='post', color=color, lw=2.1,
                    linestyle='--' if method in ('LOLBO', 'TPE') else '-',
                    label=f'{method} (to {end})')
            ax.plot(x[-1], y[-1], 'o', color=color, ms=4)
        ax.axvline(400, color='#999999', ls=':', lw=1)
        ax.set(xlim=(0, 800), title=f'{label} · {len(cohort)} queries',
               xlabel='Total counted evaluations per query (warm starts included)' if total_axis else
               'New proposals per query (warm starts excluded)',
               ylabel='Summed best query runtime (seconds)' if raw else
               'Best runtime / Bao49 best (geometric mean)')
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle('Query convergence to 800 evaluations · lower is better', fontsize=16, weight='bold')
    fig.text(.5, .016,
             'Fixed query cohorts; each curve ends at the last budget reached by every query in that panel. No extrapolation.\n'
             'LLM + FSBO / BOLT start from 30 saved BOLT plans; other methods from 49 Bao plans. Historical execution conditions differ.',
             ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout(rect=(0, .09, 1, .93))
    save(fig, stem)


def main():
    global WARM, DATA, ANCHOR
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    WARM = json.loads((HYB / 'pools/batch10_frozen_pool.json').read_text())['tasks']
    extension = json.loads((SEM / 'extension.json').read_text())
    with np.load(CACHE, allow_pickle=False) as cached:
        tasks = cached['tasks'].tolist()
        DATA = {t: {m: cached[m][i].copy() for m in ('FSBO', 'LOLBO', 'TPE')} for i, t in enumerate(tasks)}
        old_sem = {t: cached['SemanticOPT'][i].copy() for i, t in enumerate(tasks)}
        ANCHOR = {t: float(cached['FSBO'][i, 0]) for i, t in enumerate(tasks)}
    assert len(tasks) == 200 and set(tasks) == set(extension['tasks']) == set(WARM)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for task, result in pool.map(read_task, tasks):
            np.testing.assert_array_equal(result['SemanticOPT'][:201], old_sem[task])
            DATA[task].update(result)
    for task in tasks:
        for method in METHODS:
            curve = DATA[task][method]
            assert np.isfinite(curve).all() and np.all(curve > 0)
            assert np.all(np.diff(curve) <= 0), (task, method)
    active = sorted(t for t in tasks if len(DATA[t]['BOLT continuation']) > 1)
    cohorts = {'All held-out queries': tasks, 'Current BOLT query cohort': active}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    draw(cohorts, 'query_convergence_800')
    draw(cohorts, 'query_summed_runtime_800', raw=True)
    draw(cohorts, 'query_new_proposals_800', total_axis=False)
    pilot = extension['pilot_tasks']
    draw({'First 10 SemanticOPT pilot queries': pilot}, 'query_pilot10_800')
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.8))
    for ax, task in zip(axes.flat, pilot):
        for method, color in zip(METHODS, COLORS):
            curve = DATA[task][method][:801 - START[method]]
            x = np.arange(START[method], START[method] + len(curve))
            ax.step(x, curve / ANCHOR[task], where='post', color=color, lw=1.5,
                    linestyle='--' if method in ('LOLBO', 'TPE') else '-', label=method)
            ax.plot(x[-1], curve[-1] / ANCHOR[task], 'o', color=color, ms=3)
        ax.axvline(400, color='#999999', ls=':', lw=1)
        ax.set(title=task, xlim=(0, 800), xlabel='Total counted evaluations', ylabel='Runtime / Bao49 best')
        ax.grid(axis='y', alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False)
    fig.suptitle('First 10 query tasks · measured convergence only · lower is better', fontsize=16)
    fig.tight_layout(rect=(0, .06, 1, .95))
    save(fig, 'query_pilot10_per_task_800')
    rows = []
    summary = {}
    for label, cohort in {**cohorts, 'First 10 pilot queries': pilot}.items():
        summary[label] = {}
        for method in METHODS:
            x, ratio, summed = aggregate(cohort, method)
            summary[label][method] = {'queries': len(cohort), 'last_common_total_evaluations': int(x[-1]),
                                    'runtime_ratio_at_endpoint': float(ratio[-1]),
                                    'summed_runtime_seconds_at_endpoint': float(summed[-1])}
            for target in (400, 800):
                if x[0] <= target <= x[-1]:
                    i = target - int(x[0])
                    summary[label][method][f'runtime_ratio_at_{target}'] = float(ratio[i])
            rows += [dict(cohort=label, queries=len(cohort), method=method, total_evaluations=int(n),
                          geomean_runtime_over_bao49=float(r), summed_best_runtime_seconds=float(s))
                     for n, r, s in zip(x, ratio, summed)]
    write_csv('convergence.csv', rows)
    write_csv('per_query_coverage.csv', [dict(task=t, method=m, warm_start_evaluations=START[m],
        total_evaluations_available=START[m] + len(DATA[t][m]) - 1,
        final_best_runtime_seconds=float(DATA[t][m][-1])) for t in tasks for m in METHODS])
    arrays = {'tasks': np.asarray(tasks)}
    for method in METHODS:
        array = np.full((len(tasks), 801), np.nan)
        for i, task in enumerate(tasks):
            c = DATA[task][method][:801 - START[method]]
            array[i, START[method]:START[method]+len(c)] = c
        arrays[method] = array
    np.savez_compressed(OUT / 'observed_total_evaluation_curves.npz', **arrays)
    notes = [
        '800 evaluations means 800 total counted observations/slots per query, including initialization; it is 80 batches of 10 for the hybrid and BOLT. SemanticOPT proposal targets exclude its49Bao starts.',
        'LLM + FSBO is the completed retrained-VAE ORPO + FSBO hybrid (RL disabled),200/200queries,800total evaluations each.',
        'FSBO,LOLBO,TPE are historical standalone baselines; their raw49Bao+1000proposal histories are represented by the previously audited unpadded curve cache. Charged cache-hit proposals remain slots, so this is not an equal fresh-SQL-call comparison.',
        'Each aggregate uses a fixed query cohort and truncates each method at the minimum available prefix across that cohort. No forward filling, variable-cohort averaging, smoothing, or extrapolation.',
        'The full200-query BOLT trajectory currently has only its shared30-observation warm-start point. Its active-query panel and per-task pilot plots expose actual ongoing progress.',
        'Incumbents improve only on uncensored successful measured runtimes. Timeouts count as evaluation slots but cannot improve the incumbent. Decoder rejection attempts without SQL do not count as evaluations.',
        'LLM + FSBO and BOLT share exactly the same30warm measurements (verified); SemanticOPT retains its original200-record prefix (verified) and shares Bao initialization with the audited standalone baselines.',
        'Relative runtime uses a fixed per-query Bao49 best anchor, then a geometric mean; lower is better. Summed-runtime plots show rawseconds. Initial pools differ between method families.',
        'Historical devices, dates, seeds, SQL timeout and storage policies differ; these comparisons are descriptive. No seed-replication uncertainty is available.',
        'The extra new-proposals figure excludes initialization; the completed800-total hybrid therefore ends at770new proposals. This alternative axis must not be confused with the primary total-evaluation axis.',
        'SQLite journals were opened read-only in per-task consistent transactions. Running tasks can advance during the snapshot; only complete saved batch states are included for BOLT and hybrid.',
    ]
    audit = dict(snapshot_started=started, snapshot_finished=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 summary=summary, notes=notes, active_bolt_tasks=active,
                 source_campaigns=[str(HYB), str(SEM), str(BOLT)],
                 baseline_curve_cache=str(CACHE), baseline_curve_cache_sha256=sha(CACHE),
                 baseline_audit=str(CACHE.parent / 'audit.json'), builder_sha256=sha(Path(__file__)))
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# Query convergence through800total evaluations', '', started, '',
             'Primary: query_convergence_800.png/.pdf/.svg. Left: all200queries. Right: current BOLT cohort.', '',
             'Additional: query_summed_runtime_800, query_pilot10_800, query_pilot10_per_task_800, query_new_proposals_800.', '',
             *['- '+n for n in notes], '',
             'Exports: convergence.csv,per_query_coverage.csv,observed_total_evaluation_curves.npz,audit.json.',
             'Refresh read-only with: python3 build_plot.py.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(audit['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
