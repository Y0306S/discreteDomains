"""Rebuild a read-only comparison from the three complete held-out query campaigns."""
import collections
import csv
import hashlib
import json
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
BASE = Path('/root/data/cys/logs/query_test200_baselines_20261012')
FSBO = Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1')
METHODS = ['FSBO', 'TPE', 'LOLBO']
COLORS = ['#1864ab', '#e67700', '#2b8a3e']
TASKS = sorted((BASE / 'test200.txt').read_text().split())
assert len(TASKS) == len(set(TASKS)) == 200


def read_case(case):
    method, task = case
    folder = FSBO / 'runs' / task if method == 'FSBO' else BASE / 'outputs' / task / method.lower() / 'seed0'
    path = folder / 'history.json'
    raw = path.read_bytes()
    doc = json.loads(raw)
    status = json.loads((folder / 'status.json').read_text())
    rows = doc['observations']
    assert status['status'] == 'complete' and status['proposals'] == 1000
    assert status['initial_observations'] == 49 and len(rows) == 1049
    assert status['unresolved_database_calls'] == 0
    assert doc['task_id'] == task
    initial = []
    best = np.inf
    for row in rows[:49]:
        e = row['extra_info']
        value = float(e['runtime'])
        assert e['source'] == 'bao_import'
        initial.append((str(e['plan_id']), value, e['censored']))
        if not e['censored'] and np.isfinite(value) and value > 0:
            best = min(best, value)
    assert np.isfinite(best), (method, task, 'no uncensored initial runtime')
    baseline = best
    curve = [best]
    for i, row in enumerate(rows[49:]):
        e = row['extra_info']
        assert e['proposal_index'] == i and e['charged_index'] == i + 49
        value = float(e['runtime'])
        assert np.isfinite(value) and value > 0
        if not e['censored']:
            best = min(best, value)
        curve.append(best)
    assert np.isclose(best, status['best_uncensored_runtime'], rtol=1e-7), (method, task, best, status['best_uncensored_runtime'])
    return dict(method=method, task=task, curve=curve, baseline=baseline,
                initial=initial, source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                seed=status['seed'], database_calls=status['database_calls'],
                cache_hits=status['cache_hits'], vae_sha256=status['vae_sha256'])


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cases = [(m, t) for m in METHODS for t in TASKS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(read_case, cases))
    index = {(r['method'], r['task']): r for r in records}
    for task in TASKS:
        first = index['FSBO', task]
        for method in METHODS[1:]:
            other = index[method, task]
            assert first['initial'] == other['initial'], (task, method, 'Bao initialization mismatch')
            assert first['vae_sha256'] == other['vae_sha256']
    curves = np.asarray([[index[m, t]['curve'] for t in TASKS] for m in METHODS])
    baselines = curves[:, :, :1]
    relative = curves / baselines
    assert np.all(np.diff(curves, axis=2) <= 0) and np.isfinite(relative).all()
    geometric = np.exp(np.log(relative).mean(axis=1))
    ranks = rankdata(curves, axis=0, method='average').mean(axis=1)
    np.savez_compressed(OUT / 'per_task_curves.npz', methods=METHODS, tasks=TASKS,
                        best_uncensored_runtime_seconds=curves, relative_runtime=relative)
    write_csv('convergence.csv', [dict(proposals=n, **{
        f'{method.lower()}_{metric}': float(array[i, n])
        for i, method in enumerate(METHODS)
        for metric, array in [('geomean_relative_runtime', geometric), ('mean_rank', ranks)]
    }) for n in range(1001)])
    write_csv('per_task_final.csv', [dict(task=t, method=m, seed=index[m, t]['seed'],
        initial_best_seconds=curves[i, j, 0], final_best_seconds=curves[i, j, -1],
        runtime_ratio=relative[i, j, -1], speedup=1 / relative[i, j, -1],
        database_calls=index[m, t]['database_calls'], cache_hits=index[m, t]['cache_hits'])
        for j, t in enumerate(TASKS) for i, m in enumerate(METHODS)])
    summary = {m: dict(tasks=200, proposals_per_task=1000, seed=index[m, TASKS[0]]['seed'],
        geometric_mean_relative_runtime=float(geometric[i, -1]),
        geometric_mean_speedup=float(1 / geometric[i, -1]),
        runtime_reduction_percent=float(100 * (1 - geometric[i, -1])),
        mean_rank=float(ranks[i, -1]),
        tasks_improved=int(np.sum(relative[i, :, -1] < 1 - 1e-12))) for i, m in enumerate(METHODS)}
    pairwise = {}
    for i, m in enumerate(METHODS[1:], 1):
        a, b = curves[0, :, -1], curves[i, :, -1]
        ties = np.isclose(a, b, rtol=1e-9, atol=1e-12)
        pairwise[m] = dict(fsbo_wins=int(np.sum((a < b) & ~ties)),
                           ties=int(ties.sum()), fsbo_losses=int(np.sum((a > b) & ~ties)),
                           fsbo_geomean_runtime_ratio=float(np.exp(np.log(a / b).mean())))
    notes = [
        'All 200 fixed held-out queries; all three methods complete 1000 charged proposals per query.',
        'Proposal 0 is the best uncensored runtime among the same 49 imported Bao observations; equality checked by plan_id, runtime and censoring flag. Recovered token representations can differ.',
        'Only uncensored runtimes update the best-so-far incumbent. Cache hits still consume proposal slots.',
        'Primary aggregation is exp(mean(log(best_runtime / initial_best_runtime))) across queries; lower is better.',
        'Average rank uses per-query raw runtimes with average ranks for exact ties; lower is better.',
        'FSBO seed 123; TPE and LOLBO seed 0. One run per method, no repeated-seed uncertainty estimate.',
        'SQL timeout differs: FSBO 50 seconds; TPE 100 seconds; LOLBO adaptive.',
        'Campaigns ran on different dates, hosts and storage; imported Bao timings are historical. This is a descriptive comparison, not a controlled same-hardware timing experiment.',
        'FSBO is frozen-VAE pretrained transfer GP; LOLBO is the frozen-VAE latent TuRBO variant.',
        'No query evaluations, workers, or production results were modified.'
    ]
    audit = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 summary=summary, fsbo_pairwise=pairwise, notes=notes,
                 initialization_verified_equal=True, sources=[{k: r[k] for k in ['method', 'task', 'source', 'sha256']} for r in records])
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={'width_ratios': [1.25, 1]})
    x = np.arange(1001)
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        axes[0].plot(x, geometric[i], color=color, lw=2.4, label=method)
        axes[1].plot(x, ranks[i], color=color, lw=2.2, label=method)
    axes[0].set_ylabel('Best runtime / initial Bao best\n(geometric mean; lower is better)')
    axes[0].set_title('Query convergence', loc='left', fontweight='bold')
    axes[1].set_title('Average rank', loc='left', fontweight='bold')
    axes[1].set_ylabel('Average rank (lower is better)')
    axes[1].set_ylim(0.95, 3.05)
    axes[1].set_yticks([1, 1.5, 2, 2.5, 3])
    for ax in axes:
        ax.set_xlabel('New acquisition proposals (49 Bao starts excluded)')
        ax.set_xlim(0, 1000)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle('FSBO vs TPE vs LOLBO · 200 held-out CEB queries', fontsize=15, y=0.98)
    fig.text(0.5, 0.02, 'Complete runs · FSBO: 50 s SQL cap / seed 123 · TPE: 100 s / seed 0 · LOLBO: adaptive cap / seed 0',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'query_fsbo_tpe_lolbo.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Query FSBO / TPE / LOLBO comparison', '', audit['generated_at'], '',
             '| Method | Complete queries | Geometric runtime ratio ↓ | Speedup vs Bao ↑ | Average rank ↓ |',
             '|---|---:|---:|---:|---:|']
    for m in METHODS:
        s = summary[m]
        lines.append(f"| {m} | 200/200 | {s['geometric_mean_relative_runtime']:.4f} | {s['geometric_mean_speedup']:.3f}× | {s['mean_rank']:.3f} |")
    lines += ['', *['- ' + note for note in notes], '',
              'Reproduce: `python3 build_plot.py`. The script only reads production histories and writes this report directory.',
              'Exports: PNG, PDF, SVG; convergence.csv; per_task_final.csv; per_task_curves.npz; audit.json.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(summary=summary, fsbo_pairwise=pairwise), indent=2), flush=True)


if __name__ == '__main__':
    main()
