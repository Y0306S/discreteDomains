"""Two query convergence alternatives using the shared ML/TaskSet presentation."""
import ast
import csv
import hashlib
import importlib.util
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'two_styles'
SHARED = Path('/root/reports/standardized_convergence_20261101/build_plots.py')
REFERENCE = Path('/root/data/cys/llmoptimizer_benchmark/examples/benchmark/plot_mltest20_algo40_avg_rank.py')
METHOD_KEYS = {'FSBO': 'fsbo_minmax', 'TPE': 'tpe', 'LOLBO': 'lolbo'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_pool(source):
    path = Path(source['source'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256'], path
    rows = json.loads(raw)['observations']
    assert len(rows) == 1049
    values = []
    for row in rows:
        e = row['extra_info']
        if not e['censored']:
            value = float(e['runtime'])
            assert math.isfinite(value) and value > 0
            values.append(value)
    assert values
    return source['method'], source['task'], values


def reference_normalizer():
    tree = ast.parse(REFERENCE.read_text())
    names = {'normalize', 'normalize_from_pooled_drop_worst'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'math': math, 'PRIMARY_DROP_FRAC': .2}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(REFERENCE), 'exec'), namespace)
    return namespace['normalize_from_pooled_drop_worst']


def draw(shared, methods, values, stem, ylabel, ylim, subtitle, note):
    fig, ax = plt.subplots(figsize=(12.8, 7.5), facecolor='white')
    for i, method in enumerate(methods):
        # Shared helper starts its x coordinates at 1; shift to proposal 0.
        x, y = shared.interpolate(values[i])
        x = x - 1
        style = shared.STYLE[METHOD_KEYS[method]]
        ax.plot(x, y, color=style['color'], linestyle='-' if style['transfer'] else ':',
                linewidth=2.5, label=f'{method}  {values[i, -1]:.3f}')
        np.testing.assert_allclose(y[::16], values[i], atol=1e-12, rtol=0)
    ax.set(xlim=(0, 1000), ylim=ylim, xlabel='Acquisition proposals per query', ylabel=ylabel)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='lower right', ncol=3, frameon=False, fontsize=10.5,
              title='Score at proposal 1,000', title_fontsize=10.5)
    fig.suptitle('Query convergence', fontsize=20, weight='bold', y=.97)
    fig.text(.5, .905, subtitle, ha='center', color='#555555', fontsize=11)
    fig.text(.5, .04, note + '\nPCHIP preserves observed points · Solid: transfer learning · Dotted: no cross-task transfer',
             ha='center', fontsize=9.5, color='#555555')
    fig.subplots_adjust(left=.10, right=.975, top=.85, bottom=.19)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    source_audit = json.loads((ROOT / 'audit.json').read_text())
    spec = importlib.util.spec_from_file_location('shared_convergence', SHARED)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    with np.load(ROOT / 'per_task_curves.npz', allow_pickle=False) as data:
        methods = data['methods'].tolist()
        tasks = data['tasks'].tolist()
        curves = data['best_uncensored_runtime_seconds'].copy()
    assert curves.shape == (3, 200, 1001)
    assert np.all(np.diff(curves, axis=2) <= 0)
    np.testing.assert_array_equal(curves[0, :, 0], curves[1, :, 0])
    np.testing.assert_array_equal(curves[0, :, 0], curves[2, :, 0])
    cache = OUT / 'pooled_uncensored_runtimes.npz'
    sources = source_audit['sources']
    source_identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    pools = {task: [] for task in tasks}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            assert str(data['source_identity']) == source_identity
            for i, task in enumerate(tasks):
                pools[task] = data[f'q{i}'].tolist()
        print('Loaded hash-identified normalization pools.', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, (_, task, values) in enumerate(executor.map(read_pool, sources), 1):
                pools[task].extend(values)
                if i % 100 == 0:
                    print(f'Validated source histories: {i}/600', flush=True)
        np.savez_compressed(cache, source_identity=np.asarray(source_identity),
                            **{f'q{i}': np.asarray(pools[task]) for i, task in enumerate(tasks)})
    normalized = np.empty_like(curves)
    scales = []
    degenerate = []
    reference = reference_normalizer()
    for j, task in enumerate(tasks):
        runtimes = np.sort(np.asarray(pools[task]))
        dropped = int(math.floor(.2 * len(runtimes)))
        lo = float(runtimes[0])
        hi = float(runtimes[len(runtimes) - dropped - 1])
        if hi <= lo:
            degenerate.append(task)
            # No observed distinction: all incumbents are the same best runtime.
            assert np.all(curves[:, j] == lo)
            normalized[:, j] = 1.
        else:
            normalized[:, j] = np.clip((hi - curves[:, j]) / (hi - lo), 0, 1)
            # Check the exact ML reference on every integer budget.
            performance_pool = (-runtimes).tolist()
            for n in range(1001):
                direct = reference({m: -float(curves[i, j, n]) for i, m in enumerate(methods)},
                                   performance_pool, drop_worst_frac=.2)
                np.testing.assert_allclose(normalized[:, j, n], [direct[m] for m in methods],
                                           rtol=0, atol=1e-12)
        scales.append(dict(task=task, pool_size=len(runtimes), worst_values_dropped_for_bounds=dropped,
                           best_runtime_seconds=lo, worst_retained_runtime_seconds=hi,
                           degenerate=hi <= lo))
    means = normalized.mean(axis=1)
    summed = curves.sum(axis=1)
    sum_scores = 1 - summed / summed[0, 0]
    assert np.isfinite(normalized).all() and ((normalized >= 0) & (normalized <= 1)).all()
    assert np.all(np.diff(normalized, axis=2) >= -1e-12)
    assert np.all(np.diff(sum_scores, axis=1) >= -1e-12)
    np.testing.assert_allclose(means, np.asarray([[math.fsum(normalized[i, :, n])/200
                                                   for n in range(1001)] for i in range(3)]), atol=1e-12, rtol=0)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    draw(shared, methods, sum_scores, 'query_summed_runtime_score_y05_smoothed',
         '1 − normalized summed query runtime', (0, .5),
         'Same 200 held-out queries · 1,000 observed proposals · Summed runtime score',
         'Score = 1 − summed best runtime / summed initial Bao best runtime; y-axis 0–0.5.')
    draw(shared, methods, means, 'query_mean_minmax_score_smoothed',
         'Mean normalized best-so-far score', (.5, 1.),
         'Same 200 held-out queries · Equal query weights · Pooled min–max normalization',
         'ML convention: worst 20% excluded only for scale bounds; all queries retained. Values below 0.5 are off-axis.')
    write_csv('normalization_scales.csv', scales)
    write_csv('convergence.csv', [dict(method=m, proposals=n, tasks=200,
                  summed_runtime_seconds=float(summed[i, n]), summed_runtime_score=float(sum_scores[i, n]),
                  mean_minmax_score=float(means[i, n])) for i, m in enumerate(methods) for n in range(1001)])
    final = [dict(method=m, summed_runtime_score=float(sum_scores[i, -1]),
                  mean_minmax_score=float(means[i, -1]), initial_mean_minmax_score=float(means[i, 0]))
             for i, m in enumerate(methods)]
    write_csv('final_scores.csv', final)
    np.savez_compressed(OUT / 'per_query_minmax_scores.npz', methods=methods, tasks=tasks, scores=normalized)
    audit = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 methods=methods, tasks=200, proposals_per_query=1000, final_scores=final,
                 summed_score='1 - sum_q incumbent_runtime_q(n) / sum_q initial_Bao_best_runtime_q',
                 minmax_score='mean_q clip((H_q - incumbent_runtime_q(n)) / (H_q - L_q), 0, 1)',
                 bounds='L_q is pooled minimum uncensored runtime; H_q is pooled maximum after dropping the largest floor(20%) runtimes. Fixed across all budgets and methods.',
                 pool='All uncensored recorded observations, including 49 initial Bao observations in each method and cached results. Censored timeout bounds excluded.',
                 smoothing='Same ML/TaskSet PCHIP helper, 16 subdivisions per interval, every integer score preserved, no overshoot.',
                 styling='Shared ML/TaskSet palette, dimensions, labels, solid transfer and dotted no-transfer lines.',
                 degenerates=degenerate, axes=dict(summed=[0, .5], mean_minmax=[.5, 1]),
                 caveats=['SQL caps, seeds and execution environments differ across the original runs.',
                          'Minmax bounds are retrospective and depend on the selected methods. This is a different score from workload runtime reduction.',
                          'All 200 queries retained; no observations or tasks removed from score curves. Worst-20% trimming only sets normalization bounds.',
                          'The actual observations and endpoints are preserved; interpolation cannot remove real large jumps without changing the reported scores.'],
                 validation=dict(all_600_source_hashes_verified=True, minmax_matches_ml_reference=True,
                                 all_curves_finite_monotone=True, equal_query_aggregation_reproduced=True,
                                 observed_scores_preserved=True),
                 input_sha256={str(p): sha(p) for p in [ROOT / 'audit.json', ROOT / 'per_task_curves.npz',
                                                       cache, SHARED, REFERENCE, Path(__file__)]})
    audit['output_sha256'] = {p.name: sha(p) for p in OUT.iterdir()
                              if p.suffix in ['.png', '.pdf', '.svg', '.csv', '.npz']}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    table = '\n'.join(f"| {r['method']} | {r['summed_runtime_score']:.4f} | {r['mean_minmax_score']:.4f} |" for r in final)
    (OUT / 'README.md').write_text(f'''# Two query convergence styles

Both figures use the shared ML/TaskSet PCHIP interpolation and method styles. Same
200 complete held-out queries and 1,000 charged proposals per method, plus the same
49 historical Bao starts. Observed integer scores, endpoints and monotonicity are
preserved. Interpolation creates no new measurements and cannot erase real jumps.

1. Summed runtime score: `1 - sum(best runtime) / sum(initial Bao best runtime)`;
   displayed y-axis 0–0.5, with no numeric clipping of scores.
2. Mean min–max score: pool uncensored runtimes per query across FSBO, TPE and
   LOLBO, drop the slowest floor(20%) only to set bounds, and compute
   `clip((worst_retained - best_so_far) / (worst_retained - pooled_best), 0, 1)`.
   Average all 200 queries equally. Displayed y-axis 0.5–1, matching the ML view;
   any earlier scores below 0.5 remain in the CSV. This follows the ML reference
   function exactly for all nondegenerate query scales.

| Method | Summed-runtime score | Mean min–max score |
|---|---:|---:|
{table}

Normalization is fixed retrospectively using full observed trajectories, and
depends on the included methods. Timeout values are lower bounds, so they do not
enter the scale pool or update incumbent runtimes. Initial and cached uncensored
observations remain in the pool as charged observations. No tasks are excluded.
SQL caps, seeds, dates, storage and hosts differ across the original campaigns.

PNG, PDF, SVG, aggregate CSV, per-query scores, fixed normalization bounds and
hash-identified source pools are retained here. Production results are unchanged.
Reproduce: `python3 {Path(__file__)}`.
''')
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
