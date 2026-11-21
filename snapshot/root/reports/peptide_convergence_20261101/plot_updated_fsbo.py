#!/usr/bin/env python3
"""Validate the new frozen-VAE FSBO archives and update peptide plots."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo')
POOL = Path('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json')
GRID = np.arange(0, 5001, 25)
COLORS = {'FSBO': '#1764ab', 'FSBO-local': '#168fa1',
          'LOLBO': '#da7624', 'TPE': '#2b8b57'}


def similar(sequence, reference):
    """Match the repository's normalized Levenshtein similarity rule."""
    limit = len(reference) // 4
    if abs(len(sequence) - len(reference)) > limit:
        return False
    previous = list(range(len(reference) + 1))
    for i, char in enumerate(sequence, 1):
        current = [i] + [0] * len(reference)
        for j, ref_char in enumerate(reference, 1):
            current[j] = min(current[j-1] + 1, previous[j] + 1,
                             previous[j-1] + (char != ref_char))
        if min(current) > limit:
            return False
        previous = current
    return 1 - previous[-1] / len(reference) >= .75


def load_archive(item):
    task, path, recipe_hash, pool = item
    hashes, docs = {}, {}
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
                docs[member.name] = json.loads(b''.join(chunks))
    manifest = docs['manifest.json']
    for name, expected in manifest['members'].items():
        assert hashes[name] == expected, (path, 'checksum', name)
    h = docs['history.json']
    assert h['status'] == 'complete' and h['identity'] == manifest['identity'], path
    assert h['identity']['task'] == task and h['identity']['recipe_sha256'] == recipe_hash, path
    assert h['reference'] == pool['reference'], path
    assert (h['initialization'], h['proposal_budget'], h['fresh_budget']) == (100, 5000, 5000), path
    rows = h['observations']
    initial = [r for r in rows if r['phase'] == 'initialization']
    post = [r for r in rows if r['phase'] != 'initialization']
    assert len(initial) == 100, path
    assert [r['sequence'] for r in initial] == pool['sequences'], (path, 'initial sequences')
    assert np.allclose([r['score'] for r in initial], pool['scores'], rtol=0, atol=1e-6), path
    assert all(np.isfinite(r['mic']) and np.isclose(r['score'], -r['mic']) for r in rows), path
    for row in rows:
        is_similar = similar(row['sequence'], h['reference'])
        assert row['feasible'] == is_similar, (path, row['sequence'], row['feasible'], is_similar)
    fresh = sum(not r['cached'] for r in post)
    proposals = sum(r['phase'] == 'acquisition' for r in post)
    recenter = sum(not r['cached'] and r['phase'] == 'recenter' for r in post)
    assert fresh == h['fresh_evaluations'] <= 5000, path
    assert proposals == h['proposals'] <= 5000, path
    assert recenter == h['recenter_fresh'], path
    assert len(post) - fresh == h['cache_hits'], path
    assert proposals == 5000 or fresh == 5000, (path, 'budget not exhausted')
    best = min(r['mic'] for r in initial if r['feasible'])
    xs, ys, used = [0], [best], 0
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
    curve = np.asarray(ys)[np.searchsorted(xs, GRID, side='right') - 1]
    assert np.all(np.diff(curve) <= 1e-10) and curve[0] == min(r['mic'] for r in initial), path
    return task, curve, dict(task=task, initial_best_mic=curve[0], final_best_mic=best,
        fresh_evaluations=fresh, acquisition_proposals=proposals, recenter_fresh=recenter,
        cache_hits=h['cache_hits'], elapsed_seconds=h['seconds'], archive=str(path),
        history_sha256=hashes['history.json'])


def main():
    previous = np.load(OUT/'per_task_curves.npz')
    tasks = previous['tasks'].tolist()
    pool = json.loads(POOL.read_text())
    assert set(tasks) == set(pool) and len(tasks) == 200
    recipe_bytes = (ROOT/'recipe.json').read_bytes()
    recipe = json.loads(recipe_bytes)
    assert sorted(recipe['test']) == tasks and recipe['seed'] == 1
    assert recipe['similarity'] == 0.75 and recipe['batch_size'] == 50
    assert recipe['vae_policy'] == 'frozen_author' and recipe['e2e_every'] == 0
    paths = {p.name[:-7]: p for p in (ROOT/'runs').glob('*.tar.gz')}
    assert set(paths) == set(tasks), (len(paths), len(tasks))
    jobs = [(task, paths[task], hashlib.sha256(recipe_bytes).hexdigest(), pool[task]) for task in tasks]
    records, curves = [], {}
    if '--reuse-validated' in sys.argv:
        cached = np.load(OUT/'per_task_curves_updated_fsbo.npz')
        assert cached['tasks'].tolist() == tasks
        curves = dict(zip(tasks, cached['FSBO-local']))
        with (OUT/'fsbo_local_task_summary.csv').open() as stream:
            records = list(csv.DictReader(stream))
        for record in records:
            for key in ('initial_best_mic', 'final_best_mic', 'elapsed_seconds'):
                record[key] = float(record[key])
            for key in ('fresh_evaluations', 'acquisition_proposals', 'recenter_fresh', 'cache_hits'):
                record[key] = int(record[key])
        assert len(records) == 200 and {r['task'] for r in records} == set(tasks)
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            for i, (task, curve, record) in enumerate(executor.map(load_archive, jobs), 1):
                curves[task] = curve
                records.append(record)
                if i % 25 == 0:
                    print(f'Validated {i}/200 new FSBO archives', flush=True)

    methods = {'FSBO': previous['FSBO'], 'FSBO-local': np.stack([curves[t] for t in tasks]),
               'LOLBO': previous['LOLBO'], 'TPE': previous['TPE']}
    starts = methods['FSBO'][:, 0]
    assert np.array_equal(starts, methods['FSBO-local'][:, 0])
    rng = np.random.default_rng(20261105)
    weights = rng.multinomial(200, np.full(200, 1/200), size=2000)/200
    stats, rows = {}, []
    for method, matrix in methods.items():
        improvement = 100*(1-matrix/starts[:, None])
        stats[method] = {}
        for metric, values in [('mic', matrix), ('improvement_percent', improvement)]:
            stats[method][metric] = (values.mean(axis=0), *np.quantile(weights @ values, [.025, .975], axis=0))
        for j, x in enumerate(GRID):
            rows.append(dict(method=method, fresh_evaluations=int(x), tasks=200,
                **{f'{metric}_{field}': float(stats[method][metric][k][j])
                   for metric in stats[method] for k, field in enumerate(['mean', 'ci95_low', 'ci95_high'])}))

    def csv_write(path, data):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    csv_write(OUT/'convergence_updated_fsbo.csv', rows)
    csv_write(OUT/'fsbo_local_task_summary.csv', records)
    np.savez_compressed(OUT/'per_task_curves_updated_fsbo.npz', tasks=np.array(tasks),
                        fresh_evaluations=GRID, **methods)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric in zip(axes, ['mic', 'improvement_percent']):
        for method, values in stats.items():
            mean, lo, hi = values[metric]
            ax.fill_between(GRID, lo, hi, color=COLORS[method], alpha=.13, linewidth=0)
            ax.plot(GRID, mean, color=COLORS[method], lw=2.3,
                    ls='--' if method == 'FSBO-local' else '-',
                    label={'FSBO': 'FSBO (previous)', 'FSBO-local': 'FSBO (new, frozen VAE)'}.get(method, method))
        ax.set_xlim(0, 5000)
        ax.set_xticks(np.arange(0, 5001, 1000))
        ax.grid(alpha=.18)
        ax.set_xlabel('Fresh oracle evaluations after initialization')
    axes[0].set_ylabel('Mean best feasible MIC')
    axes[0].set_title('Best feasible objective · lower is better')
    axes[1].set_ylabel('Mean improvement from initial best (%)')
    axes[1].set_title('Improvement per task · higher is better')
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, loc='lower right')
    fig.suptitle('Peptide optimization · 200 held-out tasks', fontsize=17, y=.98)
    fig.text(.5, .91, 'New FSBO uses frozen-VAE local/recenter wrapper · similarity ≥ 0.75',
             ha='center', fontsize=11, color='#444444')
    fig.text(.5, .045, 'Shading: pointwise 95% bootstrap CI across tasks; one optimizer seed per task.\n'
             'New FSBO archives and feasibility were validated; prior baseline curves are carried forward.',
             ha='center', fontsize=9, color='#555555', linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.98, bottom=.21, top=.80, wspace=.29)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT/f'peptide_convergence_updated_fsbo.{ext}', dpi=200, facecolor='white')
    summary = {}
    for method, matrix in methods.items():
        if method == 'FSBO-local':
            final = matrix[:, -1]
            summary[method] = dict(tasks=200, initial_mean_best_mic=float(starts.mean()),
                final_mean_best_mic=float(final.mean()),
                mean_relative_improvement_percent=float((100*(1-final/starts)).mean()),
                final_mean_mic_ci95=[float(stats[method]['mic'][i][-1]) for i in (1, 2)],
                mean_fresh_evaluations=float(np.mean([r['fresh_evaluations'] for r in records])),
                tasks_with_any_improvement=int(sum(r['final_best_mic'] < r['initial_best_mic'] for r in records)),
                mean_acquisition_fresh_rate=float(np.mean([
                    (r['fresh_evaluations']-r['recenter_fresh'])/r['acquisition_proposals']
                    for r in records])))
        else:
            matrix0 = previous[method]
            summary[method] = dict(tasks=200, final_mean_best_mic=float(matrix0[:, -1].mean()),
                mean_relative_improvement_percent=float((100*(1-matrix0[:, -1]/starts)).mean()))
    result = dict(new_fsbo_archives_validated=200,
        new_run_recipe=str(ROOT/'recipe.json'),
        new_run_checks=['all archive member SHA256 checksums', 'complete status and frozen recipe identity',
            'same task list and common initialization', 'each recorded feasible label recomputed from edit-distance similarity >= 0.75',
            'fresh/cache/recenter/proposal accounting', '5000 fresh evaluations or 5000 acquisition proposals per task',
            'terminal incumbent matches history and curves are nonincreasing'],
        new_fsbo=dict(settings={k: recipe[k] for k in ('batch_size', 'initialization', 'proposals',
            'fresh_budget', 'similarity', 'seed', 'vae_policy', 'e2e_every', 'recenter_policy')}, summary=summary['FSBO-local']),
        methods=summary, previous_baseline_source=str(OUT/'per_task_curves.npz'),
        uncertainty='Pointwise percentile bootstrap 95% CI; paired task resamples (2000); one optimizer seed per task.')
    (OUT/'updated_fsbo_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
