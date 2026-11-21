"""Additional convergence figures in the physical units used by BOLT Figure 2."""
import csv
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PAPER = Path('/root/21295_Scaling_Multi_Task_Bayes.pdf')
SHARED = ROOT / 'standardized_convergence_20261101/build_plots.py'
spec = importlib.util.spec_from_file_location('shared_convergence', SHARED)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
METHODS = ['FSBO', 'LOLBO', 'TPE', 'SemanticOPT']
KEYS = dict(zip(METHODS, ['fsbo_minmax', 'lolbo', 'tpe', 'semanticopt']))
DOMAINS = dict(query=dict(horizon=1000, cutoff=200, unit='seconds',
    ylabel='Summed query runtime (s)', title='Query optimization', aggregate='sum'),
    peptide=dict(horizon=5000, cutoff=1000, unit='micromol/L',
    ylabel='Average predicted MIC (µmol/L)', title='Peptide optimization', aggregate='mean'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    data = {}
    for domain, settings in DOMAINS.items():
        source = ROOT / f'{domain}_convergence_four_methods_20261117' / f"standardized_{settings['horizon']}"
        with np.load(source / 'per_task_curves.npz', allow_pickle=False) as d:
            tasks = d['tasks'].tolist()
            curves = {m: d[m].copy() for m in METHODS}
        assert len(tasks) == 200
        for m, a in curves.items():
            assert a.shape == (200, settings['horizon'] + 1)
            assert np.isfinite(a).all() and np.all(a > 0) and np.all(np.diff(a, axis=1) <= 0)
            np.testing.assert_array_equal(a[:, 0], curves['FSBO'][:, 0])
        sem = curves['SemanticOPT']
        np.testing.assert_array_equal(sem[:, settings['cutoff']:],
            np.repeat(sem[:, settings['cutoff']:settings['cutoff'] + 1],
                      settings['horizon'] + 1 - settings['cutoff'], axis=1))
        aggregates = {m: a.sum(axis=0) if settings['aggregate'] == 'sum' else a.mean(axis=0)
                      for m, a in curves.items()}
        data[domain] = dict(source=source, tasks=tasks, curves=curves, values=aggregates,
            audit=json.loads((source / 'audit.json').read_text()))
    return data


def draw_axis(ax, domain, values, *, summed_mic=False):
    settings = DOMAINS[domain]
    for m in METHODS:
        style = shared.STYLE[KEYS[m]]
        checkpoints = values[m][::50]
        x, y = shared.interpolate(checkpoints)
        x = (x - 1) * 50
        np.testing.assert_allclose(y[::16], checkpoints, rtol=0, atol=1e-12)
        if m == 'SemanticOPT':
            np.testing.assert_allclose(y[(settings['cutoff'] // 50) * 16:],
                                      values[m][settings['cutoff']], rtol=0, atol=1e-12)
        label = {'FSBO': 'FSBO (frozen VAE)', 'TPE': 'TPE (global)'}.get(m, m) if domain == 'peptide' else m
        ax.plot(x, y, label=label, color=style['color'], linewidth=2.2,
                linestyle='-' if style['transfer'] else ':')
    ax.plot(settings['cutoff'], values['SemanticOPT'][settings['cutoff']], 'o',
            color=shared.STYLE['semanticopt']['color'], markersize=4)
    ylabel = 'Summed predicted MIC (µmol/L)' if summed_mic else settings['ylabel']
    ax.set(xlim=(0, settings['horizon']), ylim=(0, None),
           xlabel='New acquisition proposals per task', ylabel=ylabel,
           title=settings['title'] + ' (lower is better)')
    ax.set_xticks(np.linspace(0, settings['horizon'], 6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc='upper right')


def save(fig, stem):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    data = load_data()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for domain, item in data.items():
        s = DOMAINS[domain]
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        draw_axis(ax, domain, item['values'])
        fig.text(.5, .015, f"200 held-out tasks · SemanticOPT held flat after {s['cutoff']:,} proposals\n"
                 'Shared colours · PCHIP through 50-proposal checkpoints · Initial observations excluded from x-axis',
                 ha='center', fontsize=8.5, color='#555555')
        fig.tight_layout(rect=(0, .10, 1, 1))
        save(fig, f'{domain}_physical_units')
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9))
    for ax, (domain, item) in zip(axes, data.items()):
        draw_axis(ax, domain, item['values'])
    fig.text(.5, .015, '200 held-out tasks in each domain · SemanticOPT retained after 200 query / 1,000 peptide proposals\n'
             'Raw objective units · Shared colours and PCHIP smoothing · Completed baseline tasks retain their measured final best',
             ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, 'query_peptide_physical_units')
    # Figure 1 / Table 11 also use a sum across peptide tasks; retain that raw-unit alternative.
    summed_mic = {m: data['peptide']['curves'][m].sum(axis=0) for m in METHODS}
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    draw_axis(ax, 'peptide', summed_mic, summed_mic=True)
    fig.text(.5, .015, 'Sum of best feasible predicted MIC across 200 tasks · SemanticOPT held flat after 1,000 proposals\n'
             'Unnormalized aggregate; not the concentration of a combined peptide treatment',
             ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout(rect=(0, .10, 1, 1))
    save(fig, 'peptide_summed_mic')
    rows, endpoints = [], []
    for domain, item in data.items():
        s = DOMAINS[domain]
        for m in METHODS:
            a = item['curves'][m]
            endpoints.append(dict(domain=domain, method=m, aggregate=s['aggregate'],
                unit=s['unit'], initial=float(item['values'][m][0]), final=float(item['values'][m][-1]),
                mean_initial=float(a[:, 0].mean()), mean_final=float(a[:, -1].mean()),
                sum_initial=float(a[:, 0].sum()), sum_final=float(a[:, -1].sum())))
            for n in range(s['horizon'] + 1):
                rows.append(dict(domain=domain, method=m, new_proposals=n,
                    aggregate=s['aggregate'], unit=s['unit'], objective=float(item['values'][m][n]),
                    summed_objective=float(a[:, n].sum()), mean_objective=float(a[:, n].mean()),
                    semanticopt_carried_forward=m == 'SemanticOPT' and n > s['cutoff']))
    for name, records in [('convergence.csv', rows), ('endpoints.csv', endpoints)]:
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    notes = [
        'Metric reference: BOLT paper Figure 2, PDF page 8: summed query runtime and average predicted MIC. Units: seconds and micromol/L (paper pages 3/6). Figure 1 on PDF page 7 instead uses normalized summed objectives; these additional figures intentionally show raw objective units.',
        'Query curve = sum across 200 tasks of best uncensored query-plan runtime in seconds; it is not cumulative optimization wall time.',
        'Peptide curve = arithmetic mean across 200 tasks of best feasible APEX bacteria_0 predicted MIC, in micromol/L (equivalently micromolar, µM). These are predictions, not laboratory MIC measurements.',
        'APEX_predict.py explicitly applies 10 ** (6 - model_prediction) before averaging its eight models; saved objective values already have MIC units. No extra logarithm or back-transform is applied.',
        'Peptide summed-MIC alternative is also exported, following the aggregate used by Figure 1 / Table 11 without normalization.',
        'X preserves the requested new-acquisition-proposal accounting: query 0–1000, peptide 0–5000. This differs from the paper Figure 1 oracle-call x-axis when cache hits/recentering occur. Initialization: query49, peptide100, excluded from x.',
        'SemanticOPT measured endpoints are held flat after query200 / peptide1000. Other completed runs retain final incumbents after exhausting their proposal or fresh-oracle budget. No new observations are fabricated.',
        'Shared palette and PCHIP helper are reused exactly, with checkpoints every 50 proposals. Raw integer-budget values and aggregate sums/means are exported; smoothing is display interpolation.',
        'No per-task min–max normalization or geometric mean is used in these new physical-unit figures. Aggregation differs from the previous normalized-score plots and can change method ordering.',
        'All200 tasks are included with identical starting pools. FSBO is the frozen-VAE local/recenter run; TPE is the completed global variant. Historical campaign device/date/seed/timeout differences remain.',
        'SemanticOPT includes constrained random fallback: query32690/40000; peptide136255/200000. Refer to source audits for full method provenance.',
        'No uncertainty bands added: one optimizer seed per method/task. The paper Figure 1 includes standard-error bands, but this request changes objective units only.',
    ]
    metadata = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        paper=str(PAPER), paper_sha256=digest(PAPER), metric_reference='Figure 2, PDF page 8; units PDF pages 3 and 6',
        notes=notes, endpoints=endpoints,
        sources={d: dict(path=str(item['source'] / 'per_task_curves.npz'),
            sha256=digest(item['source'] / 'per_task_curves.npz'), audit=str(item['source'] / 'audit.json'))
            for d, item in data.items()}, style_source=str(SHARED), style_sha256=digest(SHARED))
    (OUT / 'audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (OUT / 'README.md').write_text('# Convergence in paper objective units\n\n'
        + '\n\n'.join(notes) + '\n\nOutputs: query_physical_units, peptide_physical_units, '
        'query_peptide_physical_units, peptide_summed_mic (PNG/PDF/SVG); convergence.csv; endpoints.csv; audit.json.\n'
        'Reproduce: `python3 build_plots.py`. Existing normalized-score figures are retained.\n')
    (OUT / 'captions.md').write_text(
        'Convergence across 200 held-out tasks per domain. Left: summed best uncensored query-plan runtime (seconds). '
        'Right: average best feasible APEX-predicted MIC (µmol/L). Lower is better in both panels. '
        'The x-axis counts new acquisition proposals per task, excluding the common initial observations '
        '(49 query plans or 100 peptides). Lines use monotone PCHIP interpolation through checkpoints every 50 proposals. '
        'SemanticOPT values are carried forward after its completed budgets (200 query and 1,000 peptide proposals); '
        'completed baseline tasks similarly retain their final measured incumbents when either budget is exhausted. '
        'Peptide recentering evaluations incur additional oracle calls and do not increment the acquisition-proposal axis. '
        'FSBO uses the frozen-VAE local/recenter variant; TPE uses the global independent latent-space variant. '
        'SemanticOPT includes constrained random fallback proposals (81.7% query, 68.1% peptide). '
        'The runs share task-specific initial pools but differ in optimizer seed and execution environment.\n')
    print(json.dumps(endpoints, indent=2), flush=True)


if __name__ == '__main__':
    main()
