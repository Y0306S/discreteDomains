"""Plot the previously validated old and new FSBO task curves."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
data = np.load(OUT / 'per_task_curves_updated_fsbo.npz')
x = data['fresh_evaluations']
assert len(data['tasks']) == 200
weights = np.random.default_rng(20261105).multinomial(200, np.full(200, 1 / 200), size=2000) / 200
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(9, 6))
rows = []
for key, label, color, style in [('FSBO', 'Old FSBO', '#1764ab', '-'),
                                  ('FSBO-local', 'New FSBO · local/recenter', '#d87524', '--')]:
    values = data[key]
    assert values.shape == (200, len(x)) and np.all(np.diff(values, axis=1) <= 1e-10)
    mean = values.mean(axis=0)
    low, high = np.quantile(weights @ values, [.025, .975], axis=0)
    ax.fill_between(x, low, high, color=color, alpha=.13, linewidth=0)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=2.6, label=label)
    ax.annotate(f'{mean[-1]:.2f}', (x[-1], mean[-1]), xytext=(-10, 9),
                textcoords='offset points', ha='right', color=color, weight='bold')
    rows.extend(dict(method=label, fresh_evaluations=int(n), tasks=200,
                     mean_best_feasible_mic=float(m), ci95_low=float(lo), ci95_high=float(hi))
                for n, m, lo, hi in zip(x, mean, low, high))
ax.set(xlim=(0, 5000), xlabel='Fresh oracle evaluations after initialization',
       ylabel='Mean best feasible MIC (lower is better)')
ax.set_xticks(np.arange(0, 5001, 1000))
ax.grid(alpha=.18)
ax.legend(frameon=False, loc='upper right')
fig.suptitle('Peptide optimization · old vs new FSBO', fontsize=17, y=.98)
fig.text(.5, .915, '200 matched held-out tasks · both batch 50 · both frozen VAE',
         ha='center', fontsize=11, color='#444444')
fig.text(.5, .04, 'Identical 100-peptide initialization · similarity ≥ 0.75 · shading: 95% task-bootstrap CI\n'
         'Fresh calls include recentering; completed proposal-limited runs retain their final measured best.',
         ha='center', fontsize=9, color='#555555', linespacing=1.5)
fig.subplots_adjust(left=.12, right=.97, top=.85, bottom=.20)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'peptide_fsbo_old_vs_new.{ext}', dpi=200, facecolor='white')
with (OUT / 'peptide_fsbo_old_vs_new.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(OUT / 'peptide_fsbo_old_vs_new.png')
