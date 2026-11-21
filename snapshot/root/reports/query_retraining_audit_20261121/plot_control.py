from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
d = json.loads((out / 'controlled_results.json').read_text())
names = ['original80', 'retrained80', 'original788']
labels = ['Original VAE\n80 source tasks', 'Retrained VAE\n80 source tasks', 'Original VAE\n788 source tasks']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for delta, key, label, color in [(-.19, 'without_target_gradient_updates', 'Before target fitting', '#777777'),
                                (.19, 'after_native_target_adaptation', 'After native target fitting', '#CC79A7')]:
    summaries = [d[key]['models'][n]['summary'] for n in names]
    for ax, metric, fmt in [(axes[0], 'geomean_selection_regret_ratio', '.2f'),
                            (axes[1], 'mean_spearman', '.3f')]:
        vals = [s[metric] for s in summaries]
        bars = ax.bar(np.arange(3) + delta, vals, width=.38, label=label, color=color)
        ax.bar_label(bars, fmt='%'+fmt, padding=3, fontsize=8)
for ax in axes:
    ax.set_xticks(np.arange(3), labels, fontsize=9)
    ax.spines[['right', 'top']].set_visible(False)
axes[0].set_ylabel('Selected runtime / best saved candidate runtime')
axes[0].set_title('Geometric mean regret; lower is better')
axes[0].set_ylim(0, 2.8)
axes[0].axhline(1, color='#999999', ls='--', lw=.8)
axes[0].legend(frameon=False, fontsize=8)
axes[1].set_ylabel('Mean Spearman correlation')
axes[1].set_title('Runtime ranking; higher is better')
axes[1].set_ylim(0, .28)
fig.text(.5, .015, 'Same 15 held-out queries and saved BOLT plans; up to 10 support observations. Predictive diagnostic, not BO convergence.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, .055, 1, 1))
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(out / ('controlled_transfer.' + ext), dpi=180)
