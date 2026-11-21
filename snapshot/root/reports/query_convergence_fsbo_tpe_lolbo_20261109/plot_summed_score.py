"""Plot one minus normalized summed incumbent query runtime from audited curves."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'per_task_curves.npz'
OUT = ROOT / 'summed_runtime_score'


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as data:
        methods = data['methods'].tolist()
        tasks = data['tasks'].tolist()
        curves = data['best_uncensored_runtime_seconds']
    assert curves.shape == (3, 200, 1001)
    assert len(set(tasks)) == 200
    assert np.isfinite(curves).all() and (curves > 0).all()
    assert np.all(np.diff(curves, axis=2) <= 0)
    assert np.array_equal(curves[0, :, 0], curves[1, :, 0])
    assert np.array_equal(curves[0, :, 0], curves[2, :, 0])
    summed = curves.sum(axis=1)
    initial_sum = float(summed[0, 0])
    normalized = summed / initial_sum
    scores = 1 - normalized
    assert np.allclose(scores[:, 0], 0)
    assert ((scores >= -1e-12) & (scores <= 1)).all()
    assert np.all(np.diff(scores, axis=1) >= -1e-12)
    assert np.allclose(scores, ((1 - curves / curves[:, :, :1]) * curves[:, :, :1]).sum(axis=1) / initial_sum)
    fields = ['proposals'] + [f'{method.lower()}_{metric}' for method in methods
                              for metric in ['summed_runtime_seconds', 'normalized_summed_runtime', 'score']]
    with (OUT / 'convergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n in range(1001):
            row = {'proposals': n}
            for i, method in enumerate(methods):
                row.update({f'{method.lower()}_summed_runtime_seconds': float(summed[i, n]),
                            f'{method.lower()}_normalized_summed_runtime': float(normalized[i, n]),
                            f'{method.lower()}_score': float(scores[i, n])})
            writer.writerow(row)
    summary = {method: dict(initial_summed_runtime_seconds=initial_sum,
                           final_summed_runtime_seconds=float(summed[i, -1]),
                           final_normalized_summed_runtime=float(normalized[i, -1]),
                           final_score=float(scores[i, -1])) for i, method in enumerate(methods)}
    audit = dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 source_audit=str(ROOT / 'audit.json'), tasks=tasks, methods=methods,
                 formula='score_m(n) = 1 - sum_q best_uncensored_runtime_m,q(n) / sum_q initial_Bao_best_uncensored_runtime_q',
                 denominator_seconds=initial_sum, summary=summary,
                 paper_reference=dict(path='/root/data/cys/baseline/baseline_papers/01_bolt.pdf',
                                      page=7, figure=1,
                                      evidence='Lines represent the sum across test tasks; axis is Summed query runtime (normalized).'),
                 notes=[
                     'Sum best-so-far execution runtimes across the 200 queries at each proposal count, then normalize by the common starting sum.',
                     'The sum is the runtime of the best plans for the workload at this point, not cumulative optimization/search cost.',
                     'Zero is the initial Bao baseline; one is the zero-runtime ideal limit. Endpoints are not forced to one.',
                     'This aggregation weights absolute saved seconds; it is not a geometric mean of per-query ratios.',
                     'All three runs complete 1000 proposals per query, excluding the shared 49 Bao starts. Only uncensored measurements update incumbents.',
                     'FSBO SQL cap 50 s, seed 123; TPE cap 100 s, seed 0; LOLBO adaptive cap, seed 0. Hardware, storage and dates differ.',
                     'BOLT Figure 1 (page 7) confirms aggregation by sum across test tasks. Its caption does not specify an exact normalization denominator; this figure explicitly uses the shared Bao starting total.'
                 ])
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    for i, (method, color) in enumerate(zip(methods, ['#1864ab', '#e67700', '#2b8a3e'])):
        ax.plot(np.arange(1001), scores[i], label=method, color=color, lw=2.7)
        ax.annotate(f'{scores[i, -1]:.3f}', xy=(1000, scores[i, -1]),
                    xytext=(8, 6 if method == 'TPE' else -8 if method == 'LOLBO' else 0),
                    textcoords='offset points', color=color, fontsize=11, va='center')
    ax.set_xlim(0, 1075)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel('New acquisition proposals (49 Bao starts excluded)')
    ax.set_ylabel('1 − normalized summed query runtime\n(higher is better)')
    ax.set_title('Query convergence · 200 held-out queries', loc='left', fontweight='bold', pad=15)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left', frameon=False)
    fig.text(0.5, 0.045, 'Score = 1 − (summed best runtime / summed initial Bao best runtime)',
             ha='center', fontsize=10, color='#444444')
    fig.text(0.5, 0.012, 'SQL caps: FSBO 50 s · TPE 100 s · LOLBO adaptive; seeds and execution environments differ',
             ha='center', fontsize=8.5, color='#666666')
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'query_summed_runtime_score.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    text = ['# Query convergence: one minus normalized summed runtime', '',
            '`score(n) = 1 - sum_q best_runtime_q(n) / sum_q initial_Bao_best_runtime_q`', '',
            '| Method | Initial sum (s) | Final sum (s) | Final score ↑ |',
            '|---|---:|---:|---:|']
    for method, s in summary.items():
        text.append(f"| {method} | {initial_sum:.3f} | {s['final_summed_runtime_seconds']:.3f} | {s['final_score']:.4f} |")
    text += ['', *['- ' + note for note in audit['notes']], '',
             'Reproduce: `python3 ../plot_summed_score.py`.',
             'The original geometric-mean plots remain in the parent directory.']
    (OUT / 'README.md').write_text('\n'.join(text) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
