"""Read-only comparison of the nine completed rollback experiments."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119')
OUT = Path(__file__).resolve().parent
NEW = 'Online VAE + FSBO + rollback (20 rounds, max2)'
METHODS = ['Original frozen FSBO', 'Online VAE + FSBO', 'SemanticOPT with fallback', NEW]
LABELS = ['Frozen FSBO', 'Previous online VAE + FSBO', 'SemanticOPT with fallback', 'Online VAE + FSBO: rollback, patience20, max2']
COLORS = ['#1f77b4', '#d62728', '#9467bd', '#009E73']
cfg = json.loads((ROOT/'config.json').read_text())
results = {}
source_hashes = {}
for task in cfg['tasks']:
    p = ROOT/'evaluation/adapted_online'/task/'result.json'
    if p.exists():
        payload = p.read_bytes()
        result = json.loads(payload)
        if result['status'] == 'complete':
            assert result['proposals'] == 4000 and result['rounds'] == 80
            results[task] = result
            source_hashes[str(p)] = hashlib.sha256(payload).hexdigest()
TASKS = sorted(results)
assert len(TASKS) == 9, f'Expected requested nine completed tasks, found {TASKS}'
assert 'pep_967' not in TASKS

curves = defaultdict(list)
previous = ROOT/'comparison_inputs/capped_convergence.csv'
source_hashes[str(previous)] = hashlib.sha256(previous.read_bytes()).hexdigest()
for row in csv.DictReader(previous.open()):
    if row['task'] in TASKS:
        curves[row['method'], row['task']].append({k: float(row[k]) for k in ['proposals','fresh','mic','time']})

records = []
for task, result in results.items():
    best = result['initialization_best_mic']
    fresh = proposals = 0
    curve = [dict(fresh=0., proposals=0., mic=best, time=0.)]
    groups = []
    group = []
    key = None
    for row in result['observations']:
        if row['phase'] == 'initialization':
            continue
        current = (row['phase'], row['elapsed'])
        if key is not None and key != current:
            groups.append(group)
            group = []
        group.append(row)
        key = current
    if group:
        groups.append(group)
    for group in groups:
        fresh += sum(not r['cached'] for r in group)
        proposals += sum(r['phase'] == 'acquisition' for r in group)
        best = min([best] + [r['mic'] for r in group if r['feasible']])
        curve.append(dict(fresh=float(fresh), proposals=float(proposals), mic=best,
                          time=group[-1].get('wall_elapsed', group[-1]['elapsed'])))
    assert fresh == result['fresh_evaluations'] and proposals == result['proposals']
    assert np.isclose(best, result['best_feasible_mic'])
    curves[NEW, task] = curve
    events = result['adaptation_events']
    assert len(events) <= 2 and all(e['trigger_nonimproving_rounds'] >= 20 for e in events)
    for event in events:
        if event['accepted']:
            after = event['target_loss_after']['mean_target_loss']
            assert after <= event['target_loss_before']['mean_target_loss'] + 1e-12
            assert after <= event['target_loss_initial_baseline']['mean_target_loss'] + 1e-12
        else:
            assert event['vae_change_l2'] == 0
    records.append(dict(task=task, initial_mic=result['initialization_best_mic'],
                        **{m: curves[m, task][-1]['mic'] for m in METHODS},
                        new_fresh=fresh, new_proposals=proposals,
                        new_wall_hours=result['wall_seconds']/3600,
                        accepted_updates=sum(e['accepted'] for e in events),
                        rolled_back_updates=sum(not e['accepted'] for e in events)))

for method in METHODS:
    for task in TASKS:
        c = curves[method, task]
        assert c and np.isclose(c[0]['mic'], results[task]['initialization_best_mic'])
        for key in ['fresh', 'proposals']:
            assert all(0 <= r[key] <= 4000 for r in c)
            assert np.all(np.diff([r[key] for r in c]) >= 0)
        assert np.all(np.diff([r['mic'] for r in c]) <= 1e-7)

finals = np.array([[curves[m, t][-1]['mic'] for m in METHODS] for t in TASKS])
ranks = np.array([rankdata(row, method='average') for row in finals])
method_summary = []
for i, method in enumerate(METHODS):
    method_summary.append(dict(method=method, mean_mic=float(finals[:, i].mean()),
                               median_mic=float(np.median(finals[:, i])),
                               average_rank=float(ranks[:, i].mean()),
                               first_places=int(np.sum(ranks[:, i] == 1))))
paired = []
for i, method in enumerate(METHODS[:-1]):
    difference = finals[:, i] - finals[:, -1]
    paired.append(dict(reference=method, new_wins=int(np.sum(difference > 1e-7)),
                       new_losses=int(np.sum(difference < -1e-7)),
                       ties=int(np.sum(np.abs(difference) <= 1e-7)),
                       mean_mic_reduction_pct=float(100*difference.mean()/finals[:, i].mean())))

def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

write_csv('task_results.csv', records)
write_csv('method_summary.csv', method_summary)
write_csv('paired_comparison.csv', paired)
write_csv('convergence.csv', [dict(method=m, task=t, **r) for m in METHODS for t in TASKS for r in curves[m, t]])
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT/f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

def values(method, task, grid, axis):
    c = curves[method, task]
    positions = np.asarray([r[axis] for r in c])
    indices = np.maximum(0, np.searchsorted(positions, grid, side='right') - 1)
    return np.asarray([r['mic'] for r in c])[indices]

aggregate = []
for axis in ['fresh', 'proposals']:
    grid = np.arange(4001)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for m, label, color in zip(METHODS, LABELS, COLORS):
        matrix = np.asarray([values(m, t, grid, axis) for t in TASKS])
        mean = matrix.mean(0)
        starts = np.maximum(0, grid-199)
        cs = np.r_[0., np.cumsum(mean)]
        smooth = (cs[grid+1]-cs[starts])/(grid-starts+1)
        axes[0].step(grid, mean, where='post', color=color, label=label, lw=1.8)
        axes[1].plot(grid, smooth, color=color, label=label, lw=2)
        for x in range(0, 4001, 25):
            aggregate.append(dict(axis=axis, budget=x, method=m, mean_mic=float(mean[x]),
                                  smoothed_mean_mic=float(smooth[x]),
                                  tasks_with_measured_support=sum(curves[m,t][-1][axis]>=x for t in TASKS)))
    for ax, title in zip(axes, ['Raw measured incumbent mean', 'Same causal trailing-200 smoothing']):
        ax.set(title=title, xlabel='Fresh evaluations' if axis=='fresh' else 'Acquisition proposals',
               ylabel='Mean best predicted MIC (µM) ↓', xlim=(0,4000))
        ax.grid(alpha=.2)
    fig.suptitle('Nine completed peptides · same cohort for every method', y=1.01)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8.5,
               bbox_to_anchor=(.5,-.08))
    fig.text(.5,-.12,'Shared initialization100 excluded. Earlier endpoints carry forward after stopping; no extra measurements implied.',ha='center',fontsize=8)
    fig.tight_layout()
    save(fig, f'mean_convergence_{axis}')
write_csv('aggregate_curves.csv', aggregate)

fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for task, ax in zip(TASKS, axes.flat):
    for m, label, color in zip(METHODS, LABELS, COLORS):
        c = curves[m,task]
        ax.step([r['fresh'] for r in c], [r['mic'] for r in c], where='post', color=color, lw=1.6, label=label)
        ax.scatter(c[-1]['fresh'], c[-1]['mic'], color=color, s=15, zorder=4)
    ax.set(title=task, xlim=(0,4000), xlabel='Fresh evaluations', ylabel='Best predicted MIC (µM) ↓')
    ax.grid(alpha=.2)
fig.suptitle('Nine completed peptides · exact per-task convergence')
fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=9)
fig.tight_layout(rect=(0,.06,1,.97))
save(fig,'per_task_convergence')

fig, ax = plt.subplots(figsize=(12, 5))
offsets = np.linspace(-.27,.27,4)
for i, (m,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
    ax.scatter(np.arange(9)+offsets[i], finals[:,i], color=color, s=48, label=label)
ax.set(xticks=np.arange(9), xticklabels=TASKS, yscale='log',
       ylabel='Final best predicted MIC (µM, log scale) ↓',
       title='Exact final MIC · nine completed peptides · 4,000 cap')
ax.grid(axis='y', alpha=.2, which='both')
ax.legend(fontsize=8.5, ncol=2)
fig.tight_layout()
save(fig,'final_mic')

summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='partial_cohort_9_of_10',
               tasks=TASKS, excluded=['pep_967'], methods=method_summary, paired=paired,
               accepted_updates=sum(r['accepted_updates'] for r in records),
               rolled_back_updates=sum(r['rolled_back_updates'] for r in records),
               new_wall_hours_mean=float(np.mean([r['new_wall_hours'] for r in records])),
               source_sha256=source_hashes,
               budget_note='Each history capped at 4000 acquisition proposals OR 4000 fresh evaluations, whichever first. Frozen retrospective cut uses last full batch within both caps. Endpoints may be earlier than 4000; carried incumbents are not new measurements.',
               limitations='Nine completed tasks selected by completion, not the final ten-task cohort. One seed per task/method. Predicted MIC, not wet-lab MIC. Hardware/contention differ; runtime is not a controlled method comparison. SemanticOPT includes its original random fallback policy.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines = ['# Peptide rollback comparison: nine completed tasks', '',
         'The same nine completed tasks are included for every method. `pep_967` is still running and excluded from all means, ranks and plots.', '',
         '| Method | Mean MIC (µM) ↓ | Median MIC (µM) ↓ | Average rank ↓ |', '|---|---:|---:|---:|']
for s in method_summary:
    lines.append(f"| {s['method']} | {s['mean_mic']:.3f} | {s['median_mic']:.3f} | {s['average_rank']:.2f} |")
lines += ['', 'Rollback improves on the previous online run in 7/9 tasks and reduces its mean MIC by 16.8%. It beats frozen FSBO in 8/9 tasks, but mean MIC is 0.7% worse because pep_999 remains poor. SemanticOPT has the lowest mean and median MIC; its average rank ties the new rollback run at 1.89. The new run beats SemanticOPT in 4/9 tasks.', '',
          '| Peptide | Frozen | Previous online | SemanticOPT | New rollback |', '|---|---:|---:|---:|---:|']
for r in records:
    lines.append('| '+r['task']+' | '+' | '.join(f'{r[m]:.3f}' for m in METHODS)+' |')
lines += ['', 'The new run accepted 11 of 18 VAE retraining attempts and rolled back 7. Accepted updates pass both target-loss comparisons; rejected updates have exactly zero VAE weight delta. This protects the measured reconstruction-loss criterion but does not guarantee improved optimization or MIC. pep_999 accepted both attempts and still ends at 101.007 µM, versus frozen 43.308 and SemanticOPT 29.575 µM.', '',
          f"Mean new-run wall time was {summary['new_wall_hours_mean']:.2f} h per completed peptide. Different hardware/shared workloads prevent a controlled speed comparison with prior methods.", '',
          summary['budget_note'], '', 'All aggregate displays use the same trailing-200-position causal smoothing; raw mean panels, final scores and per-task plots are unsmoothed. Initialization100 is excluded from the x-axis. '+summary['limitations'], '',
          '[Mean convergence](mean_convergence_fresh.png) · [Per-task convergence](per_task_convergence.png) · [Final MIC](final_mic.png) · [Proposal-axis view](mean_convergence_proposals.png)', '']
(OUT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},indent=2))
