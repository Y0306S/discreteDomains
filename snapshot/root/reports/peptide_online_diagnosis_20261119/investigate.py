"""Read-only investigation of measured peptide trajectories; no model or oracle calls."""
import csv
import hashlib
import json
import math
import statistics
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from Levenshtein import distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ON = Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
FR = Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo/runs')
CAP = Path('/root/reports/peptide_cap4000_graphs_20261119')
METHODS = ['Original frozen FSBO', 'Online VAE + FSBO', 'SemanticOPT with fallback']
COLORS = dict(zip(METHODS, ['#1f77b4', '#d62728', '#9467bd']))
TASKS = json.loads((ON / 'config.json').read_text())['tasks']
curves = defaultdict(list)
for r in csv.DictReader((CAP / 'capped_convergence.csv').open()):
    curves[r['task'], r['method']].append({k: float(r[k]) for k in ['fresh', 'mic', 'time', 'proposals']})

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def at(c, x):
    return min(r['mic'] for r in c if r['fresh'] <= x)

def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

intervals, plateaus, feasibility, updates, improvements, stage_rows = [], [], [], [], [], []
task_stats, focus, provenance = {}, {}, []
for task in TASKS:
    raw = (ON / 'evaluation/adapted_online' / task / 'result.json').read_bytes()
    online = json.loads(raw)
    assert online['status'] == 'complete'
    provenance.append(dict(task=task, online_result_sha256=hashlib.sha256(raw).hexdigest()))
    with tarfile.open(FR / f'{task}.tar.gz') as archive:
        frozen = json.load(archive.extractfile('history.json'))
    data = {METHODS[0]: frozen, METHODS[1]: online}
    task_stats[task] = dict(reference=online['reference'], reference_length=len(online['reference']),
                            allowed_edit_distance=math.floor(.25 * len(online['reference'])),
                            status=online['status'], rounds=online['rounds'])
    for method in METHODS:
        c = curves[task, method]
        end = c[-1]
        last = c[0]
        best = c[0]['mic']
        for point in c[1:]:
            if point['mic'] < best - 1e-8:
                last = point
                best = point['mic']
        plateaus.append(dict(task=task, method=method, initial_mic=c[0]['mic'], final_mic=end['mic'],
                             last_improvement_fresh=int(last['fresh']), actual_final_fresh=int(end['fresh']),
                             fresh_since_last_improvement=int(end['fresh']-last['fresh']),
                             final_500_actual_fresh_flat=end['fresh']-last['fresh'] >= 500,
                             improvement_percent=100*(c[0]['mic']-end['mic'])/c[0]['mic']))
        for start in range(0, 4000, 500):
            if start >= end['fresh']:
                continue
            stop = min(start + 500, end['fresh'])
            before, after = at(c, start), at(c, stop)
            intervals.append(dict(task=task, method=method, interval_start=start,
                                  interval_end=int(stop), nominal_interval_end=start+500,
                                  partial_interval=stop < start+500, mic_before=before, mic_after=after,
                                  mic_reduction=before-after, mic_reduction_percent=100*(before-after)/before))
        if method not in data:
            continue
        d = data[method]
        rows = [r for r in d['observations'] if method == METHODS[1] or r['elapsed'] <= end['time'] + 1e-7]
        acquisitions = [r for r in rows if r['phase'] == 'acquisition']
        fresh = [r for r in acquisitions if not r['cached']]
        feasible = [r for r in fresh if r['feasible']]
        fresh_all = [r for r in rows if r['phase'] != 'initialization' and not r['cached']]
        assert len(fresh_all) == end['fresh']
        assert len(acquisitions) == end['proposals']
        for r in fresh:
            assert r['feasible'] == (distance(r['sequence'], d['reference']) <= .25*len(d['reference']))
        best_mic = c[0]['mic']
        new_count, proposal_count, last_round = 0, 0, 0
        event_groups = []
        for r in rows:
            if r['phase'] == 'initialization':
                continue
            key = r['phase'], r['elapsed']
            if not event_groups or event_groups[-1][0] != key:
                event_groups.append((key, []))
            event_groups[-1][1].append(r)
        for _, batch in event_groups:
            new_count += sum(not r['cached'] for r in batch)
            proposal_count += sum(r['phase'] == 'acquisition' for r in batch)
            new_best = min([best_mic] + [r['mic'] for r in batch if r['feasible']])
            if new_best < best_mic - 1e-8:
                last_round = batch[-1]['round']
                improvements.append(dict(task=task, method=method, round=last_round,
                                         fresh=new_count, proposals=proposal_count, previous_mic=best_mic,
                                         mic=new_best, sequence=min((r for r in batch if r['feasible']), key=lambda r:r['mic'])['sequence']))
            best_mic = new_best
        for item in plateaus:
            if item['task'] == task and item['method'] == method:
                item.update(last_improvement_round=last_round,
                            actual_final_round=max(r['round'] for r in rows),
                            rounds_since_last_improvement=max(r['round'] for r in rows)-last_round)
        for lo in range(0, 80, 10):
            group = [r for r in fresh if lo < r['round'] <= lo+10]
            vals = [r for r in group if r['feasible']]
            feasibility.append(dict(task=task, method=method, round_start=lo+1, round_end=lo+10,
                                     fresh=len(group), fresh_feasible=len(vals),
                                     feasible_percent=100*len(vals)/len(group) if group else None,
                                     median_edit_distance=statistics.median(distance(r['sequence'], d['reference']) for r in group) if group else None))
        if method == METHODS[1]:
            recent = [r for r in rows if r['phase'] == 'recenter']
            task_stats[task].update(fresh_acquisitions=len(fresh), fresh_feasible_acquisitions=len(feasible),
                                    feasible_acquisition_percent=100*len(feasible)/len(fresh),
                                    adaptation_events=len(d['adaptation_events']),
                                    accepted_updates=sum(e['accepted'] for e in d['adaptation_events']),
                                    recenter_proposals=len(recent), recenter_fresh=sum(not r['cached'] for r in recent),
                                    recenter_fresh_feasible=sum(r['feasible'] and not r['cached'] for r in recent),
                                    source_refresh_fraction_of_wall=d['timing']['source_meta_retraining']/d['wall_seconds'])
            for e in d['adaptation_events']:
                updates.append(dict(task=task, round=e['round'], fresh=e['fresh_evaluations'],
                                    accepted=e['accepted'], selected_sequences=e['selected_observations'],
                                    source_feasible_before=e['source_decoder_before']['feasible'],
                                    source_feasible_after=e['source_decoder_after']['feasible'],
                                    source_probe_n=e['source_decoder_after']['sequences'],
                                    fsbo_status=e.get('source_meta_retraining',{}).get('status'),
                                    fsbo_steps=e.get('source_meta_retraining',{}).get('step'),
                                    fsbo_reloaded=e.get('source_meta_retraining',{}).get('reload_verified')))
        if task == 'pep_999':
            focus[method] = dict(rows=rows, distances=[distance(r['sequence'], d['reference']) for r in fresh],
                                  fresh=len(fresh), fresh_feasible=len(feasible),
                                  best_unconstrained_mic=min(r['mic'] for r in fresh),
                                  better_but_infeasible=sum(not r['feasible'] and r['mic'] < end['mic'] for r in fresh))
            if method == METHODS[1]:
                for stage, seconds in d['timing'].items():
                    stage_rows.append(dict(stage=stage, seconds=seconds, hours=seconds/3600))
                focus['wall_seconds'] = d['wall_seconds']
    print(task, 'read and checked', flush=True)

write_csv('mic_changes_per_500_evaluations.csv', intervals)
write_csv('plateaus.csv', plateaus)
write_csv('feasibility_per_10_rounds.csv', feasibility)
write_csv('online_update_audit.csv', updates)
write_csv('fsbo_improvements.csv', improvements)
write_csv('pep999_timing.csv', stage_rows)
write_csv('source_hashes.csv', provenance)

plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 5, figsize=(19, 7.5))
for task, ax in zip(TASKS, axes.flat):
    for method in METHODS:
        c = curves[task, method]
        end = c[-1]['fresh']
        samples = [x for x in range(0, 4001, 500) if x <= end]
        ax.step([r['fresh'] for r in c], [r['mic'] for r in c], where='post',
                color=COLORS[method], alpha=.45, lw=1.1)
        ax.plot(samples, [at(c, x) for x in samples], 'o', color=COLORS[method], ms=3.7, label=method)
        ax.plot(end, c[-1]['mic'], '^', color=COLORS[method], ms=5)
    ax.set(title=task, xlabel='Fresh evaluations', ylabel='Best predicted MIC (µM) ↓', xlim=(0,4000))
    ax.set_xticks(range(0,4001,500))
    ax.tick_params(axis='x', labelrotation=60, labelsize=8)
    ax.grid(alpha=.2)
fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper center', ncol=3)
fig.text(.5, .005, 'Dots: measured incumbent at each 500-evaluation budget; triangles: actual stops. Thin steps: full batch-causal history. No smoothing or extrapolation.', ha='center', fontsize=9)
fig.tight_layout(rect=(0,.035,1,.94))
save(fig, 'all_tasks_every_500_evaluations')

# Numeric interval reductions make plateaus explicit without smoothing.
fig, ax = plt.subplots(figsize=(12.5,6))
matrix = np.zeros((len(TASKS),8))
partial = set()
for r in intervals:
    if r['method'] == METHODS[1]:
        i,j=TASKS.index(r['task']), r['interval_start']//500
        matrix[i,j] = r['mic_reduction_percent']
        if r['partial_interval']: partial.add((i,j))
im=ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=max(1,matrix.max()), aspect='auto')
for i in range(len(TASKS)):
    for j in range(8):
        label=f'{matrix[i,j]:.1f}%' + ('*' if (i,j) in partial else '')
        ax.text(j,i,label,ha='center',va='center',color='white' if matrix[i,j]>.55*matrix.max() else 'black',fontsize=9)
ax.set(xticks=range(8),xticklabels=[f'{i:,}–{i+500:,}' for i in range(0,4000,500)],
       yticks=range(10),yticklabels=TASKS,xlabel='Fresh-evaluation interval',title='Online VAE + FSBO: MIC reduction within each interval (0% = no improvement)')
ax.tick_params(axis='x',labelrotation=25)
fig.colorbar(im,ax=ax,label='MIC reduction relative to start of interval (%)')
fig.text(.5,.005,'* Final interval ends at that task’s actual stop (3,693–4,000 fresh); percentages use exact, unsmoothed MIC. Flat tails do not prove convergence.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));save(fig,'online_improvement_per_500')

fig, axes = plt.subplots(1,3,figsize=(15,5.7),sharey=True)
for method,ax in zip(METHODS,axes):
    pp=[next(r for r in plateaus if r['task']==task and r['method']==method) for task in TASKS]
    for i,r in enumerate(pp):
        ax.barh(i,r['last_improvement_fresh'],color=COLORS[method],height=.65)
        ax.barh(i,r['fresh_since_last_improvement'],left=r['last_improvement_fresh'],color='#dddddd',height=.65)
        ax.scatter(r['last_improvement_fresh'],i,color=COLORS[method],s=20,zorder=3)
    ax.set(title=method,xlabel='Fresh evaluations',xlim=(0,4000),yticks=range(10),yticklabels=TASKS)
    ax.grid(axis='x',alpha=.2)
axes[0].invert_yaxis()
fig.text(.5,.005,'Colored segment ends at the last measured MIC improvement; gray tail = subsequent evaluations with no improvement. Each bar ends at the actual stop.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));save(fig,'last_improvement_and_flat_tail')

fig, axes=plt.subplots(2,2,figsize=(13,9))
ax=axes[0,0]
for method in METHODS:
    c=curves['pep_999',method]
    ax.step([r['fresh'] for r in c],[r['mic'] for r in c],where='post',color=COLORS[method],label=method,lw=2)
for e in updates:
    if e['task']=='pep_999':ax.axvline(e['fresh'],color=COLORS[METHODS[1]],ls=':',alpha=.35,lw=1)
ax.set(title='pep_999: MIC and online update points',xlabel='Fresh evaluations',ylabel='Best predicted MIC (µM) ↓',xlim=(0,4000))
ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[0,1];xx=np.arange(8)
for k,method in enumerate(METHODS[:2]):
    rr=[r for r in feasibility if r['task']=='pep_999' and r['method']==method]
    ax.bar(xx+(k-.5)*.36,[r['fresh_feasible'] for r in rr],width=.36,color=COLORS[method],label=method)
ax.set(title='Fresh feasible acquisition candidates',xlabel='Acquisition rounds',ylabel='Count passing similarity constraint',xticks=xx,xticklabels=[f'{i+1}–{i+10}' for i in range(0,80,10)])
ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
ax=axes[1,0]
for method in METHODS[:2]:
    dd=focus[method]['distances'];counts=Counter(dd);denom=len(dd)
    ax.step(range(0,25),[100*counts[x]/denom for x in range(25)],where='mid',lw=2,color=COLORS[method],label=method)
ax.axvspan(-.5,3.5,color='green',alpha=.1,label='Feasible: ≤3 edits')
ax.set(title='Most decoded proposals miss the allowed neighborhood',xlabel='Levenshtein edits from 13-residue reference',ylabel='Fresh acquisition candidates (%)',xlim=(-.5,24))
ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[1,1]
st={r['stage']:r['seconds'] for r in stage_rows};wall=focus['wall_seconds']
categories=['Source re-encoding + scratch FSBO','Target FSBO fitting','Other wall time']
seconds=[st['source_meta_retraining'],st['target_refit'],wall-st['source_meta_retraining']-st['target_refit']]
ax.barh(range(3),np.array(seconds)/3600,color=['#d62728','#ff9896','#bbbbbb'])
for i,s in enumerate(seconds):ax.text(s/3600+.04,i,f'{s/3600:.2f} h ({100*s/wall:.1f}%)',va='center',fontsize=9)
ax.set(yticks=range(3),yticklabels=categories,xlabel='Recorded wall hours',title='pep_999: where online runtime went',xlim=(0,4.8));ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.tight_layout();save(fig,'pep999_diagnostics')

online_plateaus=[r for r in plateaus if r['method']==METHODS[1]]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),focus_task='pep_999',
             all_online_tasks_complete=True,task_stats=task_stats,
             online_flat_final_500=sum(r['final_500_actual_fresh_flat'] for r in online_plateaus),
             online_plateaus=online_plateaus,
             focus={k:{kk:vv for kk,vv in v.items() if kk not in ['rows','distances']} if isinstance(v,dict) else v for k,v in focus.items()},
             interpretation='Observed feasibility bottleneck starts before first online update. One improvement at round52 followed by28 flat rounds. Logs do not establish that online retraining caused the score gap; initial VAE and FSBO differ too. Source-only validation does not certify target acquisition feasibility.',
             plot_scope='Existing capped histories only: 500-evaluation intervals up to4000. No500-round run, no new oracle calls, no extension beyond actual stops.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
report=['# Online peptide diagnosis and 500-evaluation trajectories','',
        'All ten online tasks are complete at the established cap; these graphs analyze existing results. Batch size50 and at most80 acquisition rounds, not500 rounds. Identical initial100 measurements are excluded from fresh budgets.','',
        '## pep_999','',
        'The reference is KLYFLSFAITLAL (13 residues). The75% similarity constraint permits at most3 Levenshtein edits. Online FSBO produced9 feasible fresh acquisition candidates out of3837 (0.235%);3828 were infeasible. Original frozen FSBO produced17 feasible fresh candidates within its capped history. Both were inefficient; online never found the better late candidate reached by frozen FSBO.','',
        'Online MIC improved only once:112.650 →109.458µM at round52 (2483 total fresh, at completion of the scoring batch). It remained flat through round80 (3841 fresh), a1358-evaluation tail without improvement. Frozen FSBO found109.463 at round69 and43.308 at round72. The slightly different oracle readings for their shared sequence are preserved rather than silently reconciled.','',
        'There were zero feasible fresh acquisitions in online rounds1–30 (1420 fresh), before and across its first updates. All seven update cycles were accepted; each completed100000-step scratch FSBO retraining with verified checkpoint reload. Every source reconstruction probe retained128/128 feasible reconstructions, but those probes cover32 source tasks, not pep_999 acquisition neighborhoods. No task-specific decoder/feasibility rollback is implemented.','',
        'Recenter attempts produced2138 decoded proposals, only4 fresh sequences, and zero fresh feasible sequences. This was largely replay of known seeds, not an expansion of the useful search neighborhood. Source re-encoding and scratch FSBO consumed69.3% of task wall; target refitting consumed29.5%. More compute did not ensure better constrained exploration.','',
        'This supports a constrained-search/decoder bottleneck, not a demonstrated training crash or proven causal regression from online updates. The initial VAE and FSBO prior also differ from frozen baseline, and only one optimizer seed was run. Short length makes the discrete constraint tight, but length alone does not explain outcomes. A controlled adapted-frozen arm with matching initial weights/seeds is needed to isolate online updates.','',
        '## Observed online plateaus','',
        '| Task | Initial MIC | Final MIC | Last improvement round | Fresh at last improvement | Subsequent flat fresh evaluations |','|---|---:|---:|---:|---:|---:|']
for r in online_plateaus:
    report.append(f"| {r['task']} | {r['initial_mic']:.3f} | {r['final_mic']:.3f} | {r['last_improvement_round']} | {r['last_improvement_fresh']} | {r['fresh_since_last_improvement']} |")
report += ['',f"{summary['online_flat_final_500']}/10 online tasks had no improvement during their final500 actual fresh evaluations. This describes the observed tail and does not establish an asymptotic plateau.",'',
           '## Graphs and numerical data','',
           '- [All tasks at500-evaluation checkpoints](all_tasks_every_500_evaluations.png)','- [Online interval improvements](online_improvement_per_500.png)','- [Last improvement and flat tails, all methods](last_improvement_and_flat_tail.png)','- [pep_999 diagnostic panels](pep999_diagnostics.png)','',
           'Exact scores and interval reductions are in mic_changes_per_500_evaluations.csv; final intervals are partial where runs stopped below4000. The plots preserve all batch-causal measured steps and actual stopping points, use the same method colors as the previous report, and apply no smoothing. These graphs intentionally expose plateaus.','',
           'Suggested next diagnostic, not launched: compare author-frozen, adapted-frozen and adapted-online from matched seeds; log constraint posterior calibration and target-specific reconstruction validity; test smaller trust-region perturbations and valid decode filtering in a separate arm. Any filtering/retry policy needs physical oracle counts reported consistently. Do not silently replace infeasible candidates or alter existing scores.','']
(OUT/'REPORT.md').write_text('\n'.join(report))
print(json.dumps({k:v for k,v in summary.items() if k!='task_stats'},indent=2),flush=True)
