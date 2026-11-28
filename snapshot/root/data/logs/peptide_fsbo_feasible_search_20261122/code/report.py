"""Live and final comparisons against the previous same-ten rollback run."""
import bisect
import csv
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def curve(result):
    best = result['initialization_best_mic']
    proposals = 0
    points = [(0, best)]
    for row in result['observations']:
        if row['phase'] == 'initialization':
            continue
        proposals += row['phase'] == 'acquisition'
        if row['feasible']:
            best = min(best, row['mic'])
        points.append((proposals, best))
    assert proposals == result['proposals']
    return points


def at(points, proposals):
    index = bisect.bisect_right([p for p, _ in points], proposals)-1
    return points[max(index, 0)][1]


def main():
    cfg = read(ROOT/'config.json')
    execution = read(ROOT/'execution_policy.json')
    baseline = Path(execution['comparison_baseline'])
    records, curves = [], {}
    for task in cfg['tasks']:
        previous = read(baseline/'evaluation/adapted_online'/task/'result.json')
        old_curve = curve(previous)
        directory = ROOT/'evaluation/adapted_online'/task
        final = directory/'result.json'
        source = final if final.exists() else directory/'status.json'
        value = read(source) if source.exists() else {}
        proposals = value.get('proposals', 0)
        best = value.get('best_feasible_mic', previous['initialization_best_mic'])
        complete = value.get('status') == 'complete'
        old_metrics = previous['phase_metrics']['acquisition']
        record = dict(task=task, status=value.get('status', 'queued'),
            rounds=value.get('rounds', 0), proposals=proposals,
            fresh_evaluations=value.get('fresh_evaluations', 0),
            new_best_mic=best, previous_mic_at_same_proposals=at(old_curve, proposals),
            previous_final_mic=previous['best_feasible_mic'],
            matched_mic_reduction=at(old_curve, proposals)-best,
            wall_seconds=value.get('wall_seconds', 0.),
            previous_wall_seconds=previous['wall_seconds'],
            acquisition_fresh=value.get('acquisition_fresh', 0),
            fresh_feasible_acquisition=value.get('fresh_feasible_acquisition', 0),
            previous_fresh_feasible_rate=old_metrics['fresh_feasible_rate'],
            update_attempts=value.get('retrain_schedule', {}).get('attempts', 0))
        record['new_fresh_feasible_rate'] = (record['fresh_feasible_acquisition']/record['acquisition_fresh']
                                             if record['acquisition_fresh'] else None)
        if complete:
            assert proposals == cfg['proposals'] or value['fresh_evaluations'] == cfg['fresh_budget']
            assert value['identity']['task'] == task
            new_curve = curve(value)
            assert abs(new_curve[-1][1]-best) < 1e-8
            metrics = value['phase_metrics']['acquisition']
            assert record['fresh_feasible_acquisition'] == metrics['fresh_feasible']
            curves[task] = dict(previous=old_curve, new=new_curve)
            record.update(updates_accepted=sum(e['accepted'] for e in value['adaptation_events']),
                          updates_rolled_back=sum(not e['accepted'] for e in value['adaptation_events']))
        records.append(record)
    done = [r for r in records if r['status'] == 'complete']
    summary = dict(status='complete' if len(done) == len(records) else 'running',
        updated_at=time.time(), completed_tasks=len(done), total_tasks=len(records),
        baseline=str(baseline), tasks=records, comparison_budget='matching acquisition proposals',
        single_seed=True)
    if done:
        summary['completed_task_comparison'] = dict(tasks=[r['task'] for r in done],
            mean_new_mic=statistics.mean(r['new_best_mic'] for r in done),
            mean_previous_mic=statistics.mean(r['previous_final_mic'] for r in done),
            wins=sum(r['previous_final_mic']-r['new_best_mic'] > 1e-8 for r in done),
            losses=sum(r['previous_final_mic']-r['new_best_mic'] < -1e-8 for r in done),
            ties=sum(abs(r['previous_final_mic']-r['new_best_mic']) <= 1e-8 for r in done))
    write(ROOT/'summary.json', summary)
    fields = list(records[0])+[k for k in ('updates_accepted', 'updates_rolled_back') if k not in records[0]]
    tmp = ROOT/'task_results.csv.tmp'
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(ROOT/'task_results.csv')
    table = '\n'.join(f"| {r['task']} | {r['status']} | {r['proposals']} | {r['new_best_mic']:.4f} | "
                      f"{r['previous_mic_at_same_proposals']:.4f} | {r['matched_mic_reduction']:+.4f} |"
                      for r in records)
    text = f'''# Same-ten peptide FSBO rerun

Completed: {len(done)}/{len(records)} tasks. Same initial100, seed1, batch50,
4000 acquisition proposals and 4000 shared fresh-evaluation cap. Lower MIC is better.
Positive reduction means the new search has a lower MIC at the same proposal count.

| Task | Status | Proposals | New MIC | Previous MIC at same budget | MIC reduction |
| --- | --- | ---: | ---: | ---: | ---: |
{table}

Baseline: {baseline}

`summary.json` and `task_results.csv` include fresh acquisition feasibility,
runtime, and retraining counts. Completed-task averages cover only the tasks
that have finished. The controller refreshes this report while workers run.
Runtime comparisons use shared A16 GPUs with different concurrent workloads.
This is one seed on ten previously examined tasks.
'''
    if len(done) == len(records):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name, label in (('previous', 'Previous rollback FSBO'), ('new', 'New feasible search')):
            x = list(range(0, cfg['proposals']+1, 50))
            y = [statistics.mean(at(curves[t][name], p) for t in cfg['tasks']) for p in x]
            ax.plot(x, y, label=label)
        ax.set(xlabel='Acquisition proposals (initial100 excluded)',
               ylabel='Mean best feasible MIC (µM, lower is better)',
               title='Same ten peptides, seed1')
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(ROOT/f'comparison.{ext}', dpi=180)
        plt.close(fig)
        text += '\n![Matched-budget comparison](comparison.png)\n'
    tmp = ROOT/'RESULTS.md.tmp'
    tmp.write_text(text)
    tmp.replace(ROOT/'RESULTS.md')
    write(ROOT/'report_status.json', dict(status=summary['status'], completed_tasks=len(done),
                                         updated_at=summary['updated_at']))


if __name__ == '__main__':
    main()
