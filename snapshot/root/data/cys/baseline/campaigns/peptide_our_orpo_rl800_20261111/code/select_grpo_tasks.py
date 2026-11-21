"""Select 32 train-only peptides with all ML-style turn checkpoints available."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from peptide_prompt import atomic_json, file_sha
from rl_common import ROOT, read


def ml_order(stats):
    """ML: Pareto-front first, then average percentile rank of both axes."""
    def dominates(a, b):
        return (a['late_n_new'] >= b['late_n_new'] and a['late_pct_sum'] >= b['late_pct_sum']
                and (a['late_n_new'] > b['late_n_new'] or a['late_pct_sum'] > b['late_pct_sum']))
    front={t for t in stats if not any(dominates(s,stats[t]) for s in stats.values())}
    for t,s in stats.items():
        s['pareto_front']=t in front
        s['rank_score']=sum(sum(v[k]<=s[k] for v in stats.values())/len(stats)
                            for k in ['late_n_new','late_pct_sum'])/2
    return sorted(stats,key=lambda t:(t not in front,-stats[t]['rank_score'],
        -stats[t]['late_n_new'],-stats[t]['late_pct_sum'],t))


def select(jobs, count=32):
    inventory = {t['task_id']: t for t in read('assets/rl_task_inventory.json')['tasks']}
    required = {k for p in read('settings.json')['phases'] for k in p['prefixes']}
    available = defaultdict(set)
    for job in jobs:
        available[job['task_id']].add(job['prefix_k'])
    training = {t['task_id'] for t in read('assets/tasks.json')}
    stats = {}
    for task in sorted(training):
        if available[task] != required:
            continue
        history = read('assets/histories/' + task + '.json')
        valid = [r for r in history['records'] if r['feasible'] and r['first_occurrence']]
        best=min(r['mic'] for r in valid if r['round']==0)
        curve={0:best};events=[]
        for turn in range(1,101):
            candidate=min((r['mic'] for r in valid if r['round']==turn),default=best)
            if candidate < best-1e-12:events.append(turn)
            best=min(best,candidate);curve[turn]=best
        pct=lambda lo,hi:(curve[lo]-curve[hi])/max(abs(curve[lo]),1e-12)*100
        # ML's interval event counts include their shared boundary at turn 80.
        n1=sum(40<=turn<=80 for turn in events);n2=sum(80<=turn<=100 for turn in events)
        descriptors=inventory[task]['descriptors']
        charge=descriptors['charge_proxy_KR_minus_DE']
        charge_bin='nonpositive' if charge<=0 else '1-2' if charge<=2 else '3+'
        hydro_bin='below0.5' if descriptors['hydrophobic_fraction_AVILMFWY']<.5 else 'atleast0.5'
        diversity_bin='|'.join([inventory[task]['length_bin'],charge_bin,hydro_bin,
                              'Cys1+' if descriptors['cysteine_count'] else 'Cys0'])
        stats[task] = dict(late_n_new=n1+n2,late_pct_sum=pct(40,80)+pct(80,100),
            n_new_40_80=n1,n_new_80_100=n2,pct_40_80=pct(40,80),pct_80_100=pct(80,100),
            new_best_rounds_40_100=sum(40<turn<=100 for turn in events),
            relative_improvement_40_100=(curve[40]-curve[100])/max(abs(curve[40]),1e-12),
            length_bin=inventory[task]['length_bin'],reference_length=inventory[task]['reference_length'],
            descriptors=descriptors,diversity_bin=diversity_bin,
            history_sha256=file_sha(ROOT/'assets/histories'/f'{task}.json'))
    histories={t:s['history_sha256'] for t,s in stats.items()}
    # Late improvement is the ML difficulty proxy; the user's end-of-run
    # requirement additionally excludes tasks already flat during turns 80-100.
    stats={t:s for t,s in stats.items() if s['late_n_new']>=1 and s['pct_80_100']>0}
    if len(stats)<count:raise ValueError('Insufficient tasks still improving at turns 80-100')
    bins = defaultdict(list)
    for task in stats:bins[stats[task]['diversity_bin']].append(task)
    for name,tasks in bins.items():
        bins[name]=ml_order({t:stats[t] for t in tasks})
    picked = []
    while len(picked) < count:
        for name in sorted(bins):
            if bins[name] and len(picked) < count:
                picked.append(bins[name].pop(0))
    ml_source=Path('/root/data/cys/llmoptimizer_benchmark/examples/benchmark/select_train32_pareto_bo_intervals.py')
    return dict(task_count=count, tasks=picked, eligible_tasks=len(stats),full_history_tasks=len(histories),source_training_tasks=len(training),
        criterion='ML Pareto-front first, then average percentile-rank fill on late_n_new and late_pct_sum, balanced across peptide diversity groups',
        ml_selection_source=str(ml_source),ml_selection_source_sha256=file_sha(ml_source),
        late_metrics='n_new(40-80)+n_new(80-100), pct(40-80)+pct(80-100), using completed peptide acquisition rounds',
        additional_user_gate='Positive improvement during turns 80-100 for every selected task',
        percentage_cap=None,percentage_cap_reason='ML cap25 controlled near-zero regression-scale artifacts; only 3 peptide tasks meet that cap, so it is not applied to positive MIC reductions',
        eligibility='All 12 requested completed-acquisition turn checkpoints exist in the frozen training history',
        heldout_tasks_used=0, selected_statistics={t:stats[t] for t in picked},
        eligible_history_sha256=histories,
        descriptors_notice='Sequence-derived diversity proxies, not experimentally verified biological subtypes',
        diversity_bin_counts=dict(Counter(stats[t]['diversity_bin'] for t in picked)),
        length_bin_counts=dict(Counter(stats[t]['length_bin'] for t in picked)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    selection = select(json.loads(args.jobs.read_text()), args.count)
    selection['source_jobs_sha256'] = file_sha(args.jobs)
    atomic_json(ROOT/'assets/grpo_task_selection.json', selection)
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
