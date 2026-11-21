"""Offline task-diversity and RL-context audit. Does not launch or queue jobs."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from Levenshtein import distance
from peptide_prompt import atomic_json, file_sha

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
AUDIT=Path('/root/reports/peptide_train800_test200_audit_20261027/train800_audit.json')


def length_bin(length):
    if length<=15:return '10-15'
    if length<=20:return '16-20'
    return '21-25'


def main():
    recipe=json.loads((SOURCE/'recipe.json').read_text())
    audit=json.loads(AUDIT.read_text())
    checks={r['task']:r for r in audit['tasks']}
    manifest=json.loads((ROOT/'manifest.json').read_text())
    train=recipe['train'];heldout=set(recipe['test'])
    assert len(train)==800 and len(set(train))==800 and not set(train)&heldout
    refs={t:recipe['references'][t] for t in train}
    tasks=[];nearest_sims=[]
    availability=Counter();late_missing=[]
    # Full coverage order interleaves short/medium/long references. It does not
    # alter the ORPO dataset, sample weights, or existing train/test split.
    bins={name:[] for name in ['10-15','16-20','21-25']}
    per_task={r['task']:r for r in manifest['per_task']}
    for task in train:
        ref=refs[task];length=len(ref);bucket=length_bin(length)
        bins[bucket].append(task)
        neighbors=sorted((dict(task_id=t,reference_sequence=other,
                               edit_similarity=1-distance(ref,other)/max(length,len(other)),
                               length_difference=len(other)-length)
                          for t,other in refs.items() if t!=task),
                         key=lambda n:(-n['edit_similarity'],abs(n['length_difference']),n['task_id']))[:3]
        nearest_sims.append(neighbors[0]['edit_similarity'])
        contexts=json.loads((ROOT/'data/tasks'/f'{task}.json').read_text())['causal_context_inventory']
        acquisition=[r for r in contexts if r['source_phase']=='acquisition']
        late=[r for r in acquisition if r['source_round']>=51]
        if not late:late_missing.append(task)
        for proposed_round in [4,6,11,21,31,41,51,61,71,81,91]:
            availability[proposed_round]+=any(r['source_round']==proposed_round for r in acquisition)
        tasks.append(dict(task_id=task,reference_sequence=ref,reference_length=length,length_bin=bucket,
            descriptors=dict(charge_proxy_KR_minus_DE=sum(ref.count(x) for x in 'KR')-sum(ref.count(x) for x in 'DE'),
                             hydrophobic_fraction_AVILMFWY=sum(ref.count(x) for x in 'AVILMFWY')/length,
                             cysteine_count=ref.count('C')),
            descriptors_notice='Sequence-derived diversity descriptors, not measured net charge or biological subtype labels.',
            source_BO_improved=checks[task]['improved'],orpo_pairs=per_task[task]['pairs'],
            train_only_reference_neighbors=neighbors,
            neighbors_notice='Nearest references only; ranking is not a verified biological relationship or a calibrated transfer predictor. No held-out reference used.',
            acquisition_contexts=[dict(proposal_round=r['source_round'],completed_acquisition_rounds=r['source_round']-1,
                                      prefix_end_exclusive=r['prefix_end_exclusive'],valid_prefix_count=r['valid_prefix_count'])
                                  for r in acquisition],
            post50_exploitation_contexts=len(late)))
    order=[]
    for i in range(max(map(len,bins.values()))):
        for bucket in bins.values():
            if i<len(bucket):order.append(bucket[i])
    assert len(order)==800 and set(order)==set(train)
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),
        train_tasks=800,heldout_tasks_used=0,all_unique_training_references=True,
        reference_length_range=[min(map(len,refs.values())),max(map(len,refs.values()))],
        reference_length_counts=dict(sorted(Counter(map(len,refs.values())).items())),
        length_bins={k:dict(tasks=len(v),source_improved=sum(checks[t]['improved'] for t in v),
                           ORPO_pairs=sum(per_task[t]['pairs'] for t in v)) for k,v in bins.items()},
        biological_type_labels_available=False,
        proposed_RL_task_scope='all 800 training tasks, including no-improvement tasks; use genuine available causal contexts only',
        nearest_reference_similarity=dict(min=min(nearest_sims),median=statistics.median(nearest_sims),max=max(nearest_sims),
                                          below_0p5=sum(s<.5 for s in nearest_sims)),
        source_context_available_by_proposal_round=dict(sorted(availability.items())),
        post50_eligible_tasks=800-len(late_missing),post50_ineligible_tasks=late_missing,
        round_semantics='proposal round r uses only observations before acquisition round r; it follows r-1 completed acquisition rounds. Imported initialization and earlier recenter measurements count in valid history, not as additional acquisition rounds.',
        selection_limit='Late phases cannot claim all 800 tasks from existing histories: do not relabel early prefixes as post-50 contexts.',
        source_recipe_sha256=file_sha(SOURCE/'recipe.json'),source_audit_sha256=file_sha(AUDIT))
    atomic_json(ROOT/'reports/peptide_task_selection_audit.json',summary)
    atomic_json(ROOT/'assets/rl_task_inventory.json',dict(status='proposed_not_training_jobs',tasks=tasks,interleaved_full_coverage_order=order))
    proposal=dict(status='prepared_for_discussion_not_queued',domain='peptide',
        selection_audit='reports/peptide_task_selection_audit.json',task_inventory='assets/rl_task_inventory.json',
        ORPO=dict(base_model='/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B',parent_adapter=None,
                  examples=1263,tasks_with_pairs=651,tasks_scanned=800,epochs=1,optimizer_steps=158,
                  config='orpo_config.json',token_preflight='orpo_preflight.json',ready=True),
        RL=dict(ready=False,parent='exact verified new peptide ORPO adapter',reference='same ORPO adapter, fixed for all four phases',
                cycles=1,phase_order=['retrieval','normal','explore','exploit'],training_passes_per_phase=2,
                task_scope='all800 early phases; only genuinely eligible tasks in later phases',
                prompts='current-task top/bottom + Recent50 + constraints; source-task top/bottom and task differences in RS/post50 exploitation; no region summaries',
                reward='measured APEX bacteria_0 MIC improvement, with the ML GRPO framework; final normalization must be verified in peptide adapter',
                output='one discrete peptide sequence per response',
                sample_rule='ML adaptive4-10 proposals, exactly4 selected responses per GRPO group; no oracle call for duplicates/invalid sequences',
                checkpoints='ORPO final; each RL phase with optimizer/RNG state; preserved cycle1 checkpoint',
                remaining=['Implement peptide rollout collection/oracle integration and exact ML reward adaptation',
                           'Finalize source-task retrieval selection; inspect weak reference neighbors rather than claiming biological similarity',
                           'Select phase contexts from genuine causal prefixes, accounting for missing post50 histories',
                           'Token-preflight actual retrieval prompts without truncating Recent50 or constraints',
                           'Run live CPU-oracle, exact ORPO-parent and GPU backward/memory smoke checks',
                           'Verify checkpoint/resume and full launch gates before automatic RL handoff']),
        concurrency=dict(dependency='current ML ORPO+RL+FSBO120 must release its 12 replicas',
            semanticopt='existing queue: query200 then peptide200, one 21GiB-capped inference replica per A800',
            training='one A800 shared with one SemanticOPT replica; ORPO48GiB allocator cap, admission requires54GiB free; actual memory smoke required',
            rollout='reuse training allocation sequentially for rollout generation and optimization; fixed-reference log probabilities via adapter swap on one base, as in ML',
            GPUs=1,CPU_training_threads=4,CPU_rollout_workers_initial=4,
            CPU_note='Only launch rollout workers after joint admission against SemanticOPT load. Its adaptive controller can yield workers at safe boundaries.',
            no_unrelated_job_preemption=True,
            admission_gates=['ML dependency audited complete','SemanticOPT stage running with all4 verified replicas',
                             'at least54GiB genuinely free on selected GPU after SemanticOPT load','shared output writable with free bytes and inodes',
                             'backward/memory smoke passes before full training']),
        scheduling_note='This is a proposal, not a running queue. No ORPO/RL process is started by this offline preparation.')
    atomic_json(ROOT/'training_setup_proposal.json',proposal)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
