"""Build training-only histories and exact causal context jobs; no oracle calls."""
from collections import Counter
import io
import json
from pathlib import Path
import sqlite3
import hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from build_pairs import load_history,SOURCE,AUDIT
from peptide_prompt import valid_first_occurrences,atomic_json,file_sha
from peptide_context import task_description
from rl_common import ROOT,read

def one(task,reference,source_sha):
    raw=load_history(task,source_sha)
    records,valid=valid_first_occurrences(raw['observations'],reference)
    h=dict(task=task,reference=reference,source_history_sha256=source_sha,records=records,completed_acquisition_rounds=raw['rounds'])
    atomic_json(ROOT/'assets/histories'/f'{task}.json',h)
    contexts=read('data/tasks/'+task+'.json')['causal_context_inventory']
    cuts={r['source_round']-1:r['prefix_end_exclusive'] for r in contexts if r['source_phase']=='acquisition'}
    # A genuine terminal checkpoint is also a valid causal prefix after its last acquisition.
    cuts[raw['rounds']]=len(records)
    jobs=[]
    for phase in read('settings.json')['phases']:
        for k in phase['prefixes']:
            if k not in cuts:continue
            group=f"{phase['id']}/{task}/k{k}"
            jobs.append(dict(task_id=task,phase=phase['id'],scenario=phase['scenario'],prefix_k=k,
                prefix_end_exclusive=cuts[k],repeat=0,group_id=group))
    ranked=sorted(valid,key=lambda r:(r['mic'],r['observation_index']))
    card=dict(task=task_description(task,reference),top=[dict(candidate=r['sequence'],measured_value=r['mic']) for r in ranked[:3]],
              bottom=[dict(candidate=r['sequence'],measured_value=r['mic']) for r in ranked[-3:]])
    return jobs,card

def main():
    recipe=json.loads((SOURCE/'recipe.json').read_text());audit=json.loads(AUDIT.read_text());checks={r['task']:r for r in audit['tasks']}
    jobs=[];cards={}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures={ex.submit(one,t,recipe['references'][t],checks[t]['member_checksums']['history.json']):t for t in recipe['train']}
        for f in as_completed(futures):
            task=futures[f];j,c=f.result();jobs.extend(j);cards[task]=c
            if len(cards)%100==0:print(json.dumps(dict(tasks=len(cards),jobs=len(jobs))),flush=True)
    order=read('assets/rl_task_inventory.json')['interleaved_full_coverage_order'];positions={t:i for i,t in enumerate(order)}
    phases={p['id']:i for i,p in enumerate(read('settings.json')['phases'])}
    jobs.sort(key=lambda j:(phases[j['phase']],positions[j['task_id']],j['prefix_k']))
    selection_path=ROOT/'assets/grpo_task_selection.json'
    selected=set(read('assets/grpo_task_selection.json')['tasks']) if selection_path.exists() else set(order)
    assert selected <= set(order)
    jobs=[j for j in jobs if j['task_id'] in selected]
    atomic_json(ROOT/'assets/tasks.json',[dict(task_id=t,reference=recipe['references'][t]) for t in order])
    atomic_json(ROOT/'assets/jobs.json',jobs);atomic_json(ROOT/'assets/source_cards.json',cards)
    bank=Path('/root/data/logs/peptide_transfer_train800_20261027_v2/rgpe/models.sqlite')
    with sqlite3.connect('file:'+str(bank)+'?mode=ro',uri=True) as db:
        inventory={t:dict(sha256=sha,metadata=json.loads(meta)) for t,sha,meta in db.execute('SELECT task,sha256,metadata FROM models')}
    assert set(inventory)==set(recipe['train']) and not set(inventory)&set(recipe['test'])
    atomic_json(ROOT/'assets/source_gp_inventory.json',inventory)
    atomic_json(ROOT/'protocol.json',dict(domain='peptide',cycles=1,phases=read('settings.json')['phases'],training=read('settings.json')['training'],
        data_split='train800 only; heldout200 excluded',task_count=len(selected),source_training_task_count=800,
        task_selection='assets/grpo_task_selection.json' if selection_path.exists() else None,job_counts=dict(Counter(j['scenario'] for j in jobs)),
        actual_prefix_semantics='completed acquisition rounds; true terminal checkpoints admitted; unavailable requested prefixes excluded',
        retrieval=read('settings.json')['retrieval'],prompt_policy='rl_prompt_policy.json',fresh_policy_each_phase=True))
    atomic_json(ROOT/'evidence/rl_assets.json',dict(passed=True,tasks=len(selected),retrieval_source_tasks=800,jobs=len(jobs),job_counts=dict(Counter(j['scenario'] for j in jobs)),
        oracle_calls=0,heldout_tasks_used=0,source_recipe_sha256=file_sha(SOURCE/'recipe.json')))
    print((ROOT/'evidence/rl_assets.json').read_text(),flush=True)

if __name__=='__main__':main()
