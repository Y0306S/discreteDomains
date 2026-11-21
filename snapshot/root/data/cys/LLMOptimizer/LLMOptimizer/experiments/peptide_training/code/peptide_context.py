"""Causal discrete peptide prompts, real APEX and the exact ML reward formula."""
import json
import math
import sys
import time
from functools import lru_cache
from rl_common import ROOT,read
from rl_prompt_policy import build_peptide_rl_prompt

def task_description(task_id,reference):
    return dict(task_id=task_id,description=f'Minimize antimicrobial MIC near reference {reference}, length {len(reference)}.',
        objective='minimize APEX bacteria_0 MIC',reference_sequence=reference,
        constraints=dict(alphabet='ACDEFGHIKLMNPQRSTVWY',min_length=1,max_length=30,max_cysteine=1,
            min_reference_similarity=.75,similarity_metric='1-Levenshtein(candidate,reference)/len(reference)'))

@lru_cache(maxsize=2)
def history(task_id):return read('assets/histories/'+task_id+'.json')

def prefix(job):
    h=history(job['task_id']);rows=h['records'][:job['prefix_end_exclusive']]
    valid=[r for r in rows if r['first_occurrence'] and r['feasible']]
    assert len(valid)>=50
    return h,rows,valid

def prompt(job):
    h,rows,valid=prefix(job)
    evidence=read('assets/retrieval/'+job['group_id']+'.json') if job['scenario'] in ('retrieval','exploit') else None
    messages,context=build_peptide_rl_prompt(task_description(job['task_id'],h['reference']),valid,
        scenario=job['scenario'],round_number=job['prefix_k']+1,cards=[] if evidence is None else evidence['cards'],
        allowed_training_tasks={t['task_id'] for t in read('assets/tasks.json')})
    if evidence is not None:context['retrieval_ranking']=evidence['ranking']
    context['phase_instruction']=dict(retrieval='Analyze source/target differences before adapting measured top/bottom peptides.',
        normal='Balance novel feasible candidates using current measured evidence.',
        explore='Explore novel sequence variations while respecting every constraint.',
        exploit='Favor improvements near measured feasible incumbents while checking source-task differences.')[job['scenario']]
    messages[1]['content']=json.dumps(context,indent=2,allow_nan=False)
    return messages,dict(task_id=job['task_id'],source_prefix_end=job['prefix_end_exclusive'],
        source_history_sha256=h['source_history_sha256'],retrieval=evidence['ranking'] if evidence else None,
        reference_length=len(h['reference']),valid_prefix_count=len(valid)),rows,valid

@lru_cache(maxsize=1)
def oracle():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,'/root/data/logs/peptide_bo_train800_20261027_v5b/upstream_cpu')
    from apex_oracle.APEX_predict import apex_wrapper
    def evaluate(sequence):
        with torch.inference_mode():value=float(apex_wrapper([sequence])[0,0])
        if not math.isfinite(value) or value<0:raise RuntimeError('Invalid APEX MIC')
        return value
    return evaluate

def measured(sequence,store):
    result=store.get('evaluation_result')
    if result is not None:
        if result['configuration']!={'sequence':sequence} or not math.isfinite(result['objective']):raise RuntimeError('Corrupt oracle record')
        return result
    if store.get('evaluation_started') is not None:raise RuntimeError('Unresolved APEX call; audit before replay: '+str(store.path))
    evaluator=oracle();store.put('evaluation_started',dict(at=time.time(),configuration=dict(sequence=sequence)))
    started=time.monotonic();value=evaluator(sequence)
    result=dict(status='success',objective=value,configuration=dict(sequence=sequence),measured=True,
        oracle='APEX_bacteria_0_CPU',elapsed=time.monotonic()-started)
    store.put('evaluation_result',result);return result

def sample(text,reference,rows,valid,earlier,evaluate):
    from peptide_prompt import parse_action
    from rl_reward_utils import RewardWeights,compute_trial_reward
    try:sequence=parse_action(text,reference,set())
    except Exception as exc:return dict(evaluation_status='PARSE_FAILURE',config=None,error=str(exc),reward=dict(reward=-.5,delta_best=0.))
    for scope,seen in [('history',{r['sequence'] for r in rows}),('sibling',{r['config']['sequence'] for r in earlier if r.get('config')})]:
        if sequence in seen:return dict(evaluation_status='DUPLICATE',config=dict(sequence=sequence),duplicate_scope=scope,reward=dict(reward=-1.,delta_best=0.))
    result=evaluate(sequence);loss=result['objective']
    reward=compute_trial_reward(perf=-loss,prior_perfs=[-r['mic'] for r in valid],parse_ok=True,duplicate=False,eval_ok=True,weights=RewardWeights()).to_dict()
    return dict(evaluation_status='SUCCESS',config=dict(sequence=sequence),loss=loss,reward=reward,evaluation=result)
