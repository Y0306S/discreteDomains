"""Offline launch checks plus deferred CPU-oracle/GPU-parent checks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from peptide_prompt import atomic_json,digest,file_sha
from rl_common import ROOT,read

def tokens():
    from transformers import AutoTokenizer
    from mindware.domains.dlhpo.taskset_llm.spec import render_nothink
    import peptide_context
    tokenizer=AutoTokenizer.from_pretrained(read('settings.json')['base_model'],trust_remote_code=True)
    lengths=[];maximum=None;identities={}
    tasks={t['task_id'] for t in read('assets/tasks.json')}
    original=json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/recipe.json').read_text())
    assert len(tasks)==800 and tasks==set(original['train']) and not tasks&set(original['test'])
    for i,job in enumerate(read('assets/jobs.json')):
        messages,meta,rows,valid=peptide_context.prompt(job)
        context=json.loads(messages[1]['content']);assert len(context['recent50'])==50
        indices=[r['observation_index'] for r in context['recent50']]
        assert indices==[r['observation_index'] for r in valid[-50:]] and max(indices)<job['prefix_end_exclusive']
        if job['scenario'] in ('retrieval','exploit'):
            ranking=context['retrieval_ranking'];assert ranking['target_future_labels_used'] is False
            assert max(ranking['target_evidence_indices'])<job['prefix_end_exclusive']
            assert not (set(ranking['selected_source_ids'])-tasks) and job['task_id'] not in ranking['selected_source_ids']
            assert all(s['differences_from_current_task'] for s in context['similar_task_evidence']['similar_tasks'])
        assert not any(k in context for k in ['good_region','bad_region','regions','latent_bounds'])
        n=len(tokenizer(render_nothink(messages),add_special_tokens=False)['input_ids'])
        if n+read('settings.json')['max_new_tokens']>read('settings.json')['max_seq_len']:raise ValueError('Full RL prompt exceeds context: '+job['group_id'])
        lengths.append(n);identities[job['group_id']]=dict(prompt_sha256=digest(messages),tokens=n)
        if maximum is None or n>maximum['tokens']:maximum=dict(job=job,messages=messages,tokens=n)
        if (i+1)%500==0:print(json.dumps(dict(prompts=i+1,max_tokens=max(lengths))),flush=True)
    atomic_json(ROOT/'assets/prompt_identities.json',identities)
    atomic_json(ROOT/'evidence/longest_rl_prompt.json',maximum)
    atomic_json(ROOT/'evidence/rl_token_preflight.json',dict(passed=True,prompts=len(lengths),min_tokens=min(lengths),max_tokens=max(lengths),max_seq_len=read('settings.json')['max_seq_len'],truncated_prompts=0,recent50_preserved=True,heldout_used=False))

def oracle_check():
    from peptide_context import oracle,history
    evaluate=oracle();h=history('pep_000');rows=[r for r in h['records'] if r['feasible'] and r['first_occurrence']]
    probes=[rows[0],min(rows,key=lambda r:r['mic'])];results=[]
    for row in probes:
        actual=evaluate(row['sequence']);expected=row['mic'];relative=abs(actual-expected)/max(abs(expected),1e-12)
        if relative>.001:raise ValueError('CPU APEX differs from source beyond audited tolerance')
        results.append(dict(sequence=row['sequence'],archived_mic=expected,CPU_mic=actual,relative_error=relative))
    atomic_json(ROOT/'evidence/cpu_oracle_check.json',dict(passed=True,samples=results,relative_tolerance=.001,oracle_calls=len(results),test_tasks_used=0))

def parent_check():
    from contextlib import nullcontext
    import torch
    from transformers import AutoTokenizer
    from train_grpo import load_model,token_logprobs
    from mindware.domains.dlhpo.taskset_llm.spec import render_nothink
    settings=read('settings.json');parent=Path(settings['parent_adapter'])
    status=read('orpo_status.json')
    if status['status']!='completed' or status['adapter_sha256']!=file_sha(parent/'adapter_model.safetensors'):raise ValueError('Wrong/incomplete ORPO parent')
    model,coverage=load_model(parent,True,settings)
    if not coverage.get('exact'):raise ValueError('RL did not inherit exact peptide ORPO weights')
    # Match the production gradient path: checkpointing is active only in train
    # mode. load_model already disables every dropout to preserve probabilities.
    model.train()
    if not model.is_gradient_checkpointing:raise ValueError('Gradient checkpointing is inactive')
    tokenizer=AutoTokenizer.from_pretrained(parent,trust_remote_code=True)
    probe=read('evidence/longest_rl_prompt.json');prompt=tokenizer(render_nothink(probe['messages']),add_special_tokens=False)['input_ids']
    h=read('assets/histories/'+probe['job']['task_id']+'.json')
    answer=tokenizer(json.dumps({'sequence':h['reference']})+'<|im_end|>\n',add_special_tokens=False)['input_ids']
    ids=torch.tensor([prompt+answer],device='cuda:0')
    offload=settings['training'].get('activation_offload',False)
    saved_tensors=torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
    with saved_tensors,torch.autocast('cuda',dtype=torch.bfloat16):
        values=token_logprobs(model,ids,len(prompt))
        loss=-values.mean()
    if not torch.isfinite(values).all():raise ValueError('Nonfinite actual peptide token probabilities')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    if float(norm)<=0:raise ValueError('No adapter gradient')
    model.zero_grad(set_to_none=True)
    atomic_json(ROOT/'evidence/rl_parent_check.json',dict(passed=True,exact_ORPO_adapter=coverage,parent_sha256=status['adapter_sha256'],
        prompt_tokens=len(prompt),answer_tokens=len(answer),gradient_norm=float(norm),probe_updates=0,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,attention=model.config._attn_implementation,
        gradient_checkpointing=model.is_gradient_checkpointing,activation_offload=offload,bf16_autocast=True,at=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tokens','oracle','parent']);a=p.parse_args()
    {'tokens':tokens,'oracle':oracle_check,'parent':parent_check}[a.action]()

if __name__=='__main__':main()
