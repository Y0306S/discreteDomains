"""Query training gates: source runtime, complete prompt tokens, exact parent gradients."""
import argparse,json,os,time
from pathlib import Path
from rl_common import ROOT,read,digest
from query_protocol import atomic

def tokens():
 from transformers import AutoTokenizer
 from mindware.domains.dlhpo.taskset_llm.spec import render_nothink
 from query_context import prompt
 tok=AutoTokenizer.from_pretrained(read('settings.json')['base_model'],trust_remote_code=True);lengths=[];maximum=None;identities={}
 jobs=read('assets/jobs.json');assert len(jobs)==384
 for j in jobs:
  msgs,meta,rows,valid=prompt(j);ctx=json.loads(msgs[1]['content']);assert ctx['recent50']==[__import__('query_data').compact(r) for r in valid[-50:]]
  assert all(r['observation_index']<j['prefix_end_exclusive'] for k in ['recent50','good','bad'] for r in ctx[k])
  if j['scenario']=='retrieval':
   ranking=ctx['retrieval_ranking'];assert ranking['target_future_labels_used'] is False;assert max(ranking['target_evidence_indices'])<j['prefix_end_exclusive'];assert j['task_id'] not in ranking['selected_source_ids']
  n=len(tok(render_nothink(msgs),add_special_tokens=False)['input_ids']);lengths.append(n)
  if n+read('settings.json')['max_new_tokens']>read('settings.json')['max_seq_len']:raise ValueError('Untruncated context limit exceeded: '+j['group_id']+' tokens='+str(n))
  identities[j['group_id']]=dict(tokens=n,prompt_sha256=__import__('hashlib').sha256(json.dumps(msgs,sort_keys=True).encode()).hexdigest())
  if maximum is None or n>maximum['tokens']:maximum=dict(job=j,messages=msgs,tokens=n)
 atomic(ROOT/'assets/prompt_identities.json',identities);atomic(ROOT/'evidence/longest_rl_prompt.json',maximum);atomic(ROOT/'evidence/rl_token_preflight.json',dict(passed=True,prompts=len(jobs),min_tokens=min(lengths),max_tokens=max(lengths),truncated_prompts=0,recent50_preserved=True,heldout_used=False))

def oracle_check():
 import math
 from resources import databases
 from query_context import oracle
 from query_data import canonicalize
 instances=databases();task=read('assets/grpo_task_selection.json')['tasks'][0];h=read('assets/histories/'+task+'.json');best=min((r for r in h['records'] if r['first_occurrence'] and not r['censored']),key=lambda r:r['runtime'])
 os.environ.update(DB_HOST='127.0.0.1',DB_PORT=str(instances[0]['port']),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb')
 actual=oracle(task)(best['plan'].copy(),120)
 if actual['censored'] or actual['status']!='completed' or not math.isfinite(actual['runtime']) or actual['runtime']<=0:raise ValueError('Real training runtime probe did not complete uncensored')
 atomic(ROOT/'evidence/cpu_oracle_check.json',dict(passed=True,task=task,archived_runtime=best['runtime'],actual=actual,oracle_calls=1,heldout_used=False,note='Wall-clock runtime is remeasured on live PostgreSQL, not expected to match bit for bit; same hints and two parallel worker policy.'))

def parent_check():
 import torch
 from transformers import AutoTokenizer
 from train_grpo import load_model,token_logprobs
 from mindware.domains.dlhpo.taskset_llm.spec import render_nothink
 settings=read('settings.json');parent=Path(settings['parent_adapter']);status=read('orpo_status.json')
 if status['status']!='completed' or status['adapter_sha256']!=digest(parent/'adapter_model.safetensors'):raise ValueError('Wrong or incomplete ORPO parent')
 model,coverage=load_model(parent,True,settings)
 if not coverage.get('exact'):raise ValueError('Exact ORPO adapter was not inherited')
 model.train();assert model.is_gradient_checkpointing
 tokenizer=AutoTokenizer.from_pretrained(parent,trust_remote_code=True);probe=read('evidence/longest_rl_prompt.json');prompt=tokenizer(render_nothink(probe['messages']),add_special_tokens=False)['input_ids'];h=read('assets/histories/'+probe['job']['task_id']+'.json');plan=max((r['plan'] for r in h['records'] if r['first_occurrence']),key=len)
 answer=tokenizer(json.dumps(dict(plan=plan),separators=(',',':'))+'<|im_end|>\n',add_special_tokens=False)['input_ids'];ids=torch.tensor([prompt+answer],device='cuda:0')
 with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.bfloat16):
  values=token_logprobs(model,ids,len(prompt));loss=-values.mean()
 if not torch.isfinite(values).all():raise ValueError('Nonfinite query probabilities')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
 if float(norm)<=0:raise ValueError('No adapter gradient')
 model.zero_grad(set_to_none=True)
 atomic(ROOT/'evidence/rl_parent_check.json',dict(passed=True,exact_ORPO_adapter=coverage,parent_sha256=status['adapter_sha256'],prompt_tokens=len(prompt),answer_tokens=len(answer),gradient_norm=float(norm),probe_updates=0,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,gradient_checkpointing=True,activation_offload=True,at=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['tokens','oracle','parent']);a=p.parse_args();{'tokens':tokens,'oracle':oracle_check,'parent':parent_check}[a.action]()
