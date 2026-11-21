"""Verify long intact retrieval prompt capacity on saved training histories only."""
import json,sys,time
from pathlib import Path
from query_data import build_prompt,task_description
from query_protocol import canonical,atomic,digest
from backends import LLM
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];TRAIN=Path('/root/data/cys/baseline/campaigns/query_our_orpo_rl800_20261112');cfg=json.loads((ROOT/'batch_recipe.json').read_text());cards=json.loads((TRAIN/'assets/source_cards.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['models']['orpo']['identity']['base'],trust_remote_code=True);maximum=None
for p in (TRAIN/'assets/histories').glob('*.json'):
 task=p.stem
 assert task in cfg['train_tasks'] and task not in cfg['tasks']
 rows=json.loads(p.read_text())['records'][:70];a=task_description(task)['constraints']['aliases'];tables={r['table'] for r in a};short=[]
 for t,v in cards.items():
  if t==task:continue
  aa=v['task']['constraints']['aliases'];tt={r['table'] for r in aa};short.append((t,len(tables&tt)/len(tables|tt),abs(len(a)-len(aa))))
 short=sorted(short,key=lambda v:(-v[1],v[2],v[0]))[:16];selected=[cards[x[0]] for x in sorted(short,key=lambda x:len(canonical(cards[x[0]])),reverse=True)[:3]]
 msgs,ctx=build_prompt(task,rows,scenario='retrieval',round_number=8,cards=selected,allowed_training_tasks=cfg['train_tasks']);ctx['retrieval_confidence']='capacity_test_only_no_rank_transfer_claim';msgs[1]['content']=canonical(ctx)
 text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False);tokens=len(tok.encode(text))
 if maximum is None or tokens>maximum[0]:maximum=(tokens,task,msgs)
tokens,task,msgs=maximum
ctx=json.loads(msgs[1]['content']);ctx['capacity_test_padding']='Ignore this synthetic capacity-test field; it contains no measurements: '+('padding '*16000);msgs[1]['content']=canonical(ctx)
rendered=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False);tokens=len(tok.encode(rendered))
print('Training long context',task,tokens,'tokens with explicit synthetic capacity padding',flush=True);assert 28000<tokens<32768
llm=LLM(cfg,'orpo','http://127.0.0.1:21811');reply=llm(dict(task=task,attempt='context_capacity_smoke',retry=0,messages=msgs,seed=192038))
atomic(ROOT/'evidence/batch_context_smoke.json',dict(passed=True,task=task,input_tokens=tokens,heldout_used=False,SQL_calls=0,response=reply,at=time.time()));print('LONG CONTEXT PASSED peakGiB',reply['peak_reserved_gib'],flush=True)
