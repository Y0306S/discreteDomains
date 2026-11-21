"""Exercise full training-only long context on expanded replicas; no SQL."""
import argparse,json,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'batch_code'))
from query_data import build_prompt,task_description
from query_protocol import canonical,atomic
from qwen_inference_transport import digest
from transformers import AutoTokenizer
from hf_qwen_client import render_s1b_prompt
p=argparse.ArgumentParser();p.add_argument('ports',nargs='+',type=int);p.add_argument('--concurrent',action='store_true');p.add_argument('--padding-words',type=int,default=16000);p.add_argument('--max-new-tokens',type=int,default=32);p.add_argument('--min-input-tokens',type=int,default=28000);p.add_argument('--output-suffix',default='');a=p.parse_args()
cfg=json.loads((ROOT/'batch_recipe.json').read_text());train=Path('/root/data/cys/baseline/campaigns/query_our_orpo_rl800_20261112')
task=json.loads((ROOT/'evidence/batch_context_smoke.json').read_text())['task'];rows=json.loads((train/'assets/histories'/f'{task}.json').read_text())['records'][:70]
cards=json.loads((train/'assets/source_cards.json').read_text());tables={r['table'] for r in task_description(task)['constraints']['aliases']};short=[]
for t,v in cards.items():
 if t==task:continue
 tt={r['table'] for r in v['task']['constraints']['aliases']};short.append((t,len(tables&tt)/len(tables|tt)))
short=sorted(short,key=lambda v:(-v[1],v[0]))[:16];selected=[cards[x[0]] for x in sorted(short,key=lambda x:len(canonical(cards[x[0]])),reverse=True)[:3]]
msgs,ctx=build_prompt(task,rows,scenario='retrieval',round_number=8,cards=selected,allowed_training_tasks=cfg['train_tasks']);ctx['capacity_test_padding']='Ignore this synthetic capacity-test field; no measurements: '+('padding '*a.padding_words);msgs[1]['content']=canonical(ctx)
tok=AutoTokenizer.from_pretrained(cfg['models']['orpo']['identity']['adapter'],trust_remote_code=True);tokens=len(tok.encode(render_s1b_prompt(tok,msgs)))
assert a.min_input_tokens<tokens<32768,(tokens,task)
base=dict(identity=cfg['models']['orpo']['identity'],messages=msgs,settings=dict(temperature=1.,max_new_tokens=a.max_new_tokens,top_p=.9,top_k=50,enable_thinking=False),rng=dict(seed=192038,cuda_rng=None),use_s1b_template=True)
def call(port):
 body=dict(base);body['request_id']=digest(body);t=time.time();request=urllib.request.Request(f'http://127.0.0.1:{port}/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(request,timeout=1200) as r:result=json.load(r)
  record=dict(passed=True,port=port,seconds=time.time()-t,input_tokens=tokens,result={k:v for k,v in result.items() if k not in ['messages','identity']},heldout_used=False,SQL_calls=0)
 except Exception as e:record=dict(passed=False,port=port,seconds=time.time()-t,error=repr(e),input_tokens=tokens)
 atomic(ROOT/'operations/reallocate_20261119'/f'inference_smoke_{port}{a.output_suffix}.json',record);print(json.dumps(record)[:1700],flush=True);return record['passed']
if a.concurrent:
 import concurrent.futures
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.ports)) as ex:passed=list(ex.map(call,a.ports))
else:passed=[call(port) for port in a.ports]
raise SystemExit(not all(passed))
