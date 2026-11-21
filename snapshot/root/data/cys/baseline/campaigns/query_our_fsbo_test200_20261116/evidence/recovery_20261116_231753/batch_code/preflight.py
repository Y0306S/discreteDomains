"""Live batch revision smoke on train800 only; no heldout tuning."""
import copy,json,os,time
from pathlib import Path
from unittest.mock import patch
from query_protocol import atomic,digest
from query_data import plan_key,canonicalize
from single_method import Journal,observed
from warm import warm_start
from method import build_batch
from backends import LLM,FSBO,oracle
from retrieval import retrieve
ROOT=Path(__file__).resolve().parents[1]
os.sched_setaffinity(0,{24});cfg=json.loads((ROOT/'batch_recipe.json').read_text());policy=json.loads((ROOT/'batch_policy.json').read_text());task=cfg['train_tasks'][0]
assert task not in cfg['tasks'];journal=Journal(ROOT/'evidence/batch_preflight.sqlite',dict(recipe_sha256=digest(ROOT/'batch_recipe.json'),policy_sha256=digest(ROOT/'batch_policy.json'),task=task))
print('Train-only live preflight',task,flush=True)
raw_evaluate=oracle(task,50);new_SQL_calls=[]
def evaluate(plan):
 new_SQL_calls.append(plan);return raw_evaluate(plan)
bolt=LLM(cfg,'warm_start','http://127.0.0.1:21810');small=copy.deepcopy(cfg);small['warm_start']['count']=3
empty=dict(rows=[],attempts=0,rejected=[],archive=[],legacy_sha256=None)
with patch('warm.legacy',return_value=empty):state=warm_start(journal,task,small,bolt,evaluate)
print('BOLT0.9',len(state['rows']),'unique measured plans',state['attempts'],'single calls',flush=True)
rows=copy.deepcopy(state['rows']);orpo=LLM(cfg,'orpo','http://127.0.0.1:21811');fsbo=FSBO(cfg,task,rows)
try:
 for number,selected in [(4,'llm'),(9,'fsbo'),(21,'fsbo')]:
  batch=journal.once(f'live_batch/{number}',dict(round=number,selected=selected),lambda:build_batch(journal,task,cfg,policy,rows,number,selected,orpo,fsbo,retrieve))
  print('Round',number,'frozen',len(batch['items']),'sources',[x['actual_arm'] for x in batch['items']],flush=True)
  assert len(batch['items'])==10 and len({plan_key(x['plan']) for x in batch['items']})==10
  # Entire batch is saved before the first SQL call, exactly as production.
  for slot,item in enumerate(batch['items']):
   p=canonicalize(task,item['plan'],strict=True);result=journal.once(f'live_measurement/{number}/{slot}',dict(plan=p),lambda:evaluate(p),measurement=True)
   rows.append(observed(p,result,len(rows),item['actual_arm']));print('SQL',number,slot,result['runtime'],result['status'],flush=True)
  if number==4:assert batch['retrieval'] is not None
  if number==21:assert any(x['actual_arm']=='fsbo' for x in batch['items']), 'FSBO must demonstrate a real valid decoded proposal'
 report=dict(passed=True,task=task,heldout_evaluations=0,bolt_smoke_count=3,bolt_unique_target30_verified_by_tests=True,batches=[4,9,21],fsbo_batch_size=10,real_measurements=len(rows),new_SQL_calls_this_invocation=len(new_SQL_calls),fsbo_demonstrated_plans=sum(x['actual_arm']=='fsbo' for x in batch['items']),fsbo_final_fit=batch['fsbo_diagnostics']['fit'],recipe_sha256=digest(ROOT/'batch_recipe.json'),policy_sha256=digest(ROOT/'batch_policy.json'),at=time.time())
 atomic(ROOT/'evidence/batch_preflight.json',report);print('BATCH PREFLIGHT PASSED',flush=True)
finally:fsbo.close();journal.close()
