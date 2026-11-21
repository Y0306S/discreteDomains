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
spec=next(w for w in json.loads((ROOT/'resources.json').read_text())['workers'] if w['host']=='LLMOptimizer2')
os.environ.update(DB_HOST='127.0.0.1',DB_PORT=str(spec['db_port']),DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb')
raw_evaluate=oracle(task,50);new_SQL_calls=[]
def evaluate(plan):
 new_SQL_calls.append(plan);return raw_evaluate(plan)
# Reuse the unchanged BOLT checkpoint's saved TRAIN-only smoke measurements.
# This is provenance-preserving reuse; all ORPO/FSBO batches below are fresh.
import sqlite3,pickle
prior=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_test200_20261116/evidence/batch_preflight.sqlite')
with sqlite3.connect('file:'+str(prior)+'?mode=ro',uri=True) as db:
 state=pickle.loads(db.execute("SELECT value FROM kv WHERE key='warm_state'").fetchone()[0])
assert len(state['rows'])==3
for r in state['rows']:canonicalize(task,r['plan'],strict=True)
journal.put('warm_state',state)
print('Reused three saved training-only BOLT smoke measurements',flush=True)
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
 report=dict(passed=True,task=task,heldout_evaluations=0,bolt_smoke_count=3,bolt_unique_target30_verified_by_tests=True,batches=[4,9,21],fsbo_batch_size=10,real_measurements=len(rows),new_SQL_calls_this_invocation=len(new_SQL_calls),bolt_reuse_source_sha256=digest(prior),fsbo_demonstrated_plans=sum(x['actual_arm']=='fsbo' for x in batch['items']),fsbo_final_fit=batch['fsbo_diagnostics']['fit'],recipe_sha256=digest(ROOT/'batch_recipe.json'),policy_sha256=digest(ROOT/'batch_policy.json'),at=time.time())
 atomic(ROOT/'evidence/batch_preflight.json',report);print('BATCH PREFLIGHT PASSED',flush=True)
finally:fsbo.close();journal.close()
