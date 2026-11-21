"""Measured CEB join plans, causal optimization prompts and strict direct actions."""
import hashlib,json,math,sys,re
from pathlib import Path
from functools import lru_cache
from query_protocol import validate_query_plan,canonical,atomic,digest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
NATIVE=SOURCE/'native'
sys.path.insert(0,str(NATIVE))
SYSTEM=('You optimize PostgreSQL query join plans using measured runtime in seconds, lower is better. '
 'Return exactly one compact JSON object {"plan":[integers]}. No predictions, prose, markdown, or latent vectors. '
 'Respect the supplied SQL and all table aliases. Use exactly N-1 component merges for N aliases. '
 'Each group is [left_table_id,left_alias,right_table_id,right_alias,operator]; table IDs are zero based, '
 'aliases are one based. Each operand identifies the component containing that alias. Merge two different components. '
 'Operators: 1=NestLoop, 2=HashJoin, 3=MergeJoin. Never repeat a measured plan. Censored runtimes are lower bounds.')
@lru_cache(maxsize=32)
def definition(task):
 from workload.workloads import get_workload_set
 from codec.codec import AliasesCodec
 m=re.fullmatch(r'CEB_(\d+[A-Z])(\d+)',task);p=NATIVE/'workload/ceb-3k'/m[1].lower()/((m[1]+m[2]).lower()+'.sql')
 sql=p.read_text();d=get_workload_set('CEB_3K').extract_query(sql,task)
 return d,AliasesCodec(d.all_tables),sql,p

def canonicalize(task,plan,strict=False):
 d,c,_,_=definition(task)
 if strict:return validate_query_plan(plan,d,c)
 tree=c.decode(d.query_tables,plan.copy());encoded=c.encode(tree)
 validate_query_plan(encoded,d,c)
 if not tree.equal(c.decode(d.query_tables,encoded.copy())):raise ValueError('Canonicalization changes execution')
 return encoded

def plan_key(plan):
 # Independent joins may be emitted in different topological orders by upstream sets.
 return canonical(sorted([plan[i:i+5] for i in range(0,len(plan),5)]))

def task_description(task):
 d,c,sql,_=definition(task)
 aliases=[dict(table=t,table_id=d.all_tables.index(t),alias=a) for t,n in d.query_tables for a in range(1,n+1)]
 return dict(task_id=task,description=f'Optimize {len(aliases)}-alias CEB SQL join tree.',objective='minimize PostgreSQL wall runtime seconds',
  constraints=dict(preserve_SQL_and_predicates=True,aliases=aliases,joins=len(aliases)-1,operators={'1':'NestLoop','2':'HashJoin','3':'MergeJoin'},no_implicit_codec_repairs=True),
  query_text=sql,search_space=dict(representation='AliasesCodec integer groups of five',table_ids='zero based',alias_ids='one based',component_merge='Each operand names an alias inside an active component; merge distinct components.'))

def compact(r):return dict(observation_index=r['observation_index'],plan=r['plan'],runtime_seconds=float(format(r['runtime'],'.6g')),censored=r['censored'])

def build_prompt(task,rows,*,sft=False,scenario='normal',round_number=None,cards=(),allowed_training_tasks=()):
 desc=task_description(task)
 if sft:return [dict(role='system',content=SYSTEM),dict(role='user',content=canonical(dict(current_task=desc,instruction='Return a fast feasible join plan for this SQL.')))],None
 valid=[r for r in rows if r['first_occurrence'] and r['feasible']]
 if not valid:raise ValueError('No uncensored causal incumbent')
 ranked=sorted(valid,key=lambda r:(r['runtime'],r['observation_index']))
 ctx=dict(current_task=desc,current_best_runtime=min(r['runtime'] for r in valid),good=[compact(r) for r in ranked[:5]],bad=[compact(r) for r in ranked[-5:]],recent50=[compact(r) for r in valid[-50:]],scenario=scenario,round_number=round_number,
  instruction='Propose one novel feasible plan that improves the incumbent. Compare join order and operators using actual measurements.')
 if cards:
  from rl_prompt_policy import retrieval_evidence
  ctx['similar_task_evidence']=retrieval_evidence('query',desc,cards,allowed_training_tasks)
  ctx['similar_task_evidence'].pop('current_task')  # Already present once at the prompt's top level.
 return [dict(role='system',content=SYSTEM),dict(role='user',content=canonical(ctx))],ctx

def parse_action(text,task):
 def unique(items):
  out={}
  for k,v in items:
   if k in out:raise ValueError('Repeated JSON key')
   out[k]=v
  return out
 value=json.loads(text.strip(),object_pairs_hook=unique)
 if not isinstance(value,dict) or set(value)!={'plan'}:raise ValueError('Exactly one plan key required')
 return canonicalize(task,value['plan'],strict=True)
