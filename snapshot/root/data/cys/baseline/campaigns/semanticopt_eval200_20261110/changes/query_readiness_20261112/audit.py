import sys,json,time,pathlib,collections,statistics,copy
r=pathlib.Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110');sys.path.insert(0,str(r/'code'))
from common import verify
from evaluate import query_setup
from batch import batch_messages,accept_candidates
from protocol import atomic
recipe=verify();out=[];bad=collections.Counter()
assert recipe['stages']==['peptide','query'];assert recipe['batch_policy']['size']==50
assert recipe['query']['proposals']==1000 and recipe['query']['rounds']==20
for task in recipe['query']['test']:
 context,initial,validate,_=query_setup(task,recipe['query'],0)
 messages=batch_messages(context,initial,[],1000,50,[],[],32);prompt=json.loads(messages[1]['content'])
 assert prompt['constraints']['preserve']=='all query tables, aliases, predicates and SQL semantics'
 assert prompt['constraints']['canonical_alias_codec'] is True
 assert prompt['constraints']['implicit_repair_allowed'] is False
 assert set(prompt['constraints']['operators'])=={'1','2','3'}
 assert prompt['query_tables'] and prompt['table_symbols'] and prompt['sql']
 assert prompt['proposal_budget']==1000 and prompt['batch_size']==50 and prompt['requested_configurations']==50
 assert 'distinct' in messages[0]['content'] and 'duplicates' in prompt['instruction']
 for row in initial:validate(row['configuration'])
 base=initial[0]['configuration'];invalid=[]
 x=copy.deepcopy(base);x['plan'][0]=999999;invalid.append(('unknown_table',x))
 x=copy.deepcopy(base);x['plan'][1]=999999;invalid.append(('unknown_alias',x))
 x=copy.deepcopy(base);x['plan'][4]=99;invalid.append(('invalid_operator',x))
 x=copy.deepcopy(base);x['plan']=x['plan'][:-5];invalid.append(('missing_join',x))
 for label,config in invalid:
  try:validate(config)
  except ValueError:bad[label]+=1
  else:raise AssertionError((task,label))
 accepted,errors=accept_candidates([base],validate,{json.dumps(base,sort_keys=True,separators=(',',':'))},50)
 assert not accepted and errors[0]['error']=='Duplicate configuration';bad['duplicate']+=1
 out.append(dict(task=task,initializations=len(initial),query_tables=len(context['query_tables']),operators=prompt['constraints']['operators']))
 if len(out)%50==0:print('audited',len(out),flush=True)
result=dict(passed=True,at=time.time(),tasks=len(out),initializations_validated=sum(x['initializations'] for x in out),rejected_invalid_cases=dict(bad),batch_size=50,rounds=20,proposals_per_task=1000,
 queue_order=recipe['stages'],constraints_in_actual_batch_prompt=True,heldout_oracle_calls=0,details=out)
atomic(r/'changes/query_readiness_20261112/constraint_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)
