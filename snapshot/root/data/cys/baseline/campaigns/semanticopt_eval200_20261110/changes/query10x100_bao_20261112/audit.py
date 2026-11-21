import sys,json,time,copy,collections
from pathlib import Path
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'staged/code'))
from common import verify,domain_recipe
from evaluate import query_setup
from batch import batch_messages,accept_candidates
from protocol import atomic,canonical
from transformers import AutoTokenizer
recipe=domain_recipe(verify(),'query');settings=recipe['query'];policy=recipe['batch_policy']
tokenizer=AutoTokenizer.from_pretrained(settings['adapter'],local_files_only=True,trust_remote_code=True)
results=[];bad=collections.Counter()
for task in settings['test']:
 context,initial,validate,_=query_setup(task,settings,0)
 assert len(initial)==49
 for row in initial:validate(row['configuration'])
 tokens=[];windows=[]
 for rows in [[],[dict(initial[i%49],evaluation=i+1) for i in range(32)]]:
  window=32
  while True:
   messages=batch_messages(context,initial,rows,1000,10,[],[],window,size=10)
   content=json.loads(messages[1]['content'])
   count=len(tokenizer(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))['input_ids'])
   if count<=policy['max_input_tokens']:break
   assert window>1,(task,count)
   window=max(1,window//2)
  assert len(content['bao_initial_observations'])==49
  assert [x['configuration'] for x in content['bao_initial_observations']]==[x['configuration'] for x in initial]
  assert content['initial_observation_count']==49
  assert content['batch_size']==10 and content['proposal_budget']==1000 and content['requested_configurations']==10
  assert content['constraints']['preserve']=='all query tables, aliases, predicates and SQL semantics'
  assert content['constraints']['canonical_alias_codec'] is True and content['constraints']['implicit_repair_allowed'] is False
  assert set(content['constraints']['operators'])=={'1','2','3'}
  assert content['sql'] and content['query_tables'] and content['table_symbols']
  tokens.append(count);windows.append(window)
 representative=max((x['configuration'] for x in initial),key=lambda x:len(canonical(x)))
 output=len(tokenizer(canonical(dict(configurations=[representative]*10)))['input_ids'])
 assert output*1.2+128 <= policy['max_new_tokens']['query']
 base=initial[0]['configuration']
 for label,index,value in [('unknown_table',0,999999),('unknown_alias',1,999999),('invalid_operator',4,99)]:
  c=copy.deepcopy(base);c['plan'][index]=value
  try:validate(c)
  except ValueError:bad[label]+=1
  else:raise AssertionError((task,label))
 c=copy.deepcopy(base);c['plan']=c['plan'][:-5]
 try:validate(c)
 except ValueError:bad['missing_join']+=1
 else:raise AssertionError(task)
 accepted,errors=accept_candidates([base],validate,{canonical(base)},10)
 assert not accepted and errors[0]['error']=='Duplicate configuration';bad['duplicate']+=1
 results.append(dict(task=task,initializations=49,first_prompt_tokens=tokens[0],later_prompt_tokens=tokens[1],feedback_windows=windows,output_tokens_estimate=output))
 if len(results)%25==0:print('audited',len(results),flush=True)
result=dict(passed=True,at=time.time(),tasks=len(results),bao_initializations=sum(x['initializations'] for x in results),all_bao_in_first_and_later_prompts=True,batch_size=10,rounds=100,proposals=1000,heldout_oracle_calls=0,rejected_invalid_cases=dict(bad),max_first_prompt_tokens=max(x['first_prompt_tokens'] for x in results),max_later_prompt_tokens=max(x['later_prompt_tokens'] for x in results),max_output_tokens_estimate=max(x['output_tokens_estimate'] for x in results),input_token_limit=policy['max_input_tokens'],details=results)
atomic(E/'prompt_constraint_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)
