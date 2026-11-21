"""Frozen causal query prompts, real PostgreSQL measurements, explicit censoring."""
import json,math,os,sys,time
from functools import lru_cache
from rl_common import ROOT,read
from query_data import build_prompt,parse_action,plan_key,definition
from campaign_io import InfrastructureError
@lru_cache(maxsize=2)
def history(task_id):return read('assets/histories/'+task_id+'.json')
def prefix(job):
 h=history(job['task_id']);rows=h['records'][:job['prefix_end_exclusive']];valid=[r for r in rows if r['first_occurrence'] and r['feasible']]
 if not valid:raise InfrastructureError('Missing causal uncensored incumbent')
 return h,rows,valid
def prompt(job):
 h,rows,valid=prefix(job);e=read('assets/retrieval/'+job['group_id']+'.json') if job['scenario']=='retrieval' else None
 messages,ctx=build_prompt(job['task_id'],rows,scenario=job['scenario'],round_number=job['prefix_k']+1,cards=e['cards'] if e else [],allowed_training_tasks={t['task_id'] for t in read('assets/tasks.json')})
 if e:ctx['retrieval_ranking']=e['ranking'];messages[1]['content']=json.dumps(ctx,separators=(',',':'),sort_keys=True)
 return messages,dict(task_id=job['task_id'],source_prefix_end=job['prefix_end_exclusive'],source_history_sha256=h['source_history_sha256'],valid_prefix_count=len(valid),retrieval=e['ranking'] if e else None),rows,valid
@lru_cache(maxsize=4)
def oracle(task):
 sys.path.insert(0,'/root/data/cys/logs/query_train800_author_fixed_20261004/code')
 from run_query_ceb_baselines import RuntimeOracle
 from workload.workloads import WorkloadSpec,OracleCodec
 d,_,_,_=definition(task)
 return RuntimeOracle(WorkloadSpec.from_definition(d,OracleCodec.Aliases),read('settings.json')['query_timeout_seconds'])
def measured(plan,store):
 task=os.environ['QUERY_RL_TASK'];result=store.get('evaluation_result')
 if result is not None:
  if result['configuration']!={'plan':plan} or not math.isfinite(result['objective']):raise InfrastructureError('Corrupt durable runtime')
  return result
 if store.get('evaluation_started') is not None:raise InfrastructureError('Unresolved PostgreSQL call: inspect durable attempt before replay '+str(store.path))
 store.put('evaluation_started',dict(at=time.time(),configuration=dict(plan=plan),task=task,port=os.environ['DB_PORT'],timeout_seconds=float(os.environ['QUERY_RL_TIMEOUT'])))
 start=time.monotonic();timeout=float(os.environ['QUERY_RL_TIMEOUT']);value=oracle(task)(plan.copy(),timeout)
 if not math.isfinite(value['runtime']) or value['runtime']<=0:raise InfrastructureError('Invalid PostgreSQL runtime')
 result=dict(status='success' if not value['censored'] else value['status'],objective=value['runtime'],configuration=dict(plan=plan),measured=True,censored=bool(value['censored']),oracle='PostgreSQL_pg_hint_plan',details=value,elapsed=time.monotonic()-start)
 store.put('evaluation_result',result);return result
def sample(text,task,rows,valid,earlier,evaluate):
 from rl_reward_utils import RewardWeights,compute_trial_reward
 try:plan=parse_action(text,task)
 except (ValueError,TypeError,KeyError,IndexError) as exc:return dict(evaluation_status='PARSE_FAILURE',config=None,error=str(exc),reward=dict(reward=-.5,delta_best=0.))
 key=plan_key(plan)
 for scope,seen in [('history',{plan_key(r['plan']) for r in rows}),('sibling',{plan_key(r['config']['plan']) for r in earlier if r.get('config')})]:
  if key in seen:return dict(evaluation_status='DUPLICATE',config=dict(plan=plan),duplicate_scope=scope,reward=dict(reward=-1.,delta_best=0.))
 os.environ['QUERY_RL_TASK']=task
 policy=read('settings.json')['query_timeout_policy'];timeout=max(policy['floor_seconds'],min(policy['maximum_seconds'],policy['multiplier']*min(r['runtime'] for r in valid)))
 os.environ['QUERY_RL_TIMEOUT']=str(timeout);result=evaluate(plan);loss=result['objective']
 if result['censored']:
  # A bound is not a measured value. Timeout has a fixed non-improvement penalty,
  # no fictitious loss, no delta-best, and remains a valid negative RL response.
  return dict(evaluation_status='CENSORED',config=dict(plan=plan),reward=dict(reward=-.5,delta_best=0.),evaluation=result)
 reward=compute_trial_reward(perf=-loss,prior_perfs=[-r['runtime'] for r in valid],parse_ok=True,duplicate=False,eval_ok=True,weights=RewardWeights()).to_dict()
 return dict(evaluation_status='SUCCESS',config=dict(plan=plan),loss=loss,reward=reward,evaluation=result)
