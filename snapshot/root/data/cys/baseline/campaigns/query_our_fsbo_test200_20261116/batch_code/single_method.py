"""Durable BOLT warm50 and independent ORPO + FSBO evaluation."""
import copy,hashlib,json,math,pickle,sqlite3,time
from pathlib import Path
import numpy as np
from hybrid_selector import HybridSelector
from query_data import parse_action,plan_key,build_prompt
from query_protocol import atomic
from query_resource_recovery import QueryInfrastructureRetryExhausted

class IncompletePool(RuntimeError): pass
class UnresolvedMeasurement(RuntimeError): pass

def seed_for(*parts):return int(hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()[:15],16)

class Journal:
 def __init__(self,path,identity):
  Path(path).parent.mkdir(parents=True,exist_ok=True)
  self.path=Path(path);self.db=sqlite3.connect(path,timeout=120)
  self.db.execute('PRAGMA journal_mode=TRUNCATE');self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value BLOB NOT NULL)')
  old=self.get('identity')
  if old is not None and old!=identity:raise ValueError('Journal identity changed')
  self.put('identity',identity)
 def get(self,key,default=None):
  r=self.db.execute('SELECT value FROM kv WHERE key=?',(key,)).fetchone()
  return pickle.loads(r[0]) if r else default
 def put(self,key,value):
  with self.db:self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',(key,pickle.dumps(value,protocol=4)))
 def once(self,key,argument,call,measurement=False):
  old=self.get(key)
  if old:
   if old['argument']!=argument:raise ValueError('Operation identity changed '+key)
   if old['status']=='complete':return old['result']
   if measurement and old['status']!='safe_failed':raise UnresolvedMeasurement('Audit interrupted SQL before replay: '+str(self.path)+' '+key)
  self.put(key,dict(status='started',argument=argument,at=time.time()))
  try:result=call()
  except QueryInfrastructureRetryExhausted as exc:
   self.put(key,dict(status='safe_failed',argument=argument,failures=exc.failures,at=time.time()));raise
  self.put(key,dict(status='complete',argument=argument,result=result,at=time.time()))
  return result
 def progress(self,state,stage):
  atomic(self.path.with_name('progress.json'),dict(stage=stage,rows=len(state['rows']),attempts=state.get('attempts',len(state.get('events',[]))),
   best_runtime=best(state['rows']),at=time.time()))
 def close(self):self.db.close()

def best(rows):return min((r['runtime'] for r in rows if not r['censored']),default=None)
def observed(plan,result,index,source):
 runtime=float(result['runtime'])
 if not math.isfinite(runtime) or runtime<=0:raise ValueError('Invalid SQL measurement')
 return dict(plan=list(plan),runtime=runtime,censored=bool(result['censored']),first_occurrence=True,
  feasible=not bool(result['censored']),observation_index=index,source_phase=source,measurement=dict(result))

def novel(text,task,seen):
 plan=parse_action(text,task)
 if plan_key(plan) in seen:raise ValueError('Duplicate canonical plan')
 return plan

def warm_start(journal,task,recipe,generate,evaluate):
 state=journal.get('warm_state',dict(rows=[],attempts=0,rejected=[]));cfg=recipe['warm_start']
 while len(state['rows'])<cfg['count'] and state['attempts']<cfg['maximum_samples']:
  i=state['attempts'];req=dict(stage='warm_start',task=task,attempt=i,seed=seed_for(task,'warm_start',i,recipe['seed']))
  reply=journal.once('generation/'+str(i),req,lambda:generate(req))
  try:plan=novel(reply['text'],task,{plan_key(r['plan']) for r in state['rows']})
  except (ValueError,TypeError,KeyError,IndexError) as exc:state['rejected'].append(dict(attempt=i,error=str(exc),text=reply['text']))
  else:
   result=journal.once('measurement/'+str(i),dict(plan=plan),lambda:evaluate(plan),measurement=True)
   state['rows'].append(observed(plan,result,len(state['rows']),'initialization'))
  state['attempts']+=1;journal.put('warm_state',state);journal.progress(state,'warm_start')
 if len(state['rows'])!=cfg['count'] or best(state['rows']) is None:raise IncompletePool(f'{task}: {len(state["rows"])} distinct warm rows, incumbent={best(state["rows"])}; no random replacement')
 return state

def selector():return HybridSelector(mode='pucb',bo_arm='fsbo',pucb_p1=.7,pucb_q='arm_best',pucb_c2=1.4,rng=np.random.RandomState(1))
def select_arm(bandit,events,attempt):
 if attempt<=8:return 'llm',dict(reason='first8_post_initialization_llm')
 proposed=bandit.choose(attempt-9,91,higher_is_better=True)
 cooldown=10<=attempt<=20 and events[-1]['selected_arm']=='llm'
 return ('fsbo' if cooldown else proposed),dict(bandit.snapshot(),pucb_proposed_arm=proposed,cooldown_active=cooldown,schedule_index=attempt-9,schedule_denominator=91)

def optimize(journal,task,recipe,pool,generate,evaluate,fsbo,stop_after=None):
 if len(pool)!=recipe['warm_start']['count'] or len({plan_key(r['plan']) for r in pool})!=len(pool):raise ValueError('Require frozen distinct warm50')
 if best(pool) is None:raise ValueError('No uncensored incumbent')
 state=journal.get('optimization_state',dict(rows=copy.deepcopy(pool),events=[],selector=selector()))
 while len(state['events'])<recipe['optimization']['attempts']:
  attempt=len(state['events'])+1;bandit=copy.deepcopy(state['selector']);selected,details=select_arm(bandit,state['events'],attempt)
  seen={plan_key(r['plan']) for r in state['rows']};plan=None;actual=selected;rejected=[];diag=None
  if selected=='llm':
   for retry in range(recipe['optimization']['llm_generation_limit']):
    messages,_=build_prompt(task,state['rows'],scenario='normal',round_number=attempt)
    req=dict(stage='orpo',task=task,attempt=attempt,retry=retry,messages=messages,seed=seed_for(task,'orpo',attempt,retry,recipe['seed']))
    reply=journal.once(f'generation/{attempt}/{retry}',req,lambda:generate(req))
    try:plan=novel(reply['text'],task,seen);break
    except (ValueError,TypeError,KeyError,IndexError) as exc:rejected.append(dict(retry=retry,text=reply['text'],error=str(exc)))
   if plan is None:actual='fsbo'
  if actual=='fsbo':
   req=dict(task=task,attempt=attempt,seed=seed_for(task,'orpo',attempt,'fsbo',recipe['seed']))
   proposal=journal.once(f'fsbo/{attempt}',req,lambda:fsbo(state['rows'],req['seed']))
   plan,diag=proposal['plan'],proposal['diagnostics']
   if plan is not None:plan=novel(json.dumps(dict(plan=plan)),task,seen)
  prior=best(state['rows']);event=dict(attempt=attempt,selected_arm=selected,actual_arm=actual,selection=details,rejected_generations=rejected,fsbo_diagnostics=diag)
  perf=float('-inf')
  if plan is None:event.update(status='NO_VALID_PROPOSAL',plan=None,runtime=None)
  else:
   result=journal.once(f'measurement/{attempt}',dict(plan=plan),lambda:evaluate(plan),measurement=True)
   row=observed(plan,result,len(state['rows']),'acquisition');state['rows'].append(row)
   if not row['censored']:perf=-row['runtime']
   event.update(status=result['status'],plan=plan,runtime=row['runtime'],censored=row['censored'])
  bandit.update(actual,perf,beat_incumbent=perf>-prior)
  if actual!=selected:bandit.record_attempt(selected)
  event['best_runtime']=best(state['rows']);state['selector']=bandit;state['events'].append(event)
  journal.put('optimization_state',state);journal.progress(state,'orpo')
  if stop_after is not None and len(state['events'])>=stop_after:break
 return state
