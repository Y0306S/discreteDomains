"""Reuse chronological legacy BOLT measurements; bounded single-plan completion."""
import copy,hashlib,json,pickle,random,re,sqlite3,zlib
from pathlib import Path
from query_data import canonicalize,definition,plan_key
from query_protocol import atomic,digest
from single_method import seed_for,observed,best,novel,IncompletePool
ROOT=Path(__file__).resolve().parents[1]
HIST=Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1')
def legacy(task):
 p=ROOT/'warm_start/runs'/task/'runtime.sqlite'
 if not p.exists():return dict(rows=[],attempts=0,rejected=[],archive=[],legacy_sha256=None)
 with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as db:kv={k:pickle.loads(v) for k,v in db.execute('SELECT key,value FROM kv')}
 rows=[];archive=[];rejected=[];seen=set()
 for key,v in sorted(kv.items(),key=lambda x:int(x[0].split('/')[1]) if x[0].startswith('measurement/') else -1):
  if not key.startswith('measurement/'):continue
  if v['status']!='complete':raise RuntimeError('Unresolved legacy SQL '+task+' '+key)
  try:
   plan=canonicalize(task,v['argument']['plan'],strict=True)
   if plan_key(plan) in seen:raise ValueError('Duplicate legacy plan')
  except (ValueError,TypeError,KeyError,IndexError) as exc:rejected.append(dict(key=key,error=str(exc)));continue
  seen.add(plan_key(plan));row=observed(plan,v['result'],len(rows)+len(archive),'bolt_legacy')
  row['provenance']=dict(generation_key=key.replace('measurement/','generation/'),temperature=1.0,legacy_journal=str(p),legacy_sql_measurement_reused=True)
  if len(rows)<30:rows.append(row)
  else:archive.append(row)
 attempts=max((int(k.split('/')[1])+1 for k in kv if k.startswith('generation/')),default=0)
 return dict(rows=rows,attempts=attempts,rejected=rejected,archive=archive,legacy_sha256=digest(p))
def historical_plans(task):
 metadata=json.loads((HIST/'tasks.json').read_text())[task]
 if metadata['sql_sha256']!=digest(definition(task)[3]):raise ValueError('Previous BO SQL differs')
 candidates=[r['plan'] for r in metadata['initial']]
 p=HIST/'runs'/task/'artifacts.sqlite3'
 if p.exists():
  with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as db:
   for name,blob,sha in db.execute('SELECT name,payload,sha256 FROM artifacts ORDER BY name'):
    if not re.fullmatch(r'proposal_\d+\.json',name):continue
    raw=zlib.decompress(blob)
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Previous BO artifact checksum')
    item=json.loads(raw)
    if item.get('plan') is not None:candidates.append(item['plan'])
 out={}
 for p in candidates:
  try:plan=canonicalize(task,p,strict=True)
  except (ValueError,TypeError,KeyError,IndexError):continue
  out.setdefault(plan_key(plan),plan)
 return list(out.values())
def warm_progress(journal,state):
 atomic(journal.path.with_name('progress.json'),dict(stage='warm30',rows=len(state['rows']),attempts=state['attempts'],reused=sum(r['source_phase']=='bolt_legacy' for r in state['rows']),random_fallback=sum(r['source_phase']=='warm_random_previous_BO' for r in state['rows']),best_runtime=best(state['rows'])))
def warm_start(journal,task,recipe,generate,evaluate,pool_loader=historical_plans,paused=lambda:False):
 state=journal.get('warm_state')
 if state is None:
  old=legacy(task);state=dict(rows=old['rows'],attempts=old['attempts'],rejected=old['rejected'],legacy_calls=old['attempts'],legacy_sha256=old['legacy_sha256'])
  atomic(journal.path.with_name('legacy_archive.json'),dict(task=task,excluded_from_comparison=old['archive'],reason='measurements_after_first30_valid_generation_order'))
  journal.put('warm_state',state)
 cfg=recipe['warm_start'];seen={plan_key(r['plan']) for r in state['rows']}
 while len(state['rows'])<cfg['count'] and state['attempts']<cfg['maximum_samples']:
  if paused():
   from method import SafePause
   raise SafePause('Paused before next BOLT generation')
  i=state['attempts'];req=dict(stage='warm_start',task=task,attempt=i,seed=seed_for(task,'bolt30_v2',i,recipe['seed']))
  reply=journal.once('generation/'+str(i),req,lambda:generate(req))
  try:plan=novel(reply['text'],task,seen)
  except (ValueError,TypeError,KeyError,IndexError) as exc:state['rejected'].append(dict(attempt=i,error=str(exc),text=reply['text']))
  else:
   if paused():
    from method import SafePause
    raise SafePause('Paused before next BOLT SQL')
   result=journal.once('measurement/'+str(i),dict(plan=plan),lambda:evaluate(plan),measurement=True)
   row=observed(plan,result,len(state['rows']),'bolt_new');row['provenance']=dict(temperature=.9,generation_attempt=i)
   state['rows'].append(row);seen.add(plan_key(plan))
  state['attempts']+=1;journal.put('warm_state',state);warm_progress(journal,state)
 if len(state['rows'])<cfg['count']:
  def choose():
   plans=pool_loader(task);random.Random(seed_for(task,'warm_random_previous_BO',recipe['seed'])).shuffle(plans)
   plans=[p for p in plans if plan_key(p) not in seen]
   if len(plans)<cfg['count']-len(state['rows']):raise IncompletePool('Insufficient unique prior BO plans '+task)
   return plans[:cfg['count']-len(state['rows'])]
  # Stable argument, since rows change during measurement/resume.
  plans=journal.once('warm_fallback_plans',dict(task=task,policy='random_without_old_scores'),choose)
  for slot,plan in enumerate(plans):
   if paused():
    from method import SafePause
    raise SafePause('Paused before next fallback SQL')
   if plan_key(plan) in seen:continue
   result=journal.once(f'warm_fallback_measurement/{slot}',dict(plan=plan),lambda:evaluate(plan),measurement=True)
   state['rows'].append(observed(plan,result,len(state['rows']),'warm_random_previous_BO'));seen.add(plan_key(plan))
   journal.put('warm_state',state);warm_progress(journal,state)
 if len(state['rows'])!=cfg['count'] or best(state['rows']) is None:raise IncompletePool('Require30 valid unique measurements and an uncensored incumbent '+task)
 warm_progress(journal,state);return state
