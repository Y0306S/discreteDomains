"""80 global rounds, ten frozen proposals per round; causal batch feedback."""
import copy,json,random
from query_data import build_prompt,plan_key,canonicalize,definition
from query_protocol import atomic,canonical
from single_method import Journal,seed_for,observed,best,novel,selector,select_arm as old_select_arm

def select_arm(bandit,events,global_round):
 arm,details=old_select_arm(bandit,events,global_round)
 if global_round<=8:details["reason"]="global_rounds4..8_retrieval_llm"
 return arm,details
from random_plans import query_proposal
class UnderfilledBatch(RuntimeError):pass
class SafePause(RuntimeError):pass

def validate_pool(task,pool,count=30):
 if len(pool)!=count or len({plan_key(r['plan']) for r in pool})!=count or best(pool) is None:raise ValueError('Invalid frozen warm pool')
 for row in pool:canonicalize(task,row['plan'],strict=True)

def build_batch(journal,task,recipe,policy,rows,round_number,selected,generate,fsbo,retrieve=None):
 count=policy['size'];seen={plan_key(r['plan']) for r in rows};items=[];rejected=[];diagnostics=None;retrieval=None
 if selected=='llm':
  if 4<=round_number<=8:
   if retrieve is None:raise ValueError('Retrieval required in rounds4..8')
   retrieval=journal.once(f'retrieval/{round_number}',dict(task=task,round_number=round_number),lambda:retrieve(task,rows,fsbo,recipe['train_tasks']))
  for retry in range(policy['llm_generation_limit']):
   messages,context=build_prompt(task,rows,scenario='retrieval' if retrieval else 'normal',round_number=round_number,cards=retrieval['cards'] if retrieval else (),allowed_training_tasks=recipe['train_tasks'])
   # These are explicitly unmeasured, and never enter history/runtime targets.
   context['unmeasured_avoidance']=dict(accepted_plans=[r['plan'] for r in items],recent_rejected=rejected[-5:],instruction='Avoid these accepted proposals and rejected answers. These are not measured outcomes; do not infer their runtimes.')
   if retrieval:context['retrieval_confidence']=retrieval['ranking']['confidence']
   messages[1]['content']=canonical(context)
   req=dict(stage='orpo',task=task,attempt=round_number,retry=retry,messages=messages,seed=seed_for(task,'orpo_batch10_v2',round_number,retry,recipe['seed']))
   reply=journal.once(f'generation/{round_number}/{retry}',req,lambda:generate(req))
   try:plan=novel(reply['text'],task,seen)
   except (ValueError,TypeError,KeyError,IndexError) as exc:rejected.append(dict(call=retry+1,text=reply['text'],reason=str(exc)));continue
   items.append(dict(plan=plan,actual_arm='llm'));seen.add(plan_key(plan))
   if len(items)==count:break
 if len(items)<count:
  req=dict(task=task,round_number=round_number,count=count-len(items),excluded=[r['plan'] for r in items],seed=seed_for(task,'orpo_batch10_v2',round_number,'fsbo',recipe['seed']))
  proposal=journal.once(f'fsbo/{round_number}',req,lambda:fsbo(rows,req['seed'],req['count'],req['excluded']))
  diagnostics=proposal['diagnostics']
  if len(proposal['plans'])>req['count']:raise ValueError('Oversized FSBO batch')
  for candidate in proposal['plans']:
   try:plan=novel(json.dumps(dict(plan=candidate)),task,seen)
   except (ValueError,TypeError,KeyError,IndexError) as exc:
    rejected.append(dict(source='fsbo',plan=candidate,reason=str(exc)));continue
   items.append(dict(plan=plan,actual_arm='fsbo'));seen.add(plan_key(plan))
 random_draws=0
 if len(items)<count:
  d,c,_,_=definition(task);rng=random.Random(seed_for(task,'random_fallback',round_number,recipe['seed']))
  for _ in range(policy['random_draw_limit']):
   random_draws+=1
   try:plan=novel(json.dumps(query_proposal(d,c,rng)),task,seen)
   except (ValueError,TypeError,KeyError,IndexError):continue
   items.append(dict(plan=plan,actual_arm='random'));seen.add(plan_key(plan))
   if len(items)==count:break
 if len(items)!=count:raise UnderfilledBatch(f'{task} round{round_number}: {len(items)}/10 after bounded fallbacks; no measurements started')
 return dict(items=items,rejected_generations=rejected,fsbo_diagnostics=diagnostics,retrieval=retrieval,random_draws=random_draws)

def progress(journal,state,number=None,partial=()):
 rows=state['rows']+[r['row'] for r in partial]
 atomic(journal.path.with_name('progress.json'),dict(stage='orpo_batch10',batch_size=10,completed_rounds=3+len(state['events']),optimization_rounds=len(state['events']),active_round=number,round_offset=len(partial),attempts=len(state['events'])*10+len(partial),rows=len(rows),best_runtime=best(rows),measurements=len(rows)-30))

def optimize(journal,task,recipe,policy,pool,generate,evaluate,fsbo,retrieve=None,stop_after=None,paused=lambda:False):
 validate_pool(task,pool,recipe['warm_start']['count'])
 state=journal.get('optimization_state',dict(rows=copy.deepcopy(pool),events=[],selector=selector()))
 while len(state['events'])<policy['rounds']-policy['warm_rounds']:
  if paused():raise SafePause('Paused at durable boundary')
  number=len(state['events'])+4;bandit=copy.deepcopy(state['selector']);selected,details=select_arm(bandit,state['events'],number)
  batch=journal.once(f'proposal_batch/{number}',dict(task=task,round_number=number,selected_arm=selected),lambda:build_batch(journal,task,recipe,policy,state['rows'],number,selected,generate,fsbo,retrieve))
  if len(batch['items'])!=policy['size']:raise ValueError('Wrong proposal batch length')
  results=[];prior=best(state['rows'])
  for slot,item in enumerate(batch['items']):
   if paused():raise SafePause('Paused before next SQL at durable boundary')
   plan=item['plan'];measured=journal.once(f'measurement/{number}/{slot}',dict(plan=plan),lambda:evaluate(plan),measurement=True)
   row=observed(plan,measured,len(state['rows'])+len(results),item['actual_arm'])
   results.append(dict(slot=slot,actual_arm=item['actual_arm'],plan=plan,status=measured['status'],row=row));progress(journal,state,number,results)
  arm_updates={}
  for arm in sorted({item['actual_arm'] for item in batch['items']}):
   exact=[r['row']['runtime'] for r in results if r['actual_arm']==arm and not r['row']['censored']];perf=-min(exact) if exact else float('-inf')
   if arm!='random':bandit.update(arm,perf,beat_incumbent=perf>-prior)
   arm_updates[arm]=dict(best_runtime=min(exact) if exact else None,measurements=sum(r['actual_arm']==arm for r in results),credited_to_selector=arm!='random')
  if selected not in arm_updates:bandit.record_attempt(selected)
  state['rows'].extend(r['row'] for r in results)
  event=dict(round=number,selected_arm=selected,actual_arms=sorted(arm_updates),selection=details,results=results,arm_updates=arm_updates,best_runtime=best(state['rows']),rejected_generations=batch['rejected_generations'],fsbo_diagnostics=batch['fsbo_diagnostics'],retrieval=batch['retrieval'],random_draws=batch['random_draws'])
  state['selector']=bandit;state['events'].append(event);journal.put('optimization_state',state);progress(journal,state)
  if stop_after is not None and len(state['events'])>=stop_after:break
 return state
