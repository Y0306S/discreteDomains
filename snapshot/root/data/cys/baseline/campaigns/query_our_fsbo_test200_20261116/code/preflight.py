import json,os,sys,time,copy
from pathlib import Path
from query_protocol import atomic,digest
from query_data import build_prompt,plan_key
from method import Journal,warm_start,novel,observed,seed_for,best
from backends import LLM,oracle,FSBO
ROOT=Path(__file__).resolve().parents[1]
os.sched_setaffinity(0,{24})
cfg=json.loads((ROOT/'recipe.json').read_text());task=cfg['train_tasks'][0]
assert task not in cfg['tasks']
small=copy.deepcopy(cfg);small['warm_start']['count']=3;small['warm_start']['maximum_samples']=30
journal=Journal(ROOT/'evidence/preflight.sqlite',dict(recipe_sha256=digest(ROOT/'recipe.json'),task=task))
evaluate=oracle(task,50)
print('Training-only preflight',task,flush=True)
llm=LLM(cfg,'warm_start','http://127.0.0.1:21810')
state=warm_start(journal,task,small,llm,evaluate)
print('BOLT',len(state['rows']),'rows',state['attempts'],'generations; best',best(state['rows']),flush=True)
rows=list(state['rows']);llm=LLM(cfg,'orpo','http://127.0.0.1:21811');orpo=None
for i in range(10):
 messages,_=build_prompt(task,rows,scenario='normal',round_number=1)
 request=dict(stage='orpo',task=task,attempt=1,retry=i,messages=messages,seed=seed_for(task,'preflight_orpo',i))
 reply=journal.once(f'orpo_generation/{i}',request,lambda:llm(request))
 try:p=novel(reply['text'],task,{plan_key(r['plan']) for r in rows})
 except (ValueError,TypeError,KeyError,IndexError) as exc:print('ORPO rejected',str(exc),reply['text'][:400],flush=True);continue
 result=journal.once(f'orpo_measurement/{i}',dict(plan=p),lambda:evaluate(p),measurement=True);orpo=observed(p,result,len(rows),'preflight_orpo');rows.append(orpo);break
if orpo is None:raise ValueError('ORPO failed to produce a distinct valid training plan')
print('ORPO real measurement',orpo['runtime'],orpo['censored'],flush=True)
f=FSBO(cfg,task,rows)
try:
 proposal=journal.once('fsbo_proposal',dict(rows=rows),lambda:f(rows,12345))
 if proposal['plan'] is None:raise ValueError('FSBO failed smoke decode')
 result=journal.once('fsbo_measurement',dict(plan=proposal['plan']),lambda:evaluate(proposal['plan']),measurement=True)
 print('FSBO real measurement',result,flush=True)
finally:f.close()
report=dict(passed=True,task=task,heldout_evaluations=0,warm_rows=state['rows'],orpo_row=orpo,fsbo_proposal=proposal,fsbo_measurement=result,recipe_sha256=digest(ROOT/'recipe.json'),at=time.time())
atomic(ROOT/'evidence/preflight.json',report);journal.close();print('PREFLIGHT PASSED',flush=True)
