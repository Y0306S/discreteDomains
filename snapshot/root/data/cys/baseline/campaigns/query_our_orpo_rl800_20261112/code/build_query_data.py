"""Export all 800 source histories without oracle calls, preserving source measurement identity."""
import json,math,itertools,time,sys,sqlite3
from pathlib import Path
from collections import Counter,defaultdict
from query_data import *
from Levenshtein import distance
SFT=ROOT.parent/'query_warmstart_full800_20261112'
sys.path.insert(0,str(ROOT.parent/'peptide_our_orpo_rl800_20261111/code'))
from select_grpo_tasks import ml_order

def negative(chosen,pool):
 gaps=[r['runtime']-chosen['runtime'] for r in pool];lo,hi=min(gaps),max(gaps);scores=[]
 for r,g in zip(pool,gaps):
  sim=1-distance(chosen['plan'],r['plan'])/max(len(chosen['plan']),len(r['plan']));norm=1. if hi<=lo+1e-15 else (g-lo)/(hi-lo)
  scores.append((r,sim,norm))
 front=[v for v in scores if not any(w[1]>=v[1] and w[2]>=v[2] and (w[1]>v[1] or w[2]>v[2]) for w in scores)]
 r,sim,norm=max(front,key=lambda v:(.5*(v[1]+v[2]),v[1],v[2],-v[0]['observation_index']))
 return r,dict(similarity_to_chosen=sim,normalized_worse_gap=norm,pool_size=len(pool),pareto_size=len(front),rule='ML 0.5 sequence-edit similarity of integer plan + 0.5 normalized runtime gap')

def one(task):
 dest=ROOT/'assets/histories'/f'{task}.json';pairs_path=ROOT/'data/tasks'/f'{task}.json'
 if dest.exists() and pairs_path.exists():return json.loads(dest.read_text()),json.loads(pairs_path.read_text())
 source=SOURCE/'outputs'/task/'lolbo/seed0/history.json';raw=source.read_bytes();h=json.loads(raw);meta=h['meta_info']
 assert h['task_id']==task and meta['status']=='complete' and meta['split']=='train' and len(h['observations'])==1049
 assert digest(definition(task)[3])==meta['sql_sha256']
 cache={};seen=set();records=[]
 for i,o in enumerate(h['observations']):
  e=o['extra_info'];runtime=float(o['objectives'][0]);assert e['charged_index']==i and e['oracle']=='postgres_runtime'
  assert runtime==e['runtime'] and runtime>0 and math.isfinite(runtime) and o['trial_state']==int(e['censored'])
  rawplan=tuple(e['plan'])
  if rawplan not in cache:cache[rawplan]=canonicalize(task,list(rawplan))
  plan=cache[rawplan];key=plan_key(plan)
  # A censored first attempt does not prevent a subsequent actual completed measurement becoming training evidence.
  feasible=not bool(e['censored']);first=feasible and key not in seen
  if first:seen.add(key)
  records.append(dict(observation_index=i,plan=plan,runtime=runtime,censored=bool(e['censored']),first_occurrence=first,feasible=feasible,round=0 if i<49 else (i-49)//10+1,status=e['status']))
 valid=[r for r in records if r['first_occurrence']];pairs=[]
 for turn in range(1,101):
  start=49+(turn-1)*10;prefix=[r for r in valid if r['observation_index']<start]
  if not prefix:continue
  best=min(r['runtime'] for r in prefix);old={plan_key(r['plan']) for r in records[:start]}
  pos=[r for r in valid if start<=r['observation_index']<start+10 and r['runtime']<best and plan_key(r['plan']) not in old]
  pool=[r for r in valid if r['observation_index']>=start and r['runtime']>=best and plan_key(r['plan']) not in old]
  if not pos or not pool:continue
  chosen=min(pos,key=lambda r:(r['runtime'],r['observation_index']));rejected,nmeta=negative(chosen,pool)
  messages,ctx=build_prompt(task,records[:start],round_number=turn)
  pairs.append(dict(conversations=[{'from':'system','value':messages[0]['content']},{'from':'human','value':messages[1]['content']}],chosen={'from':'gpt','value':canonical(dict(plan=chosen['plan']))},rejected={'from':'gpt','value':canonical(dict(plan=rejected['plan']))},metadata=dict(task=task,source_round=turn,prefix_end_exclusive=start,current_best_runtime=best,chosen_index=chosen['observation_index'],rejected_index=rejected['observation_index'],chosen_runtime=chosen['runtime'],rejected_runtime=rejected['runtime'],negative_selection=nmeta,source_history_sha256=hashlib.sha256(raw).hexdigest())))
 out=dict(task=task,reference=task,source_history_sha256=hashlib.sha256(raw).hexdigest(),records=records,completed_acquisition_rounds=100)
 atomic(dest,out);atomic(pairs_path,pairs)
 return out,pairs

def main():
 train=(SOURCE/'train800.txt').read_text().split();test=(SOURCE/'test200.txt').read_text().split();assert len(train)==800 and len(test)==200 and not set(train)&set(test)
 allpairs=[];sft=[];prov=[];stats={};cards={};counts={};sources={};started=time.time()
 from concurrent.futures import ProcessPoolExecutor,as_completed
 pending=[t for t in train if not (ROOT/'assets/histories'/f'{t}.json').exists() or not (ROOT/'data/tasks'/f'{t}.json').exists()]
 with ProcessPoolExecutor(max_workers=4) as ex:
  futures=[ex.submit(one,t) for t in pending]
  for i,f in enumerate(as_completed(futures)):
   f.result()
   if (i+1)%20==0:print(json.dumps(dict(parallel_source_tasks=i+1,remaining_initial=len(pending))),flush=True)
 for ti,task in enumerate(train):
  h,pairs=one(task);allpairs.extend(pairs);records=h['records'];valid=[r for r in records if r['first_occurrence']];sources[task]=h['source_history_sha256'];counts[task]=dict(uncensored_distinct=len(valid),pairs=len(pairs))
  ranked=sorted(valid,key=lambda r:(r['runtime'],r['observation_index']))
  msgs,_=build_prompt(task,[],sft=True)
  for rank,r in enumerate(ranked[:10],1):
   sft.append(dict(messages=msgs+[dict(role='assistant',content=canonical(dict(plan=r['plan'])))]));prov.append(dict(task_id=task,rank=rank,source_index=r['observation_index'],runtime=r['runtime'],plan=r['plan'],source_sha256=h['source_history_sha256']))
  cards[task]=dict(task=task_description(task),top=[dict(candidate=dict(plan=r['plan']),measured_value=r['runtime']) for r in ranked[:3]],bottom=[dict(candidate=dict(plan=r['plan']),measured_value=r['runtime']) for r in ranked[-3:]])
  best=min(r['runtime'] for r in valid if r['round']==0);curve={0:best};events=[]
  for turn in range(1,101):
   v=min((r['runtime'] for r in valid if r['round']==turn),default=best)
   if v<best-1e-12:events.append(turn)
   best=min(best,v);curve[turn]=best
  pct=lambda a,b:(curve[a]-curve[b])/max(abs(curve[a]),1e-12)*100
  desc=task_description(task);joins=desc['constraints']['joins'];template=task.split('_')[1].split('A')[0]+'A' if 'A' in task else re.match(r'CEB_(\d+[A-Z])',task)[1]
  stats[task]=dict(late_n_new=sum(40<=r<=80 for r in events)+sum(80<=r<=100 for r in events),late_pct_sum=pct(40,80)+pct(80,100),pct_80_100=pct(80,100),n_new_40_80=sum(40<=r<=80 for r in events),n_new_80_100=sum(80<=r<=100 for r in events),pct_40_80=pct(40,80),join_count=joins,template=template,diversity_bin=f'{template}|joins{joins}',history_sha256=digest(ROOT/'assets/histories'/f'{task}.json'))
  if (ti+1)%20==0:
   progress=dict(status='building',tasks=ti+1,pairs=len(allpairs),sft_examples=len(sft),seconds=time.time()-started);atomic(ROOT/'reports/build_progress.json',progress);print(json.dumps(progress),flush=True)
 eligible={t:s for t,s in stats.items() if s['late_n_new']>=1 and s['pct_80_100']>0}
 if len(eligible)<32:raise ValueError(f'Only {len(eligible)} late improving queries; need32')
 ml_order(eligible);bins=defaultdict(list)
 for t in ml_order(eligible):bins[eligible[t]['diversity_bin']].append(t)
 picked=[]
 while len(picked)<32:
  for b in sorted(bins,key=lambda b:(not any(eligible[t]['pareto_front'] for t in bins[b]),-max((eligible[t]['rank_score'] for t in bins[b]),default=-1),b)):
   if bins[b] and len(picked)<32:picked.append(bins[b].pop(0))
 selection=dict(tasks=picked,task_count=32,eligible_tasks=len(eligible),source_training_tasks=800,heldout_tasks_used=0,criterion='Same ML late-improvement 2D Pareto then mean percentile rank; round-robin template and join-count diversity; positive improvement80-100',selected_statistics={t:eligible[t] for t in picked},all_statistics=stats,diversity_bin_counts=dict(Counter(eligible[t]['diversity_bin'] for t in picked)))
 atomic(ROOT/'assets/grpo_task_selection.json',selection);settings=json.loads((ROOT/'settings.json').read_text());jobs=[]
 for phase in settings['phases']:
  for task in picked:
   for k in phase['prefixes']:jobs.append(dict(task_id=task,phase=phase['id'],scenario=phase['scenario'],prefix_k=k,prefix_end_exclusive=49+k*10,repeat=0,group_id=f"{phase['id']}/{task}/k{k}"))
 atomic(ROOT/'assets/jobs.json',jobs);atomic(ROOT/'assets/tasks.json',[dict(task_id=t) for t in train]);atomic(ROOT/'assets/source_cards.json',cards)
 for dest,rows in [(ROOT/'data/orpo_train.jsonl',allpairs),(SFT/'data/train.jsonl',sft),(SFT/'data/provenance.jsonl',prov)]:
  with dest.open('w') as f:
   for r in rows:f.write(canonical(r)+'\n')
 atomic(ROOT/'data/dataset_info.json',{'query_our_single':dict(file_name='orpo_train.jsonl',formatting='sharegpt',ranking=True,columns=dict(messages='conversations',chosen='chosen',rejected='rejected'))})
 atomic(SFT/'data/dataset_info.json',{'bolt_query_top10_train800':dict(file_name='train.jsonl',formatting='sharegpt',columns=dict(messages='messages'),tags=dict(role_tag='role',content_tag='content',user_tag='user',assistant_tag='assistant',system_tag='system'))})
 manifest=dict(status='built_pending_independent_audit',tasks_scanned=800,pairs=len(allpairs),tasks_with_pairs=sum(c['pairs']>0 for c in counts.values()),sft_examples=len(sft),sft_tasks=len(cards),sft_tasks_below10=sum(c['uncensored_distinct']<10 for c in counts.values()),split=dict(train=train,test=test),source=str(SOURCE),source_history_hashes=sources,counts=counts,oracle_calls=0,training_test_overlap=0,grpo_groups=len(jobs),eligible_late_improvers=len(eligible),seconds=time.time()-started)
 atomic(ROOT/'manifest.json',manifest);atomic(SFT/'data/manifest.json',manifest)
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['split','source_history_hashes','counts']}),flush=True)
if __name__=='__main__':main()
