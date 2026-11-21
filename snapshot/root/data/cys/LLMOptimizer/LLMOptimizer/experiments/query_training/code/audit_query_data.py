"""Independent source, causal-label, split and top-ten validation."""
import json,math,time
from collections import Counter,defaultdict
from query_data import *
SFT=ROOT.parent/'query_warmstart_full800_20261112'
def main():
 m=json.loads((ROOT/'manifest.json').read_text());train=set((SOURCE/'train800.txt').read_text().split());test=set((SOURCE/'test200.txt').read_text().split())
 assert len(train)==800 and len(test)==200 and not train&test and train==set(m['split']['train'])
 pairs=[json.loads(x) for x in (ROOT/'data/orpo_train.jsonl').open()];by=defaultdict(list)
 for p in pairs:by[p['metadata']['task']].append(p)
 sf=[json.loads(x) for x in (SFT/'data/train.jsonl').open()];prov=[json.loads(x) for x in (SFT/'data/provenance.jsonl').open()];assert len(sf)==len(prov)
 sp=defaultdict(list)
 for p,r in zip(sf,prov):sp[r['task_id']].append((p,r))
 assert set(sp)==train and set(by)<=train
 for task in sorted(train):
  h=json.loads((ROOT/'assets/histories'/f'{task}.json').read_text());source=SOURCE/'outputs'/task/'lolbo/seed0/history.json';raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==m['source_history_hashes'][task]==h['source_history_sha256'];original=json.loads(raw)['observations']
  rows=h['records'];valid=[r for r in rows if r['first_occurrence'] and not r['censored']];assert len(valid)==len({plan_key(r['plan']) for r in valid})
  arm_indices={p['metadata'][k] for p in by[task] for k in ['chosen_index','rejected_index']}|{r['source_index'] for _,r in sp[task]}
  for i in arm_indices:
   row=rows[i];o=original[i];assert row['observation_index']==i and not o['extra_info']['censored'] and row['runtime']==o['objectives'][0]
   d,c,_,_=definition(task);target_tree=c.decode(d.query_tables,row['plan'].copy());source_tree=c.decode(d.query_tables,o['extra_info']['plan'].copy());assert target_tree.equal(source_tree);canonicalize(task,row['plan'],strict=True)
  for p in by[task]:
   meta=p['metadata'];start=meta['prefix_end_exclusive'];chosen=rows[meta['chosen_index']];rejected=rows[meta['rejected_index']];prefix=rows[:start];old={plan_key(r['plan']) for r in prefix};best=min(r['runtime'] for r in prefix if r['first_occurrence'] and not r['censored'])
   assert start==49+(meta['source_round']-1)*10 and start<=chosen['observation_index']<start+10 and rejected['observation_index']>=start
   assert chosen['runtime']<best<=rejected['runtime'] and best==meta['current_best_runtime']
   assert plan_key(chosen['plan']) not in old and plan_key(rejected['plan']) not in old
   assert plan_key(chosen['plan'])!=plan_key(rejected['plan'])
   assert json.loads(p['chosen']['value'])==dict(plan=chosen['plan']) and json.loads(p['rejected']['value'])==dict(plan=rejected['plan'])
   context=json.loads(p['conversations'][1]['value']);indices={r['observation_index'] for k in ['good','bad','recent50'] for r in context[k]}
   assert indices and max(indices)<start and context['current_task']==task_description(task)
   assert context['recent50']==[compact(r) for r in [r for r in prefix if r['first_occurrence'] and not r['censored']][-50:]]
   assert all(context[k]==[compact(r) for r in (sorted([v for v in prefix if v['first_occurrence'] and not v['censored']],key=lambda r:(r['runtime'],r['observation_index']))[:5] if k=='good' else sorted([v for v in prefix if v['first_occurrence'] and not v['censored']],key=lambda r:(r['runtime'],r['observation_index']))[-5:])] for k in ['good','bad'])
  expected=sorted(valid,key=lambda r:(r['runtime'],r['observation_index']))[:10];assert len(sp[task])==len(expected)
  for (p,r),e,rank in zip(sp[task],expected,range(1,11)):
   assert r['rank']==rank and r['runtime']==e['runtime'] and r['source_index']==e['observation_index'] and r['plan']==e['plan']
   assert p['messages'][-1]==dict(role='assistant',content=canonical(dict(plan=e['plan'])))
   assert p['messages'][:2]==build_prompt(task,[],sft=True)[0]
 selection=json.loads((ROOT/'assets/grpo_task_selection.json').read_text());jobs=json.loads((ROOT/'assets/jobs.json').read_text());settings=json.loads((ROOT/'settings.json').read_text());picked=selection['tasks'];assert len(picked)==len(set(picked))==32 and set(picked)<=train and len(jobs)==384
 for t in picked:assert selection['selected_statistics'][t]['pct_80_100']>0
 for phase in settings['phases']:assert {(j['task_id'],j['prefix_k']) for j in jobs if j['phase']==phase['id']}=={(t,k) for t in picked for k in phase['prefixes']}
 evidence=dict(passed=True,at=time.time(),pairs=len(pairs),pair_tasks=sum(bool(v) for v in by.values()),sft_examples=len(sf),sft_tasks=len(sp),train_test_overlap=0,oracle_calls=0,causal_prefix_verified=True,measured_uncensored_targets_verified=True,strict_plans_verified=True,top_ten_per_task_verified=True,data_sha256=digest(ROOT/'data/orpo_train.jsonl'),sft_sha256=digest(SFT/'data/train.jsonl'),source_hashes_verified=800)
 atomic(ROOT/'reports/independent_audit.json',evidence);atomic(SFT/'evidence/data_audit.json',evidence);atomic(ROOT/'evidence/grpo_scope_check.json',dict(passed=True,tasks=32,groups=384,selection_sha256=digest(ROOT/'assets/grpo_task_selection.json'),jobs_sha256=digest(ROOT/'assets/jobs.json')))
 print(json.dumps(evidence),flush=True)
if __name__=='__main__':main()
