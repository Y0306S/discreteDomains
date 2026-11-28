import hashlib,json,shutil,sqlite3,time
from pathlib import Path
SRC=Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110')
HY=SRC.parent/'query_our_fsbo_vae80_test200_20261117'
ROOT=SRC.parent/'semanticopt_query_400_pilot800_20261123'
PLAN=Path('/root/reports/query_semanticopt_bolt_allocation_20261123/proposal.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def prefix(db):return hashlib.sha256(json.dumps(db.execute('SELECT n,value FROM records WHERE n<200 ORDER BY n').fetchall(),separators=(',',':')).encode()).hexdigest()
assert not ROOT.exists(),'Do not overwrite an existing continuation'
plan=json.loads(PLAN.read_text());tasks=plan['tasks'];pilot=plan['semanticopt']['pilot_tasks']
for sub in ['code','services','query/services','logs','evidence','control','leases']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
for folder in ['code','services']:
 for p in (SRC/folder).glob('*.py'):shutil.copy2(p,ROOT/folder/p.name)
for name in ['recipe.json','runtime_policy.json','batch_policy.json','budget_policy.json','query_policy.json','query_cache_policy.json','peptide_oracle_policy.json']:shutil.copy2(SRC/name,ROOT/name)
shutil.copy2(SRC/'identity.json',ROOT/'evidence/source_identity.json')
rounds=json.loads((SRC/'query_round_policy.json').read_text());rounds.update(rounds=80,authorization='User authorized launch 2026-11-23: all 200 queries to 400 proposals, first ten to 800; batch size ten',scope='Isolated continuation; per-task cap is pinned in extension.json')
write(ROOT/'query_round_policy.json',rounds)
ext=dict(source=str(SRC),tasks=tasks,pilot_tasks=pilot,source_budget=200,default_budget=400,pilot_budget=800,additional_evaluations=44000,new_budget_policy_sha256=sha(ROOT/'query_round_policy.json'),source_budget_policy_sha256=sha(SRC/'query_round_policy.json'),authorized_at=time.time(),batch_size=10)
write(ROOT/'extension.json',ext)
p=ROOT/'code/evaluate.py';s=p.read_text();old="    recipe=domain_recipe(verify(),domain);settings=recipe[domain]";assert old in s
s=s.replace(old,old+"\n    if domain == 'query':\n        extension=load(ROOT/'extension.json')\n        target=extension['pilot_budget'] if task in extension['pilot_tasks'] else extension['default_budget']\n        recipe['query']=dict(settings,proposals=target,rounds=target//10)\n        settings=recipe['query']",1);p.write_text(s)
# Retain exact SemanticOPT generation behavior; only add explicit model placement.
p=ROOT/'services/hf_qwen_client.py';s=p.read_text().replace('device_map=device_map,\n                attn_implementation=',"device_map=getattr(self, 'device_map', device_map),\n                attn_implementation=");s=s.replace('return self._model.get_input_embeddings().weight.device',"return getattr(self, 'input_device', self._model.get_input_embeddings().weight.device)");p.write_text(s)
# Proven sharding/admission/overflow implementation, original cache/client retained.
shutil.copy2(HY/'batch_code/qwen_inference_server.py',ROOT/'services/qwen_inference_server.py')
identities=json.loads((SRC/'query/services/client.json').read_text())['identities']
pairs=plan['gpus']['LLMOptimizer3']['gpu_pairs'];replicas=[]
for i in range(20):
 local=i<12;gpu=i//3 if local else None
 cfg=dict(name=f'semanticopt_query_cont_replica{i:02}',base=identities[0]['base'],adapter=identities[0]['adapter'],identities=identities,cpus=[32+i%8] if local else [20+(i-12)%8],port=23100+i,memory_gib=30 if local else 14.4,max_input_tokens=15360 if local else 8192,max_new_tokens=16384,prompt_cache=dict(enabled=True,max_bytes=1610612736,min_tokens=128),cuda_allocator='expandable_segments:True')
 if local:cfg.update(gpu=gpu,gpu_admission=dict(lock_file=str(ROOT/f'query/services/gpu_admission/gpu{gpu}.lock'),shared_max_input_tokens=0,long_context='exclusive until canary'))
 else:cfg.update(gpu_devices=pairs[i-12],text_layer_splits=[18],cpu_embeddings=True,fallback_max_input_tokens=15360,overflow_replicas=[f'http://127.0.0.1:{23100+k}' for k in range(12)])
 write(ROOT/f'query/services/replica{i:02}.json',cfg);replicas.append(dict(index=i,host='LLMOptimizer2' if local else 'LLMOptimizer3',config=str(ROOT/f'query/services/replica{i:02}.json'),port=cfg['port']))
# Key used by existing overflow implementation.
for item in replicas[12:]:
 p=Path(item['config']);cfg=json.loads(p.read_text());cfg['overflow_replicas']=cfg['overflow_replicas'];write(p,cfg)
write(ROOT/'query/services/broker.json',dict(name='semanticopt_query_cont_broker',port=23090,cpus=[39],identities=identities,replicas=[f'http://127.0.0.1:{23100+i}' for i in range(20)],journal=str(ROOT/'query/services/requests.sqlite')))
write(ROOT/'query/services/client.json',dict(url='http://127.0.0.1:23090',identities=identities,request_timeout_seconds=7200))
resources=json.loads((HY/'resources.json').read_text());workers=[]
for host,n in [('LLMOptimizer2',8),('LLMOptimizer4',12)]:
 for idx,w in enumerate([w for w in resources['workers'] if w['host']==host][:n]):workers.append(dict(w,method='semanticopt',slot=len(workers),optimizer_cpu=4*idx+3,postgres_cpus=list(range(4*idx,4*idx+3))))
bolt=[]
for host,n,offset in [('LLMOptimizer4',3,48),('LLMOptimizer3',4,0)]:
 used={w['index'] for w in workers}
 for idx,w in enumerate([w for w in resources['workers'] if w['host']==host and w['index'] not in used][:n]):bolt.append(dict(w,method='bolt',slot=len(bolt),optimizer_cpu=offset+4*idx+3,postgres_cpus=list(range(offset+4*idx,offset+4*idx+3))))
write(ROOT/'allocation.json',dict(workers=workers,bolt_workers=bolt,replicas=replicas,hybrid_source=str(HY),service_cpus={'LLMOptimizer2':list(range(32,40)),'LLMOptimizer4':list(range(60,64)),'LLMOptimizer3':list(range(20,28))}))
audits=[]
for task in tasks:
 source=SRC/'query/runs'/task;dest=ROOT/'query/runs'/task;dest.mkdir(parents=True)
 db=sqlite3.connect(f'file:{source}/observations.sqlite?mode=ro',uri=True)
 def get(k):
  r=db.execute('SELECT value FROM state WHERE key=?',(k,)).fetchone();return json.loads(r[0]) if r else None
 assert db.execute('SELECT count(*) FROM records').fetchone()[0]==200
 assert get('pending') is None and len(get('initial'))==49
 assert get('budget_policy_sha256')==ext['source_budget_policy_sha256']
 h=prefix(db);states=db.execute('SELECT key,value FROM state ORDER BY key').fetchall();dst=sqlite3.connect(dest/'observations.sqlite');db.backup(dst)
 assert dst.execute('SELECT key,value FROM state ORDER BY key').fetchall()==states
 amendment=dict(at=time.time(),source_checkpoint=str(source/'observations.sqlite'),source_prefix_sha256=h,old_budget_policy_sha256=ext['source_budget_policy_sha256'],new_budget_policy_sha256=ext['new_budget_policy_sha256'],retained_evaluations=200,new_budget=800 if task in pilot else 400)
 with dst:
  dst.execute('INSERT OR REPLACE INTO state VALUES (?,?)',('budget_policy_sha256',json.dumps(ext['new_budget_policy_sha256'])))
  dst.execute('INSERT INTO state VALUES (?,?)',('extension_amendment',json.dumps(amendment)))
 assert prefix(dst)==h;dst.close();db.close()
 write(dest/'status.json',dict(status='queued',task=task,evaluations=200,budget=amendment['new_budget'],completed_rounds=20,batch_size=10))
 audits.append(dict(task=task,**amendment,pending=None,initial_count=49))
write(ROOT/'evidence/import_audit.json',dict(passed=True,tasks=audits,retained_evaluations=40000))
files={name:h for name,h in json.loads((SRC/'identity.json').read_text())['files'].items() if not name.startswith('/root/reports/')}
for sub in ['code','services']:
 for p in (ROOT/sub).glob('*.py'):files[str(p)]=sha(p)
for p in ROOT.glob('*.json'):
 if p.name!='identity.json':files[str(p)]=sha(p)
for p in (ROOT/'query/services').glob('*.json'):files[str(p)]=sha(p)
write(ROOT/'identity.json',dict(files=files,source_campaign=str(SRC),source_identity_sha256=sha(SRC/'identity.json')))
print(json.dumps(dict(root=str(ROOT),imported_tasks=200,retained_evaluations=40000,additional=44000)))
