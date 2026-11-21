import hashlib,json,re,time,urllib.request
from pathlib import Path
import numpy as np
OUT=Path('/root/data/cys/tmp/query_vae_control_20261121')
NEW=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
pools=json.loads((NEW/'pools/batch10_frozen_pool.json').read_text())['tasks'];recipe=json.loads((NEW/'batch_recipe.json').read_text())
families={}
for task in recipe['tasks']:families.setdefault(re.match(r'CEB_(\d+[A-Z])',task)[1],task)
token=(NEW/'services/token').read_text().strip()
folder=OUT/'target_probe';folder.mkdir(exist_ok=True)
report={'selection':'first task per15templates; saved BOLT30 observations in generation order; first half up to10 uncensored distinct plans support, remainder query; no SQL or score-based task selection','at':time.time(),'oracle_calls':0,'meta_training_updates':0,'production_promoted':False,'tasks':[]}
for task in families.values():
 rows=[];seen=set()
 for row in pools[task]:
  key=tuple(row['plan'])
  if not row['censored'] and key not in seen:rows.append(row);seen.add(key)
 if len(rows)<4:
  report['tasks'].append({'task':task,'skipped':'fewer than4distinct uncensored warm observations'});continue
 plans=[r['plan'] for r in rows];features={}
 for name,op in [('original','encode_retrieval_mean'),('retrained','encode_mean')]:
  req=urllib.request.Request('http://127.0.0.1:21843',data=json.dumps({'op':op,'plans':plans,'task':task,'seed':61021}).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as response:value=json.load(response)
  features[name]=np.asarray(value['z'],dtype=np.float32)
 y=np.array([r['runtime'] for r in rows]);n=min(10,len(rows)//2)
 np.savez(folder/(task+'.npz'),original=features['original'],retrained=features['retrained'],y=y,support_size=n)
 report['tasks'].append({'task':task,'support':n,'query':len(rows)-n,'plans_sha256':hashlib.sha256(json.dumps(plans).encode()).hexdigest()})
 print(task,n,len(rows)-n,flush=True)
 (folder/'manifest.json').write_text(json.dumps(report,indent=2))
