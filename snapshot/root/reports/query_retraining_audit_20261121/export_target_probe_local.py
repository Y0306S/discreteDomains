import hashlib,json,os,re,sys,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.025)
OUT=Path('/root/data/cys/tmp/query_vae_control_20261121');NEW=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117');NATIVE=Path('/root/data/cys/logs/query_train800_author_fixed_20261004/native')
sys.path.insert(0,str(NATIVE));from vae.model import VAEModule,encode_batch
pools=json.loads((NEW/'pools/batch10_frozen_pool.json').read_text())['tasks'];recipe=json.loads((NEW/'batch_recipe.json').read_text());families={}
for task in recipe['tasks']:families.setdefault(re.match(r'CEB_(\d+[A-Z])',task)[1],task)
folder=OUT/'target_probe';folder.mkdir(exist_ok=True);selected={};features={}
report={'selection':'first task per15templates; saved BOLT30 in generation order; first half up to10 uncensored distinct plans support, remainder query; no SQL or score-based task selection','at':time.time(),'oracle_calls':0,'meta_training_updates':0,'production_promoted':False,'tasks':[],'encoding':'direct GPU float32 highest, matching serving','checkpoints':{}}
for task in families.values():
 seen=set();rows=[]
 for r in pools[task]:
  key=tuple(r['plan'])
  if not r['censored'] and key not in seen:rows.append(r);seen.add(key)
 if len(rows)<4:report['tasks'].append({'task':task,'skipped':'fewer than4uncensored distinct warm plans'});continue
 selected[task]=rows;features[task]={}
for name,ckpt in [('original',NATIVE/'vae/CEB_64_author.ckpt'),('retrained',NEW/'training/vae/best.ckpt')]:
 model=VAEModule.load_from_checkpoint(str(ckpt),map_location='cuda:0').eval().requires_grad_(False)
 report['checkpoints'][name]={'path':str(ckpt),'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()}
 with torch.inference_mode():
  for task,rows in selected.items():
   mu,_=model.model.encode(encode_batch([r['plan'] for r in rows],model.vocab).cuda());features[task][name]=mu.reshape(-1,64).cpu().numpy().astype(np.float32)
 del model;torch.cuda.empty_cache();print('encoded',name,flush=True)
for task,rows in selected.items():
 n=min(10,len(rows)//2);np.savez(folder/(task+'.npz'),original=features[task]['original'],retrained=features[task]['retrained'],y=np.asarray([r['runtime'] for r in rows]),support_size=n)
 report['tasks'].append({'task':task,'support':n,'query':len(rows)-n,'plans_sha256':hashlib.sha256(json.dumps([r['plan'] for r in rows]).encode()).hexdigest()})
report['complete']=True;(folder/'manifest.json').write_text(json.dumps(report,indent=2));print('complete',len(selected),flush=True)
