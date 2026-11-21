"""Rebuild the FSBO corpus using the exact GPU/highest-precision serving path."""
import hashlib,json,os,sqlite3,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from train import load,atomic,digest,SOURCE
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
import peptide_transfer as shared
sys.path.insert(0,str(SOURCE/'native'))
from vae.model import VAEModule,encode_batch
def main():
 cfg=load(ROOT/'training/config.json');plans=load(ROOT/'training/plans.json');status=load(ROOT/'training/vae/status.json')
 assert status['status']=='complete' and status['reload_verified']
 os.sched_setaffinity(0,{24,25});torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 torch.cuda.set_per_process_memory_fraction(.06)
 checkpoint=ROOT/'training/vae/best.ckpt';assert digest(checkpoint)==status['checkpoint_sha256']
 model=VAEModule.load_from_checkpoint(str(checkpoint),map_location='cuda:0').eval().requires_grad_(False)
 folder=ROOT/'training/corpus';folder.mkdir(exist_ok=True);db=shared.connect(folder/'corpus.sqlite')
 lower=np.full(64,np.inf);upper=np.full(64,-np.inf);hashes={};source_hashes={};observations=0
 for task in cfg['source_tasks']:
  rows=plans['tasks'][task];features=[]
  with torch.inference_mode():
   for start in range(0,len(rows['plans']),128):
    mean,_=model.model.encode(encode_batch(rows['plans'][start:start+128],model.vocab).cuda());features.append(mean.reshape(-1,64).cpu().numpy().astype(np.float32))
  x=np.concatenate(features);assert x.shape==(len(rows['y']),64) and np.isfinite(x).all()
  payload=shared.pack_arrays(X=x,y=np.asarray(rows['y'],np.float64),source_rows=np.asarray(rows['source_rows'],np.int32))
  sha=hashlib.sha256(payload).hexdigest()
  with db:db.execute('INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?)',(task,len(x),rows['source_sha256'],sha,sqlite3.Binary(payload)))
  lower=np.minimum(lower,x.min(0));upper=np.maximum(upper,x.max(0));hashes[task]=sha;source_hashes[task]=rows['source_sha256'];observations+=len(x)
 manifest=dict(domain='ceb_query_runtime',policy='uncensored_first_distinct_plan_min_two',status='complete',purpose='production_training',
  representation='finetuned_train80_VAE_posterior_mean_64',normalization='selected_training_tasks_coordinate_minmax',
  dimension=64,source=str(SOURCE),source_tasks=cfg['source_tasks'],fsbo_train_tasks=cfg['fsbo_train_tasks'],fsbo_validation_tasks=cfg['fsbo_validation_tasks'],
  train_split_sha256=cfg['train_split_sha256'],test_split_sha256=cfg['test_split_sha256'],
  config_sha256=digest(ROOT/'training/config.json'),plans_sha256=digest(ROOT/'training/plans.json'),
  vae_checkpoint=str(checkpoint),vae_sha256=digest(checkpoint),lower=lower.tolist(),upper=upper.tolist(),
  task_hashes=hashes,source_hashes=source_hashes,observations=observations,oracle_calls=0,acquisition_wrapper=None,
  exporter_sha256=digest(__file__),device='cuda:0',float32_matmul_precision='highest')
 assert not set(manifest['source_tasks'])&set(cfg['excluded_test_tasks']) and np.all(upper>lower)
 with db:shared.put_meta(db,'manifest',manifest)
 db.close();atomic(folder/'manifest.json',manifest);print('GPU REEMBEDDED',len(hashes),observations,flush=True)
if __name__=='__main__':main()
