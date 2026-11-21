"""Paired transfer-prediction diagnostic on15fixed heldout warm pools; no SQL or checkpoint promotion."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,{int(os.environ['PROBE_CPU'])})
sys.path[:0]=['/root/data/cys/llmoptimizer_benchmark/tools','/root/data/cys/baseline/repos/FSBO']
import torch
from scipy.stats import spearmanr
from query_fsbo_model import QueryFSBOModel
OUT=Path(__file__).resolve().parent
NEW=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((OUT/'target_probe/manifest.json').read_text());results=[]
report={'at':time.time(),'selection':manifest['selection'],'oracle_calls':0,'production_promoted':False,'maximum_target_updates':10000,'models':{},'per_task':results,'limitations':['Saved BOLT30 candidate pool only; predictive transfer test rather than BO convergence.','One task per15templates; one model-training seed; diagnostic holdouts do not select production checkpoints.','Original representation80-source FSBO matched to retrained80-source split and labels.','Original normalization recomputed on same80sources; pretrained full788source baseline tested separately.']}
variants=[('original80',OUT/'original/fsbo/weights',OUT/'original/corpus/manifest.json','original'),('retrained80',NEW/'training/fsbo/weights',NEW/'training/corpus/manifest.json','retrained'),('original788',Path('/root/data/logs/query_transfer_train800_20261106_v1/fsbo/weights'),Path('/root/data/logs/query_transfer_train800_20261106_v1/corpus/manifest.json'),'original')]
variants=[v for v in variants if v[0]==os.environ['PROBE_MODEL']]
result_path=OUT/('target_probe_'+os.environ['PROBE_MODEL']+'_'+os.environ['PROBE_SHARD']+'.json')
previous=json.loads((OUT/'target_probe_results.json').read_text())['per_task']
for name,weights,mp,feature in variants:
 m=json.loads(mp.read_text());lower=np.asarray(m['lower']);span=np.subtract(m['upper'],lower)
 cfg={'seed':1,'fsbo_repo':'/root/data/cys/baseline/repos/FSBO','fsbo_weights':str(weights),'fsbo_weights_sha256':sha(weights),'target_max_updates':10000}
 model=QueryFSBOModel(cfg);report['models'][name]={'checkpoint':str(weights),'sha256':sha(weights),'corpus_manifest_sha256':sha(mp)}
 for idx,entry in enumerate(manifest['tasks']):
  if 'skipped' in entry or idx%2!=int(os.environ['PROBE_SHARD']):continue
  done=next((r for r in previous if r['model']==name and r['task']==entry['task']),None)
  if done:
   results.append(done);continue
  task=entry['task'];a=np.load(OUT/'target_probe'/(task+'.npz'));n=int(a['support_size']);x=((a[feature]-lower)/span).astype(np.float32);y=a['y'];score=-y
  seed=int(hashlib.sha256((task+'target_probe').encode()).hexdigest()[:7],16)
  fit=model.fit(x[:n],score[:n],seed)
  with torch.no_grad():
   p=model.model.likelihood(model.model.model(model.model.feature_extractor(torch.as_tensor(x[n:]))))
   truth=(score[n:]-model.offset)/model.scale
   mean=p.mean.numpy();var=p.variance.numpy()
   nll=float(-p.log_prob(torch.as_tensor(truth,dtype=torch.float32))/len(truth))
  rho=float(spearmanr(mean,truth)[0]) if np.ptp(mean)>0 and np.ptp(truth)>0 else None
  picked=int(np.argmax(mean));runtime_picked=float(y[n:][picked]);best=float(y[n:].min())
  row={'task':task,'model':name,'support':n,'query':len(truth),'spearman':rho,'rmse_normalized':float(np.sqrt(np.mean((mean-truth)**2))),'predictive_nll':nll,'selected_query_runtime':runtime_picked,'best_query_runtime':best,'selection_regret_ratio':runtime_picked/best,'fit_steps':fit['optimizer_updates'],'fit_seconds':fit['seconds']}
  results.append(row);print(json.dumps(row),flush=True)
  result_path.write_text(json.dumps(report,indent=2))
for name,*_ in variants:
 rows=[r for r in results if r['model']==name];vals=[r['spearman'] for r in rows if r['spearman'] is not None and np.isfinite(r['spearman'])]
 report['models'][name]['summary']={'tasks':len(rows),'mean_spearman':float(np.mean(vals)),'median_spearman':float(np.median(vals)),'mean_normalized_rmse':float(np.mean([r['rmse_normalized'] for r in rows])),'mean_predictive_nll':float(np.mean([r['predictive_nll'] for r in rows])),'geomean_selection_regret_ratio':float(np.exp(np.mean(np.log([r['selection_regret_ratio'] for r in rows]))))}
report['status']='complete';result_path.write_text(json.dumps(report,indent=2));print(json.dumps(report['models'],indent=2),flush=True)
