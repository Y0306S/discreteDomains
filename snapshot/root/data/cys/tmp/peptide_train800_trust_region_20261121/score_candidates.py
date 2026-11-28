"""Fixed-pool APEX evaluation with fresh initial baselines; not sequential BO."""
import gzip,hashlib,json,os,sys,time
from pathlib import Path
os.sched_setaffinity(0,{42,43})
OUT=Path(__file__).resolve().parent
ROOT=Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119')
APEX=Path('/root/data/cys/BOLT/optimization/peptides/apex_oracle')
sys.path.insert(0,str(APEX))
import torch,numpy as np
from utils import make_vocab,onehot_encoding
import APEX_models
torch.set_num_threads(2);torch.set_num_interop_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.15)
paths=list((APEX/'APEX_pathogen_models').glob('APEX_*'));assert len(paths)==8
models=[torch.load(p,map_location='cpu',weights_only=False).cuda().eval() for p in paths]
vocab,_=make_vocab()
cfg=json.loads((ROOT/'config.json').read_text());tasks=sorted(cfg['train_tasks'])
pool=json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/initialization.json').read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def ledger(d):
 with (OUT/'oracle_prediction_ledger.jsonl').open('a') as f:f.write(json.dumps(d)+'\n');f.flush();os.fsync(f.fileno())
def predict(seqs):
 result=[]
 with torch.inference_mode():
  for start in range(0,len(seqs),128):
   x=torch.as_tensor(onehot_encoding(seqs[start:start+128],52,vocab),device='cuda',dtype=torch.long)
   total=None
   for model in models:
    # Native APEX performs the power transformation in numpy, after copying to CPU.
    pred=10**(6-model(x).cpu().numpy())
    total=pred if total is None else total+pred
   result.extend((total/len(models))[:,0].astype(float).tolist())
 assert len(result)==len(seqs) and np.isfinite(result).all() and all(v>0 for v in result)
 return result
identity=dict(oracle_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 precision='fp32_cuda_no_tf32_native_numpy_transform',target='native APEX ensemble bacteria0 MIC',cohort='training800',selection='fixed generated pools; no GP selection or feedback',baseline='rescore all100 initial sequences on same oracle/device')
write(OUT/'candidate_scoring_identity.json',identity)
# Verify directly against the unmodified native predict_APEX function on this device.
import ast
node=ast.parse((APEX/'APEX_predict.py').read_text())
func=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='predict_APEX')
import math
ns=dict(APEX_models=models,batch_size=3000,onehot_encoding=onehot_encoding,max_len=52,word2idx=vocab,torch=torch,np=np,math=math)
exec(compile(ast.Module(body=[func],type_ignores=[]),str(APEX/'APEX_predict.py'),'exec'),ns)
checks=[pool[t]['sequences'][0] for t in tasks[:32]]
ledger(dict(phase='native_parity',state='requested',predictions=64,time=time.time()))
a=np.array(predict(checks));b=ns['predict_APEX'](checks)[:,0]
assert np.allclose(a,b,rtol=1e-6,atol=1e-6),(a,b)
write(OUT/'native_scorer_parity.json',dict(status='passed',sequences=32,physical_predictions=64,max_relative_error=float(np.max(np.abs(a-b)/b)),native_source_sha256=hashlib.sha256((APEX/'APEX_predict.py').read_bytes()).hexdigest()))
ledger(dict(phase='native_parity',state='complete',predictions=64,time=time.time()))
start=time.monotonic();calls=0;finished=set()
(OUT/'scored').mkdir(exist_ok=True)
while len(finished)<800:
 progress=False
 for task in tasks:
  if task in finished:continue
  dest=OUT/'scored'/(task+'.json')
  if dest.exists():
   old=json.loads(dest.read_text());assert old['status']=='complete' and old['baseline']=='fresh_rescored100';calls+=old['physical_candidate_predictions'];finished.add(task);continue
  source=OUT/'tasks/adapted'/(task+'.json')
  if not source.exists():continue
  meta=json.loads(source.read_text())
  with gzip.open(source.with_suffix('.jsonl.gz'),'rt') as f:draws=[json.loads(line) for line in f]
  seqs=list(dict.fromkeys(r['sequence'] for r in draws if r['valid'] and r['novel']))
  init=pool[task]['sequences'];assert len(init)==100
  ledger(dict(phase='fixed_pool',task=task,state='requested',candidate_predictions=len(seqs),initial_predictions=100,time=time.time()))
  vals=predict(init+seqs)
  expected=-np.array(pool[task]['scores'],dtype=float)
  actual=np.array(vals[:100]);relative=np.abs(actual-expected)/expected
  # Historical precision/hardware agreement is diagnostic, not the new baseline.
  # A1% guard catches substantial oracle/configuration drift; all errors retained.
  assert np.max(relative)<.01,(task,float(np.max(relative)))
  ledger(dict(phase='fixed_pool',task=task,state='complete',candidate_predictions=len(seqs),initial_predictions=100,time=time.time()))
  pred=dict(zip(seqs,vals[100:]));calls+=len(seqs)
  initial_best=float(min(actual));measures=[]
  for radius in sorted({r['radius'] for r in draws}):
   for k in [1,2,3,100]:
    selected=list(dict.fromkeys(r['sequence'] for r in draws if r['radius']==radius and r['valid'] and r['novel'] and r['edits_center']<=k))
    scores=[pred[s] for s in selected]
    for budget in [5,10,20,100000]:
     prefix=scores[:budget];best=min([initial_best]+prefix)
     measures.append(dict(radius=radius,local_edits=k,nominal_candidate_budget=budget,actual_candidates_scored=min(len(scores),budget),filled_budget=len(scores)>=budget,best_mic=best,best_over_initial=best/initial_best))
  write(dest,dict(status='complete',task=task,reference_length=meta['reference_length'],global_edit_budget=meta['global_edit_budget'],development_split=meta['development_split'],baseline='fresh_rescored100',initial_best_mic=initial_best,historical_initial_best_mic=float(min(expected)),physical_candidate_predictions=len(seqs),initial_predictions=100,max_calibration_relative_error=float(max(relative)),initial_predictions_values=vals[:100],scores=pred,measurements=measures,decoder_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
  finished.add(task);progress=True
  write(OUT/'scoring_status.json',dict(status='running',complete_tasks=len(finished),candidate_predictions=calls,initial_predictions=100*len(finished),elapsed_seconds=time.monotonic()-start))
  if len(finished)%25==0:print('scored',len(finished),'tasks',calls,'candidate predictions',round(time.monotonic()-start,1),'seconds',flush=True)
 if not progress and len(finished)<800:
  if time.monotonic()-start>7200:raise TimeoutError('Decoder dependency not complete within2hours')
  time.sleep(5)
write(OUT/'scoring_status.json',dict(status='complete',complete_tasks=len(finished),candidate_predictions=calls,initial_predictions=100*len(finished),elapsed_seconds=time.monotonic()-start))
print('COMPLETE',len(finished),calls,flush=True)
