"""Score fixed generated source800 pools with the pinned APEX ensemble; not BO."""
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
def predict(seqs):
 result=[]
 with torch.inference_mode():
  for start in range(0,len(seqs),128):
   x=torch.as_tensor(onehot_encoding(seqs[start:start+128],52,vocab),device='cuda',dtype=torch.long)
   # Match native order: transform each member to MIC, average, select bacteria0.
   total=None
   for model in models:
    pred=10**(6-model(x)).cpu().numpy()
    total=pred if total is None else total+pred
   result.extend((total/len(models))[:,0].astype(float).tolist())
 assert len(result)==len(seqs) and np.isfinite(result).all() and all(v>0 for v in result)
 return result
identity=dict(oracle_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 precision='fp32_cuda_no_tf32',target='native APEX ensemble bacteria0 MIC',cohort='training800',selection='fixed generated pools; no GP selection or feedback')
write(OUT/'candidate_scoring_identity.json',identity)
start=time.monotonic();calls=0;finished=set()
(OUT/'scored').mkdir(exist_ok=True)
while len(finished)<800:
 progress=False
 for task in tasks:
  if task in finished:continue
  dest=OUT/'scored'/(task+'.json')
  if dest.exists():
   old=json.loads(dest.read_text());assert old['status']=='complete';calls+=old['physical_candidate_predictions'];finished.add(task);continue
  source=OUT/'tasks/adapted'/(task+'.json')
  if not source.exists():continue
  meta=json.loads(source.read_text())
  with gzip.open(source.with_suffix('.jsonl.gz'),'rt') as f:draws=[json.loads(line) for line in f]
  # Only exact-global-valid, syntax-valid and initially unseen sequences are scored.
  seqs=list(dict.fromkeys(r['sequence'] for r in draws if r['valid'] and r['novel']))
  checks=pool[task]['sequences'][:4]
  vals=predict(checks+seqs)
  expected=np.array(pool[task]['scores'][:4],dtype=float)*-1
  actual=np.array(vals[:4]);assert np.allclose(actual,expected,rtol=1e-4,atol=1e-4),(task,actual,expected)
  pred=dict(zip(seqs,vals[4:]));calls+=len(seqs)
  initial_best=-max(pool[task]['scores'])
  measures=[]
  for radius in sorted({r['radius'] for r in draws}):
   for k in [1,2,3,100]:
    selected=list(dict.fromkeys(r['sequence'] for r in draws if r['radius']==radius and r['valid'] and r['novel'] and r['edits_center']<=k))
    scores=[pred[s] for s in selected]
    for budget in [5,10,20,100000]:
     prefix=scores[:budget]
     measures.append(dict(radius=radius,local_edits=k,nominal_candidate_budget=budget,actual_candidates_scored=min(len(scores),budget),filled_budget=len(scores)>=budget,
      best_mic=min([initial_best]+prefix),best_over_initial=min([initial_best]+prefix)/initial_best))
  write(dest,dict(status='complete',task=task,reference_length=meta['reference_length'],global_edit_budget=meta['global_edit_budget'],
   development_split=meta['development_split'],initial_best_mic=initial_best,physical_candidate_predictions=len(seqs),calibration_predictions=4,
   max_calibration_relative_error=float(np.max(np.abs(actual-expected)/expected)),scores=pred,measurements=measures,
   decoder_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
  finished.add(task);progress=True
  write(OUT/'scoring_status.json',dict(status='running',complete_tasks=len(finished),candidate_predictions=calls,calibration_predictions=4*len(finished),elapsed_seconds=time.monotonic()-start))
  if len(finished)%25==0:print('scored',len(finished),'tasks',calls,'candidate predictions',round(time.monotonic()-start,1),'seconds',flush=True)
 if not progress and len(finished)<800:
  if time.monotonic()-start>7200:raise TimeoutError('Decoder dependency not complete within2hours')
  time.sleep(5)
write(OUT/'scoring_status.json',dict(status='complete',complete_tasks=len(finished),candidate_predictions=calls,calibration_predictions=4*len(finished),elapsed_seconds=time.monotonic()-start))
print('COMPLETE',len(finished),calls,flush=True)
