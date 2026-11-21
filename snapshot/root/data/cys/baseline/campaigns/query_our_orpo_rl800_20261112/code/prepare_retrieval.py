"""Rank training source GPs on causal target runtime evidence; no new oracle calls."""
import hashlib,io,json,sqlite3,time
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from query_data import task_description,plan_key,atomic,digest,canonical
from query_context import prefix
from rl_common import ROOT,read
CORPUS=__import__('pathlib').Path(read('settings.json')['retrieval']['corpus'])
@lru_cache(maxsize=64)
def source_gp(task):
 p=ROOT/'assets/source_gp_mean'/f'{task}.npz';m=json.loads(p.with_suffix('.json').read_text());expected=read('assets/source_gp_inventory.json')[task]['sha256']
 if m['source_model_sha256']!=expected or not m['reload_verified'] or digest(p)!=m['compact_sha256']:raise ValueError('Source GP provenance')
 with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def prediction(model,x):
 radius=np.sqrt(5.)*cdist(np.asarray(x,dtype=np.float64)/model['length_scale'],model['X']/model['length_scale'])
 return (model['amplitude']*(1+radius+radius**2/3)*np.exp(-radius)).dot(model['alpha'])
def shortlist(task,bank):
 target=task_description(task);aliases=target['constraints']['aliases'];tables={r['table'] for r in aliases};n=len(aliases);out=[]
 for t in bank:
  if t==task:continue
  # Query syntax/template identity and actual tables, not a biological proxy.
  desc=read('assets/source_cards.json')[t]['task'];other={r['table'] for r in desc['constraints']['aliases']}
  sim=len(tables&other)/len(tables|other);diff=abs(n-len(desc['constraints']['aliases']))
  out.append((t,sim,diff))
 return sorted(out,key=lambda v:(-v[1],v[2],v[0]))[:16]
def main():
 manifest=json.loads((CORPUS/'manifest.json').read_text());bank=read('assets/source_gp_inventory.json');cards=read('assets/source_cards.json');lower,upper=np.array(manifest['lower']),np.array(manifest['upper']);grouped={}
 for j in read('assets/jobs.json'):
  if j['scenario']=='retrieval':grouped.setdefault(j['task_id'],[]).append(j)
 count=0
 for task,jobs in grouped.items():
  with sqlite3.connect('file:'+str(CORPUS/'corpus.sqlite')+'?mode=ro',uri=True) as db:sha,raw,source_sha=db.execute('SELECT payload_sha256,payload,source_sha256 FROM tasks WHERE id=?',(task,)).fetchone()
  if hashlib.sha256(raw).hexdigest()!=sha or sha!=manifest['task_hashes'][task]:raise ValueError('Feature payload changed')
  with np.load(io.BytesIO(raw),allow_pickle=False) as f:x=f['X'];indices=f['source_rows']
  x=((x-lower)/np.where(upper>lower,upper-lower,1.)).astype(np.float32);lookup={int(i):p for p,i in enumerate(indices)};short=shortlist(task,bank)
  for job in jobs:
   h,rows,valid=prefix(job)
   if h['source_history_sha256']!=source_sha:raise ValueError('Target features from different history')
   # Corpus dedup and this exporter agree by executed canonical tree. A join
   # ordering variant can share one corpus row; map by the whole plan key.
   bykey={plan_key(h['records'][int(i)]['plan']):p for p,i in enumerate(indices)}
   rankedvalid=sorted(valid,key=lambda r:(r['runtime'],r['observation_index']));evidence=sorted({r['observation_index']:r for r in valid[-50:]+rankedvalid[:5]+rankedvalid[-5:]}.values(),key=lambda r:r['observation_index'])
   matrix=x[[bykey[plan_key(r['plan'])] for r in evidence]];observed=rankdata([r['runtime'] for r in evidence]);ranked=[]
   for source,sim,diff in short:
    pred=rankdata(prediction(source_gp(source),matrix));rho=None if np.std(pred)==0 or np.std(observed)==0 else float(np.corrcoef(pred,observed)[0,1]);ranked.append(dict(task=source,spearman=rho,table_jaccard=sim,alias_count_difference=diff,source_gp_sha256=bank[source]['sha256']))
   ranked.sort(key=lambda r:(-(r['spearman'] if r['spearman'] is not None else -2),-r['table_jaccard'],r['alias_count_difference'],r['task']));strong=[r for r in ranked if r['spearman'] is not None and r['spearman']>=.5];selected=(strong or ranked)[:3]
   ranking=dict(method='table_Jaccard_alias_count_shortlist16_then_exact_source_GP_Spearman',scores=ranked,selected_source_ids=[r['task'] for r in selected],strong_agreement_count=len(strong),confidence='rank_agreement_ge_0.5' if strong else 'weak_or_undefined_DO_NOT_ASSUME_TRANSFER',floor=.5,target_evidence_indices=[r['observation_index'] for r in evidence],target_future_labels_used=False,source_label_scope='other training tasks only')
   atomic(ROOT/'assets/retrieval'/f"{job['group_id']}.json",dict(job_identity=job,cards=[cards[r['task']] for r in selected],ranking=ranking));count+=1
  print(json.dumps(dict(retrieval_groups=count,task=task)),flush=True)
 atomic(ROOT/'evidence/retrieval_preflight.json',dict(passed=True,groups=count,tasks=len(grouped),oracle_calls=0,heldout_used=False,target_future_labels_used=False,source_GP_predictions_verified=True))
if __name__=='__main__':main()
