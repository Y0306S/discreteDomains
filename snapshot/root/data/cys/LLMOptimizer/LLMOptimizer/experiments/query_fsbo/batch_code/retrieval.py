"""Training-only source GP ranking on the current measured query history."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from query_data import task_description,plan_key
from query_protocol import digest
TRAIN=Path('/root/data/cys/baseline/campaigns/query_our_orpo_rl800_20261112')
CORPUS=Path('/root/data/logs/query_transfer_train800_20261106_v1/corpus')
@lru_cache(maxsize=1)
def assets():
 return tuple(json.loads(p.read_text()) for p in (TRAIN/'assets/source_cards.json',TRAIN/'assets/source_gp_inventory.json',CORPUS/'manifest.json'))
@lru_cache(maxsize=64)
def source_gp(task):
 _,bank,_=assets();p=TRAIN/'assets/source_gp_mean'/f'{task}.npz';m=json.loads(p.with_suffix('.json').read_text())
 if m['source_model_sha256']!=bank[task]['sha256'] or not m['reload_verified'] or digest(p)!=m['compact_sha256']:raise ValueError('Source GP provenance')
 with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def prediction(model,x):
 radius=np.sqrt(5.)*cdist(np.asarray(x,dtype=np.float64)/model['length_scale'],model['X']/model['length_scale'])
 return (model['amplitude']*(1+radius+radius**2/3)*np.exp(-radius)).dot(model['alpha'])
def retrieve(task,rows,fsbo,allowed):
 cards,bank,manifest=assets()
 if set(cards)!=set(allowed) or not set(bank)<=set(allowed):raise ValueError('Retrieval training split differs')
 target=task_description(task);aliases=target['constraints']['aliases'];tables={r['table'] for r in aliases};short=[]
 for t in bank:
  if t==task:continue
  other=cards[t]['task']['constraints']['aliases'];other_tables={r['table'] for r in other}
  short.append((t,len(tables&other_tables)/len(tables|other_tables),abs(len(aliases)-len(other))))
 short=sorted(short,key=lambda v:(-v[1],v[2],v[0]))[:16]
 valid=[r for r in rows if r['first_occurrence'] and r['feasible'] and not r['censored']];rankedvalid=sorted(valid,key=lambda r:(r['runtime'],r['observation_index']))
 evidence=sorted({r['observation_index']:r for r in valid[-50:]+rankedvalid[:5]+rankedvalid[-5:]}.values(),key=lambda r:r['observation_index'])
 if not evidence:raise ValueError('No causal retrieval evidence')
 if not hasattr(fsbo,'retrieval_encodings'):fsbo.retrieval_encodings={}
 missing=[r['plan'] for r in evidence if plan_key(r['plan']) not in fsbo.retrieval_encodings]
 if missing:
  encoded=np.asarray(fsbo.vae.pool.request('encode_retrieval_mean',plans=missing,task=task)['z'],dtype=np.float32)
  if encoded.shape!=(len(missing),64) or not np.isfinite(encoded).all():raise ValueError('Bad retrieval VAE encoding')
  fsbo.retrieval_encodings.update((plan_key(p),v.tolist()) for p,v in zip(missing,encoded))
 x=np.asarray([fsbo.retrieval_encodings[plan_key(r['plan'])] for r in evidence]);lo=np.asarray(manifest['lower']);hi=np.asarray(manifest['upper']);x=((x-lo)/np.where(hi>lo,hi-lo,1.)).astype(np.float32)
 observed=rankdata([r['runtime'] for r in evidence]);ranked=[]
 for source,sim,diff in short:
  pred=rankdata(prediction(source_gp(source),x));rho=None if np.std(pred)==0 or np.std(observed)==0 else float(np.corrcoef(pred,observed)[0,1]);ranked.append(dict(task=source,spearman=rho,table_jaccard=sim,alias_count_difference=diff,source_gp_sha256=bank[source]['sha256']))
 ranked.sort(key=lambda r:(-(r['spearman'] if r['spearman'] is not None else -2),-r['table_jaccard'],r['alias_count_difference'],r['task']))
 strong=[r for r in ranked if r['spearman'] is not None and r['spearman']>=.5];selected=(strong or ranked)[:3]
 return dict(cards=[cards[r['task']] for r in selected],ranking=dict(method='table_Jaccard_alias_count_shortlist16_then_exact_source_GP_Spearman',scores=ranked,selected_source_ids=[r['task'] for r in selected],strong_agreement_count=len(strong),confidence='rank_agreement_ge_0.5' if strong else 'weak_or_undefined_DO_NOT_ASSUME_TRANSFER',target_evidence_indices=[r['observation_index'] for r in evidence],target_future_labels_used=False,source_label_scope='training_tasks_only',retrieval_vae_sha256=manifest['vae_sha256'],representation_compatible_with_source_GPs=True,available_source_GPs=len(bank),training_tasks_without_GP=sorted(set(allowed)-set(bank))))
