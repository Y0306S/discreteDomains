"""Train-only source-GP agreement at causal target observations; no oracle calls."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import numpy as np
from Levenshtein import distance
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from peptide_prompt import atomic_json,digest,file_sha
from peptide_context import prefix
from rl_common import ROOT,read

CORPUS=Path('/root/data/logs/peptide_transfer_train800_20261027_v2/corpus')

@lru_cache(maxsize=16)
def source_gp(task):
    p=ROOT/'assets/source_gp_mean'/f'{task}.npz';m=json.loads(p.with_suffix('.json').read_text())
    inventory=read('assets/source_gp_inventory.json')
    if m['source_model_sha256']!=inventory[task]['sha256'] or not m['reload_verified'] or file_sha(p)!=m['compact_sha256']:
        raise ValueError('Source GP provenance failed')
    with np.load(p,allow_pickle=False) as f:result={k:f[k].copy() for k in f.files}
    result['scaled_X']=result['X']/result['length_scale']
    return result

def prediction(model,x):
    radius=np.sqrt(5.)*cdist(np.asarray(x,dtype=np.float64)/model['length_scale'],model['scaled_X'])
    return (model['amplitude']*(1+radius+radius**2/3)*np.exp(-radius)).dot(model['alpha'])

def correlation(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite GP/evidence')
    ra,rb=rankdata(a),rankdata(b)
    if np.std(ra)==0 or np.std(rb)==0:return None
    return float(np.corrcoef(ra,rb)[0,1])

def task_jobs(task,jobs):
    # All outputs are atomic and preparation makes no oracle calls. A fully
    # committed target can be reused when its job/prefix identity is intact.
    saved=[]
    for job in jobs:
        p=ROOT/'assets/retrieval'/f"{job['group_id']}.json"
        if not p.exists():break
        value=json.loads(p.read_text())
        if value.get('job_identity')!=job:break
        saved.append(job['group_id'])
    if len(saved)==len(jobs):return saved
    settings=read('settings.json')['retrieval'];tasks=read('assets/tasks.json');refs={t['task_id']:t['reference'] for t in tasks}
    all_cards=read('assets/source_cards.json');manifest=json.loads((CORPUS/'manifest.json').read_text())
    assert task in manifest['source_tasks']
    with sqlite3.connect('file:'+str(CORPUS/'corpus.sqlite')+'?mode=ro',uri=True) as db:
        sha,raw=db.execute('SELECT payload_sha256,payload FROM tasks WHERE id=?',(task,)).fetchone()
    if sha!=manifest['task_hashes'][task] or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Target frozen-feature payload changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        # Only frozen features and strings are consumed, never future target labels.
        x=f['X'];sequences=f['sequences'].astype(str)
    lower,upper=np.asarray(manifest['lower']),np.asarray(manifest['upper'])
    x=((x-lower)/np.where(upper>lower,upper-lower,1.)).astype(np.float32)
    lookup={s:i for i,s in enumerate(sequences)}
    reference=refs[task]
    shortlist=sorted((t for t in refs if t!=task),key=lambda t:(-(1-distance(reference,refs[t])/max(len(reference),len(refs[t]))),abs(len(reference)-len(refs[t])),t))[:settings['shortlist']]
    outputs=[]
    for job in jobs:
        h,rows,valid=prefix(job)
        by_mic=sorted(valid,key=lambda r:(r['mic'],r['observation_index']))
        evidence=sorted({r['observation_index']:r for r in valid[-50:]+by_mic[:5]+by_mic[-5:]}.values(),key=lambda r:r['observation_index'])
        assert len(evidence)<=60 and all(r['observation_index']<job['prefix_end_exclusive'] for r in evidence)
        matrix=x[[lookup[r['sequence']] for r in evidence]];observed=np.asarray([r['mic'] for r in evidence])
        ranked=[]
        for source in shortlist:
            rho=correlation(prediction(source_gp(source),matrix),observed)
            sim=1-distance(reference,refs[source])/max(len(reference),len(refs[source]))
            ranked.append(dict(task=source,spearman=rho,reference_edit_similarity=sim,source_gp_sha256=read('assets/source_gp_inventory.json')[source]['sha256']))
        ranked.sort(key=lambda s:(-(s['spearman'] if s['spearman'] is not None else -2),-s['reference_edit_similarity'],s['task']))
        strong=[s for s in ranked if s['spearman'] is not None and s['spearman']>=settings['spearman_floor']]
        selected=(strong if strong else ranked)[:settings['top_k']]
        ranking=dict(method='reference_shortlist16_then_exact_source_GP_mean_Spearman',shortlist_count=len(shortlist),
            scores=ranked,selected_source_ids=[s['task'] for s in selected],strong_agreement_count=len(strong),
            confidence='rank_agreement_ge_0.5' if strong else 'weak_or_undefined_agreement_DO_NOT_ASSUME_TRANSFER',
            floor=settings['spearman_floor'],target_evidence_indices=[r['observation_index'] for r in evidence],
            target_evidence_sha256=digest(evidence),source_label_scope='other training tasks only',target_future_labels_used=False,
            weak_evidence_instruction='Reference or ranking agreement is tentative. Analyze the source/current task differences and prioritize measured current-task evidence; never blindly copy.')
        output=dict(job_identity=job,cards=[all_cards[s['task']] for s in selected],ranking=ranking)
        atomic_json(ROOT/'assets/retrieval'/f"{job['group_id']}.json",output)
        outputs.append(job['group_id'])
    return outputs

def main():
    grouped={}
    for j in read('assets/jobs.json'):
        if j['scenario'] in ('retrieval','exploit'):grouped.setdefault(j['task_id'],[]).append(j)
    done=0;groups=0
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures=[ex.submit(task_jobs,t,j) for t,j in grouped.items()]
        for f in as_completed(futures):
            groups+=len(f.result());done+=1
            if done%25==0:print(json.dumps(dict(tasks=done,retrieval_groups=groups)),flush=True)
    atomic_json(ROOT/'evidence/retrieval_preflight.json',dict(passed=True,tasks=done,groups=groups,oracle_calls=0,
        heldout_used=False,target_future_labels_used=False,source_GP_predictions_verified=True))

if __name__=='__main__':main()
