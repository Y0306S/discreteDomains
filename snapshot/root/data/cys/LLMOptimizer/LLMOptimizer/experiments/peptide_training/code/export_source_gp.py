"""Python3.7: export exact mean-only Matern2.5 source GPs, with reload checks."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0,'/root/data/cys/baseline/repos/transfer-hpo-framework')
ROOT=Path(__file__).resolve().parents[1]

def predict(model,x):
    radius=np.sqrt(5.)*cdist(np.asarray(x,dtype=float)/model['length_scale'],model['X']/model['length_scale'])
    return (model['amplitude']*(1+radius+radius**2/3)*np.exp(-radius)).dot(model['alpha'])

def export(task,expected):
    destination=ROOT/'assets/source_gp_mean'/str(task+'.npz')
    side=destination.with_suffix('.json')
    if side.exists():
        metadata=json.loads(side.read_text())
        if metadata['source_model_sha256']!=expected:raise ValueError('Source inventory changed')
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=metadata['compact_sha256']:raise ValueError('Compact GP damaged')
        return metadata
    with sqlite3.connect('file:/root/data/logs/peptide_transfer_train800_20261027_v2/rgpe/models.sqlite?mode=ro',uri=True,timeout=120) as db:
        row=db.execute('SELECT metadata,sha256,payload FROM models WHERE task=?',(task,)).fetchone()
    if row[1]!=expected or hashlib.sha256(row[2]).hexdigest()!=expected:raise ValueError('GP checksum mismatch')
    identity=json.loads(row[0]);model=pickle.loads(row[2]);gp=model.gp
    if not identity['reload_verified'] or np.any(np.asarray(model.types)!=0):raise ValueError('Unsupported GP')
    kernel=gp.kernel_
    if kernel.has_conditions or kernel.k1.k2.nu!=2.5:raise ValueError('Unexpected source kernel')
    compact=dict(X=gp.X_train_.copy(),alpha=gp.alpha_.copy(),length_scale=np.asarray(kernel.k1.k2.length_scale),amplitude=np.asarray(kernel.k1.k1.constant_value))
    probe=gp.X_train_[::max(1,len(gp.X_train_)//16)][:16]
    actual=predict(compact,probe);expected_mean=gp.predict(probe)
    error=float(np.max(np.abs(actual-expected_mean)))
    if not np.allclose(actual,expected_mean,rtol=1e-8,atol=1e-8):raise ValueError('Exact mean export differs from source posterior: '+str(error))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**compact)
    os.replace(str(temp),str(destination))
    with np.load(str(destination),allow_pickle=False) as saved:after=predict(saved,probe)
    if not np.allclose(after,expected_mean,rtol=1e-8,atol=1e-8):raise ValueError('Reload failed')
    metadata=dict(task=task,source_model_sha256=row[1],compact_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        source_identity=identity,observations=len(compact['X']),max_normalized_mean_error=error,reload_verified=True,
        variance_exported=False,purpose='source_GP_Spearman_rerank_only',normalization='positive-affine standardization leaves ranks unchanged')
    temp=side.with_name(side.name+'.tmp');temp.write_text(json.dumps(metadata,indent=2));os.replace(str(temp),str(side))
    return metadata

def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    inventory=json.loads((ROOT/'assets/source_gp_inventory.json').read_text())
    tasks=sorted(inventory);limit=int(sys.argv[1]) if len(sys.argv)>1 else len(tasks)
    started=time.time();count=0
    with ProcessPoolExecutor(max_workers=2) as ex:
        jobs=[ex.submit(export,t,inventory[t]['sha256']) for t in tasks[:limit]]
        for future in as_completed(jobs):
            future.result();count+=1
            if count%25==0 or limit<5:print(json.dumps(dict(exported=count,total=limit,seconds=time.time()-started)),flush=True)
    if limit==800:
        path=ROOT/'evidence/source_gp_export.json';temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(dict(passed=True,models=count,exact_mean_only=True,seconds=time.time()-started)));os.replace(str(temp),str(path))

if __name__=='__main__':main()
