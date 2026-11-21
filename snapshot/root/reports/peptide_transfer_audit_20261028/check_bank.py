#!/usr/bin/env python3
"""Verify a saved source GP against its full frozen-coordinate training data."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys

os.sched_setaffinity(0,{18})
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0,'/root/data/cys/baseline/repos/transfer-hpo-framework')
import numpy as np
import peptide_transfer as transfer

BASE=Path('/root/data/logs/peptide_transfer_train800_20261027_v2')


def main():
    with sqlite3.connect('file:'+str(BASE/'rgpe/models.sqlite')+'?mode=ro',uri=True,timeout=120) as db:
        task,metadata,expected,payload=db.execute('SELECT task,metadata,sha256,payload FROM models ORDER BY task LIMIT 1').fetchone()
    assert hashlib.sha256(payload).hexdigest()==expected
    meta=json.loads(metadata)
    model=pickle.loads(payload)
    del payload
    manifest,db=transfer.load_corpus(BASE/'corpus')
    data=transfer.read_task(db,task); db.close()
    x=transfer.normalize(data['X'],manifest).astype(np.float64); y=-data['y']
    assert task in manifest['source_tasks'] and meta['source_payload_sha256']==data['payload_sha256']
    assert model.is_trained and model.normalize_y and not model.gp.normalize_y
    assert np.array_equal(model.gp.X_train_,x)
    assert np.isclose(model.mean_y_,y.mean()) and np.isclose(model.std_y_,y.std())
    assert np.allclose(model.gp.y_train_,(y-y.mean())/y.std())
    assert model.kernel.k1.k2.nu==2.5 and len(model.kernel.k1.k2.length_scale)==256
    assert model.n_opt_restarts==10
    probe=np.vstack((x[:8],(x[8:16]+x[16:24])/2))
    mean,var=model.predict(probe)
    assert np.isfinite(mean).all() and np.isfinite(var).all() and np.all(var>=0)
    report=dict(passed=True,task=task,observations=len(y),dimensions=256,
        payload_checksum_verified=True,full_frozen_normalized_features_match=True,
        mic_minimization_sign_verified=True,source_standardization_verified=True,
        kernel='Matern-5/2 ARD',optimizer_restarts=model.n_opt_restarts,
        likelihood_evaluations=model._n_ll_evals,finite_posterior_verified=True,
        elapsed_seconds=meta['elapsed_seconds'],peak_rss_gib=meta['peak_rss_gib'],oracle_calls=0)
    transfer.atomic_json(Path(__file__).with_name('bank_validation.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
