#!/usr/bin/env python3
"""Refit an already completed source on CUDA and compare to its CPU model."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import time

import numpy as np
import peptide_transfer as transfer
from peptide_rgpe_cuda_objective import cuda_objective


def main(args):
    os.sched_setaffinity(0, {args.cpu})
    sys.path.insert(0, str(transfer.RGPE_REPO))
    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
    from smac.epm.util_funcs import get_types
    from rgpe.utils import get_gaussian_process
    with sqlite3.connect('file:'+str(args.campaign/'rgpe/models.sqlite')+'?mode=ro', uri=True, timeout=120) as db:
        task, payload, metadata = db.execute('SELECT task,payload,metadata FROM models ORDER BY rowid DESC LIMIT 1').fetchone()
    reference = pickle.loads(payload)
    del payload
    metadata = json.loads(metadata)
    manifest, db = transfer.load_corpus(args.campaign/'corpus')
    data = transfer.read_task(db, task)
    db.close()
    assert data['payload_sha256'] == metadata['source_payload_sha256']
    assert transfer.digest(args.campaign/'corpus/manifest.json') == metadata['corpus_sha256']
    x = transfer.normalize(data['X'], manifest).astype(np.float64)
    y = -data['y']
    cs = ConfigurationSpace()
    cs.add_hyperparameters([UniformFloatHyperparameter('z%03d'%i, 0., 1.) for i in range(256)])
    types, bounds = get_types(cs, None)
    model = get_gaussian_process(cs, types, bounds, np.random.RandomState(metadata['seed']), None)
    assert model.n_opt_restarts == metadata['optimizer_restarts'] == 10
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (args.output/'worker.log').open('ab') as log:
        with cuda_objective(model, args.cuda_python, args.gpu, log,
                            callback=lambda value:print(json.dumps(value),flush=True)) as performance:
            model.train(x, y)
    elapsed = time.monotonic()-started
    model_bytes = pickle.dumps(model, protocol=4)
    restored = pickle.loads(model_bytes)
    probes = np.vstack([x[:32], (x[:32]+x[32:64])/2,
                        np.random.RandomState(139).uniform(0,1,(32,256))])
    cpu_prediction = reference.predict(probes, cov_return_type='full_cov')
    cuda_prediction = model.predict(probes, cov_return_type='full_cov')
    reloaded = restored.predict(probes, cov_return_type='full_cov')
    errors = [float(np.max(np.abs(a-b))) for a,b in zip(cpu_prediction,cuda_prediction)]
    posterior_passed = all(np.allclose(a,b,rtol=1e-6,atol=1e-6) for a,b in zip(cpu_prediction,cuda_prediction))
    reload_passed = all(np.array_equal(a,b) for a,b in zip(cuda_prediction,reloaded))
    result = dict(task=task, observations=len(x), dimensions=256, seed=metadata['seed'],
        optimizer_restarts=model.n_opt_restarts, cuda_seconds=elapsed,
        cpu_seconds=metadata['elapsed_seconds'], cpu_evaluations=metadata['likelihood_evaluations'],
        cuda_evaluations=performance['evaluations'], posterior_max_abs_errors=errors,
        posterior_passed=bool(posterior_passed), reload_passed=bool(reload_passed),
        passed=bool(posterior_passed and reload_passed), production_enabled=False,
        theta_max_abs_difference=float(np.max(np.abs(reference.hypers-model.hypers))),
        cpu_theta=reference.hypers.tolist(),cuda_theta=model.hypers.tolist(),
        source_metadata=metadata, completed_at=time.time())
    transfer.atomic_json(args.output/'validation.json',result)
    (args.output/'model.pkl').write_bytes(model_bytes)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cpu_theta','cuda_theta','source_metadata')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=Path('/root/data/logs/peptide_transfer_train800_20261027_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cuda-python', default='/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python')
    main(parser.parse_args())
