#!/usr/bin/env python3
"""Fit all observations with the unchanged RGPE GP and a bounded-memory objective."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sqlite3
import time
import sys

import numpy as np
import peptide_transfer as transfer
from peptide_rgpe_exact import exact_objective

EXECUTION_SHA256 = transfer.digest(__file__)
OBJECTIVE_SHA256 = transfer.digest(Path(__file__).with_name('peptide_rgpe_exact.py'))


def fit(args):
    sys.path.insert(0, str(transfer.RGPE_REPO))
    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
    from smac.epm.util_funcs import get_types
    from rgpe.utils import get_gaussian_process
    manifest, db = transfer.load_corpus(args.corpus)
    if args.task not in manifest['source_tasks']:
        raise ValueError('RGPE source must be in train800')
    data = transfer.read_task(db, args.task)
    db.close()
    if data['payload_sha256'] != manifest['task_hashes'][args.task]:
        raise ValueError('Source payload differs from manifest')
    x = transfer.normalize(data['X'], manifest).astype(np.float64)
    y = -data['y']
    if y.shape != (len(x),) or np.any(x < -1e-6) or np.any(x > 1 + 1e-6):
        raise ValueError('Invalid aligned normalized training data')
    cs = ConfigurationSpace()
    cs.add_hyperparameters([UniformFloatHyperparameter('z%03d' % i, 0.0, 1.0) for i in range(256)])
    types, bounds = get_types(cs, None)
    seed = args.seed + manifest['source_tasks'].index(args.task)
    model = get_gaussian_process(cs, types, bounds, np.random.RandomState(seed), None)
    args.output.mkdir(parents=True, exist_ok=True)
    with transfer.lock(args.output / 'fit_locks' / (args.task + '.lock')):
        bank_path = args.output / 'models.sqlite'
        if bank_path.exists():
            with sqlite3.connect(str(bank_path), timeout=120) as bank:
                row = bank.execute('SELECT metadata FROM models WHERE task=?', (args.task,)).fetchone()
            if row:
                if json.loads(row[0])['source_payload_sha256'] != data['payload_sha256']:
                    raise ValueError('Existing source GP has incompatible provenance')
                print(json.dumps(dict(task=args.task, status='already_complete')), flush=True)
                return
        started = time.monotonic()
        def progress(value):
            print(json.dumps(dict(value, task=args.task, status='fitting',
                elapsed_seconds=time.monotonic() - started,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)), flush=True)
        with exact_objective(model, callback=progress) as performance:
            model.train(x, y)
        count = min(16, len(x) // 2)
        probe = np.vstack((x[:count], (x[:count] + x[count:2*count]) / 2))
        mean, variance = model.predict(probe)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError('Invalid fitted posterior')
        payload = pickle.dumps(model, protocol=4)
        restored = pickle.loads(payload)
        after = restored.predict(probe)
        if not np.allclose(mean, after[0]) or not np.allclose(variance, after[1]):
            raise ValueError('Source GP predictions changed after reload')
        identity = dict(domain='peptide', method='rgpe_source_gp', task=args.task,
            corpus_sha256=transfer.digest(args.corpus / 'manifest.json'),
            source_payload_sha256=data['payload_sha256'], representation=manifest['representation'],
            kernel='upstream_Matern2.5_ARD', normalize_y=True, seed=seed,
            trainer_sha256=transfer.digest(transfer.__file__),
            gp_implementation_sha256=transfer.digest(transfer.RGPE_REPO / 'rgpe/utils.py'),
            execution_engine='exact_trace_contraction', execution_sha256=EXECUTION_SHA256,
            objective_sha256=OBJECTIVE_SHA256,
            optimizer_restarts=model.n_opt_restarts, likelihood_evaluations=model._n_ll_evals,
            nll_seconds=performance['seconds'], observations=len(y), reload_verified=True,
            elapsed_seconds=time.monotonic() - started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        with transfer.lock(args.output / 'bank.lock', blocking=True):
            with sqlite3.connect(str(bank_path), timeout=120) as bank:
                bank.execute('PRAGMA synchronous=FULL')
                bank.execute('CREATE TABLE IF NOT EXISTS models (task TEXT PRIMARY KEY, metadata TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL)')
                bank.execute('INSERT INTO models VALUES (?,?,?,?)', (args.task, json.dumps(identity),
                    hashlib.sha256(payload).hexdigest(), sqlite3.Binary(payload)))
                completed = bank.execute('SELECT COUNT(*) FROM models').fetchone()[0]
            transfer.atomic_json(args.output / 'status.json', dict(status='complete' if completed == 800 else 'training',
                completed=completed, total=800, last_task=args.task, last_identity=identity))
        print(json.dumps(identity), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--cpu', required=True, type=int)
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU-only fitting requires empty CUDA_VISIBLE_DEVICES')
    fit(args)


if __name__ == '__main__':
    main()
