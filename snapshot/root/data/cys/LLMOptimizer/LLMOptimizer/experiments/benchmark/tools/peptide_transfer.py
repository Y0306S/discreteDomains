#!/usr/bin/env python3
"""Resumable frozen-VAE train800 corpus and compact peptide FSBO/RGPE training."""

from __future__ import print_function

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import resource
import sqlite3
import sys
import tarfile
import time

import numpy as np

SOURCE = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
AUDIT = Path('/root/reports/peptide_train800_test200_audit_20261027')
RELEASE = Path('/root/data/logs/peptide_bo_release_20261027_v5')
TRAINER = Path('/root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains')
RGPE_REPO = Path('/root/data/cys/baseline/repos/transfer-hpo-framework')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


@contextlib.contextmanager
def lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def connect(path):
    db = sqlite3.connect(str(path), timeout=120)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, observations INTEGER NOT NULL, source_sha256 TEXT NOT NULL, payload_sha256 TEXT NOT NULL, payload BLOB NOT NULL)')
    db.commit()
    return db


def get_meta(db, key):
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def put_meta(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value, allow_nan=False)))


def pack_arrays(**values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def read_task(db, task):
    row = db.execute('SELECT observations,source_sha256,payload_sha256,payload FROM tasks WHERE id=?', (task,)).fetchone()
    if row is None or hashlib.sha256(row[3]).hexdigest() != row[2]:
        raise ValueError('Missing or damaged source: ' + task)
    with np.load(io.BytesIO(row[3]), allow_pickle=False) as stored:
        data = {k: stored[k].copy() for k in stored.files}
    if data['X'].shape != (row[0], 256) or not np.isfinite(data['X']).all() or not np.isfinite(data['y']).all():
        raise ValueError('Invalid source feature/outcome alignment')
    data.update(source_sha256=row[1], payload_sha256=row[2])
    return data


def export(args):
    sys.path.insert(0, str(RELEASE))
    sys.path.insert(0, str(RELEASE / 'tools'))
    import peptide_bo_v5 as native
    from mindware.domains.protein.peptide_rpc import RemoteObjective
    from types import SimpleNamespace
    import torch
    torch.set_num_threads(1)
    recipe = native.verify(SOURCE)
    audit = json.loads((AUDIT / 'train800_audit.json').read_text())
    if not audit['summary']['passed']:
        raise ValueError('Source audit must pass')
    records = {r['task']: r for r in audit['tasks']}
    tasks = recipe['train']
    if len(tasks) != 800 or set(tasks) & set(recipe['test']):
        raise ValueError('Only train800 is permitted as an offline source')
    identity = dict(domain='peptide', source_recipe_sha256=digest(SOURCE / 'recipe.json'),
        split_sha256=digest(SOURCE / 'split.json'), vae_sha256=digest(recipe['vae']),
        exporter_sha256=digest(__file__), source_audit_sha256=digest(AUDIT / 'train800_audit.json'),
        representation='frozen_author_VAE_posterior_mean_256', normalization='train800_coordinate_minmax',
        observation_policy='all_distinct_finite_measured_first_value_float32_equivalent_cache_variants_preserved', oracle_calls=0)
    args.output.mkdir(parents=True, exist_ok=True)
    with lock(args.output / 'export.lock'):
        db = connect(args.output / 'corpus.sqlite')
        previous = get_meta(db, 'identity')
        if previous is not None and previous != identity:
            raise ValueError('Export resume identity changed')
        with db:
            put_meta(db, 'identity', identity)
        sys.path.insert(0, recipe['cpu_upstream'])
        os.chdir(recipe['cpu_upstream'])
        from uniref_vae.load_vae import load_vae
        vae, _ = load_vae(recipe['vae'], max_string_length=30)
        objective = SimpleNamespace(vae=vae, vae_forward=None)
        client = RemoteObjective(objective, args.server, 'peptide_train800_export:' + identity['vae_sha256'])
        lower, upper = np.full(256, np.inf), np.full(256, -np.inf)
        observations, started = 0, time.monotonic()
        try:
            for index, task in enumerate(tasks):
                cached = db.execute('SELECT source_sha256 FROM tasks WHERE id=?', (task,)).fetchone()
                expected = records[task]['member_checksums']['history.json']
                if cached:
                    if cached[0] != expected:
                        raise ValueError('Changed archived source history')
                    data = read_task(db, task)
                else:
                    with tarfile.open(str(SOURCE / 'runs' / (task + '.tar.gz')), 'r:gz') as packed:
                        raw = packed.extractfile('history.json').read()
                    if hashlib.sha256(raw).hexdigest() != expected:
                        raise ValueError('Source history checksum changed')
                    history = json.loads(raw)
                    if history['reference'] != recipe['references'][task] or history['status'] != 'complete':
                        raise ValueError('Wrong source task or incomplete history')
                    distinct, rounding_variants = {}, []
                    for row_index, row in enumerate(history['observations']):
                        sequence = row['sequence']
                        if sequence in distinct and distinct[sequence]['score'] != row['score']:
                            first = distinct[sequence]['score']
                            if np.float32(first) != np.float32(row['score']):
                                atomic_json(args.output / 'conflicting_outcomes.json', dict(task=task,
                                    sequence=sequence, first_score=first, repeated_score=row['score'], row=row_index))
                                raise ValueError('Conflicting repeated sequence outcome beyond float32 equivalence')
                            rounding_variants.append((row_index, first, row['score']))
                        distinct.setdefault(sequence, row)
                    if len(distinct) != records[task]['distinct_sequences'] or len(distinct) < 2:
                        raise ValueError('Source distinct-observation count differs from audit')
                    sequences = list(distinct)
                    features = []
                    with torch.no_grad():
                        for start in range(0, len(sequences), 256):
                            batch = sequences[start:start + 256]
                            result = client.inference('encode_mean', sequences=batch)['z']
                            features.append(result.numpy().astype(np.float32, copy=False))
                    data = dict(X=np.concatenate(features),
                        y=np.asarray([distinct[x]['score'] for x in sequences], dtype=np.float64),
                        constraints=np.asarray([recipe['similarity'] - native.bo.similarity(x, history['reference']) for x in sequences], dtype=np.float32),
                        sequences=np.asarray(sequences), reference=np.asarray(history['reference']),
                        float32_equivalent_repeats=np.asarray(rounding_variants, dtype=np.float64).reshape(-1, 3))
                    if data['X'].shape != (len(sequences), 256) or not np.isfinite(data['X']).all():
                        raise ValueError('Invalid frozen-VAE representation')
                    payload = pack_arrays(**data)
                    with db:
                        db.execute('INSERT INTO tasks VALUES (?,?,?,?,?)', (task, len(sequences), expected,
                            hashlib.sha256(payload).hexdigest(), sqlite3.Binary(payload)))
                lower = np.minimum(lower, data['X'].min(axis=0))
                upper = np.maximum(upper, data['X'].max(axis=0))
                observations += len(data['y'])
                progress = dict(status='exporting', completed=index + 1, total=800, observations=observations,
                    elapsed_seconds=time.monotonic() - started, oracle_calls=0, identity=identity)
                atomic_json(args.output / 'status.json', progress)
                print(json.dumps({k: v for k, v in progress.items() if k != 'identity'}), flush=True)
            rng = np.random.RandomState(123)
            order = rng.permutation(800)
            valid = {tasks[i] for i in order[:160]}
            manifest = dict(identity, status='complete', purpose='production_training', source_tasks=tasks,
                fsbo_train_tasks=[t for t in tasks if t not in valid],
                fsbo_validation_tasks=[t for t in tasks if t in valid], observations=observations,
                lower=lower.tolist(), upper=upper.tolist(), database=str(args.output / 'corpus.sqlite'),
                fsbo_dim=256, task_hashes={t: db.execute('SELECT payload_sha256 FROM tasks WHERE id=?', (t,)).fetchone()[0] for t in tasks})
            with db:
                put_meta(db, 'manifest', manifest)
            atomic_json(args.output / 'manifest.json', manifest)
            atomic_json(args.output / 'status.json', dict(progress, status='complete'))
        finally:
            client.close()
            db.close()


def load_corpus(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    source = json.loads((SOURCE / 'recipe.json').read_text())
    if (manifest['status'] != 'complete' or manifest['source_tasks'] != source['train']
            or len(manifest['source_tasks']) != 800 or set(manifest['source_tasks']) & set(source['test'])
            or manifest['source_recipe_sha256'] != digest(SOURCE / 'recipe.json')
            or manifest['vae_sha256'] != digest(source['vae'])):
        raise ValueError('Full train800 corpus/source provenance mismatch')
    db = connect(directory / 'corpus.sqlite')
    if get_meta(db, 'manifest') != manifest:
        raise ValueError('Database/manifest identities differ')
    return manifest, db


def normalize(x, manifest):
    lower, upper = np.asarray(manifest['lower']), np.asarray(manifest['upper'])
    scale = np.where(upper > lower, upper - lower, 1.0)
    return ((x - lower) / scale).astype(np.float32)


def fsbo(args):
    sys.path.insert(0, str(TRAINER))
    import train_taskset_transfer as trainer
    manifest, db = load_corpus(args.corpus)
    tasks = {}
    for task in manifest['source_tasks']:
        data = read_task(db, task)
        if data['payload_sha256'] != manifest['task_hashes'][task]:
            raise ValueError('Source payload differs from manifest')
        tasks[task] = dict(X=normalize(data['X'], manifest), y_maximize=data['y'],
                          y_minimize=-data['y'], source_sha256=data['source_sha256'])
    db.close()
    def identity(settings, _manifest):
        return dict(domain='peptide', method='fsbo', purpose=settings.purpose,
                    corpus_sha256=digest(args.corpus / 'manifest.json'),
                    split_sha256=manifest['split_sha256'], representation=manifest['representation'],
                    seed=args.seed, trainer_sha256=digest(__file__),
                    shared_training_loop_sha256=digest(TRAINER / 'train_taskset_transfer.py'),
                    device='cpu', threads=args.threads)
    trainer.identity = identity
    args.method = 'fsbo'
    args.purpose = 'production_training' if args.steps == 100000 else 'component_check'
    args.resume = True
    args.output.mkdir(parents=True, exist_ok=True)
    with lock(args.output / '.lock'):
        trainer.fsbo(args, manifest, tasks)


def rgpe_task(args):
    sys.path.insert(0, str(RGPE_REPO))
    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
    from smac.epm.util_funcs import get_types
    from rgpe.utils import get_gaussian_process
    manifest, db = load_corpus(args.corpus)
    if args.task not in manifest['source_tasks']:
        raise ValueError('RGPE source is not in train800')
    data = read_task(db, args.task)
    db.close()
    if data['payload_sha256'] != manifest['task_hashes'][args.task]:
        raise ValueError('Source payload differs from manifest')
    x, y = normalize(data['X'], manifest).astype(np.float64), -data['y']
    cs = ConfigurationSpace()
    cs.add_hyperparameters([UniformFloatHyperparameter('z%03d' % i, 0.0, 1.0) for i in range(256)])
    types, bounds = get_types(cs, None)
    task_index = manifest['source_tasks'].index(args.task)
    model = get_gaussian_process(cs, types, bounds, np.random.RandomState(args.seed + task_index), None)
    started = time.monotonic()
    # Full exact source fits: no observation cap or substituted approximate GP.
    model.train(x, y)
    probe = x[:16]
    mean, variance = model.predict(probe)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError('Invalid fitted source GP posterior')
    payload = pickle.dumps(model, protocol=4)
    restored = pickle.loads(payload)
    after = restored.predict(probe)
    if not np.allclose(mean, after[0]) or not np.allclose(variance, after[1]):
        raise ValueError('Source predictions changed after reload')
    identity = dict(domain='peptide', method='rgpe_source_gp', task=args.task,
        corpus_sha256=digest(args.corpus / 'manifest.json'), source_payload_sha256=data['payload_sha256'],
        representation=manifest['representation'], kernel='upstream_Matern2.5_ARD', normalize_y=True,
        trainer_sha256=digest(__file__), gp_implementation_sha256=digest(RGPE_REPO / 'rgpe/utils.py'),
        seed=args.seed + task_index, observations=len(y), reload_verified=True,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    args.output.mkdir(parents=True, exist_ok=True)
    with lock(args.output / 'bank.lock', blocking=True):
        bank = sqlite3.connect(str(args.output / 'models.sqlite'), timeout=120)
        bank.execute('PRAGMA synchronous=FULL')
        bank.execute('CREATE TABLE IF NOT EXISTS models (task TEXT PRIMARY KEY, metadata TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL)')
        with bank:
            bank.execute('INSERT INTO models VALUES (?,?,?,?)', (args.task, json.dumps(identity),
                hashlib.sha256(payload).hexdigest(), sqlite3.Binary(payload)))
        completed = bank.execute('SELECT COUNT(*) FROM models').fetchone()[0]
        bank.close()
        atomic_json(args.output / 'status.json', dict(status='complete' if completed == 800 else 'training',
            completed=completed, total=800, last_task=args.task, last_identity=identity))
    print(json.dumps(identity), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('export')
    p.add_argument('--server', default='http://127.0.0.1:18871')
    for name in ('fsbo', 'rgpe-task'):
        p = sub.add_parser(name)
        p.add_argument('--corpus', type=Path, required=True)
        p.add_argument('--seed', type=int, default=123)
        p.add_argument('--threads', type=int, default=2 if name == 'fsbo' else 1)
        if name == 'fsbo':
            p.add_argument('--steps', type=int, default=100000)
        else:
            p.add_argument('--task', required=True)
    for p in sub.choices.values():
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--cpus', required=True)
    args = parser.parse_args()
    cpus = {int(i) for i in args.cpus.split(',')}
    os.sched_setaffinity(0, cpus)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Training/export coordinators require CPU-only visibility')
    if args.action == 'export':
        export(args)
    elif args.action == 'fsbo':
        fsbo(args)
    else:
        rgpe_task(args)


if __name__ == '__main__':
    main()
