#!/usr/bin/env python3
"""Query-specific FSBO/RGPE training, without an acquisition/search wrapper."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys
import time

import numpy as np
import peptide_transfer as shared

SOURCE = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
TRAINER = Path(__file__).resolve().parents[1] / 'examples/benchmark/domains/train_taskset_transfer.py'
DIM = 64
POLICY = 'uncensored_first_distinct_plan_min_two'


def source_rows(history, task, plan, canonicalize=None):
    meta = history['meta_info']
    expected = dict(task_id=task, method='lolbo', split='train', seed=0,
                    status='complete', charged_observations=1049,
                    vae_sha256=plan['vae_sha256'], bao_sha256=plan['bao_sha256'],
                    train_split_sha256=plan['train_sha256'], test_split_sha256=plan['test_sha256'],
                    unresolved_database_calls=0, update_e2e=False)
    if history['task_id'] != task or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Incomplete or incompatible query source: ' + task)
    rows = history['observations']
    if len(rows) != 1049:
        raise ValueError('Incomplete source observations: ' + task)
    distinct = {}
    censored = duplicates = canonicalized = 0
    for index, row in enumerate(rows):
        extra = row['extra_info']
        if extra['charged_index'] != index or extra['oracle'] != 'postgres_runtime':
            raise ValueError('Non-measured or misaligned query observation')
        runtime = float(row['objectives'][0])
        if not np.isfinite(runtime) or runtime <= 0 or runtime != float(extra['runtime']):
            raise ValueError('Invalid query runtime')
        if row['trial_state'] != int(extra['censored']):
            raise ValueError('Censoring metadata mismatch')
        if extra['censored']:
            censored += 1
            continue
        key = tuple(extra['plan'])
        if not key or len(key) % 5 or any(type(v) is not int for v in key):
            raise ValueError('Invalid encoded query plan')
        if canonicalize is not None:
            canonical = tuple(canonicalize(list(key)))
            canonicalized += int(canonical != key)
            key = canonical
        if key in distinct:
            duplicates += 1
            continue
        distinct[key] = (index, runtime)
    counts = dict(total=len(rows), censored=censored, duplicates=duplicates, retained=len(distinct))
    if canonicalize is not None:
        counts['canonicalized'] = canonicalized
    return distinct, counts


def read_task(db, task):
    row = db.execute('SELECT observations,source_sha256,payload_sha256,payload FROM tasks WHERE id=?', (task,)).fetchone()
    if row is None or hashlib.sha256(row[3]).hexdigest() != row[2]:
        raise ValueError('Missing or damaged query source: ' + task)
    with np.load(io.BytesIO(row[3]), allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    if (result['X'].shape != (row[0], DIM) or result['y'].shape != (row[0],)
            or row[0] < 2 or not np.isfinite(result['X']).all()
            or not np.isfinite(result['y']).all() or np.any(result['y'] <= 0)):
        raise ValueError('Invalid query feature/runtime alignment')
    result.update(source_sha256=row[1], payload_sha256=row[2])
    return result


def load_corpus(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    train, test = (SOURCE / 'train800.txt').read_text().split(), (SOURCE / 'test200.txt').read_text().split()
    if (manifest['status'] != 'complete' or manifest['domain'] != 'ceb_query_runtime'
            or manifest['policy'] != POLICY or manifest['acquisition_wrapper'] is not None
            or manifest['train_split_sha256'] != shared.digest(SOURCE / 'train800.txt')
            or manifest['test_split_sha256'] != shared.digest(SOURCE / 'test200.txt')
            or manifest['vae_sha256'] != shared.digest(SOURCE / 'native/vae/CEB_64_author.ckpt')
            or manifest['codec_sha256'] != shared.digest(SOURCE / 'native/codec/codec.py')
            or set(manifest['source_tasks']) & set(test)
            or set(manifest['source_tasks']) | set(manifest['excluded_tasks']) != set(train)
            or set(manifest['fsbo_train_tasks']) & set(manifest['fsbo_validation_tasks'])
            or set(manifest['fsbo_train_tasks']) | set(manifest['fsbo_validation_tasks']) != set(manifest['source_tasks'])):
        raise ValueError('Query corpus provenance or split mismatch')
    db = sqlite3.connect('file:' + str(directory / 'corpus.sqlite') + '?mode=ro', uri=True, timeout=120)
    if shared.get_meta(db, 'manifest') != manifest:
        raise ValueError('Query database/manifest mismatch')
    return manifest, db


def export(args):
    import torch
    torch.set_num_threads(args.threads)
    plan = json.loads((SOURCE / 'plan.json').read_text())
    train, test = (SOURCE / 'train800.txt').read_text().split(), (SOURCE / 'test200.txt').read_text().split()
    if len(train) != 800 or len(set(train)) != 800 or len(test) != 200 or len(set(test)) != 200 or set(train) & set(test):
        raise ValueError('Invalid query train800/test200 split')
    for name, expected in [('train800.txt', plan['train_sha256']), ('test200.txt', plan['test_sha256']),
                           ('native/vae/CEB_64_author.ckpt', plan['vae_sha256'])]:
        if shared.digest(SOURCE / name) != expected:
            raise ValueError('Changed query source asset: ' + name)
    native = SOURCE / 'native'
    sys.path.insert(0, str(native))
    from vae.model import VAEModule, encode_batch
    from codec.codec import AliasesCodec
    from workload.workloads import get_workload_set
    workloads = get_workload_set('CEB_3K')
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.06)
    vae = VAEModule.load_from_checkpoint(str(native / 'vae/CEB_64_author.ckpt'), map_location=args.device)
    vae.eval().requires_grad_(False).to(args.device)
    identity = dict(domain='ceb_query_runtime', policy=POLICY,
        representation='frozen_author_VAE_posterior_mean_64', normalization='train800_coordinate_minmax',
        source=str(SOURCE), source_plan_sha256=shared.digest(SOURCE / 'plan.json'),
        train_split_sha256=plan['train_sha256'], test_split_sha256=plan['test_sha256'],
        vae_sha256=plan['vae_sha256'], encoder_sha256=shared.digest(native / 'vae/model.py'),
        plan_representation='oracle_canonical_alias_encoding',
        codec_sha256=shared.digest(native / 'codec/codec.py'),
        exporter_sha256=shared.digest(__file__), oracle_calls=0, acquisition_wrapper=None)
    args.output.mkdir(parents=True, exist_ok=True)
    with shared.lock(args.output / 'export.lock'):
        db = shared.connect(args.output / 'corpus.sqlite')
        previous = shared.get_meta(db, 'identity')
        if previous is not None and previous != identity:
            raise ValueError('Export resume identity changed')
        with db:
            shared.put_meta(db, 'identity', identity)
        lower, upper = np.full(DIM, np.inf), np.full(DIM, -np.inf)
        tasks, excluded, hashes, source_hashes, counts, sql_hashes = [], {}, {}, {}, {}, {}
        observations, started = 0, time.monotonic()
        for index, task in enumerate(train):
            history_path = SOURCE / 'outputs' / task / 'lolbo/seed0/history.json'
            raw = history_path.read_bytes()
            source_hash = hashlib.sha256(raw).hexdigest()
            history = json.loads(raw)
            import re
            match = re.fullmatch(r'CEB_(\d+[A-Z])(\d+)', task)
            sql_path = native / 'workload/ceb-3k' / match[1].lower() / ((match[1] + match[2]).lower() + '.sql')
            sql_hashes[task] = shared.digest(sql_path)
            if sql_hashes[task] != history['meta_info']['sql_sha256']:
                raise ValueError('Query SQL differs from measured source')
            definition = workloads.extract_query(sql_path.read_text(), task)
            codec = AliasesCodec(definition.all_tables)
            def canonicalize(encoded):
                tree = codec.decode(definition.query_tables, encoded)
                canonical = codec.encode(tree)
                restored = codec.decode(definition.query_tables, canonical.copy())
                if not tree.equal(restored):
                    raise ValueError('Canonical encoding changed the executed query plan')
                return canonical
            distinct, counts[task] = source_rows(history, task, plan, canonicalize)
            if len(distinct) < 2:
                excluded[task] = dict(reason='fewer_than_two_distinct_uncensored_plans', counts=counts[task])
                continue
            cached = db.execute('SELECT source_sha256 FROM tasks WHERE id=?', (task,)).fetchone()
            if cached:
                if cached[0] != source_hash:
                    raise ValueError('Query history changed during resume')
                data = read_task(db, task)
            else:
                plans = list(distinct)
                features = []
                with torch.inference_mode():
                    for start in range(0, len(plans), args.batch_size):
                        tokens = encode_batch(plans[start:start + args.batch_size], vae.vocab).to(args.device)
                        mean, _ = vae.model.encode(tokens)
                        features.append(mean.reshape(-1, DIM).cpu().numpy().astype(np.float32))
                data = dict(X=np.concatenate(features),
                    y=np.asarray([distinct[p][1] for p in plans], dtype=np.float64),
                    source_rows=np.asarray([distinct[p][0] for p in plans], dtype=np.int32))
                if data['X'].shape != (len(plans), DIM) or not np.isfinite(data['X']).all():
                    raise ValueError('Invalid query encoder output')
                payload = shared.pack_arrays(**data)
                with db:
                    db.execute('INSERT INTO tasks VALUES (?,?,?,?,?)', (task, len(plans), source_hash,
                        hashlib.sha256(payload).hexdigest(), sqlite3.Binary(payload)))
                data = read_task(db, task)
            lower = np.minimum(lower, data['X'].min(axis=0))
            upper = np.maximum(upper, data['X'].max(axis=0))
            tasks.append(task)
            hashes[task], source_hashes[task] = data['payload_sha256'], source_hash
            observations += len(data['y'])
            status = dict(status='exporting', completed=index + 1, total=800,
                eligible_tasks=len(tasks), observations=observations,
                elapsed_seconds=time.monotonic() - started, device=args.device, oracle_calls=0)
            shared.atomic_json(args.output / 'status.json', status)
            if (index + 1) % 10 == 0:
                print(json.dumps(status), flush=True)
        if len(tasks) < 2:
            raise ValueError('Insufficient source tasks')
        rng = np.random.RandomState(123)
        valid = {tasks[i] for i in rng.permutation(len(tasks))[:max(1, len(tasks) // 5)]}
        manifest = dict(identity, status='complete', purpose='production_training', source_tasks=tasks,
            excluded_tasks=excluded, fsbo_train_tasks=[t for t in tasks if t not in valid],
            fsbo_validation_tasks=[t for t in tasks if t in valid], observations=observations,
            lower=lower.tolist(), upper=upper.tolist(), dimension=DIM, task_hashes=hashes,
            source_hashes=source_hashes, sql_hashes=sql_hashes, observation_audit=counts)
        with db:
            shared.put_meta(db, 'manifest', manifest)
        db.close()
        shared.atomic_json(args.output / 'manifest.json', manifest)
        shared.atomic_json(args.output / 'status.json', dict(status, status='complete'))


def fsbo(args):
    sys.path.insert(0, str(TRAINER.parent))
    import train_taskset_transfer as trainer
    manifest, db = load_corpus(args.corpus)
    tasks = {}
    for task in manifest['source_tasks']:
        data = read_task(db, task)
        if data['payload_sha256'] != manifest['task_hashes'][task]:
            raise ValueError('Source payload differs from manifest')
        tasks[task] = dict(X=shared.normalize(data['X'], manifest), y_maximize=-data['y'],
                          y_minimize=data['y'], source_sha256=data['source_sha256'])
    db.close()
    def identity(settings, _manifest):
        return dict(domain='ceb_query_runtime', method='fsbo', purpose=settings.purpose,
            corpus_sha256=shared.digest(args.corpus / 'manifest.json'),
            train_split_sha256=manifest['train_split_sha256'], test_split_sha256=manifest['test_split_sha256'],
            representation=manifest['representation'], seed=args.seed,
            trainer_sha256=shared.digest(__file__), shared_training_loop_sha256=shared.digest(TRAINER),
            device='cpu', threads=args.threads, acquisition_wrapper=None)
    trainer.identity = identity
    args.method, args.purpose, args.resume = 'fsbo', 'production_training' if args.steps == 100000 else 'component_check', True
    args.output.mkdir(parents=True, exist_ok=True)
    with shared.lock(args.output / '.lock'):
        trainer.fsbo(args, manifest, tasks)


def rgpe_fit(args):
    from peptide_rgpe_exact import exact_objective
    sys.path.insert(0, str(shared.RGPE_REPO))
    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
    from smac.epm.util_funcs import get_types
    from rgpe.utils import get_gaussian_process
    manifest, db = load_corpus(args.corpus)
    if args.task not in manifest['source_tasks']:
        raise ValueError('RGPE source is outside query training tasks')
    data = read_task(db, args.task)
    db.close()
    if data['payload_sha256'] != manifest['task_hashes'][args.task]:
        raise ValueError('Query payload differs from manifest')
    x, y = shared.normalize(data['X'], manifest).astype(np.float64), data['y']
    cs = ConfigurationSpace()
    cs.add_hyperparameters([UniformFloatHyperparameter('z%03d' % i, 0.0, 1.0) for i in range(DIM)])
    types, bounds = get_types(cs, None)
    seed = args.seed + manifest['source_tasks'].index(args.task)
    model = get_gaussian_process(cs, types, bounds, np.random.RandomState(seed), None)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(domain='ceb_query_runtime', method='rgpe_source_gp', task=args.task,
        corpus_sha256=shared.digest(args.corpus / 'manifest.json'), source_payload_sha256=data['payload_sha256'],
        representation=manifest['representation'], kernel='upstream_Matern2.5_ARD', normalize_y=True,
        seed=seed, trainer_sha256=shared.digest(__file__),
        gp_implementation_sha256=shared.digest(shared.RGPE_REPO / 'rgpe/utils.py'),
        execution_engine='exact_trace_contraction', objective_sha256=shared.digest(Path(__file__).with_name('peptide_rgpe_exact.py')),
        acquisition_wrapper=None)
    with shared.lock(args.output / 'fit_locks' / (args.task + '.lock')):
        bank_path = args.output / 'models.sqlite'
        with shared.lock(args.output / 'bank.lock', blocking=True):
            with sqlite3.connect(str(bank_path), timeout=120) as bank:
                bank.execute('CREATE TABLE IF NOT EXISTS models (task TEXT PRIMARY KEY, metadata TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL)')
                row = bank.execute('SELECT metadata,sha256,payload FROM models WHERE task=?', (args.task,)).fetchone()
        if row:
            meta = json.loads(row[0])
            if any(meta.get(k) != v for k, v in identity.items()) or hashlib.sha256(row[2]).hexdigest() != row[1] or not meta['reload_verified']:
                raise ValueError('Existing query GP provenance mismatch')
            print(json.dumps(dict(task=args.task, status='already_complete')), flush=True)
            return
        started = time.monotonic()
        def progress(value):
            shared.atomic_json(args.output / 'progress' / (args.task + '.json'),
                dict(value, task=args.task, status='fitting', elapsed_seconds=time.monotonic() - started))
        (args.output / 'progress').mkdir(exist_ok=True)
        with exact_objective(model, callback=progress) as performance:
            model.train(x, y)
        count = min(16, len(x) // 2)
        probe = np.vstack((x[:count], (x[:count] + x[count:2 * count]) / 2))
        mean, variance = model.predict(probe)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError('Invalid query GP posterior')
        payload = pickle.dumps(model, protocol=4)
        restored = pickle.loads(payload)
        after = restored.predict(probe)
        if not np.allclose(mean, after[0]) or not np.allclose(variance, after[1]):
            raise ValueError('Query GP posterior changed after reload')
        identity.update(optimizer_restarts=model.n_opt_restarts, likelihood_evaluations=model._n_ll_evals,
            nll_seconds=performance['seconds'], observations=len(y), reload_verified=True,
            elapsed_seconds=time.monotonic() - started)
        with shared.lock(args.output / 'bank.lock', blocking=True):
            with sqlite3.connect(str(bank_path), timeout=120) as bank:
                bank.execute('PRAGMA synchronous=FULL')
                bank.execute('INSERT INTO models VALUES (?,?,?,?)', (args.task, json.dumps(identity),
                    hashlib.sha256(payload).hexdigest(), sqlite3.Binary(payload)))
                completed = bank.execute('SELECT COUNT(*) FROM models').fetchone()[0]
            shared.atomic_json(args.output / 'status.json', dict(status='complete' if completed == len(manifest['source_tasks']) else 'training',
                completed=completed, total=len(manifest['source_tasks']), last_task=args.task))
        print(json.dumps(identity), flush=True)


def dispatch(args):
    manifest, db = load_corpus(args.corpus)
    db.close()
    args.output.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))[:args.workers]
    def worker(cpu, tasks):
        failures = []
        for task in tasks:
            log = args.output / (task + '.log')
            with log.open('a') as stream:
                command = [sys.executable, __file__, 'rgpe-fit', '--corpus', str(args.corpus),
                    '--output', str(args.output), '--task', task, '--cpu', str(cpu), '--seed', str(args.seed)]
                code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                failures.append(dict(task=task, returncode=code))
        return failures
    with shared.lock(args.output / 'dispatch.lock'):
        shared.atomic_json(args.output / 'dispatch.json', dict(status='running', pid=os.getpid(), workers=len(cpus), total=len(manifest['source_tasks'])))
        with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
            jobs = [pool.submit(worker, cpu, manifest['source_tasks'][i::len(cpus)]) for i, cpu in enumerate(cpus)]
            failures = [failure for job in as_completed(jobs) for failure in job.result()]
        shared.atomic_json(args.output / 'dispatch.json', dict(status='failed' if failures else 'complete', failures=failures, total=len(manifest['source_tasks'])))
        if failures:
            raise RuntimeError('Query RGPE fits failed; inspect per-task logs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('export', 'fsbo', 'rgpe-fit', 'rgpe-dispatch'))
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--task')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.command != 'export' and (args.corpus is None or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        parser.error('Model training requires --corpus and CUDA_VISIBLE_DEVICES empty')
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    functions = {'export': export, 'fsbo': fsbo, 'rgpe-fit': rgpe_fit, 'rgpe-dispatch': dispatch}
    try:
        functions[args.command](args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        name = ('failure_' + args.task) if args.task else ('failure_' + args.command)
        shared.atomic_json(args.output / (name + '.json'), dict(error=repr(exc), command=args.command))
        raise


if __name__ == '__main__':
    main()
