#!/usr/bin/env python3
"""Causal CPU RF annotations of existing train80 histories; no ML oracle or SFT."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import importlib.metadata
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from examples.benchmark.semanticopt_ml_sft import (
    TRAIN80, canonical, digest, history_prefix_digest, original_history_length,
    validate_observations, validate_steps,
)
from mindware.components.optimizers.transfer_hpo.protocol import (
    HISTORY_ROOT, allowed_task_ids, atomic_json, checked_source_path, sha256_file,
)

OUTPUT = REPO / 'benchmark_data/semanticopt_ml_train80_v1'
INVENTORY = REPO / 'benchmark_data/bolt_ml_top10_train80_v1/tasks.json'
RF_SETTINGS = {
    'log_y': False, 'num_trees': 10, 'do_bootstrapping': True,
    'n_points_per_tree': -1, 'ratio_features': 5 / 6, 'min_samples_split': 3,
    'min_samples_leaf': 3, 'max_depth': 2**20, 'eps_purity': 1e-8,
    'max_num_nodes': 2**20, 'n_jobs': 1,
}
POLICY = {
    'initial_observations': 5,
    'empty_success_prefix': 'extend initial observations through first successful outcome',
    'horizon': 'all original observations after warmup; no windows or subsampling',
    'short_history': 'explicit exclusion if no decision remains after warmup',
    'objective_targets': 'original full-precision successful objectives, no transforms or sentinels',
    'encoding': 'OpenBox ConfigSpace array, inactive coordinates imputed with defaults',
    'uncertainty': 'population standard deviation of tree predictions; no variance floor',
    'failure_model': 'sklearn RandomForestClassifier on prior trial states only; emitted after first failure',
    'topups': 'exclude trailing topup_* observations from fitting, context and supervised targets',
    'seed': '42 + SHA256(task_id) first 32 bits, modulo 2**31-1',
    'allocation_leaf_cap': 'min(2**20, max(2, floor(prefix_fit_rows / min_samples_leaf))); same feasible trees',
}


class InsufficientHistory(ValueError):
    pass


def allocation_leaf_cap(n_rows):
    return min(RF_SETTINGS['max_num_nodes'], max(2, n_rows // RF_SETTINGS['min_samples_leaf']))


def task_seed(task_id):
    import hashlib
    return (42 + int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)) % (2**31 - 1)


def replay_steps(observations, cs, algorithm, task_type, seed):
    import numpy as np
    from ConfigSpace import Configuration
    from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
    from openbox.utils.config_space import convert_configurations_to_array
    from openbox.utils.util_funcs import get_types
    from sklearn.ensemble import RandomForestClassifier

    end = original_history_length(observations)
    original = observations[:end]
    recorded = validate_observations(original, cs, algorithm, task_type)
    successes = [i for i, obs in enumerate(original) if obs['trial_state'] == 0]
    if not successes:
        raise InsufficientHistory('No successful original observation for the objective forest')
    warmup = max(POLICY['initial_observations'], successes[0] + 1)
    if warmup >= end:
        raise InsufficientHistory(f'{end} original observations leave no decision after {warmup} initial observations')
    configs = [Configuration(cs, values=row['configuration']) for row in recorded]
    X = convert_configurations_to_array(configs)
    if not np.isfinite(X).all():
        raise ValueError('Non-finite ConfigSpace encoding')
    types, bounds = get_types(cs)
    model = skRandomForestWithInstances(types, bounds, seed=seed, **RF_SETTINGS)
    failure_model = RandomForestClassifier(
        n_estimators=RF_SETTINGS['num_trees'], bootstrap=True, n_jobs=1,
        max_features=max(1, int(len(types) * RF_SETTINGS['ratio_features'])),
        min_samples_split=RF_SETTINGS['min_samples_split'],
        min_samples_leaf=RF_SETTINGS['min_samples_leaf'],
        max_depth=RF_SETTINGS['max_depth'], max_leaf_nodes=RF_SETTINGS['max_num_nodes'],
        min_impurity_decrease=RF_SETTINGS['eps_purity'],
        random_state=np.random.RandomState((seed + 1) % (2**31 - 1)),
    )
    steps = []
    for index in range(warmup, end):
        # Neither the current outcome nor any later outcome enters either fit.
        successful = [i for i in range(index) if original[i]['trial_state'] == 0]
        y = np.asarray([original[i]['objectives'][0] for i in successful], dtype=float)
        # sklearn preallocates its best-first tree builder from max_leaf_nodes.
        # No tree can have more leaves than rows/min_samples_leaf, so bounding
        # this allocation preserves the same feasible trees and random draws.
        model.max_num_nodes = allocation_leaf_cap(len(successful))
        model.train(X[successful], y)
        mean, variance = model.predict(X[index:index + 1])
        mean, variance = float(mean.item()), float(variance.item())
        if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0:
            raise ValueError('Invalid RF predictive mean/variance')
        step = {'observation_index': index, 'prefix_sha256': history_prefix_digest(observations[:index]),
                'mean': mean, 'std': math.sqrt(variance),
                'objective_fit_observations': len(successful)}
        failures = np.asarray([int(obs['trial_state'] != 0) for obs in original[:index]])
        if failures.any():
            failure_model.set_params(max_leaf_nodes=allocation_leaf_cap(index))
            failure_model.fit(X[:index], failures)
            failure_column = list(failure_model.classes_).index(1)
            probability = float(failure_model.predict_proba(X[index:index + 1])[0, failure_column])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid failure probability')
            step['failure_probability'] = probability
            step['failure_fit_observations'] = index
        steps.append(step)
    return warmup, steps


def recipe_identity():
    from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
    return {
        'train_manifest_sha256': sha256_file(TRAIN80), 'inventory_sha256': sha256_file(INVENTORY),
        'code_sha256': sha256_file(__file__),
        'validator_sha256': sha256_file(REPO / 'examples/benchmark/semanticopt_ml_sft.py'),
        'backend': 'openbox.surrogate.base.rf_with_instances_sklearn.skRandomForestWithInstances',
        'backend_sha256': sha256_file(inspect.getfile(skRandomForestWithInstances)),
        'versions': {name: importlib.metadata.version(name)
                     for name in ('openbox', 'scikit-learn', 'numpy', 'ConfigSpace')},
        'rf_settings': RF_SETTINGS, 'policy': POLICY,
    }


def source_path(task_id):
    tt, dataset, algorithm = task_id.split('/')
    metric = 'bal_acc' if tt == 'cls' else 'mse'
    return checked_source_path(HISTORY_ROOT / tt / metric / dataset / f'{algorithm}.json')


def load_inventory():
    if os.environ.get('ML_TRANSFER_HISTORY_MIRROR'):
        raise ValueError('Replay requires the original history archive')
    allowed = allowed_task_ids(TRAIN80)
    rows = json.loads(INVENTORY.read_text())
    records = {row['task_id']: row for row in rows}
    if len(rows) != len(records) or set(records) != allowed:
        raise ValueError('Schema inventory must match the exact fixed train80 task set')
    for task_id, record in records.items():
        path = source_path(task_id)
        if path != Path(record['source_path']).resolve() or sha256_file(path) != record['source_sha256']:
            raise ValueError(f'Source differs from audited inventory: {task_id}')
    return records


def replay_task(record, identity):
    from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs, schema_fingerprint

    started = time.monotonic()
    task_id = record['task_id']
    if task_id not in allowed_task_ids(TRAIN80):
        raise ValueError('Task is outside train80')
    path = source_path(task_id)
    source_hash = sha256_file(path)
    if source_hash != record['source_sha256']:
        raise ValueError(f'Source changed before replay: {task_id}')
    payload = json.loads(path.read_text())
    observations = payload['observations']
    end = original_history_length(observations)
    if payload.get('topup_meta') and any(
            (obs.get('extra_info') or {}).get('source') != 'original_bo' for obs in observations[:end]):
        raise ValueError('Top-up archive has unmarked original observations')
    tt, _, algorithm = task_id.split('/')
    cs = build_hpo_cs(tt, algorithm, **record['context_source']['schema_kwargs'])
    if schema_fingerprint(cs) != record['schema_fingerprint']:
        raise ValueError(f'ConfigSpace differs from audited inventory: {task_id}')
    result = {'task_id': task_id, 'source_path': str(path), 'source_sha256': source_hash,
              'raw_observations': len(observations), 'original_observations': end,
              'excluded_topup_observations': len(observations) - end,
              'source_inventory_sha256': identity['inventory_sha256'],
              'schema_fingerprint': record['schema_fingerprint'],
              'schema_kwargs': record['context_source']['schema_kwargs'],
              'recipe_sha256': digest(identity)}
    seed = task_seed(task_id)
    try:
        warmup, steps = replay_steps(observations, cs, algorithm, tt, seed)
    except InsufficientHistory as exc:
        result.update(status='excluded', reason=str(exc))
    else:
        row = {
            'task_id': task_id, 'train_manifest_sha256': identity['train_manifest_sha256'],
            'source_path': str(path), 'source_sha256': source_hash,
            'objective_scale': 'archive_minimize', 'initial_observations': warmup,
            'teacher': {'name': identity['backend'], 'causal_prefix_only': True,
                        'settings': {**RF_SETTINGS, 'seed': seed, 'policy': POLICY},
                        'code_sha256': identity['code_sha256'], 'recipe_sha256': digest(identity),
                        'reconstructed_statistics': True},
            'steps': steps,
        }
        validate_steps(row, observations, cs, algorithm, tt)
        result.update(status='complete', annotation=row)
    if sha256_file(path) != source_hash:
        raise ValueError(f'Source changed during replay: {task_id}')
    result['seconds'] = time.monotonic() - started
    return result


def initialize_worker(cpu_queue):
    os.sched_setaffinity(0, {cpu_queue.get()})


def save_record(path, record):
    atomic_json(path, {'record_sha256': digest(record), 'record': record})


def load_record(path, record, identity):
    payload = json.loads(path.read_text())
    cached = payload['record']
    if (payload['record_sha256'] != digest(cached) or cached['task_id'] != record['task_id']
            or cached['source_sha256'] != record['source_sha256']
            or cached['recipe_sha256'] != digest(identity)
            or cached['status'] not in ('complete', 'excluded')):
        raise ValueError(f'Stale or changed replay result: {path}')
    return cached


def execute(args):
    cpus = [int(value) for value in args.cpus.split(',')]
    if len(cpus) != len(set(cpus)) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('CPU list must be unique and within the allowed affinity')
    if not 1 <= args.workers <= len(cpus):
        raise ValueError('Assign at least one distinct CPU per worker')
    os.sched_setaffinity(0, set(cpus))
    os.nice(10)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    records = load_inventory()
    identity = recipe_identity()
    selected = sorted(set(args.task_id or records))
    if not set(selected) <= set(records):
        raise ValueError('Requested task is not in the fixed train80 split')
    output = args.output.resolve()
    if output.is_relative_to(HISTORY_ROOT.resolve()):
        raise ValueError('Replay output cannot be inside the source archive')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'replay.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe_path = output / 'replay_recipe.json'
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != identity:
            raise ValueError('Output belongs to a different recipe; use a separate output directory')
        atomic_json(recipe_path, identity)
        (output / 'tasks').mkdir(exist_ok=True)
        started = time.monotonic()
        results, errors, pending = {}, {}, []
        for task_id in selected:
            path = output / 'tasks' / (digest(task_id) + '.json')
            if path.exists():
                results[task_id] = load_record(path, records[task_id], identity)
            else:
                pending.append(task_id)

        def status(state):
            payload = {
                'status': state, 'pid': os.getpid(), 'cpus': cpus, 'workers': args.workers,
                'selected_tasks': len(selected), 'allowed_tasks': len(records),
                'processed_tasks': len(results), 'annotated_tasks': sum(
                    r['status'] == 'complete' for r in results.values()),
                'excluded_tasks': {t: r['reason'] for t, r in results.items() if r['status'] == 'excluded'},
                'errors': errors, 'prediction_steps': sum(
                    len(r.get('annotation', {}).get('steps', [])) for r in results.values()),
                'excluded_topup_observations': sum(r['excluded_topup_observations'] for r in results.values()),
                'elapsed_seconds': time.monotonic() - started,
                'updated_unix': time.time(), 'ml_evaluator_calls': 0, 'llm_calls': 0,
                'qwen_training_started': False,
            }
            atomic_json(output / 'replay_status.json', payload)
            return payload

        status('running')
        context = multiprocessing.get_context('spawn')
        cpu_queue = context.Queue()
        for cpu in cpus[:args.workers]:
            cpu_queue.put(cpu)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                 initializer=initialize_worker, initargs=(cpu_queue,)) as pool:
            futures = {pool.submit(replay_task, records[t], identity): t for t in pending}
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    save_record(output / 'tasks' / (digest(task_id) + '.json'), result)
                    results[task_id] = result
                except Exception as exc:
                    errors[task_id] = f'{type(exc).__name__}: {exc}'
                status('running')
                print(canonical({'processed': len(results), 'total': len(selected),
                                 'task_id': task_id, 'error': errors.get(task_id)}), flush=True)
        cpu_queue.close()
        if errors:
            status('failed')
            return 1
        for task_id in selected:
            if sha256_file(source_path(task_id)) != records[task_id]['source_sha256']:
                errors[task_id] = 'Source changed before finalization'
        if sha256_file(TRAIN80) != identity['train_manifest_sha256']:
            errors['train_manifest'] = 'Train split changed during replay'
        if errors:
            status('failed')
            return 1
        annotations = [results[t]['annotation'] for t in selected if results[t]['status'] == 'complete']
        destination = output / 'annotated_trajectories.jsonl'
        temporary = output / f'annotations.{os.getpid()}.tmp'
        try:
            with temporary.open('w') as stream:
                for row in annotations:
                    stream.write(canonical(row) + '\n')
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        final_state = 'complete' if set(selected) == set(records) else 'sample_complete'
        report = status(final_state)
        manifest = {
            'format_version': 1, 'status': final_state, 'dataset_sha256': sha256_file(destination),
            'train_manifest_sha256': identity['train_manifest_sha256'], 'recipe_sha256': digest(identity),
            'represented_tasks': [row['task_id'] for row in annotations],
            'excluded_tasks': report['excluded_tasks'], 'trajectories': len(annotations),
            'horizon_policy': POLICY['horizon'], 'initial_observation_policy': POLICY['empty_success_prefix'],
            'raw_observations': sum(r['raw_observations'] for r in results.values()),
            'original_observations': sum(r['original_observations'] for r in results.values()),
            'excluded_topup_observations': report['excluded_topup_observations'],
            'prediction_steps': report['prediction_steps'], 'reconstructed_statistics': True,
            'original_source_hashes_rechecked': True, 'ml_evaluator_calls': 0, 'llm_calls': 0,
        }
        atomic_json(destination.with_suffix('.manifest.json'), manifest)
        print(canonical(report), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cpus', default='40,41,42,43,44,45,46,47')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--task-id', action='append', help='Timing sample only; cannot produce a complete SFT manifest')
    args = parser.parse_args()
    if not args.execute:
        print(canonical({'output': str(args.output), 'policy': POLICY, 'rf_settings': RF_SETTINGS,
                         'cpus': args.cpus, 'workers': args.workers, 'replay_started': False}))
        return 0
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
