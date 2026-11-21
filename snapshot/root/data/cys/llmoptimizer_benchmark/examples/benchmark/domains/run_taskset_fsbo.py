#!/usr/bin/env python3
"""TaskSet evaluation using the existing isolated paper-FSBO inference worker."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ConfigSpace import Configuration
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.dlhpo.taskset_lolbo import TaskSetCodec
from mindware.domains.history import observation, history_exists, load_openbox_history, write_rotating_openbox_history
from mindware.domains.search import _optuna_distributions
from taskset_transfer_common import atomic_json, digest, METRIC, PROTOCOL, read_split


def signature(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def successful(rows):
    return [r for r in rows if r['trial_state'] == 0 and math.isfinite(r['objectives'][0])]


def validate_resume(history, identity):
    if history['task_id'] != identity['task_id'] or history['meta_info'] != identity:
        raise ValueError('FSBO resume identity, checkpoint, or evaluation protocol changed')
    if len(history['observations']) > identity['budget']:
        raise ValueError('Saved history exceeds the requested evaluation budget')
    return history['observations']


def suggest(cs, codec, rows, directory, assets, seed, trial, n_startup):
    cs.seed(seed + 1009 * trial)
    seen = {signature(r['config']) for r in rows}
    candidates = []
    need = 1 if len(rows) < n_startup or len(successful(rows)) < 2 else 256
    for _ in range(10000):
        cfg = dict(cs.sample_configuration().get_dictionary())
        key = signature(cfg)
        if key not in seen:
            candidates.append(cfg)
            seen.add(key)
        if len(candidates) == need:
            break
    if len(candidates) != need:
        raise RuntimeError('Could not construct the distinct FSBO candidate pool')
    if need == 1:
        return candidates[0], {'phase': 'random_initialization', 'candidate_count': 1}
    source = successful(rows)
    meta = json.loads((assets / 'meta.json').read_text())
    request = dict(schema_fingerprint=meta['schema_fingerprint'],
                   X_obs=[codec.encode(r['config']).tolist() for r in source],
                   y_obs=[-r['objectives'][0] for r in source],
                   X_pen=[codec.encode(c).tolist() for c in candidates], seed=seed,
                   finetune_epochs=100, target_scaling=meta.get('target_scaling', 'raw'))
    work = directory / '_surrogate'
    work.mkdir(exist_ok=True)
    atomic_json(work / 'request.json', request)
    result_path = work / 'response.json'
    result_path.unlink(missing_ok=True)
    command = ['/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python',
               str(ROOT / 'examples/benchmark/fsbo_worker.py'), 'suggest', '--ckpt', str(assets),
               '--request', str(work / 'request.json'), '--out', str(result_path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH='/root/data/cys/baseline/repos/FSBO')
    with (work / 'worker.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    result = json.loads(result_path.read_text())
    if result['checkpoint_sha256'] != meta['weights_sha256']:
        raise ValueError('The inference worker used a different FSBO checkpoint')
    if result.get('target_scaling', 'raw') != request['target_scaling']:
        raise ValueError('The inference worker ignored the requested target scaling')
    index = result['index']
    scores = result['acquisition']['values']
    if (not isinstance(index, int) or not 0 <= index < len(candidates)
            or len(scores) != len(candidates) or not all(math.isfinite(x) for x in scores)
            or index != max(range(len(scores)), key=scores.__getitem__)):
        raise ValueError('FSBO selected index does not match saved candidate scores')
    return candidates[index], dict(result, phase='fsbo_ei', candidates=candidates,
                                   source_observation_indices=[i + 1 for i, r in enumerate(rows)
                                                               if r in source])


def run(args):
    split = read_split()
    if args.task not in split['test']:
        raise ValueError('FSBO target is outside the pinned 210-task test split')
    recipe = json.loads((args.campaign / 'recipe.json').read_text())
    assets = args.campaign / 'assets'
    meta = json.loads((assets / 'meta.json').read_text())
    if (digest(assets / 'weights') != recipe['weights_sha256']
            or meta['weights_sha256'] != recipe['weights_sha256']):
        raise ValueError('FSBO weights changed')
    if not set(meta['train_tasks'] + meta['validation_tasks']) <= set(split['train']):
        raise ValueError('FSBO source tasks are outside train838')
    directory = args.campaign / ('preflight' if args.check else 'test') / args.task / 'fsbo/seed1'
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(domain='taskset', optimizer='fsbo', task_id=args.task, seed=1,
                    budget=args.trials, protocol=PROTOCOL, objective_metric=METRIC,
                    weights_sha256=recipe['weights_sha256'], schema_fingerprint=meta['schema_fingerprint'],
                    runner_sha256=recipe['runner_sha256'], worker_sha256=recipe['worker_sha256'],
                    initial_trials=3, candidates=256, finetune_epochs=100, dummy=bool(args.dummy))
    if 'target_scaling' in meta:
        identity['target_scaling'] = meta['target_scaling']
    history_path = directory / 'history.json'
    rows = validate_resume(load_openbox_history(history_path), identity) if history_exists(history_path) else []
    if len(rows) == args.trials:
        return
    adapter = TaskSetAdapter(task={'task_id': args.task}, dummy=args.dummy,
                             oracle_work_dir=directory / '_oracle', baseline_cache=args.campaign/'task_baselines.json',
                             objective_metric=METRIC, warm_timeout=3600, replica_timeout=2400, **PROTOCOL)
    cs = adapter.search_space()
    codec = TaskSetCodec(cs, _optuna_distributions(cs))
    if codec.dim != meta['dim']:
        raise ValueError('FSBO target feature dimension changed')
    # Re-encoding a pinned training example verifies conditional feature semantics.
    example = meta['encoding_probe']
    import numpy as np
    if not np.allclose(codec.encode(example['config']), example['vector'], atol=1e-7, rtol=0):
        raise ValueError('FSBO training and target encodings differ')
    pending_path = directory / 'pending.json'
    while len(rows) < args.trials:
        trial = len(rows) + 1
        if pending_path.exists():
            pending = json.loads(pending_path.read_text())
            if pending['identity'] != identity:
                raise ValueError('Pending proposal belongs to another run')
            if pending['trial'] == trial - 1 and rows[-1]['config'] == pending['config']:
                pending_path.unlink()
                continue
            if pending['trial'] != trial:
                raise ValueError('Pending proposal is not the next evaluation')
            if pending.get('evaluation_started_at'):
                pending.setdefault('interruptions', []).append(dict(
                    started_at=pending['evaluation_started_at'], noticed_at=time.time(),
                    elapsed_upper_bound_seconds=time.time()-pending['evaluation_started_at']))
        else:
            started = time.monotonic()
            config, diagnostic = suggest(cs, codec, rows, directory, assets, 1, trial, 3)
            cs.check_configuration(Configuration(cs, values=config))
            pending = dict(identity=identity, trial=trial, config=config, selection=diagnostic,
                           proposal_seconds=time.monotonic()-started, interruptions=[])
        pending['evaluation_started_at'] = time.time()
        atomic_json(pending_path, pending)
        started = time.monotonic()
        result = adapter.evaluate_config(pending['config'])
        ok = bool(result.feasible and math.isfinite(result.objective))
        info = dict(result.info, optimizer_backend='fsbo_paper_cash', selection=pending['selection'],
                    proposal_seconds=pending['proposal_seconds'], interruptions=pending['interruptions'])
        rows.append(observation(pending['config'], result.objective, trial_state=0 if ok else 1,
                                elapsed_time=time.monotonic()-started,
                                extra={'source': pending['selection']['phase'], 'info': info}))
        write_rotating_openbox_history(history_path, task_id=args.task, observations=rows, meta=identity)
        pending_path.unlink()
        atomic_json(directory/'progress.json', dict(identity=identity, attempts=len(rows),
            successful=len(successful(rows)), status='complete' if len(rows)==args.trials else 'running',
            updated_at=time.time()))
        print(json.dumps(dict(task=args.task, trial=trial, objective=rows[-1]['objectives'][0],
                              successful=ok, phase=pending['selection']['phase'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    args = parser.parse_args()
    if (args.dummy or args.trials != 100) and not args.check:
        parser.error('Short or dummy runs require a separate preflight directory')
    run(args)
