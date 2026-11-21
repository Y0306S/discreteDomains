#!/usr/bin/env python3
"""TaskSet RGPE evaluations with durable proposals and complete selection evidence."""

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
from ConfigSpace.read_and_write import json as cs_json
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.history import observation, history_exists, load_openbox_history, write_rotating_openbox_history
from taskset_transfer_common import atomic_json, digest, METRIC, PROTOCOL, read_split


def signature(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def successful(rows):
    return [r for r in rows if r['trial_state'] == 0 and math.isfinite(r['objectives'][0])]


def validate_resume(history, identity):
    if history['task_id'] != identity['task_id'] or history['meta_info'] != identity:
        raise ValueError('RGPE resume identity, checkpoint, or protocol changed')
    if len(history['observations']) > identity['budget']:
        raise ValueError('Saved history exceeds the requested budget')
    return history['observations']


def validate_selection(result, candidates, meta):
    scores, index = result['acquisition']['values'], result['index']
    weights = result['weights']
    if (result['cache_sha256'] != meta['cache_sha256'] or result['n_sources'] != 838
            or len(weights) != 839 or not all(math.isfinite(w) and w >= 0 for w in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-8)
            or [r['task'] for r in result['source_weights']] != meta['source_tasks']
            or [r['weight'] for r in result['source_weights']] != weights[:-1]
            or result['target_weight'] != weights[-1]
            or not isinstance(index, int) or not 0 <= index < len(candidates)
            or len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores)
            or index != max(range(len(scores)), key=scores.__getitem__)):
        raise ValueError('RGPE selection evidence does not match the evaluated proposal')


def suggest(cs, rows, directory, assets, task, trial):
    cs.seed(1 + 1009 * trial)
    seen = {signature(r['config']) for r in rows}
    candidates = []
    need = 1 if len(rows) < 3 or len(successful(rows)) < 3 else 256
    for _ in range(10000):
        config = dict(cs.sample_configuration().get_dictionary())
        key = signature(config)
        if key not in seen:
            candidates.append(config)
            seen.add(key)
        if len(candidates) == need:
            break
    if len(candidates) != need:
        raise ValueError('Could not construct distinct unseen RGPE candidates')
    if need == 1:
        return candidates[0], dict(phase='random_initialization', candidate_count=1)
    source = [(i+1, r) for i,r in enumerate(rows) if r in successful(rows)]
    meta = json.loads((assets/'ensemble.json').read_text())
    request = dict(task_id=task, cache_sha256=meta['cache_sha256'], seed=1+1009*trial, n_trials=100,
                   target_configs=[r['config'] for _,r in source], y_minimize=[r['objectives'][0] for _,r in source],
                   pending_configs=candidates)
    work = directory/'_surrogate'
    work.mkdir(exist_ok=True)
    atomic_json(work/'request.json', request)
    result_path = work/'response.json'
    result_path.unlink(missing_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH='/root/data/cys/baseline/repos/transfer-hpo-framework')
    command = ['/root/data/cys/baseline/envs/bl_rgpe/bin/python', str(Path(__file__).with_name('taskset_rgpe_worker.py')),
               'suggest', '--assets', str(assets), '--request', str(work/'request.json'), '--out', str(result_path)]
    with (work/'worker.log').open('w') as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=2400)
    result = json.loads(result_path.read_text())
    validate_selection(result, candidates, meta)
    return candidates[result['index']], dict(result, phase='rgpe_taf', candidates=candidates,
                                             source_observation_indices=[i for i,_ in source])


def run(args):
    split = read_split()
    if args.task not in split['test']:
        raise ValueError('RGPE target is outside test210')
    recipe = json.loads((args.campaign/'recipe.json').read_text())
    assets = args.campaign/'assets'
    meta = json.loads((assets/'ensemble.json').read_text())
    if (digest(assets/'ensemble.pkl') != recipe['cache_sha256']
            or meta['cache_sha256'] != recipe['cache_sha256'] or set(meta['source_tasks']) != set(split['train'])):
        raise ValueError('RGPE source checkpoint identity changed')
    directory = args.campaign/('preflight' if args.check else 'test')/args.task/'rgpe/seed1'
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(domain='taskset', optimizer='rgpe', task_id=args.task, seed=1, budget=args.trials,
                    protocol=PROTOCOL, objective_metric=METRIC, cache_sha256=recipe['cache_sha256'],
                    schema_sha256=meta['schema_sha256'], runner_sha256=recipe['runner_sha256'],
                    worker_sha256=recipe['worker_sha256'], ranking_samples=1000, initial_trials=3,
                    candidates=256, sampling_mode='bootstrap', acquisition='TAF', dummy=False)
    history_path = directory/'history.json'
    rows = validate_resume(load_openbox_history(history_path), identity) if history_exists(history_path) else []
    if len(rows) == args.trials:
        return
    adapter = TaskSetAdapter(task={'task_id': args.task}, dummy=False, oracle_work_dir=directory/'_oracle',
                             baseline_cache=args.campaign/'task_baselines.json', objective_metric=METRIC,
                             warm_timeout=3600, replica_timeout=2400, **PROTOCOL)
    cs = adapter.search_space()
    if cs != cs_json.read((assets/'configspace.json').read_text()):
        raise ValueError('Target and source ConfigSpace differ')
    pending_path = directory/'pending.json'
    while len(rows) < args.trials:
        trial = len(rows)+1
        if pending_path.exists():
            pending = json.loads(pending_path.read_text())
            if pending['identity'] != identity:
                raise ValueError('Pending proposal belongs to another run')
            if rows and pending['trial'] == trial-1 and rows[-1]['config'] == pending['config']:
                pending_path.unlink()
                continue
            if pending['trial'] != trial:
                raise ValueError('Pending proposal is not the next evaluation')
            if pending.get('evaluation_started_at'):
                pending.setdefault('interruptions', []).append(dict(started_at=pending['evaluation_started_at'],
                    noticed_at=time.time(), elapsed_upper_bound_seconds=time.time()-pending['evaluation_started_at']))
        else:
            started = time.monotonic()
            config, diagnostic = suggest(cs, rows, directory, assets, args.task, trial)
            cs.check_configuration(Configuration(cs, values=config))
            pending = dict(identity=identity, trial=trial, config=config, selection=diagnostic,
                           proposal_seconds=time.monotonic()-started, interruptions=[])
        pending['evaluation_started_at'] = time.time()
        atomic_json(pending_path, pending)
        started = time.monotonic()
        result = adapter.evaluate_config(pending['config'])
        ok = bool(result.feasible and math.isfinite(result.objective))
        info = dict(result.info, optimizer_backend='rgpe_taf_taskset', selection=pending['selection'],
                    proposal_seconds=pending['proposal_seconds'], interruptions=pending['interruptions'])
        rows.append(observation(pending['config'], result.objective, trial_state=0 if ok else 1,
                                elapsed_time=time.monotonic()-started,
                                extra={'source':pending['selection']['phase'], 'info':info}))
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
    args = parser.parse_args()
    if args.trials != 100 and not args.check:
        parser.error('Short runs require an isolated preflight directory')
    run(args)
