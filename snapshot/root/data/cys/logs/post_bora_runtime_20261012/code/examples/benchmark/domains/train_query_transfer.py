#!/usr/bin/env python3
"""CPU training of CEB FSBO and per-query RGPE source GPs, with resumable artifacts."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from query_transfer_common import atomic_json, digest, load_corpus, run_lock

FSBO_REPO = Path('/root/data/cys/baseline/repos/FSBO')
RGPE_REPO = Path('/root/data/cys/baseline/repos/transfer-hpo-framework')


def identity(args, manifest):
    return dict(domain='ceb_query_runtime', method=args.method, format_version=1,
                purpose=args.purpose, corpus=str(args.corpus.resolve()),
                corpus_sha256=digest(args.corpus / 'manifest.json'),
                source_tasks=manifest['source_tasks'], seed=args.seed,
                representation=manifest['representation'], device='cpu',
                trainer_sha256=digest(__file__),
                contract_sha256=digest(Path(__file__).with_name('query_transfer_common.py')))


def dump_pickle(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


def rgpe(args, manifest, tasks):
    sys.path.insert(0, str(RGPE_REPO))
    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
    from smac.epm.util_funcs import get_types
    from rgpe.utils import get_gaussian_process
    info = identity(args, manifest)
    info['config'] = dict(kernel='Matern2.5_ARD', external_normalization='None',
                          internal_normalize_y=True, implementation_sha256=digest(RGPE_REPO / 'rgpe/utils.py'),
                          ranking_samples=1000, weight_dilution='probabilistic-ld')
    selected = list(tasks)
    if args.component_tasks:
        selected = args.component_tasks
        if args.purpose != 'component_check' or not set(selected).issubset(tasks):
            raise ValueError('A source subset is only permitted for component checks')
    info['fitted_tasks'] = selected
    meta_path = args.output / 'status.json'
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous['identity'] != info or not args.resume:
            raise ValueError('Existing RGPE run requires --resume and identical inputs')
    cs = ConfigurationSpace(seed=args.seed)
    for i in range(64):
        cs.add_hyperparameter(UniformFloatHyperparameter('z_%02d' % i, 0.0, 1.0))
    types, bounds = get_types(cs, None)
    completed, started = [], time.monotonic()
    atomic_json(meta_path, dict(identity=info, status='training', completed=0, total=len(selected)))
    for task in selected:
        source = tasks[task]
        x, y = source['X'], source['y_minimize']
        cache = args.output / (task + '.pkl')
        sidecar = cache.with_suffix('.json')
        source_identity = dict(identity=info, task_id=task, source_sha256=source['source_sha256'])
        resumed = cache.exists() and sidecar.exists()
        if resumed:
            meta = json.loads(sidecar.read_text())
            if meta['identity'] != source_identity or meta['sha256'] != digest(cache):
                raise ValueError('Source GP checkpoint identity mismatch: ' + task)
            with cache.open('rb') as stream:
                model = pickle.load(stream)
        else:
            # Per-task seeds make interrupted banks reproducible without refitting earlier GPs.
            task_seed = (args.seed + manifest['source_tasks'].index(task)) % (2 ** 31)
            model = get_gaussian_process(configspace=cs, types=types, bounds=bounds,
                                         rng=np.random.RandomState(task_seed), kernel=None)
            model.train(x, y)
        probe = x[:min(len(x), 16)]
        mean, variance = model.predict(probe)
        if (not np.isfinite(mean).all() or not np.isfinite(variance).all()
                or np.any(variance < 0)):
            raise RuntimeError('Invalid source GP posterior: ' + task)
        if not resumed:
            dump_pickle(cache, model)
            with cache.open('rb') as stream:
                restored = pickle.load(stream)
            check_mean, check_variance = restored.predict(probe)
            if not (np.allclose(mean, check_mean) and np.allclose(variance, check_variance)):
                raise RuntimeError('Source GP changed after reload: ' + task)
            atomic_json(sidecar, dict(identity=source_identity, status='complete',
                                     observations=len(y), sha256=digest(cache), reload_verified=True))
        completed.append(task)
        elapsed = time.monotonic() - started
        atomic_json(meta_path, dict(identity=info, status='training', completed=len(completed),
                                   total=len(selected), last_task=task, elapsed_seconds=elapsed))
        print(json.dumps(dict(task=task, observations=len(y), completed=len(completed),
                              total=len(selected), resumed=resumed, elapsed_seconds=elapsed)), flush=True)
    atomic_json(meta_path, dict(identity=info, status='complete', completed=len(completed),
                               total=len(selected), elapsed_seconds=time.monotonic() - started,
                               all_checkpoints_reload_verified=True))


def fsbo(args, manifest, tasks):
    sys.path.insert(0, str(FSBO_REPO))
    import torch
    from fsbo_modules import FSBO
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        raise ValueError('This launch is CPU-only; set CUDA_VISIBLE_DEVICES empty')
    info = identity(args, manifest)
    info['config'] = dict(hidden_size=[128, 128], kernel='rbf', ard=False,
                          lr=0.001, batch_size=50, augmentation='paper_eq10_11',
                          selection='fixed_final_checkpoint', validation_tasks=[],
                          implementation_sha256=digest(FSBO_REPO / 'fsbo_modules.py'))
    train_data = {t: dict(X=r['X'], y=r['y_maximize'][:, None]) for t, r in tasks.items()}
    model = FSBO(train_data, {}, str(args.output), hidden_size=[128, 128],
                 kernel='rbf', ard=False, batch_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ids = list(tasks)
    rng = np.random.RandomState(args.seed)
    seen = {t: np.zeros(len(r['X']), dtype=bool) for t, r in tasks.items()}
    meta_path, state_path = args.output / 'status.json', args.output / 'training_state.pt'
    start = 0
    if state_path.exists():
        saved = torch.load(str(state_path), map_location='cpu')
        if not args.resume or saved['identity'] != info:
            raise ValueError('Existing FSBO run requires --resume and identical inputs')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng'])
        torch.set_rng_state(saved['torch_rng'])
        start, seen = saved['step'], saved['seen']
    elif meta_path.exists():
        raise ValueError('FSBO status exists without a checkpoint; inspect before restarting')
    if args.steps <= start:
        raise ValueError('Requested steps must exceed the saved checkpoint')
    lower = min(r['y_maximize'].min() for r in tasks.values())
    upper = max(r['y_maximize'].max() for r in tasks.values())
    if not lower < upper:
        raise ValueError('Task augmentation requires a nonconstant corpus')
    atomic_json(meta_path, dict(identity=info, status='training', step=start, requested_steps=args.steps))
    started = time.monotonic()
    for step in range(start + 1, args.steps + 1):
        task = ids[rng.randint(len(ids))]
        source = tasks[task]
        indices = rng.choice(len(source['X']), size=min(50, len(source['X'])), replace=False)
        seen[task][indices] = True
        lo, hi = np.sort(rng.uniform(lower, upper, size=2))
        while hi - lo <= np.finfo(float).eps * max(1, abs(lo), abs(hi)):
            lo, hi = np.sort(rng.uniform(lower, upper, size=2))
        x = torch.as_tensor(source['X'][indices], dtype=torch.float32)
        y = torch.as_tensor((source['y_maximize'][indices] - lo) / (hi - lo), dtype=torch.float32)
        model.train()
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        model.model.set_train_data(features, y, strict=False)
        loss = -model.mll(model.model(features), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite FSBO loss at step %d' % step)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite FSBO gradient at step %d' % step)
        optimizer.step()
        if step % args.checkpoint_every == 0 or step == args.steps:
            saved = dict(identity=info, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         step=step, rng=rng.get_state(), torch_rng=torch.get_rng_state(), seen=seen)
            temp = state_path.with_suffix('.tmp')
            torch.save(saved, str(temp))
            os.replace(str(temp), str(state_path))
        if step in (1, 10) or step % 100 == 0 or step == args.steps:
            status = dict(identity=info, status='training', step=step, requested_steps=args.steps,
                          loss=float(loss.detach()), elapsed_seconds=time.monotonic() - started,
                          distinct_observations_sampled=sum(int(s.sum()) for s in seen.values()),
                          tasks_sampled=sum(bool(s.any()) for s in seen.values()), resumed_from=start)
            atomic_json(meta_path, status)
            print(json.dumps({k: v for k, v in status.items() if k != 'identity'}), flush=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite final FSBO parameters')
    temporary = args.output / 'weights.tmp'
    model.save_checkpoint(str(temporary))
    weights = args.output / 'weights'
    os.replace(str(temporary), str(weights))
    probe_task = tasks[ids[0]]
    support = torch.as_tensor(probe_task['X'][:1], dtype=torch.float32)
    labels = torch.as_tensor(probe_task['y_maximize'][:1], dtype=torch.float32)
    query = torch.as_tensor(probe_task['X'][1:2], dtype=torch.float32)

    def predict():
        model.eval()
        with torch.no_grad():
            model.model.set_train_data(model.feature_extractor(support), labels, strict=False)
            p = model.likelihood(model.model(model.feature_extractor(query)))
            return p.mean.clone(), p.variance.clone()

    before = predict()
    model.load_checkpoint(str(weights))
    after = predict()
    if not all(torch.isfinite(a).all() and torch.allclose(a, b) for a, b in zip(before, after)):
        raise RuntimeError('FSBO checkpoint posterior verification failed')
    status.update(status='complete', weights_sha256=digest(weights), reload_verified=True)
    atomic_json(meta_path, status)
    model.train_summary_writer.close()
    model.valid_summary_writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['fsbo', 'rgpe'])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--purpose', choices=['production_training', 'component_check'], default='production_training')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--component-tasks', nargs='+')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.checkpoint_every < 1:
        parser.error('Positive step and checkpoint intervals are required')
    if args.purpose == 'production_training' and (args.steps != 100000 or args.component_tasks):
        parser.error('Production requires the full schedule and complete eligible source set')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Set CUDA_VISIBLE_DEVICES empty for the authorized CPU jobs')
    with run_lock(args.output):
        manifest, tasks = load_corpus(args.corpus)
        try:
            (fsbo if args.method == 'fsbo' else rgpe)(args, manifest, tasks)
        except BaseException as exc:
            atomic_json(args.output / 'failure.json', dict(error=repr(exc), method=args.method))
            raise


if __name__ == '__main__':
    main()
