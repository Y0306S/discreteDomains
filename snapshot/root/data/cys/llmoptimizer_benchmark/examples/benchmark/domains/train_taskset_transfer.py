#!/usr/bin/env python3
"""Full TaskSet FSBO meta-training and upstream RGPE source-GP fitting on CPU."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from taskset_transfer_common import atomic_json, digest, load_corpus, run_lock

FSBO_REPO = Path('/root/data/cys/baseline/repos/FSBO')
RGPE_REPO = Path('/root/data/cys/baseline/repos/transfer-hpo-framework')


def identity(args, manifest):
    return dict(domain='taskset_cash', method=args.method, purpose=args.purpose,
                corpus_sha256=digest(args.corpus / 'manifest.json'),
                split_sha256=manifest['split_sha256'], seed=args.seed,
                trainer_sha256=digest(__file__), device='cpu', threads=args.threads)


def rgpe(args, manifest, tasks):
    sys.path.insert(0, str(RGPE_REPO))
    from ConfigSpace import Configuration
    from ConfigSpace.read_and_write import json as cs_json
    from smac.configspace import convert_configurations_to_array
    from smac.epm.util_funcs import get_types
    from rgpe.utils import get_gaussian_process

    cs = cs_json.read((args.corpus / 'configspace.json').read_text())
    types, bounds = get_types(cs, None)
    info = identity(args, manifest)
    info['config'] = dict(kernel='upstream_Matern2.5_times_Hamming', normalize_y=True,
                          encoding='native_conditional_ConfigSpace',
                          implementation_sha256=digest(RGPE_REPO / 'rgpe/utils.py'))
    status_path = args.output / 'status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if not args.resume or previous['identity'] != info:
            raise ValueError('RGPE resume requires identical source data and settings')
    started = time.monotonic()
    for index, (task, source) in enumerate(tasks.items()):
        configs = [Configuration(cs, values=values) for values in source['configs']]
        for config in configs:
            cs.check_configuration(config)
        x = convert_configurations_to_array(configs)
        y = source['y_minimize']
        cache = args.output / (task + '.pkl')
        sidecar = cache.with_suffix('.json')
        task_identity = dict(identity=info, task_id=task, source_sha256=source['source_sha256'])
        if cache.exists() and sidecar.exists():
            meta = json.loads(sidecar.read_text())
            if not args.resume or meta['identity'] != task_identity or digest(cache) != meta['sha256']:
                raise ValueError('RGPE checkpoint identity mismatch: ' + task)
            with cache.open('rb') as stream:
                model = pickle.load(stream)
            resumed = True
        elif cache.exists() or sidecar.exists():
            raise ValueError('Incomplete RGPE checkpoint pair requires inspection: ' + task)
        else:
            model = get_gaussian_process(configspace=cs, types=types, bounds=bounds,
                                         rng=np.random.RandomState(args.seed + index), kernel=None)
            model.train(x, y)
            resumed = False
        probe = x[:min(len(x), 16)]
        mean, variance = model.predict(probe)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            raise RuntimeError('Invalid RGPE source posterior: ' + task)
        if not resumed:
            temporary = cache.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            with temporary.open('rb') as stream:
                restored = pickle.load(stream)
            after = restored.predict(probe)
            if not np.allclose(mean, after[0]) or not np.allclose(variance, after[1]):
                raise RuntimeError('RGPE source changed after reload: ' + task)
            os.replace(str(temporary), str(cache))
            atomic_json(sidecar, dict(identity=task_identity, sha256=digest(cache),
                                     observations=len(y), reload_verified=True))
        status = dict(identity=info, status='training', completed=index + 1, total=len(tasks),
                      last_task=task, elapsed_seconds=time.monotonic() - started)
        atomic_json(status_path, status)
        print(json.dumps(dict(task=task, completed=index + 1, total=len(tasks), resumed=resumed)), flush=True)
    status.update(status='complete', all_checkpoints_reload_verified=True)
    atomic_json(status_path, status)


def fsbo(args, manifest, tasks):
    sys.path.insert(0, str(FSBO_REPO))
    import torch
    from fsbo_modules import FSBO

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        raise ValueError('FSBO training must be CPU-only')
    info = identity(args, manifest)
    info['config'] = dict(hidden_size=[128, 128], kernel='rbf', ard=False,
                          learning_rate=0.001, batch_size=50, augmentation='none',
                          label_scaling='per_task_minmax_upstream_get_batch',
                          validate_every=100, selection='minimum_validation_predictive_nll',
                          implementation_sha256=digest(FSBO_REPO / 'fsbo_modules.py'))
    train_ids, valid_ids = manifest['fsbo_train_tasks'], manifest['fsbo_validation_tasks']
    info.update(training_tasks=len(train_ids), validation_tasks=len(valid_ids))
    train = {t: dict(X=tasks[t]['X'], y=tasks[t]['y_maximize'][:, None]) for t in train_ids}
    valid = {t: dict(X=tasks[t]['X'], y=tasks[t]['y_maximize'][:, None]) for t in valid_ids}
    model = FSBO(train, valid, str(args.output), hidden_size=[128, 128], kernel='rbf', ard=False, batch_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.RandomState(args.seed)
    state_path, status_path = args.output / 'training_state.pt', args.output / 'status.json'
    seen = {task: np.zeros(len(tasks[task]['X']), dtype=bool) for task in train_ids}
    best, best_step, start = float('inf'), None, 0
    if state_path.exists():
        saved = torch.load(str(state_path), map_location='cpu')
        if not args.resume or saved['identity'] != info:
            raise ValueError('FSBO resume requires identical sources and settings')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng'])
        torch.set_rng_state(saved['torch_rng'])
        start, best, best_step, seen = saved['step'], saved['best'], saved['best_step'], saved['seen']
    elif status_path.exists():
        raise ValueError('FSBO status exists without a recoverable checkpoint')
    if args.steps < start:
        raise ValueError('Requested budget is smaller than the checkpoint')
    # Upstream fsbo_modules.get_batch / get_support_and_queries: MinMaxScaler on each
    # task's full response vector. Global-range augmentation (2026-10-02 run) exploded
    # labels and never matched the raw-scale validation, so selection stopped at step 100.
    def minmax(y):
        lo, hi = float(np.min(y)), float(np.max(y))
        return (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)

    scaled = {t: minmax(np.asarray(tasks[t]['y_maximize'], dtype=float)) for t in train_ids + valid_ids}
    valid_rng = np.random.RandomState(args.seed + 1)
    validation = []
    for task in valid_ids:
        source = tasks[task]
        order = valid_rng.permutation(len(source['X']))
        split = min(50, max(1, len(order) // 2))
        support, query = order[:split], order[split:split + 50]
        validation.append((source['X'][support], scaled[task][support],
                           source['X'][query], scaled[task][query]))

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32)

    def validate():
        model.eval()
        losses = []
        with torch.no_grad():
            for sx, sy, qx, qy in validation:
                model.model.set_train_data(model.feature_extractor(tensor(sx)), tensor(sy), strict=False)
                posterior = model.likelihood(model.model(model.feature_extractor(tensor(qx))))
                losses.append(float(-posterior.log_prob(tensor(qy)) / len(qy)))
        value = float(np.mean(losses))
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite FSBO validation loss')
        return value

    started = time.monotonic()
    atomic_json(status_path, dict(identity=info, status='training', step=start, requested_steps=args.steps))
    for step in range(start + 1, args.steps + 1):
        task = train_ids[rng.randint(len(train_ids))]
        source = tasks[task]
        indices = rng.choice(len(source['X']), size=min(50, len(source['X'])), replace=False)
        seen[task][indices] = True
        x, y = tensor(source['X'][indices]), tensor(scaled[task][indices])
        model.train()
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        model.model.set_train_data(features, y, strict=False)
        loss = -model.mll(model.model(features), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite FSBO loss at step %d' % step)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite FSBO gradient')
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            validation_loss = validate()
            if validation_loss < best:
                best, best_step = validation_loss, step
                temporary = args.output / 'weights.tmp'
                model.save_checkpoint(str(temporary))
                os.replace(str(temporary), str(args.output / 'weights'))
            saved = dict(identity=info, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
                         best=best, best_step=best_step, seen=seen)
            temporary = state_path.with_suffix('.tmp')
            torch.save(saved, str(temporary))
            os.replace(str(temporary), str(state_path))
            status = dict(identity=info, status='training', step=step, requested_steps=args.steps,
                          loss=float(loss.detach()), best_step=best_step, validation_loss=validation_loss,
                          best_validation_loss=best, elapsed_seconds=time.monotonic() - started,
                          tasks_sampled=sum(bool(v.any()) for v in seen.values()),
                          observations_sampled=sum(int(v.sum()) for v in seen.values()), resumed_from=start)
            atomic_json(status_path, status)
            print(json.dumps({k: v for k, v in status.items() if k != 'identity'}), flush=True)
    weights = args.output / 'weights'
    model.load_checkpoint(str(weights))
    score = validate()
    if not np.isclose(score, best, rtol=1e-5, atol=1e-5):
        raise RuntimeError('FSBO selected checkpoint changed after reload')
    status = json.loads(status_path.read_text())
    status.update(status='complete', weights_sha256=digest(weights), reload_verified=True)
    atomic_json(status_path, status)
    model.train_summary_writer.close()
    model.valid_summary_writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('fsbo', 'rgpe'))
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--purpose', choices=('production_training', 'component_check'), default='production_training')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or (args.purpose == 'production_training' and args.steps != 100000):
        parser.error('Production uses the full 100000-update schedule')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Set CUDA_VISIBLE_DEVICES empty for CPU training')
    with run_lock(args.output):
        manifest, tasks = load_corpus(args.corpus)
        if manifest['purpose'] != args.purpose:
            raise ValueError('Component-check sources cannot be used for production training')
        try:
            (fsbo if args.method == 'fsbo' else rgpe)(args, manifest, tasks)
        except BaseException as exc:
            atomic_json(args.output / 'failure.json', dict(error=repr(exc), method=args.method))
            raise


if __name__ == '__main__':
    main()
