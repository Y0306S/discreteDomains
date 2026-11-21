#!/usr/bin/env python3
"""Pack verified TaskSet source GPs and apply the existing strict RGPE/TAF worker."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.read_and_write import json as cs_json
from smac.configspace import convert_configurations_to_array

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/benchmark'))
from rgpe_worker import make_model, TAF
from mindware.components.optimizers.transfer_hpo.protocol import acquisition_diagnostics
from taskset_transfer_common import atomic_json, digest, load_corpus, read_split


def pack(args):
    manifest, sources = load_corpus(args.corpus, check_sources=False)
    status = json.loads((args.training / 'status.json').read_text())
    if (status['status'] != 'complete' or status['completed'] != 838
            or not status['all_checkpoints_reload_verified']
            or status['identity']['corpus_sha256'] != digest(args.corpus / 'manifest.json')):
        raise ValueError('RGPE source training is incomplete or belongs to another corpus')
    cs_text = (args.corpus / 'configspace.json').read_text()
    cs = cs_json.read(cs_text)
    models, records = [], []
    for task, source in sources.items():
        path = args.training / (task + '.pkl')
        meta = json.loads(path.with_suffix('.json').read_text())
        expected = dict(identity=status['identity'], task_id=task, source_sha256=source['source_sha256'])
        if (meta['identity'] != expected or not meta['reload_verified']
                or meta['observations'] != len(source['y_minimize']) or digest(path) != meta['sha256']):
            raise ValueError('Source checkpoint mismatch: ' + task)
        with path.open('rb') as stream:
            model = pickle.load(stream)
        if model.configspace != cs:
            raise ValueError('Source ConfigSpace mismatch: ' + task)
        configs = [Configuration(cs, values=v) for v in source['configs'][:2]]
        mean, variance = model.predict(convert_configurations_to_array(configs))
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError('Source posterior is invalid: ' + task)
        models.append(model)
        records.append(dict(task=task, checkpoint_sha256=meta['sha256'],
                            source_history_sha256=source['source_sha256']))
    identity = dict(domain='taskset', method='rgpe_taf', format_version=1,
                    source_tasks=list(sources), corpus_sha256=digest(args.corpus/'manifest.json'),
                    split_sha256=manifest['split_sha256'], schema_sha256=digest(args.corpus/'configspace.json'),
                    training_identity=status['identity'], normalization='None')
    if args.output.exists():
        raise ValueError('Refusing to replace an existing source ensemble')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(dict(identity=identity, base_models=models), stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(args.output))
    atomic_json(args.output.with_suffix('.json'), dict(identity, status='complete',
                cache_sha256=digest(args.output), source_checkpoints=records))
    print(json.dumps(dict(sources=len(models), cache=str(args.output), bytes=args.output.stat().st_size)), flush=True)


def configs(cs, values):
    output = []
    for index, row in enumerate(values):
        try:
            config = Configuration(cs, values=row)
            cs.check_configuration(config)
        except Exception as exc:
            raise ValueError('Invalid configuration at original index %d: %s' % (index, exc)) from exc
        output.append(config)
    return output


def suggest(args):
    assets = args.assets
    meta = json.loads((assets/'ensemble.json').read_text())
    request = json.loads(args.request.read_text())
    split = read_split()
    if (meta['status'] != 'complete' or set(meta['source_tasks']) != set(split['train'])
            or len(meta['source_tasks']) != 838 or request['task_id'] not in split['test']
            or meta['cache_sha256'] != request['cache_sha256']
            or digest(assets/'ensemble.pkl') != meta['cache_sha256']
            or digest(assets/'configspace.json') != meta['schema_sha256']):
        raise ValueError('RGPE source, target, or checkpoint provenance mismatch')
    with (assets/'ensemble.pkl').open('rb') as stream:
        payload = pickle.load(stream)
    if payload['identity'] != {k:v for k,v in meta.items() if k not in ('status','cache_sha256','source_checkpoints')}:
        raise ValueError('Packed ensemble metadata mismatch')
    cs = cs_json.read((assets/'configspace.json').read_text())
    observed = configs(cs, request['target_configs'])
    pending = configs(cs, request['pending_configs'])
    y = np.asarray(request['y_minimize'], dtype=float)
    if len(observed) < 3 or len(observed) != len(y) or not np.isfinite(y).all() or len(pending) != 256:
        raise ValueError('Invalid RGPE target observations or candidate count')
    model = make_model(cs, {}, request['seed'], 'None', request['n_trials'], 1000)
    model.base_models = payload['base_models']
    x = convert_configurations_to_array(observed)
    model.train(x, y)
    acq = TAF(model)
    acq.update(model=model, X=x)
    scores = acq._compute(convert_configurations_to_array(pending)).reshape(-1)
    if not np.isfinite(scores).all() or not np.isfinite(model.weights_).all():
        raise ValueError('RGPE produced non-finite weights or acquisition values')
    atomic_json(args.out, dict(index=int(np.argmax(scores)), n_sources=len(meta['source_tasks']),
                cache_sha256=meta['cache_sha256'], weights=model.weights_.tolist(),
                source_weights=[dict(task=t, weight=float(w)) for t,w in zip(meta['source_tasks'], model.weights_[:-1])],
                target_weight=float(model.weights_[-1]), acquisition=acquisition_diagnostics('TAF', scores),
                ranking_samples=1000, normalization='None', sampling_mode='bootstrap',
                weight_dilution_strategy='probabilistic-ld'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('pack')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('suggest')
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (pack if args.action == 'pack' else suggest)(args)
