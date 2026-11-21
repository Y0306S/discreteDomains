#!/usr/bin/env python3
"""Predict target configurations with the verified TaskSet RGPE source GPs.

Runs in the bl_rgpe environment (same as taskset_rgpe_worker.py). Used by the
LLM-RGPE task rerank: for each requested source task, the base GP's predicted
mean loss at every target configuration.

request: {"cache_sha256", "source_tasks": [...], "target_configs": [...]}
response: {"cache_sha256", "predictions": {task: [mean, ...]}}
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.read_and_write import json as cs_json
from smac.configspace import convert_configurations_to_array

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taskset_transfer_common import atomic_json, digest, read_split


def predict(args):
    meta = json.loads((args.assets / 'ensemble.json').read_text())
    request = json.loads(args.request.read_text())
    split = read_split()
    if (meta['status'] != 'complete' or set(meta['source_tasks']) != set(split['train'])
            or meta['cache_sha256'] != request['cache_sha256']
            or digest(args.assets / 'ensemble.pkl') != meta['cache_sha256']
            or not set(request['source_tasks']) <= set(meta['source_tasks'])):
        raise ValueError('Source GP provenance mismatch')
    with (args.assets / 'ensemble.pkl').open('rb') as stream:
        payload = pickle.load(stream)
    cs = cs_json.read((args.assets / 'configspace.json').read_text())
    configs = []
    for row in request['target_configs']:
        config = Configuration(cs, values=row)
        cs.check_configuration(config)
        configs.append(config)
    x = convert_configurations_to_array(configs)
    index = {task: i for i, task in enumerate(meta['source_tasks'])}
    out = {}
    for task in request['source_tasks']:
        mean, _ = payload['base_models'][index[task]].predict(x)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        if not np.isfinite(mean).all():
            raise ValueError('Non-finite source prediction: ' + task)
        out[task] = mean.tolist()
    atomic_json(args.out, dict(cache_sha256=meta['cache_sha256'], predictions=out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    predict(parser.parse_args())
