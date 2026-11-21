#!/usr/bin/env python3
"""Run the pinned peptide evaluation loop through verified GPU FSBO services."""

import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import urllib.request

import numpy as np


def runtime(path):
    config = json.loads(Path(path).read_text())
    if not config['enabled']:
        raise ValueError('GPU FSBO runtime is disabled')
    for name, expected in config['code_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('GPU FSBO execution code changed: ' + name)
    validation = config['validation']
    payload = Path(validation['path']).read_bytes()
    report = json.loads(payload)
    if (hashlib.sha256(payload).hexdigest() != validation['sha256']
            or not report['passed'] or report['code_sha256'] != config['code_sha256']):
        raise ValueError('GPU FSBO runtime has not passed its validation')
    return config


class GPUModelClient:
    def __init__(self, recipe_path, recipe, config, endpoint):
        self.recipe_hash = hashlib.sha256(Path(recipe_path).read_bytes()).hexdigest()
        if recipe['method'] != 'fsbo' or config['recipe_sha256'] != self.recipe_hash:
            raise ValueError('GPU FSBO configuration has the wrong recipe identity')
        self.endpoint = endpoint
        self.token = Path(config['token_file']).read_text().strip()
        self.current_snapshot = self.last_sample = None

    def request(self, op, **values):
        if op == 'snapshot':
            if self.current_snapshot is None:
                raise ValueError('Cannot snapshot an unfitted GPU FSBO model')
            return self.current_snapshot
        if op == 'restore':
            self.current_snapshot = values['payload']
            return None
        if op == 'close':
            return None
        if op == 'sample':
            if self.current_snapshot is None:
                raise ValueError('Cannot sample an unfitted GPU FSBO model')
            values['snapshot'] = self.current_snapshot
        values.update(op=op, recipe_sha256=self.recipe_hash)
        request = urllib.request.Request(self.endpoint + '/rpc',
            data=pickle.dumps(values, protocol=4),
            headers={'Content-Type': 'application/octet-stream',
                     'Authorization': 'Bearer ' + self.token})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=1800) as response:
            reply = pickle.loads(response.read())
        if reply.get('error'):
            raise RuntimeError('GPU FSBO service: ' + reply['error'])
        if reply.get('snapshot') is not None:
            self.current_snapshot = reply['snapshot']
        result = reply['result']
        info = result if op == 'fit' else result['info']
        info['rpc_seconds'] = time.monotonic() - started
        if op == 'fit':
            info['seconds'] = info['rpc_seconds']
        if op == 'sample' and not np.isfinite(result['draws']).all():
            raise ValueError('Nonfinite GPU FSBO reply')
        return result

    def close(self):
        self.current_snapshot = None


def main():
    import peptide_transfer_eval as adapter

    if 'run-task' not in sys.argv or '--campaign' not in sys.argv:
        raise ValueError('GPU FSBO wrapper is only for task evaluation')
    campaign = Path(sys.argv[sys.argv.index('--campaign') + 1])
    config = runtime(campaign / 'gpu_runtime.json')
    cpu = int(sys.argv[sys.argv.index('--cpu') + 1])
    endpoint = config['endpoints'][cpu % len(config['endpoints'])]
    adapter.ModelClient = lambda path, recipe: GPUModelClient(path, recipe, config, endpoint)
    adapter.main()


if __name__ == '__main__':
    main()
