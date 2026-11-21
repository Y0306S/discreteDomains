#!/usr/bin/env python3
"""Validate full-sized GPU target fitting, task isolation and CPU snapshots."""

import argparse
import copy
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
from types import SimpleNamespace

import numpy as np

from peptide_fsbo_gpu_eval import GPUModelClient
from peptide_transfer_policy import effective_recipe

TOOLS = Path(__file__).resolve().parent
CODE = ('peptide_fsbo_cuda.py', 'peptide_fsbo_gpu_service.py',
        'peptide_fsbo_gpu_eval.py', 'queue_peptide_transfer.py')


class StateUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'lolbo.utils.bo_utils.turbo' and name == 'TurboState':
            return SimpleNamespace
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--task', default='pep_151')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--token-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpu', type=int, default=45)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    recipe = effective_recipe(args.campaign / 'recipe.json')
    config = dict(recipe_sha256=hashlib.sha256((args.campaign/'recipe.json').read_bytes()).hexdigest(),
                  token_file=str(args.token_file))
    with sqlite3.connect('file:' + str(args.campaign/'runs'/args.task/'runtime.sqlite') + '?mode=ro', uri=True) as db:
        payload = db.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0]
    saved = StateUnpickler(io.BytesIO(payload)).load()
    features = np.asarray(saved['features'])
    low, high = np.asarray(recipe['lower']), np.asarray(recipe['upper'])
    x = ((features-low)/np.where(high > low, high-low, 1.)).astype(np.float32)
    scores = np.asarray([row['score'] for row in saved['rows']])
    # Exercise the campaign's maximum observation count using real cached rows.
    indices = np.arange(5100) % len(x)
    x, scores = x[indices], scores[indices]
    client = GPUModelClient(args.campaign/'recipe.json', recipe, config, args.endpoint)
    started = time.monotonic()
    fit = client.request('fit', x=x, score=scores, seed=1337)
    assert fit['device'] == 'cuda' and fit['optimizer_updates'] <= recipe['fsbo_target_epochs']
    assert abs(fit['best_loss'] - fit['restored_loss']) <= 0.00001
    snapshot = client.request('snapshot')
    candidates = np.random.RandomState(24).uniform(0, 1, (5000, 256)).astype(np.float32)
    draws = client.request('sample', candidates=candidates, count=50, seed=99)['draws']
    assert draws.shape == (50, 5000) and np.isfinite(draws).all()
    sample = client.request('sample', candidates=candidates[:64], count=3, seed=133)['draws']
    second = GPUModelClient(args.campaign/'recipe.json', recipe, config, args.endpoint)
    second.request('fit', x=x[:100], score=scores[:100] + 3.0, seed=7)
    repeated = client.request('sample', candidates=candidates[:64], count=3, seed=133)['draws']
    assert np.array_equal(sample, repeated), 'A second task contaminated the first posterior'
    fresh = GPUModelClient(args.campaign/'recipe.json', recipe, config, args.endpoint)
    fresh.request('restore', payload=snapshot)
    resumed = fresh.request('sample', candidates=candidates[:64], count=3, seed=133)['draws']
    assert np.array_equal(sample, resumed), 'Snapshot recovery changed the posterior'
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output/'snapshot.pt'
    snapshot_path.write_bytes(snapshot)
    cpu_script = '''import sys, torch
sys.path.insert(0, sys.argv[1])
from peptide_transfer_models import FSBOBackend
from peptide_transfer_policy import effective_recipe
b = FSBOBackend(effective_recipe(sys.argv[2]))
b.restore(open(sys.argv[3], 'rb').read())
assert b.x.device.type == 'cpu' and b.y.device.type == 'cpu'
assert all(p.device.type == 'cpu' for p in b.model.parameters())
print('CPU checkpoint recovery passed')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([recipe['model_python'], '-c', cpu_script, str(TOOLS),
        str(args.campaign/'recipe.json'), str(snapshot_path)], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)
    report = dict(passed=True, task=args.task, source_rows=len(saved['rows']),
        padded_real_rows=5100, candidates=5000, batch_size=50, full_fit=fit,
        checks=dict(full_target_fit=True, selected_loss_reproduced=True,
                    full_correlated_batch=True, finite_draws=True, task_isolation=True,
                    restart_snapshot=True, cpu_checkpoint_compatible=True),
        code_sha256={str(TOOLS/name):hashlib.sha256((TOOLS/name).read_bytes()).hexdigest() for name in CODE},
        recipe_sha256=config['recipe_sha256'],
        settings=dict(refit_every_round=True, maximum_updates=recipe['fsbo_target_epochs'],
                      patience=recipe['fsbo_target_patience'], tf32_enabled=False),
        elapsed_seconds=time.monotonic()-started, completed_at=time.time())
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
