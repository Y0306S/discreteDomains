#!/usr/bin/env python3
"""Verify the completed FSBO artifact, source task split and optimizer state."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

os.sched_setaffinity(0,{18})
import numpy as np
import torch
torch.set_num_threads(1)
BASE=Path('/root/data/logs/peptide_transfer_train800_20261027_v2')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status=json.loads((BASE/'fsbo/status.json').read_text())
    manifest=json.loads((BASE/'corpus/manifest.json').read_text())
    recipe=json.loads(Path('/root/data/logs/peptide_bo_train800_20261027_v5b/recipe.json').read_text())
    assert manifest['source_tasks']==recipe['train'] and not set(manifest['source_tasks'])&set(recipe['test'])
    train=set(manifest['fsbo_train_tasks']); valid=set(manifest['fsbo_validation_tasks'])
    assert len(train)==640 and len(valid)==160 and not train&valid and train|valid==set(recipe['train'])
    assert status['status']=='complete' and status['step']==status['requested_steps']==100000
    assert status['reload_verified'] and status['weights_sha256']==digest(BASE/'fsbo/weights')
    state=torch.load(str(BASE/'fsbo/training_state.pt'),map_location='cpu')
    weights=torch.load(str(BASE/'fsbo/weights'),map_location='cpu')
    assert state['identity']==status['identity'] and state['step']==100000
    assert set(state['seen'])==train and all(np.any(v) for v in state['seen'].values())
    assert sum(int(v.sum()) for v in state['seen'].values())==status['observations_sampled']
    def finite_state(values):
        for name,value in values.items():
            # Positive GP constraints intentionally serialize an infinite upper bound.
            if name.endswith('_constraint.upper_bound') or name.endswith('_constraint.lower_bound'):
                assert not torch.isnan(value).any()
            else:
                assert torch.isfinite(value).all(),name
    for group in ('gp','likelihood','net'):
        assert weights[group]
        finite_state(weights[group])
    finite_state(state['model'])
    for params in state['optimizer']['state'].values():
        for value in params.values():
            if torch.is_tensor(value): assert torch.isfinite(value).all()
    info=status['identity']
    assert info['shared_training_loop_sha256']==digest(Path('/root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains/train_taskset_transfer.py'))
    assert info['config']['implementation_sha256']==digest(Path('/root/data/cys/baseline/repos/FSBO/fsbo_modules.py'))
    with sqlite3.connect('file:'+str(BASE/'corpus/corpus.sqlite')+'?mode=ro',uri=True) as db:
        counts=db.execute('SELECT COUNT(*),MIN(observations),MAX(observations),SUM(observations) FROM tasks').fetchone()
    report=dict(passed=True,full_updates=100000,training_tasks=640,validation_tasks=160,
        train800_only=True,test200_excluded=True,all_training_tasks_sampled=True,
        distinct_observations_sampled=status['observations_sampled'],checkpoint_hash_verified=True,
        finite_model_and_adam_state=True,selected_checkpoint_reload_verified=status['reload_verified'],
        best_step=status['best_step'],source_tasks=counts[0],min_observations_per_task=counts[1],
        max_observations_per_task=counts[2],total_source_observations=counts[3],
        kernel=info['config']['kernel'],hidden_sizes=info['config']['hidden_size'],
        batch_size=50,learning_rate=0.001,purpose='completed_training_artifact_audit',
        evaluation_adapter_complete=False,oracle_calls=0)
    output=Path(__file__).with_name('fsbo_validation.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
