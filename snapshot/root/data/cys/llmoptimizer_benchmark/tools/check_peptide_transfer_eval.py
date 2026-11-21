#!/usr/bin/env python3
"""Real batch-50 peptide transfer checks, including fresh-process recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time

import numpy as np
import peptide_transfer_eval as adapter
from peptide_transfer_eval import bo, storage, torch


def saved_state(case, task):
    with sqlite3.connect('file:' + str(case / 'runs' / task / 'runtime.sqlite') + '?mode=ro', uri=True) as db:
        return storage.get(db, 'state'), storage.get(db, 'pending')


def archived_state(case, task):
    with tarfile.open(case / 'runs' / (task + '.tar.gz'), 'r:gz') as archive:
        db = sqlite3.connect(':memory:')
        db.deserialize(archive.extractfile('runtime.sqlite').read())
        saved = storage.get(db, 'state')
        db.close()
    return saved


def production_pool_check(case, task, recipe, saved, server):
    from lolbo.utils.bo_utils.turbo import generate_batch
    client = model = None
    try:
        objective, client = adapter.create_objective(recipe, task, server, saved['identity'])
        model = adapter.ModelClient(case / 'recipe.json', recipe)
        model.request('restore', payload=saved['model_snapshot'])
        constraint, _ = adapter.constraint_model(recipe, saved['features'], saved['constraint_weights'])
        x = torch.as_tensor(saved['features'])
        c = objective.compute_constraints([r['sequence'] for r in saved['rows']]).flatten()
        y = torch.tensor([r['score'] for r in saved['rows']]).reshape(-1, 1)
        bo.restore_rng(saved['rng'])
        started = time.monotonic()
        with torch.no_grad():
            z = generate_batch(saved['trust_region'], adapter.BridgeModel(model, recipe),
                x[c <= 0], y[c <= 0], 50, n_candidates=5000,
                device=torch.device('cpu'), constraint_model_list=[constraint])
            sequences = objective.vae_decode(z)
        assert z.shape == (50, 256) and len(sequences) == 50
        for sequence in sequences:
            bo.validate_sequence(sequence)
        return dict(passed=True, candidates=5000, batch_size=50,
                    seconds=time.monotonic() - started, posterior=model.last_sample,
                    new_oracle_calls=0)
    finally:
        if model is not None:
            model.close()
        if client is not None:
            client.close()


def model_reload_check(case, recipe, saved):
    query = adapter.normalized(saved['features'][:8], recipe)
    results = []
    for _ in range(2):
        client = adapter.ModelClient(case / 'recipe.json', recipe)
        try:
            client.request('restore', payload=saved['model_snapshot'])
            results.append(client.request('sample', candidates=query, count=50, seed=919)['draws'])
        finally:
            client.close()
    assert np.array_equal(results[0], results[1])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', default='fsbo,rgpe')
    parser.add_argument('--cpu', type=int, default=18)
    parser.add_argument('--server', default='http://127.0.0.1:18871')
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    reports = {}
    for method in args.methods.split(','):
        case = args.campaign / method
        recipe = adapter.verify(case)
        sys.path.insert(0, recipe['cpu_upstream'])
        if recipe['purpose'] != 'component_check' or recipe['proposals'] != 51:
            raise ValueError('This test requires an isolated 51-proposal component recipe')
        task = 'pep_235'
        report = dict(passed=False, method=method, task=task, checks={},
            recipe_sha256=bo.digest(case / 'recipe.json'),
            source_models=len(recipe['rgpe_sources']), production_scale_passed=False)
        command = [sys.executable, str(Path(adapter.__file__)), 'run-task', '--campaign', str(case),
                   '--task', task, '--cpu', str(args.cpu), '--server', args.server]
        subprocess.run(command + ['--stop-after-stage', 'proposal'], env=env, check=True)
        initial, pending = saved_state(case, task)
        assert len(initial['rows']) == 100 and pending['proposed_z'].shape == (50, 256)
        assert pending['fit']['target_observations'] == 100
        report['checks']['native_batch50'] = True
        report['first_fit'] = pending['fit']
        report['checks']['model_reload'] = model_reload_check(case, recipe, initial)
        bad = subprocess.run(command[:command.index('--task') + 1] + [recipe['train'][0]] +
            command[command.index('--task') + 2:], env=env, capture_output=True, text=True)
        assert bad.returncode and 'outside the held-out test200' in bad.stderr
        report['checks']['task_isolation'] = True
        subprocess.run(command + ['--stop-after-stage', 'score'], env=env, check=True)
        initial, pending = saved_state(case, task)
        assert 'scored' in pending
        clone = args.campaign / (method + '_recovery_copy')
        clone.mkdir(parents=True, exist_ok=False)
        shutil.copy2(case / 'recipe.json', clone / 'recipe.json')
        clone_run = clone / 'runs' / task
        clone_run.mkdir(parents=True)
        with sqlite3.connect(str(case / 'runs' / task / 'runtime.sqlite')) as source:
            with sqlite3.connect(str(clone_run / 'runtime.sqlite')) as target:
                source.backup(target)
        replay = '''import sys; from pathlib import Path; from types import SimpleNamespace
sys.path.insert(0,sys.argv[1]); import peptide_transfer_eval as a
original=a.PersistentRemoteObjective.request
def request(self,value):
    if value['op']=='oracle': raise RuntimeError('Scored pending batch was evaluated again')
    return original(self,value)
a.PersistentRemoteObjective.request=request
a.run(SimpleNamespace(campaign=Path(sys.argv[2]),task=sys.argv[3],server=sys.argv[4],cpu=int(sys.argv[5]),stop_after_events=1,stop_after_stage=None))
'''
        subprocess.run([sys.executable, '-c', replay, str(Path(adapter.__file__).parent),
            str(case), task, args.server, str(args.cpu)], env=env, check=True)
        report['checks']['pending_without_rescore'] = True
        subprocess.run(command, env=env, check=True)
        clone_command = command.copy()
        clone_command[clone_command.index('--campaign') + 1] = str(clone)
        subprocess.run(clone_command, env=env, check=True)
        identity = dict(format=adapter.FORMAT, recipe_sha256=bo.digest(case / 'recipe.json'), task=task)
        history = storage.verify_tpe_archive(case / 'runs' / (task + '.tar.gz'), identity)
        copied = storage.verify_tpe_archive(clone / 'runs' / (task + '.tar.gz'), identity)
        signature = lambda h: [(r['sequence'], r['score'], r['cached'], r['feasible'], r.get('proposed_z'))
                               for r in h['observations']]
        assert signature(history) == signature(copied)
        assert history['proposals'] == 51 and history['rounds'] == 2 and history['e2e_updates'] == 0
        assert not (case / 'runs' / task).exists()
        assert [r['fit']['target_observations'] for r in history['observations'] if r.get('fit')] == [100, 150]
        state = archived_state(case, task)
        assert np.array_equal(state['features'][:100], initial['features'])
        client = None
        try:
            objective, client = adapter.create_objective(recipe, task, args.server, identity)
            check_indices = [0, 99, 100, 149, 150]
            fresh_z = adapter.encode_mean(client, [state['rows'][i]['sequence'] for i in check_indices])
            report['encoding_max_abs_batch_difference'] = float(np.max(np.abs(fresh_z-state['features'][check_indices])))
            assert np.allclose(fresh_z, state['features'][check_indices], rtol=1e-5, atol=1e-5)
            known = {}
            for row, feature in zip(state['rows'], state['features']):
                if row['sequence'] in known:
                    assert np.array_equal(feature, known[row['sequence']])
                known.setdefault(row['sequence'], feature)
        finally:
            if client is not None:
                client.close()
        report['checks'].update(real_inference=True, target_refit=True, frozen_encoding=True,
            resume=True, reproducible_resume=True, archive=True, accounting=True,
            repeat_encoding_reused=True)
        report['production_pool_check'] = production_pool_check(case, task, recipe, state, args.server)
        report['production_scale_passed'] = (method == 'fsbo' or len(recipe['rgpe_sources']) == 800)
        report['budget'] = {k: history[k] for k in ('proposals', 'fresh_evaluations', 'cache_hits', 'rounds')}
        report['target_checkpoint_bytes'] = len(state['model_snapshot'])
        report['maximum_distinct_oracle_sequences_requested'] = 51
        report['passed'] = all(report['checks'].values())
        bo.atomic_json(args.output / (method + '_report.json'), report)
        reports[method] = report
        bo.atomic_json(args.output / 'report.json', reports)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
