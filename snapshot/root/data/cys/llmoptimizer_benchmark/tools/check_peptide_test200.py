#!/usr/bin/env python3
"""Real test200 inference and recovery checks before production launch."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import peptide_test200 as adapter
from peptide_test200 import bo, protocol, torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:18871')
    parser.add_argument('--cpu', type=int, default=16)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for method in ('tpe', 'lolbo'):
        production = args.campaign / method
        recipe = adapter.verify(production)
        report = dict(passed=False, recipe_sha256=bo.digest(production / 'recipe.json'), checks={})
        case = args.output / method
        diagnostic = dict(recipe, purpose='component_check', proposals=3 if method == 'tpe' else 11,
                          fresh_budget=400, batch_size=1 if method == 'tpe' else 4, e2e_every=2)
        bo.atomic_json(case / 'recipe.json', diagnostic)
        task = 'pep_235' if method == 'tpe' else recipe['test'][0]
        command = [sys.executable, str(Path(adapter.__file__)), 'run-task', '--campaign', str(case),
                   '--task', task, '--cpu', str(args.cpu), '--server', args.server]
        subprocess.run(command + ['--stop-after-events', '1'], env=env, check=True)
        report['checks']['real_inference'] = True
        if method == 'tpe':
            import optuna
            from optuna.distributions import FloatDistribution
            from optuna.trial import create_trial
            directory = case / 'runs' / task
            db = adapter.connect(directory / 'runtime.sqlite')
            saved = adapter.get(db, 'state')
            records = adapter.trials(db)
            distributions = {f'z{i:03d}': FloatDistribution(*recipe['tpe_bounds']) for i in range(256)}
            study = optuna.create_study(direction='maximize', sampler=saved['sampler'])
            for record in records:
                study.add_trial(create_trial(params=record['params'], distributions=distributions,
                    value=record['score'], user_attrs={'constraints': record['constraints']},
                    system_attrs={'constraints': record['constraints']}))
            state, backend = protocol.create_state(diagnostic, task, remote_url=args.server)
            state.objective.xs_to_scores_dict = {r['row']['sequence']: r['score'] for r in records}
            bo.restore_rng(saved['rng'])
            trial = study.ask(fixed_distributions=distributions)
            z = torch.tensor([[trial.params[k] for k in distributions]], dtype=torch.float32)
            with torch.no_grad():
                sequence = state.objective.vae_decode(z)[0]
            pending = dict(number=trial.number, params=trial.params, sequence=sequence,
                           proposal_seconds=0.0, oracle_seconds=0.0)
            before = adapter.state_meta(study, saved['progress'], saved['identity'])
            pending['scored'] = bo.score_batch(state.objective, [sequence])
            with db:
                adapter.put(db, 'pending', pending)
                adapter.put(db, 'state', before)
            db.close()
            backend.close()
            replay = """import sys; from pathlib import Path; from types import SimpleNamespace
sys.path.insert(0, sys.argv[1]); import peptide_test200 as a
def bad(*args, **kwargs): raise RuntimeError('Recorded pending score must not be rescored')
a.bo.score_batch=bad
a.torch.set_num_threads(1)
a.run(SimpleNamespace(campaign=Path(sys.argv[2]), task='pep_235', server=sys.argv[3], launch=False, stop_after_events=2))
"""
            subprocess.run([sys.executable, '-c', replay, str(Path(adapter.__file__).parent), str(case), args.server],
                           env=env, check=True)
            report['checks']['pending_without_rescore'] = True
        subprocess.run(command, env=env, check=True)
        report['checks']['resume'] = True
        if method == 'tpe':
            identity = dict(format=adapter.TPE_FORMAT, recipe_sha256=bo.digest(case / 'recipe.json'), task=task)
            history = adapter.verify_tpe_archive(case / 'runs' / (task + '.tar.gz'), identity)
            assert history['rounds'] == 3 and history['e2e_updates'] == 0
        else:
            identity = dict(format=protocol.FORMAT, recipe_sha256=bo.digest(case / 'recipe.json'), task=task)
            history = adapter.native.verify_archive(case / 'runs' / (task + '.tar.gz'), identity)
            report['joint_updates'] = history['e2e_updates']
        report['checks'].update(archive=True, accounting=True, task_isolation=history['reference'] == recipe['references'][task])
        report['budget'] = {k: history[k] for k in ('proposals', 'fresh_evaluations', 'cache_hits', 'recenter_fresh')}
        report['passed'] = all(report['checks'].values())
        path = args.output / (method + '_report.json')
        bo.atomic_json(path, report)
        bo.atomic_json(production / 'preflight_passed.json', dict(recipe_sha256=report['recipe_sha256'],
            report=str(path), report_sha256=bo.digest(path)))
        results[method] = report
    bo.atomic_json(args.output / 'report.json', results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
