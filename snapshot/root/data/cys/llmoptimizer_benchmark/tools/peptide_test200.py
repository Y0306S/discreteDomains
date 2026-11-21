#!/usr/bin/env python3
"""Separate test200 TPE/LOLBO queues using the audited V5 peptide protocol."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time

RELEASE = Path('/root/data/logs/peptide_bo_release_20261027_v5')
SOURCE = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
AUDIT = Path('/root/reports/peptide_train800_test200_audit_20261027')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'tools'))
import peptide_bo_v5 as native
from mindware.domains.protein import verified_bo as bo
from mindware.domains.protein import peptide_v5 as protocol
import numpy as np
import torch

TPE_FORMAT = 'peptide-tpe-test200-v1'


def verify(campaign):
    recipe = native.verify(campaign)
    if recipe['adapter'] != str(Path(__file__).resolve()):
        raise ValueError('Adapter identity mismatch')
    if recipe['optuna_version'] != importlib.metadata.version('optuna'):
        raise ValueError('Optuna version differs from recipe')
    if len(recipe['test']) != 200 or set(recipe['test']) & set(recipe['train']):
        raise ValueError('Incorrect held-out task split')
    return recipe


def preflight(campaign):
    gate = json.loads((campaign / 'preflight_passed.json').read_text())
    report = Path(gate['report'])
    value = json.loads(report.read_text())
    if (gate['recipe_sha256'] != bo.digest(campaign / 'recipe.json')
            or gate['report_sha256'] != bo.digest(report) or not value['passed']
            or value['recipe_sha256'] != gate['recipe_sha256']):
        raise ValueError('Recipe-bound component check is missing or failed')
    required = {'real_inference', 'resume', 'archive', 'accounting', 'task_isolation'}
    if not required <= value['checks'].keys() or not all(value['checks'][k] for k in required):
        raise ValueError('Missing component checks')
    native.preflight_source(campaign=SOURCE)


native.preflight_source = native.preflight
native.preflight = preflight


def prepare(args):
    source = native.verify(SOURCE)
    native.preflight_source(SOURCE)
    audit = json.loads((AUDIT / 'train800_audit.json').read_text())
    rescore = json.loads((AUDIT / 'test200_rescore_report.json').read_text())
    pool = AUDIT / 'test200_revalidated_initialization.json'
    if not audit['summary']['passed'] or not rescore['passed']:
        raise ValueError('Starting-pool and source-integrity audits must pass')
    if bo.digest(pool) != rescore['revalidated_pool_sha256']:
        raise ValueError('Revalidated initialization changed')
    args.campaign.mkdir(parents=True, exist_ok=True)
    for method in ('tpe', 'lolbo'):
        directory = args.campaign / method
        if (directory / 'recipe.json').exists():
            verify(directory)
            continue
        recipe = dict(source, method=method, adapter=str(Path(__file__).resolve()),
                      optuna_version=importlib.metadata.version('optuna'),
                      prepared_pool=str(pool), batch_size=1 if method == 'tpe' else 50,
                      prepared_at=time.time(), source_recipe_sha256=bo.digest(SOURCE / 'recipe.json'),
                      evaluation_scope='test200', tpe_bounds=[-4.0, 4.0],
                      tpe_vae_policy='frozen_author',
                      initialization_revalidation_sha256=bo.digest(AUDIT / 'test200_rescore_report.json'))
        recipe['code'] = dict(source['code'], **{str(Path(__file__).resolve()): bo.digest(Path(__file__))})
        recipe['assets'] = dict(source['assets'], **{str(pool): bo.digest(pool)})
        bo.atomic_json(directory / 'recipe.json', recipe)
    print(json.dumps(dict(prepared=str(args.campaign), test_tasks=200, launched=False)), flush=True)


def connect(path):
    db = sqlite3.connect(str(path), timeout=60)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS trials (number INTEGER PRIMARY KEY, value BLOB NOT NULL)')
    db.commit()
    return db


def get(db, key):
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return pickle.loads(row[0]) if row else None


def put(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
               (key, sqlite3.Binary(pickle.dumps(value, protocol=4))))


def trials(db):
    return [pickle.loads(row[0]) for row in db.execute('SELECT value FROM trials ORDER BY number')]


def add_trial(db, number, record):
    db.execute('INSERT INTO trials VALUES (?,?)',
               (number, sqlite3.Binary(pickle.dumps(record, protocol=4))))


def state_meta(study, progress, identity):
    sampler = copy.copy(study.sampler)
    sampler._constraints_func = None
    return dict(identity=identity, progress=progress, sampler=sampler,
                rng=bo.capture_rng())


def tpe_archive(directory, identity):
    paths = [directory / name for name in ('runtime.sqlite', 'history.json', 'progress.json')]
    manifest = dict(identity=identity, members={p.name: bo.digest(p) for p in paths})
    target = directory.with_suffix('.tar.gz')
    temporary = target.with_name(target.name + '.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as packed:
        for path in paths:
            packed.add(path, arcname=path.name, recursive=False)
        payload = json.dumps(manifest).encode()
        member = tarfile.TarInfo('manifest.json')
        member.size = len(payload)
        packed.addfile(member, io.BytesIO(payload))
    verify_tpe_archive(temporary, identity)
    os.replace(temporary, target)
    cleanup_tpe(directory, target, identity)


def verify_tpe_archive(path, identity):
    with tarfile.open(path, 'r:gz') as packed:
        members = packed.getmembers()
        if (len(members) != 4 or not all(m.isfile() for m in members)
                or {m.name for m in members} != {'runtime.sqlite', 'history.json', 'progress.json', 'manifest.json'}):
            raise ValueError('Invalid TPE archive inventory')
        manifest = json.load(packed.extractfile('manifest.json'))
        if manifest['identity'] != identity:
            raise ValueError('TPE archive identity mismatch')
        if set(manifest['members']) != {'runtime.sqlite', 'history.json', 'progress.json'}:
            raise ValueError('TPE archive checksum inventory mismatch')
        for name, expected in manifest['members'].items():
            h = hashlib.sha256()
            with packed.extractfile(name) as stream:
                for chunk in iter(lambda: stream.read(2**20), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError('TPE archive member checksum mismatch')
        history = json.load(packed.extractfile('history.json'))
    if history['status'] != 'complete' or history['identity'] != identity:
        raise ValueError('Incomplete TPE archive')
    protocol.check_accounting(dict(initialization=history['initialization'],
        proposals=history['proposal_budget'], fresh_budget=history['fresh_budget']),
        history, history['observations'])
    if history['proposals'] < history['proposal_budget'] and history['fresh_evaluations'] < history['fresh_budget']:
        raise ValueError('TPE run ended without exhausting a budget')
    feasible = [r for r in history['observations'] if r['feasible']]
    if not feasible or not np.isclose(min(r['mic'] for r in feasible), history['best_feasible_mic']):
        raise ValueError('Invalid TPE incumbent')
    return history


def cleanup_tpe(directory, archive, identity):
    verify_tpe_archive(archive, identity)
    with tarfile.open(archive, 'r:gz') as packed:
        manifest = json.load(packed.extractfile('manifest.json'))
    if directory.exists():
        if any(p.name not in manifest['members'] for p in directory.iterdir()):
            raise ValueError('Unexpected active TPE files at cleanup')
        for name, expected in manifest['members'].items():
            path = directory / name
            if path.exists():
                if bo.digest(path) != expected:
                    raise ValueError('TPE active file changed after archival')
                path.unlink()
        directory.rmdir()


def run_tpe(args, recipe):
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.trial import TrialState, create_trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    identity = dict(format=TPE_FORMAT, recipe_sha256=bo.digest(args.campaign / 'recipe.json'), task=args.task)
    with native.legacy.lock(args.campaign / 'locks' / (args.task + '.lock')):
        directory = args.campaign / 'runs' / args.task
        archive = directory.with_suffix('.tar.gz')
        if archive.exists():
            cleanup_tpe(directory, archive, identity)
            return
        directory.mkdir(parents=True, exist_ok=True)
        db = connect(directory / 'runtime.sqlite')
        saved = get(db, 'state')
        if saved and saved['identity'] != identity:
            raise ValueError('TPE recovery identity changed')
        # Only the objective is retained; no GP training or VAE updates occur in TPE.
        state, backend = protocol.create_state(recipe, args.task, remote_url=args.server)
        objective = state.objective
        distributions = {f'z{i:03d}': FloatDistribution(*recipe['tpe_bounds']) for i in range(256)}
        reference, sequences, scores = bo.load_pool(recipe, args.task)
        if saved is None:
            z = state.train_z.numpy()
            low, high = recipe['tpe_bounds']
            if np.any(z < low) or np.any(z > high):
                raise ValueError(f'Initialization outside TPE bounds: {z.min()}..{z.max()}; clipping forbidden')
            sampler = optuna.samplers.TPESampler(seed=recipe['seed'], constraints_func=tpe_constraints)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            progress = dict(rounds=0, proposals=0, fresh_evaluations=0, cache_hits=0,
                acquisition_fresh=0, recenter_fresh=0, recenter_proposals=0, events=0, seconds=0.0)
            with db:
                for i, (sequence, score) in enumerate(zip(sequences, scores)):
                    params = dict(zip(distributions, map(float, z[i])))
                    row = dict(sequence=sequence, score=float(score), mic=-float(score), phase='initialization',
                               round=0, feasible=True, cached=True, elapsed=0.0)
                    record = dict(params=params, score=float(score), constraints=[recipe['similarity'] - bo.similarity(sequence, reference)], row=row)
                    add_trial(db, i, record)
                    study.add_trial(create_trial(params=params, distributions=distributions,
                        value=score, user_attrs={'constraints': record['constraints']},
                        system_attrs={'constraints': record['constraints']}))
                put(db, 'state', state_meta(study, progress, identity))
        else:
            study = optuna.create_study(direction='maximize', sampler=saved['sampler'])
            study.sampler._constraints_func = tpe_constraints
            progress = saved['progress']
            records = trials(db)
            for record in records:
                study.add_trial(create_trial(params=record['params'], distributions=distributions,
                    value=record['score'], user_attrs={'constraints': record['constraints']},
                    system_attrs={'constraints': record['constraints']}))
            pending = get(db, 'pending')
            if pending is not None:
                study.add_trial(create_trial(params=pending['params'], distributions=distributions,
                                             state=TrialState.RUNNING))
            objective.xs_to_scores_dict = {r['row']['sequence']: r['score'] for r in records}
            objective.num_calls = progress['fresh_evaluations']
            bo.restore_rng(saved['rng'])
        stopping = False
        def stop(_sig, _frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while progress['proposals'] < recipe['proposals'] and progress['fresh_evaluations'] < recipe['fresh_budget']:
                pending = get(db, 'pending')
                if pending is None:
                    started = time.monotonic()
                    trial = study.ask(fixed_distributions=distributions)
                    z = torch.tensor([[trial.params[k] for k in distributions]], dtype=torch.float32)
                    with torch.no_grad():
                        recommended = objective.vae_decode(z)
                    if len(recommended) != 1:
                        raise ValueError('TPE must decode exactly one proposal')
                    pending = dict(number=trial.number, params=trial.params, sequence=recommended[0],
                                   proposal_seconds=time.monotonic() - started)
                    with db:
                        put(db, 'pending', pending)
                        put(db, 'state', state_meta(study, progress, identity))
                if 'scored' not in pending:
                    started = time.monotonic()
                    pending['scored'] = bo.score_batch(objective, [pending['sequence']])
                    pending['oracle_seconds'] = time.monotonic() - started
                    with db:
                        put(db, 'pending', pending)
                else:
                    scored = pending['scored']
                    objective.xs_to_scores_dict.update(scored['new_scores'])
                    objective.num_calls += scored['fresh_evaluations']
                    bo.restore_rng(scored['rng'])
                scored = pending['scored']
                constraint = scored['constraints'][0].tolist()
                trial_id = study._storage.get_trial_id_from_study_id_trial_number(
                    study._study_id, pending['number'])
                study._storage.set_trial_user_attr(trial_id, 'constraints', constraint)
                study.tell(pending['number'], scored['scores'][0])
                progress['proposals'] += 1
                progress['events'] += 1
                progress['rounds'] += 1
                progress['fresh_evaluations'] += scored['fresh_evaluations']
                progress['acquisition_fresh'] += scored['fresh_evaluations']
                progress['cache_hits'] += int(scored['cache_hits'][0])
                progress['seconds'] += pending['proposal_seconds'] + pending['oracle_seconds']
                sequence, score = pending['sequence'], scored['scores'][0]
                feasible = bo.similarity(sequence, reference) >= recipe['similarity'] - 1e-8
                if feasible != bool(constraint[0] <= 0):
                    raise ValueError('TPE feasibility/constraint mismatch')
                row = dict(sequence=sequence, score=score, mic=-score, phase='acquisition', round=progress['rounds'],
                           feasible=feasible, cached=bool(scored['cache_hits'][0]), elapsed=progress['seconds'])
                record = dict(params=pending['params'], score=score, constraints=constraint, row=row)
                with db:
                    add_trial(db, pending['number'], record)
                    put(db, 'pending', None)
                    put(db, 'state', state_meta(study, progress, identity))
                if progress['events'] % 25 == 0 or stopping or args.stop_after_events:
                    rows = [r['row'] for r in trials(db)]
                    protocol.check_accounting(recipe, progress, rows)
                    best = min(r['mic'] for r in rows if r['feasible'])
                    bo.atomic_json(directory / 'progress.json', dict(progress, identity=identity,
                        status='running', best_feasible_mic=best, e2e_updates=0))
                    print(json.dumps(dict(task=args.task, **progress, best_feasible_mic=best)), flush=True)
                if stopping or (args.stop_after_events and progress['events'] >= args.stop_after_events):
                    return
            rows = [r['row'] for r in trials(db)]
            protocol.check_accounting(recipe, progress, rows)
            best = min((r for r in rows if r['feasible']), key=lambda row: row['mic'])
            values = dict(progress, identity=identity, status='complete', e2e_updates=0,
                          best_feasible_mic=best['mic'], best_feasible_sequence=best['sequence'])
            bo.atomic_json(directory / 'progress.json', values)
            bo.atomic_json(directory / 'history.json', dict(values, observations=rows, reference=reference,
                initialization=recipe['initialization'], proposal_budget=recipe['proposals'],
                fresh_budget=recipe['fresh_budget'], initialization_actual_calls=0,
                initialization_source='freshly_revalidated_measured_test200_pool',
                budget_definition=recipe['budget_definition'], vae_policy='frozen_author'))
            db.close()
            tpe_archive(directory, identity)
            print(json.dumps(dict(task=args.task, status='complete', archive=str(archive))), flush=True)
        finally:
            db.close()
            with contextlib.suppress(Exception):
                backend.close()


def tpe_constraints(trial):
    return trial.user_attrs['constraints']


def run(args):
    recipe = verify(args.campaign)
    if args.task not in recipe['test']:
        raise ValueError('Evaluation queue accepts held-out test200 tasks only')
    if recipe['purpose'] == 'production':
        if not args.launch:
            raise ValueError('Production requires --launch')
        preflight(args.campaign)
    if recipe['method'] == 'lolbo':
        native.run_task(args)
    else:
        run_tpe(args, recipe)


def worker(args):
    recipe = verify(args.campaign)
    preflight(args.campaign)
    tasks = recipe['test']
    assigned = tasks[args.index::args.workers]
    ordered = assigned + [task for task in tasks if task not in assigned]
    child, stopping = None, False
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for task in ordered:
        if stopping:
            return
        if (args.campaign / 'runs' / (task + '.tar.gz')).exists():
            continue
        command = [sys.executable, str(Path(__file__).resolve()), 'run-task', '--campaign', str(args.campaign),
                   '--task', task, '--cpu', str(args.cpu), '--server', args.server, '--launch']
        child = subprocess.Popen(command, env=os.environ.copy())
        result = child.wait()
        if result not in (0, 73):
            bo.atomic_json(args.campaign / 'worker_errors' / f'{args.index}.json',
                           dict(task=task, returncode=result, time=time.time()))
            raise RuntimeError('Worker stopped after task error; no silent skipping')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    for action in ('run-task', 'worker'):
        p = sub.add_parser(action)
        p.add_argument('--cpu', required=True, type=int)
        p.add_argument('--server', required=True)
        p.add_argument('--launch', action='store_true')
        p.add_argument('--stop-after-events', type=int)
        if action == 'run-task':
            p.add_argument('--task', required=True)
        else:
            p.add_argument('--index', type=int, required=True)
            p.add_argument('--workers', type=int, required=True)
    for p in sub.choices.values():
        p.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    else:
        os.sched_setaffinity(0, {args.cpu})
        torch.set_num_threads(1)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            parser.error('CPU workers require CUDA_VISIBLE_DEVICES empty')
        try:
            (run if args.action == 'run-task' else worker)(args)
        except BlockingIOError:
            sys.exit(73)


if __name__ == '__main__':
    main()
