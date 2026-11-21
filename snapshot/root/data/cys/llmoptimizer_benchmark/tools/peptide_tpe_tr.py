#!/usr/bin/env python3
"""Isolated peptide TPE-TR: one proposal/evaluation at a time, frozen VAE, no TS."""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import ModuleType

import numpy as np
import peptide_tpe_tr_policy as policy

# The pinned peptide release uses namespace packages. The repository's unrelated
# regular mindware package otherwise shadows them when invoked from the repo root.
RELEASE = Path('/root/data/logs/peptide_bo_release_20261027_v5')
for package in ('mindware', 'mindware.domains', 'mindware.domains.protein'):
    location = str(RELEASE.joinpath(*package.split('.')))
    if package not in sys.modules:
        module = ModuleType(package)
        module.__package__ = package
        module.__path__ = [location]
        module.__spec__ = ModuleSpec(package, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[package] = module
        if '.' in package:
            parent, child = package.rsplit('.', 1)
            setattr(sys.modules[parent], child, module)
    elif location not in sys.modules[package].__path__:
        sys.modules[package].__path__ = [location, *sys.modules[package].__path__]
import peptide_test200 as storage
from peptide_test200 import bo, protocol, torch
from peptide_tpe_rpc import PersistentRemoteObjective

ROOT = Path(__file__).resolve().parent
PARENT = Path('/root/data/logs/peptide_test200_baselines_20261027_v2/tpe')
FORMAT = 'peptide-sequential-tpe-tr-v1'


def prepare(args):
    """Pin a new recipe; no worker, inference or oracle requests are launched."""
    target = args.campaign.resolve()
    if target == PARENT or (target/'recipe.json').exists():
        raise ValueError('Use a new campaign directory for TPE-TR')
    source = storage.verify(PARENT)
    settings = dict(policy.DEFAULTS)
    policy.validate(settings)
    recipe = dict(source, method='tpe_tr', variant=FORMAT, adapter=str(Path(__file__).resolve()),
        parent_recipe=str(PARENT/'recipe.json'), parent_sha256=bo.digest(PARENT/'recipe.json'),
        prepared_at=time.time(), batch_size=1, e2e_every=0, e2e_policy='disabled',
        center_policy='best_observed_feasible_proposed_latent',
        acquisition='constrained_TPE_density_ratio', thompson_sampling=False,
        reconstruction=False, vae_policy='frozen_author', trust_region=settings,
        optuna_version=importlib.metadata.version('optuna'))
    recipe['code'] = dict(source['code'])
    for name in ('peptide_tpe_tr.py', 'peptide_tpe_tr_policy.py', 'peptide_tpe_rpc.py'):
        recipe['code'][str(ROOT/name)] = bo.digest(ROOT/name)
    bo.atomic_json(target/'recipe.json', recipe)
    bo.atomic_json(target/'launch_status.json', dict(prepared=True, launched=False,
        tasks=len(recipe['test']), batch_size=1, method='TPE-TR'))
    print(json.dumps(dict(campaign=str(target), tasks=len(recipe['test']), launched=False)))


def verify(campaign):
    recipe = storage.native.verify(campaign)
    if (recipe.get('variant') != FORMAT or recipe.get('method') != 'tpe_tr'
            or recipe['adapter'] != str(Path(__file__).resolve())
            or recipe['batch_size'] != 1 or recipe.get('thompson_sampling') is not False
            or recipe.get('reconstruction') is not False or recipe['vae_policy'] != 'frozen_author'):
        raise ValueError('Incorrect TPE-TR method identity')
    if bo.digest(Path(recipe['parent_recipe'])) != recipe['parent_sha256']:
        raise ValueError('Parent recipe changed')
    if importlib.metadata.version('optuna') != recipe['optuna_version']:
        raise ValueError('Optuna version differs from pinned recipe')
    if (len(recipe['test']) != len(set(recipe['test']))
            or len(recipe['test']) != 200 or set(recipe['test']) & set(recipe['train'])):
        raise ValueError('Wrong held-out task cohort')
    if (recipe['initialization'], recipe['proposals'], recipe['fresh_budget'],
            recipe['similarity'], recipe['tpe_bounds']) != (100, 5000, 5000, .75, [-4., 4.]):
        raise ValueError('TPE-TR comparison protocol changed')
    policy.validate(recipe['trust_region'])
    return recipe


def create_objective(recipe, task, server, identity):
    """Use the same frozen decoder/oracle backend as the existing peptide baselines."""
    sys.path.insert(0, recipe['cpu_upstream'])
    os.chdir(recipe['cpu_upstream'])
    from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
    objective = ApexConstrainedDiverseObjective(task_id='apex', task_specific_args=['bacteria_0'],
        path_to_vae_statedict=recipe['vae'], max_string_length=30)
    objective.compute_constraints = lambda xs: torch.tensor([[recipe['similarity']-
        bo.similarity(s, recipe['references'][task])] for s in xs], dtype=torch.float32)
    objective.vae.eval()
    for parameter in objective.vae.parameters():
        parameter.requires_grad_(False)
    client = PersistentRemoteObjective(objective, server, task+':'+identity['recipe_sha256'])
    return objective, client


def initialize(recipe, task, objective, identity):
    reference, sequences, scores = bo.load_pool(recipe, task)
    if reference != recipe['references'][task]:
        raise ValueError('Task/reference mismatch')
    z = bo.encode(objective, sequences).numpy()
    if z.shape != (recipe['initialization'], 256) or not np.isfinite(z).all():
        raise ValueError('Incorrect frozen-VAE initialization coordinates')
    lower, upper = recipe['tpe_bounds']
    if np.any(z < lower) or np.any(z > upper):
        raise ValueError('Initialization outside global bounds; clipping forbidden')
    records = []
    for latent, sequence, score in zip(z, sequences, scores):
        constraint = recipe['similarity']-bo.similarity(sequence, reference)
        row = dict(sequence=sequence, score=float(score), mic=-float(score), feasible=constraint<=0,
            cached=True, phase='initialization', round=0, elapsed=0.)
        records.append(dict(z=latent.tolist(), score=float(score), constraints=[constraint], row=row))
    state = dict(identity=identity, region=policy.new_region(records, recipe['trust_region']),
        sampler=None, rng=bo.capture_rng(), progress=dict(rounds=0, proposals=0, events=0,
        fresh_evaluations=0, cache_hits=0, acquisition_fresh=0, recenter_fresh=0,
        recenter_proposals=0, seconds=0.))
    return state, records


def history(state, records, recipe, complete=False):
    best = records[policy.incumbent(records)]['row']
    return dict(state['progress'], identity=state['identity'], method='TPE-TR',
        status='complete' if complete else 'running', observations=[r['row'] for r in records],
        initialization=recipe['initialization'], proposal_budget=recipe['proposals'],
        fresh_budget=recipe['fresh_budget'], initialization_actual_calls=0,
        initialization_source='freshly_revalidated_measured_test200_pool',
        reference=recipe['references'][state['identity']['task']],
        best_feasible_mic=best['mic'], best_feasible_sequence=best['sequence'],
        trust_region=dict(state['region']), trust_region_settings=recipe['trust_region'],
        batch_size=1, thompson_sampling=False, reconstruction=False,
        vae_policy='frozen_author', e2e_updates=0, budget_definition=recipe['budget_definition'])


def write_progress(directory, state, records, recipe, complete=False):
    h = history(state, records, recipe, complete)
    bo.atomic_json(directory/'progress.json', {k:v for k,v in h.items() if k!='observations'})
    bo.atomic_json(directory/'history.json', h)
    return h


def score_pending(client, pending, cache):
    """Persisted request payload allows RPC receipt replay after an unknown response."""
    sequence = pending['sequence']
    requested = pending['oracle_request']['sequences']
    reply = client.request(pending['oracle_request']) if requested else {'scores':[]}
    values = np.asarray(reply['scores'], dtype=float)
    if values.shape != (len(requested),) or not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite oracle result')
    fresh = dict(zip(requested, values.tolist()))
    return dict(score=float(fresh[sequence] if requested else cache[sequence]),
                new_scores=fresh, fresh_evaluations=len(fresh), cached=not bool(requested))


def run_task(args, recipe=None):
    recipe = verify(args.campaign) if recipe is None else recipe
    if args.task not in recipe['test']:
        raise ValueError('Only the held-out200 task IDs are allowed')
    identity = dict(format=FORMAT, recipe_sha256=bo.digest(args.campaign/'recipe.json'), task=args.task)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    prior_handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with storage.native.legacy.lock(args.campaign/'locks'/(args.task+'.lock')):
            directory = args.campaign/'runs'/args.task
            archive = directory.with_suffix('.tar.gz')
            if archive.exists():
                storage.cleanup_tpe(directory, archive, identity)
                return storage.verify_tpe_archive(archive, identity)
            directory.mkdir(parents=True, exist_ok=True)
            db = storage.connect(directory/'runtime.sqlite')
            client = None
            try:
                random.seed(recipe['seed']); np.random.seed(recipe['seed']); torch.manual_seed(recipe['seed'])
                state = storage.get(db, 'state')
                if state is not None and state['identity'] != identity:
                    raise ValueError('Recovery identity changed')
                objective, client = create_objective(recipe, args.task, args.server, identity)
                if state is None:
                    state, records = initialize(recipe, args.task, objective, identity)
                    with db:
                        for i, record in enumerate(records):
                            storage.add_trial(db, i, record)
                        storage.put(db, 'state', state)
                else:
                    records = storage.trials(db)
                protocol.check_accounting(recipe, state['progress'], [r['row'] for r in records])
                cache = {r['row']['sequence']:r['score'] for r in records}
                session = policy.SequentialTPE(recipe['trust_region'], recipe['tpe_bounds'],
                                               recipe['seed'], state['sampler'])
                bo.restore_rng(state['rng'])
                pending = storage.get(db, 'pending')
                if pending is not None:
                    session.restore_pending(records, state['region'], pending['proposal'])
                while (state['progress']['proposals'] < recipe['proposals']
                       and state['progress']['fresh_evaluations'] < recipe['fresh_budget']):
                    if pending is None:
                        if stopped:
                            break
                        started = time.monotonic()
                        proposal = session.ask(records, state['region'])
                        z = torch.tensor([proposal['z']], dtype=torch.float32)
                        with torch.no_grad():
                            sequences = list(objective.vae_decode(z))
                        if len(sequences) != 1:
                            raise ValueError('TPE-TR must decode exactly one recommendation')
                        sequence = sequences[0]
                        bo.validate_sequence(sequence)
                        constraint = recipe['similarity']-bo.similarity(sequence, recipe['references'][args.task])
                        pending = dict(event=state['progress']['events']+1, proposal=proposal,
                            region_before=copy.deepcopy(state['region']), sequence=sequence,
                            constraints=[constraint], seconds=time.monotonic()-started,
                            oracle_request=dict(op='oracle', sequences=[] if sequence in cache else [sequence],
                                nonce=int(torch.randint(0, 2**62, (1,)).item())))
                        state.update(sampler=session.checkpoint(), rng=bo.capture_rng())
                        with db:
                            storage.put(db, 'state', state); storage.put(db, 'pending', pending)
                    if getattr(args, 'stop_after_stage', None) == 'proposal' and 'scored' not in pending:
                        return write_progress(directory, state, records, recipe)
                    if 'scored' not in pending:
                        started = time.monotonic()
                        pending['scored'] = score_pending(client, pending, cache)
                        pending['seconds'] += time.monotonic()-started
                        with db:
                            storage.put(db, 'pending', pending)
                    if getattr(args, 'stop_after_stage', None) == 'score':
                        return write_progress(directory, state, records, recipe)
                    if pending['event'] != state['progress']['events']+1:
                        raise ValueError('Pending event has already committed or is out of order')
                    scored = pending['scored']; score = scored['score']
                    session.tell(score, pending['constraints'])
                    progress = state['progress']
                    for key in ('proposals', 'rounds', 'events'):
                        progress[key] += 1
                    progress['fresh_evaluations'] += scored['fresh_evaluations']
                    progress['acquisition_fresh'] += scored['fresh_evaluations']
                    progress['cache_hits'] += int(scored['cached'])
                    progress['seconds'] += pending['seconds']
                    row = dict(sequence=pending['sequence'], score=score, mic=-score,
                        feasible=max(pending['constraints'])<=0, cached=scored['cached'],
                        phase='acquisition', round=progress['rounds'], elapsed=progress['seconds'],
                        region_revision=pending['proposal']['region_revision'],
                        region_length=pending['region_before']['length'],
                        region_center_index=pending['region_before']['center_index'],
                        local_observations=pending['proposal']['local_observations'])
                    record = dict(z=pending['proposal']['z'], u=pending['proposal']['u'],
                        region_revision=pending['proposal']['region_revision'], score=score,
                        constraints=pending['constraints'], row=row)
                    records.append(record)
                    policy.observe(state['region'], records, recipe['trust_region'])
                    row['region_action'] = state['region']['last_action']
                    cache.update(scored['new_scores'])
                    protocol.check_accounting(recipe, progress, [r['row'] for r in records])
                    state.update(sampler=session.checkpoint(), rng=bo.capture_rng())
                    with db:
                        storage.add_trial(db, len(records)-1, record)
                        storage.put(db, 'state', state); storage.put(db, 'pending', None)
                    pending = None
                    if progress['events'] % 25 == 0 or stopped or args.stop_after_events:
                        write_progress(directory, state, records, recipe)
                        print(json.dumps(dict(task=args.task, method='TPE-TR', **progress,
                            region_length=state['region']['length'])), flush=True)
                    if stopped or (args.stop_after_events and progress['events']>=args.stop_after_events):
                        break
                complete = (state['progress']['proposals']>=recipe['proposals']
                            or state['progress']['fresh_evaluations']>=recipe['fresh_budget'])
                result = write_progress(directory, state, records, recipe, complete)
            finally:
                db.close()
                if client is not None:
                    with contextlib.suppress(Exception):
                        client.close()
            if complete:
                storage.tpe_archive(directory, identity)
            return result
    finally:
        for s, handler in prior_handlers.items():
            signal.signal(s, handler)


def worker(args):
    recipe = verify(args.campaign)
    if not 0 <= args.index < args.workers:
        raise ValueError('Invalid evaluation worker index/count')
    assigned = recipe['test'][args.index::args.workers]
    ordered = assigned+[t for t in recipe['test'] if t not in assigned]
    stopping, child = False, None
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    for task in ordered:
        if stopping:
            return
        if (args.campaign/'runs'/(task+'.tar.gz')).exists():
            continue
        command = [sys.executable, str(Path(__file__).resolve()), 'run-task', '--campaign',
            str(args.campaign), '--task', task, '--cpu', str(args.cpu), '--server', args.server, '--launch']
        child = subprocess.Popen(command, env=os.environ.copy())
        code = child.wait()
        if code not in (0, 73):
            bo.atomic_json(args.campaign/'worker_errors'/f'{args.index}.json',
                dict(task=task, returncode=code, at=time.time()))
            raise RuntimeError('TPE-TR worker stopped after a task error')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--campaign', type=Path, required=True)
    for action in ('run-task', 'worker'):
        p = sub.add_parser(action)
        p.add_argument('--campaign', type=Path, required=True)
        p.add_argument('--cpu', type=int, required=True)
        p.add_argument('--server', required=True)
        p.add_argument('--launch', action='store_true')
        p.add_argument('--stop-after-events', type=int)
        p.add_argument('--stop-after-stage', choices=['proposal', 'score'])
        if action == 'run-task':
            p.add_argument('--task', required=True)
        else:
            p.add_argument('--index', type=int, required=True)
            p.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.action == 'prepare':
        prepare(args)
        return
    if not args.launch:
        parser.error('Evaluation requires --launch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU coordinators require CUDA_VISIBLE_DEVICES empty; inference is remote')
    if args.stop_after_events is not None and args.stop_after_events < 1:
        parser.error('--stop-after-events must be positive')
    os.sched_setaffinity(0, {args.cpu}); torch.set_num_threads(1)
    try:
        (run_task if args.action=='run-task' else worker)(args)
    except BlockingIOError:
        sys.exit(73)


if __name__ == '__main__':
    main()
