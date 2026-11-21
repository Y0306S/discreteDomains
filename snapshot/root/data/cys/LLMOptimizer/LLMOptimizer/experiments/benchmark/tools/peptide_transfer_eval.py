#!/usr/bin/env python3
"""Frozen-VAE peptide FSBO/RGPE evaluation with native constrained batch TS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import peptide_transfer as transfer
import peptide_test200 as storage
from peptide_test200 import bo, protocol, torch
from peptide_tpe_rpc import PersistentRemoteObjective
from peptide_transfer_models import receive, send
from peptide_transfer_policy import effective_recipe

ROOT = Path(__file__).resolve().parent
TRANSFER = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
PYTHONS = dict(fsbo='/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python',
               rgpe='/root/data/cys/baseline/envs/bl_rgpe/bin/python')
FORMAT = 'peptide-frozen-transfer-v1'


def bank_inventory(bank):
    if not bank.exists():
        return {}
    with sqlite3.connect('file:' + str(bank) + '?mode=ro', uri=True, timeout=120) as db:
        return {task: dict(metadata=json.loads(meta), sha256=sha) for task, meta, sha in
                db.execute('SELECT task,metadata,sha256 FROM models')}


def prepare(args):
    source = storage.native.verify(transfer.SOURCE)
    manifest, db = transfer.load_corpus(args.transfer / 'corpus')
    db.close()
    rescore = json.loads((transfer.AUDIT / 'test200_rescore_report.json').read_text())
    pool = transfer.AUDIT / 'test200_revalidated_initialization.json'
    if not rescore['passed'] or bo.digest(pool) != rescore['revalidated_pool_sha256']:
        raise ValueError('Held-out initialization revalidation failed')
    bank = args.transfer / 'rgpe/models.sqlite'
    inventory = bank_inventory(bank)
    code = dict(source['code'])
    for name in ('peptide_transfer_eval.py', 'peptide_transfer_models.py',
                 'peptide_transfer_policy.py', 'fsbo_target_refit.py',
                 'peptide_rgpe_posterior.py', 'peptide_rgpe_exact.py',
                 'peptide_transfer.py', 'peptide_test200.py', 'peptide_tpe_rpc.py'):
        code[str(ROOT / name)] = bo.digest(ROOT / name)
    vendor_paths = [Path('/root/data/cys/baseline/repos/FSBO') / name
                    for name in ('fsbo_modules.py', 'fsbo_utils.py')]
    vendor_paths += [transfer.RGPE_REPO / 'rgpe' / name for name in ('utils.py', 'methods/rgpe.py')]
    packages = Path('/root/data/cys/baseline/envs/bl_rgpe/lib/python3.7/site-packages')
    vendor_paths += [packages / name for name in ('smac/epm/gaussian_process.py',
        'smac/epm/base_epm.py', 'smac/epm/gp_kernels.py', 'smac/epm/gp_base_prior.py',
        'skopt/learning/gaussian_process/gpr.py', 'botorch/sampling/qmc.py')]
    for path in vendor_paths:
        code[str(path)] = bo.digest(path)
    for method in args.methods.split(','):
        if method not in PYTHONS:
            raise ValueError('Unknown method: ' + method)
        directory = args.campaign / method
        if (directory / 'recipe.json').exists():
            verify(directory)
            continue
        sources = [t for t in manifest['source_tasks'] if t in inventory]
        if method == 'rgpe' and not args.component and len(sources) != 800:
            bo.atomic_json(directory / 'plan.json', dict(status='awaiting_source_training',
                source_models=len(sources), required_source_models=800,
                corpus_sha256=bo.digest(args.transfer / 'corpus/manifest.json'),
                command='Re-run prepare when all 800 source models are saved', launched=False))
            print(json.dumps(dict(method=method, prepared_plan=True, source_models=len(sources), ready=False)), flush=True)
            continue
        if method == 'rgpe' and not sources:
            raise ValueError('RGPE component check needs at least one completed real source GP')
        if args.component:
            sources = sources[:args.source_limit]
        recipe = dict(source, format=FORMAT, method=method,
            purpose='component_check' if args.component else 'production',
            adapter=str(Path(__file__).resolve()), transfer=str(args.transfer),
            prepared_pool=str(pool), evaluation_scope='test200',
            representation=manifest['representation'], vae_policy='frozen_author', e2e_every=0,
            e2e_policy='disabled_to_preserve_shared_transfer_coordinates',
            initialization=100, proposals=args.proposals if args.component else 5000,
            fresh_budget=args.proposals if args.component else 5000,
            batch_size=50, n_candidates=args.candidates if args.component else 5000,
            fsbo_target_epochs=args.target_epochs if args.component else 1000,
            fsbo_target_learning_rate=0.001, fsbo_target_patience=16,
            rgpe_rank_samples=args.rank_samples if args.component else 256,
            rgpe_sampling_mode='correct', rgpe_weight_dilution='probabilistic-ld',
            coordinate_policy='canonical_frozen_posterior_mean_for_all_observations',
            candidate_policy='native_LOLBO_Sobol_mask_raw_latent_trust_region',
            corpus=str(args.transfer / 'corpus/manifest.json'),
            corpus_sha256=bo.digest(args.transfer / 'corpus/manifest.json'),
            lower=manifest['lower'], upper=manifest['upper'],
            fsbo_weights=str(args.transfer / 'fsbo/weights'),
            fsbo_weights_sha256=bo.digest(args.transfer / 'fsbo/weights'),
            fsbo_status=str(args.transfer / 'fsbo/status.json'),
            rgpe_bank=str(bank), rgpe_sources=sources if method == 'rgpe' else [],
            rgpe_model_hashes={t: inventory[t]['sha256'] for t in sources} if method == 'rgpe' else {},
            source_payload_hashes=manifest['task_hashes'],
            initialization_revalidation_sha256=bo.digest(transfer.AUDIT / 'test200_rescore_report.json'),
            code=code, assets=dict(source['assets'], **{str(pool): bo.digest(pool)}),
            model_python=PYTHONS[method], prepared_at=time.time())
        bo.atomic_json(directory / 'recipe.json', recipe)
        verify(directory)
        print(json.dumps(dict(method=method, purpose=recipe['purpose'], source_models=len(recipe['rgpe_sources']),
            prepared=str(directory), launched=False)), flush=True)


def verify(directory):
    recipe = effective_recipe(directory / 'recipe.json')
    if recipe['format'] != FORMAT or recipe['method'] not in PYTHONS:
        raise ValueError('Wrong transfer recipe')
    if (len(recipe['train']) != 800 or len(recipe['test']) != 200
            or set(recipe['train']) & set(recipe['test']) or recipe['vae_policy'] != 'frozen_author'):
        raise ValueError('Invalid source split or representation')
    for section in ('code', 'assets'):
        for path, expected in recipe[section].items():
            if bo.digest(Path(path)) != expected:
                raise ValueError('Pinned file changed: ' + path)
    if bo.digest(Path(recipe['corpus'])) != recipe['corpus_sha256']:
        raise ValueError('Training corpus changed')
    if recipe['method'] == 'fsbo':
        status = json.loads(Path(recipe['fsbo_status']).read_text())
        if (status['status'] != 'complete' or status['step'] != 100000 or not status['reload_verified']
                or status['identity']['corpus_sha256'] != recipe['corpus_sha256']
                or status['weights_sha256'] != recipe['fsbo_weights_sha256']
                or bo.digest(Path(recipe['fsbo_weights'])) != recipe['fsbo_weights_sha256']):
            raise ValueError('FSBO production checkpoint is not verified')
    else:
        inventory = bank_inventory(Path(recipe['rgpe_bank']))
        if (not recipe['rgpe_sources'] or len(recipe['rgpe_sources']) != len(set(recipe['rgpe_sources']))
                or not set(recipe['rgpe_sources']) <= set(recipe['train'])):
            raise ValueError('RGPE inventory must contain train800 tasks only')
        for task in recipe['rgpe_sources']:
            row = inventory.get(task)
            if (row is None or row['sha256'] != recipe['rgpe_model_hashes'][task]
                    or row['metadata']['corpus_sha256'] != recipe['corpus_sha256']
                    or row['metadata']['source_payload_sha256'] != recipe['source_payload_hashes'][task]
                    or not row['metadata']['reload_verified']):
                raise ValueError('Invalid source-bank identity: ' + task)
        if recipe['purpose'] == 'production' and set(recipe['rgpe_sources']) != set(recipe['train']):
            raise ValueError('Production requires all 800 source GPs')
    if recipe['purpose'] == 'production' and (recipe['initialization'], recipe['proposals'],
            recipe['fresh_budget'], recipe['batch_size'], recipe['n_candidates']) != (100, 5000, 5000, 50, 5000):
        raise ValueError('Production budget or candidate recipe changed')
    return recipe


def preflight(directory):
    gate = json.loads((directory / 'preflight_passed.json').read_text())
    report = json.loads(Path(gate['report']).read_text())
    required = {'real_inference', 'target_refit', 'frozen_encoding', 'native_batch50',
                'resume', 'pending_without_rescore', 'archive', 'accounting', 'task_isolation'}
    if (gate['recipe_sha256'] != bo.digest(directory / 'recipe.json')
            or gate['report_sha256'] != bo.digest(Path(gate['report'])) or not report['passed']
            or not required <= report['checks'].keys() or not all(report['checks'][k] for k in required)):
        raise ValueError('Transfer component check has not passed')
    if not report.get('production_scale_passed', False):
        raise ValueError('Full source inventory/candidate-pool performance check is still required')


class ModelClient:
    def __init__(self, recipe_path, recipe):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        self.child = subprocess.Popen([recipe['model_python'], str(ROOT / 'peptide_transfer_models.py'),
            '--recipe', str(recipe_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self.last_sample = None

    def request(self, op, **values):
        send(self.child.stdin, dict(values, op=op))
        reply = receive(self.child.stdout)
        if reply is None or reply.get('error'):
            raise RuntimeError('Transfer model worker: ' + (reply.get('error', 'EOF') if reply else 'EOF'))
        return reply.get('result')

    def close(self):
        try:
            if self.child.poll() is None:
                self.request('close')
        finally:
            self.child.stdin.close()
            self.child.stdout.close()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.terminate()
                self.child.wait(timeout=10)


def normalized(x, recipe):
    low, high = np.asarray(recipe['lower']), np.asarray(recipe['upper'])
    return ((np.asarray(x) - low) / np.where(high > low, high - low, 1.)).astype(np.float32)


class BridgePosterior:
    def __init__(self, client, x, recipe):
        self.client, self.x, self.recipe = client, x, recipe

    def rsample(self, sample_shape):
        result = self.client.request('sample', candidates=normalized(self.x.cpu().numpy(), self.recipe),
            count=int(sample_shape[0]), seed=int(torch.randint(0, 2**31 - 1, (1,)).item()))
        draws = np.asarray(result['draws'])
        if draws.shape != (int(sample_shape[0]), len(self.x)) or not np.isfinite(draws).all():
            raise ValueError('Invalid transferred posterior draws')
        self.client.last_sample = result['info']
        return torch.as_tensor(draws, dtype=self.x.dtype).unsqueeze(-1)


class BridgeModel:
    def __init__(self, client, recipe):
        self.client, self.recipe = client, recipe

    def posterior(self, x, observation_noise=False):
        return BridgePosterior(self.client, x, self.recipe)


def encode_mean(client, sequences):
    pieces = []
    for start in range(0, len(sequences), 128):
        result = client.inference('encode_mean', sequences=list(sequences[start:start + 128]))['z']
        pieces.append(result.cpu().numpy().astype(np.float32))
    z = np.concatenate(pieces)
    if z.shape != (len(sequences), 256) or not np.isfinite(z).all():
        raise ValueError('Invalid canonical frozen-VAE encoding')
    return z


def check_memory(recipe, observations):
    root = Path('/sys/fs/cgroup')
    maximum = (root / 'memory.max').read_text().strip()
    if maximum == 'max':
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
    else:
        stats = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
        available = int(maximum) - int((root / 'memory.current').read_text()) + int(stats.get('inactive_file', 0))
    dense = 8 * (6 * observations**2 + 4 * recipe['n_candidates']**2)
    worker = max(4 * 2**30, dense + 2 * 2**30)
    if available < 16 * 2**30 + worker:
        raise RuntimeError('Transfer memory guard: keep 16 GiB headroom plus %.1f GiB for this worker' % (worker / 2**30))


def create_objective(recipe, task, server, identity):
    sys.path.insert(0, recipe['cpu_upstream'])
    os.chdir(recipe['cpu_upstream'])
    from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
    objective = ApexConstrainedDiverseObjective(task_id='apex', task_specific_args=['bacteria_0'],
        path_to_vae_statedict=recipe['vae'], max_string_length=30)
    objective.compute_constraints = lambda xs: torch.tensor([[recipe['similarity'] -
        bo.similarity(x, recipe['references'][task])] for x in xs], dtype=torch.float32)
    client = PersistentRemoteObjective(objective, server, task + ':' + identity['recipe_sha256'])
    return objective, client


def constraint_model(recipe, features, saved=None):
    import gpytorch
    from lolbo.utils.bo_utils.ppgpr import GPModelDKL
    from gpytorch.mlls import PredictiveLogLikelihood
    count = min(recipe['initialization'], 1024)
    model = GPModelDKL(torch.as_tensor(features[:count]), gpytorch.likelihoods.GaussianLikelihood())
    if saved is not None:
        model.load_state_dict(saved, strict=True)
    mll = PredictiveLogLikelihood(model.likelihood, model, num_data=len(features))
    return model, mll


def history_value(saved, recipe, task, complete=False):
    feasible = [r for r in saved['rows'] if r['feasible']]
    best = min(feasible, key=lambda r: r['mic'])
    return dict(saved['progress'], identity=saved['identity'], status='complete' if complete else 'running',
        task=task, method=recipe['method'], reference=recipe['references'][task],
        initialization=recipe['initialization'], proposal_budget=recipe['proposals'],
        fresh_budget=recipe['fresh_budget'], observations=saved['rows'], e2e_updates=0,
        best_feasible_mic=best['mic'], best_feasible_sequence=best['sequence'],
        representation=recipe['representation'], vae_policy=recipe['vae_policy'])


def persist(db, directory, saved, recipe, task):
    saved['rng'] = bo.capture_rng()
    with db:
        storage.put(db, 'state', saved)
    bo.atomic_json(directory / 'progress.json', dict(saved['progress'], identity=saved['identity']))
    bo.atomic_json(directory / 'history.json', history_value(saved, recipe, task))


def run(args):
    recipe = verify(args.campaign)
    if args.task not in recipe['test']:
        raise ValueError('Evaluation task is outside the held-out test200 split')
    if recipe['purpose'] == 'production':
        preflight(args.campaign)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Transfer coordinators are CPU-only; use the inference service')
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {args.cpu})
    identity = dict(format=FORMAT, recipe_sha256=bo.digest(args.campaign / 'recipe.json'), task=args.task)
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with transfer.lock(args.campaign / 'locks' / (args.task + '.lock')):
        directory = args.campaign / 'runs' / args.task
        archive = directory.with_suffix('.tar.gz')
        if archive.exists():
            storage.cleanup_tpe(directory, archive, identity)
            return storage.verify_tpe_archive(archive, identity)
        directory.mkdir(parents=True, exist_ok=True)
        db = storage.connect(directory / 'runtime.sqlite')
        client = model_client = None
        try:
            random.seed(recipe['seed'])
            np.random.seed(recipe['seed'])
            torch.manual_seed(recipe['seed'])
            objective, client = create_objective(recipe, args.task, args.server, identity)
            saved = storage.get(db, 'state')
            if saved is None:
                reference, sequences, scores = bo.load_pool(recipe, args.task)
                for sequence in sequences:
                    bo.validate_sequence(sequence)
                    if bo.similarity(sequence, reference) < recipe['similarity']:
                        raise ValueError('Revalidated initialization violates the reference constraint')
                features = encode_mean(client, sequences)
                rows = [dict(sequence=x, score=float(y), mic=-float(y), phase='initialization',
                    round=0, feasible=True, cached=True, elapsed=0.) for x, y in zip(sequences, scores)]
                from lolbo.utils.bo_utils.turbo import TurboState
                saved = dict(identity=identity, rows=rows, features=features,
                    constraint_weights=None, model_snapshot=None,
                    trust_region=TurboState(dim=256, batch_size=50, best_value=max(scores),
                        best_constraint_values=torch.tensor([0.])),
                    progress=dict(rounds=0, proposals=0, fresh_evaluations=0, cache_hits=0,
                        acquisition_fresh=0, recenter_fresh=0, recenter_proposals=0, events=0, seconds=0.))
                persist(db, directory, saved, recipe, args.task)
            if saved['identity'] != identity:
                raise ValueError('Target-task recovery identity changed')
            if (saved['features'].shape != (len(saved['rows']), 256)
                    or not np.isfinite(saved['features']).all()
                    or not all(np.isfinite(r['score']) and r['mic'] == -r['score'] for r in saved['rows'])):
                raise ValueError('Recovered coordinates and outcomes are misaligned')
            protocol.check_accounting(recipe, saved['progress'], saved['rows'])
            objective.xs_to_scores_dict = {r['sequence']: r['score'] for r in saved['rows']}
            objective.num_calls = saved['progress']['fresh_evaluations']
            constraint, mll = constraint_model(recipe, saved['features'], saved['constraint_weights'])
            model_client = ModelClient(args.campaign / 'recipe.json', recipe)
            bo.restore_rng(saved['rng'])
            from lolbo.utils.bo_utils.turbo import generate_batch, update_state, TurboState
            from lolbo.utils.utils import update_surr_model
            while (saved['progress']['proposals'] < recipe['proposals']
                    and saved['progress']['fresh_evaluations'] < recipe['fresh_budget']):
                pending = storage.get(db, 'pending')
                if pending is not None and args.stop_after_stage == 'proposal':
                    return history_value(saved, recipe, args.task)
                if pending is not None and 'scored' in pending and args.stop_after_stage == 'score':
                    return history_value(saved, recipe, args.task)
                if pending is None:
                    if stopped:
                        break
                    progress = saved['progress']
                    check_memory(recipe, len(saved['rows']))
                    started = time.monotonic()
                    fit = model_client.request('fit', x=normalized(saved['features'], recipe),
                        score=np.asarray([r['score'] for r in saved['rows']]),
                        seed=recipe['seed'] + progress['rounds'])
                    x = torch.as_tensor(saved['features'], dtype=torch.float32)
                    c = torch.tensor([recipe['similarity'] - bo.similarity(r['sequence'], recipe['references'][args.task])
                        for r in saved['rows']], dtype=torch.float32)
                    if progress['rounds'] == 0:
                        cx, cy, epochs = x, c, recipe['gp_initial_epochs']
                    else:
                        cx, cy, epochs = x[-recipe['batch_size']:], c[-recipe['batch_size']:], recipe['gp_update_epochs']
                    mll.num_data = len(saved['features'])
                    constraint = update_surr_model(constraint, mll, recipe['learning_rate'], cx, cy, epochs)
                    feasible = c <= 0
                    y = torch.tensor([r['score'] for r in saved['rows']], dtype=torch.float32).reshape(-1, 1)
                    remaining = recipe['fresh_budget'] - progress['fresh_evaluations']
                    count = min(recipe['batch_size'], recipe['proposals'] - progress['proposals'], remaining)
                    with torch.no_grad():
                        z = generate_batch(saved['trust_region'], BridgeModel(model_client, recipe),
                            x[feasible], y[feasible], count, n_candidates=recipe['n_candidates'],
                            device=torch.device('cpu'), constraint_model_list=[constraint])
                        sequences = list(objective.vae_decode(z))
                    if z.shape != (count, 256) or len(sequences) != count:
                        raise ValueError('Malformed native acquisition batch')
                    for sequence in sequences:
                        bo.validate_sequence(sequence)
                    sequences = protocol.bounded_prefix(sequences, objective.xs_to_scores_dict, remaining)
                    z = z[:len(sequences)]
                    new = list(dict.fromkeys(s for s in sequences if s not in objective.xs_to_scores_dict))
                    pending = dict(sequences=sequences, proposed_z=z.cpu().numpy(), fit=fit,
                        posterior=model_client.last_sample, proposal_seconds=time.monotonic() - started,
                        oracle_request=dict(op='oracle', sequences=new,
                            nonce=int(torch.randint(0, 2**62, (1,)).item())))
                    saved['model_snapshot'] = model_client.request('snapshot')
                    saved['constraint_weights'] = bo.cpu_copy(constraint.state_dict())
                    saved['rng'] = bo.capture_rng()
                    with db:
                        storage.put(db, 'state', saved)
                        storage.put(db, 'pending', pending)
                    if args.stop_after_stage == 'proposal':
                        return history_value(saved, recipe, args.task)
                if 'scored' not in pending:
                    started = time.monotonic()
                    requested = pending['oracle_request']['sequences']
                    result = client.request(pending['oracle_request']) if requested else {'scores': []}
                    values = np.asarray(result['scores'], dtype=float)
                    if values.shape != (len(requested),) or not np.isfinite(values).all():
                        raise ValueError('Invalid APEX oracle response')
                    new_scores = dict(zip(requested, values.tolist()))
                    cache = dict(objective.xs_to_scores_dict, **new_scores)
                    seen, hits = set(objective.xs_to_scores_dict), []
                    for sequence in pending['sequences']:
                        hits.append(sequence in seen)
                        seen.add(sequence)
                    pending['scored'] = dict(scores=[cache[s] for s in pending['sequences']],
                        new_scores=new_scores, cache_hits=hits, fresh_evaluations=len(requested))
                    pending['oracle_seconds'] = time.monotonic() - started
                    with db:
                        storage.put(db, 'pending', pending)
                    if args.stop_after_stage == 'score':
                        return history_value(saved, recipe, args.task)
                if 'canonical_z' not in pending:
                    known = {}
                    for row, feature in zip(saved['rows'], saved['features']):
                        known.setdefault(row['sequence'], feature)
                    new_sequences = list(dict.fromkeys(s for s in pending['sequences'] if s not in known))
                    if new_sequences:
                        known.update(zip(new_sequences, encode_mean(client, new_sequences)))
                    pending['canonical_z'] = np.stack([known[s] for s in pending['sequences']])
                    pending['rng'] = bo.capture_rng()
                    with db:
                        storage.put(db, 'pending', pending)
                else:
                    bo.restore_rng(pending['rng'])
                scored = pending['scored']
                objective.xs_to_scores_dict.update(scored['new_scores'])
                objective.num_calls += scored['fresh_evaluations']
                progress = saved['progress']
                progress['rounds'] += 1
                progress['events'] += 1
                progress['proposals'] += len(pending['sequences'])
                progress['fresh_evaluations'] += scored['fresh_evaluations']
                progress['acquisition_fresh'] += scored['fresh_evaluations']
                progress['cache_hits'] += sum(scored['cache_hits'])
                progress['seconds'] += pending['proposal_seconds'] + pending['oracle_seconds']
                constraints = objective.compute_constraints(pending['sequences'])
                update_state(saved['trust_region'], torch.tensor(scored['scores']).reshape(-1, 1), constraints)
                if saved['trust_region'].restart_triggered:
                    best = max(r['score'] for r in saved['rows'] if r['feasible'])
                    saved['trust_region'] = TurboState(dim=256, batch_size=50,
                        best_value=best, best_constraint_values=torch.tensor([0.]))
                for i, (sequence, score) in enumerate(zip(pending['sequences'], scored['scores'])):
                    saved['rows'].append(dict(sequence=sequence, score=float(score), mic=-float(score),
                        phase='acquisition', round=progress['rounds'], feasible=bool(constraints[i, 0] <= 0),
                        cached=bool(scored['cache_hits'][i]), elapsed=progress['seconds'],
                        proposed_z=pending['proposed_z'][i].tolist(),
                        fit=pending['fit'] if i == 0 else None,
                        posterior=pending['posterior'] if i == 0 else None))
                saved['features'] = np.concatenate([saved['features'], pending['canonical_z']])
                protocol.check_accounting(recipe, progress, saved['rows'], SimpleNamespace(objective=objective))
                saved['rng'] = bo.capture_rng()
                with db:
                    storage.put(db, 'state', saved)
                    storage.put(db, 'pending', None)
                persist(db, directory, saved, recipe, args.task)
                print(json.dumps(dict(method=recipe['method'], task=args.task, **progress)), flush=True)
                if stopped or (args.stop_after_events and progress['events'] >= args.stop_after_events):
                    break
            complete = (saved['progress']['proposals'] >= recipe['proposals']
                        or saved['progress']['fresh_evaluations'] >= recipe['fresh_budget'])
            history = history_value(saved, recipe, args.task, complete)
            bo.atomic_json(directory / 'history.json', history)
        finally:
            if model_client is not None:
                model_client.close()
            if client is not None:
                client.close()
            db.close()
        if complete:
            storage.tpe_archive(directory, identity)
        return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--transfer', type=Path, default=TRANSFER)
    prepare_parser.add_argument('--methods', default='fsbo,rgpe')
    prepare_parser.add_argument('--component', action='store_true')
    prepare_parser.add_argument('--proposals', type=int, default=51)
    prepare_parser.add_argument('--candidates', type=int, default=64)
    prepare_parser.add_argument('--target-epochs', type=int, default=25)
    prepare_parser.add_argument('--rank-samples', type=int, default=32)
    prepare_parser.add_argument('--source-limit', type=int, default=2)
    run_parser = sub.add_parser('run-task')
    run_parser.add_argument('--task', required=True)
    run_parser.add_argument('--cpu', type=int, default=18)
    run_parser.add_argument('--server', default='http://127.0.0.1:18871')
    run_parser.add_argument('--stop-after-events', type=int, default=0)
    run_parser.add_argument('--stop-after-stage', choices=('proposal', 'score'))
    for item in sub.choices.values():
        item.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        if (args.proposals < 1 or args.candidates < 50 or args.target_epochs < 1
                or args.rank_samples < 1 or args.source_limit < 1):
            parser.error('Invalid component-check settings')
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
