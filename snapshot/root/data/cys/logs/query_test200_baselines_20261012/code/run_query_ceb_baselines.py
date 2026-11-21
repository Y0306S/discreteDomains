#!/usr/bin/env python3
"""CEB runtime baselines using the released BayesQO frozen VAE.

The budget is 49 imported Bao observations plus 1,000 proposals. Cached plans
consume proposal slots, but never a second database call. No services are started.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

if __package__:
    from .query_transfer_common import VAE_FILE, validate_assets
    from .run_artifact_store import JsonJournal
else:
    from query_transfer_common import VAE_FILE, validate_assets
    from run_artifact_store import JsonJournal


QP_ROOT = Path('/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization')
SPLIT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = SPLIT_ROOT / 'train800.txt'
TEST_FILE = SPLIT_ROOT / 'test200.txt'
INITIAL_OBSERVATIONS = 49
TOTAL_BUDGET = 1049
BATCH_SIZE = 10


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(payload, stream, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_split(train_file=TRAIN_FILE, test_file=TEST_FILE):
    train = Path(train_file).read_text().split()
    test = Path(test_file).read_text().split()
    if len(train) != 800 or len(set(train)) != 800:
        raise ValueError('Expected 800 distinct training query IDs')
    if len(test) != 200 or len(set(test)) != 200 or set(train) & set(test):
        raise ValueError('Expected 200 distinct held-out IDs, disjoint from training')
    if any(not re.fullmatch(r'CEB_\d+[A-Z]\d+', task) for task in train + test):
        raise ValueError('Invalid CEB task ID')
    return train, test


def sql_path(root, task):
    match = re.fullmatch(r'CEB_(\d+[A-Z])(\d+)', task)
    if not match:
        raise ValueError(f'Invalid CEB task ID: {task}')
    name = (match[1] + match[2]).lower()
    path = Path(root) / 'workload/ceb-3k' / match[1].lower() / (name + '.sql')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_initializations(path, tasks):
    selected = set(tasks)
    result = {task: [] for task in tasks}
    with Path(path).open(newline='') as stream:
        for raw in csv.DictReader(stream):
            task = raw['query_name']
            if task not in selected:
                continue
            runtime = abs(float(raw['runtime_secs']))
            plan = [int(value) for value in raw['encoded_plan'].split(',')]
            flag = raw['timed_out'].lower()
            if not plan or len(plan) % 5 or not math.isfinite(runtime) or runtime <= 0:
                raise ValueError(f'Invalid Bao observation for {task}')
            if flag not in {'0', '1', 'false', 'true'}:
                raise ValueError(f'Invalid Bao censoring flag for {task}')
            result[task].append({
                'plan': plan, 'runtime': runtime,
                'censored': int(flag in {'1', 'true'}),
                'status': 'bao_import', 'plan_id': raw['plan_id'],
            })
    for task, rows in result.items():
        if len(rows) != INITIAL_OBSERVATIONS:
            raise ValueError(f'{task}: expected 49 Bao rows, found {len(rows)}')
        if len({row['plan_id'] for row in rows}) != INITIAL_OBSERVATIONS:
            raise ValueError(f'{task}: duplicate Bao plan IDs')
        if all(row['censored'] for row in rows):
            raise ValueError(f'{task}: no uncensored Bao incumbent')
    return result


def native_path(root):
    root = Path(root).resolve()
    for name in ('lolbo', 'vae', 'oracle', 'workload', 'codec'):
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, '__file__', None):
            if not Path(loaded.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f'{name} already imported from a different repository')
    sys.path.insert(0, str(root))


def validate_initial_encodings(model, initializations):
    from vae.model import encode_batch
    checked = 0
    for task, rows in initializations.items():
        try:
            tokens = encode_batch([row['plan'] for row in rows], model.vocab)
        except (KeyError, ValueError) as exc:
            raise ValueError(f'{task}: Bao initialization cannot be encoded by the selected VAE') from exc
        if tokens.shape[0] != len(rows) or tokens.shape[1] > model.model.enc_pos_enc.pe.shape[1]:
            raise ValueError(f'{task}: Bao initialization exceeds the VAE sequence capacity')
        checked += len(rows)
    return checked


class FrozenBackend:
    def __init__(self, root, checkpoint, task, initial, device):
        import torch
        native_path(root)
        from vae.model import VAEModule
        from your_tasks.your_objective_functions import DatabaseObjective
        from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec

        self.torch = torch
        self.device = device
        self.vae = VAEModule.load_from_checkpoint(str(checkpoint), map_location=device)
        self.vae.eval().requires_grad_(False)
        self.worst_runtime = max(row['runtime'] for row in initial)
        self.fallback = max(initial, key=lambda row: row['runtime'])['plan']
        self.db = DatabaseObjective(task, worst_runtime_observed=self.worst_runtime)
        # Reparse this exact SQL so a stale workload pickle cannot select a query.
        definition = get_workload_set('CEB_3K').extract_query(sql_path(root, task).read_text(), task)
        self.spec = WorkloadSpec.from_definition(definition, OracleCodec.Aliases)
        self.db.full_workload_spec = self.spec
        from oracle.oracle import plan_has_crossjoin
        self.has_crossjoin = lambda plan: plan_has_crossjoin(self.spec, plan)
        if self.has_crossjoin(self.fallback):
            raise ValueError('Historical fallback plan contains a cross join')

    def encode(self, plans):
        with self.torch.no_grad():
            return self.vae(plans)['z'].reshape(len(plans), 64).cpu().tolist()

    def decode(self, vectors):
        torch = self.torch
        z = torch.tensor(vectors, dtype=torch.float32, device=self.device)
        output = []
        with torch.no_grad():
            samples = self.vae.sample(z=z)
            for index, sample in enumerate(samples):
                retries = 0
                fallback = False
                while self.has_crossjoin(sample):
                    sample = self.vae.sample(z=z[index])[0]
                    retries += 1
                    if retries > 50:
                        sample = self.fallback
                        fallback = True
                        break
                output.append({'plan': list(sample), 'decode_retries': retries,
                               'fallback': fallback})
        return output


class RuntimeOracle:
    """Native encoded SQL with explicit timeout/error provenance and hint loading."""

    def __init__(self, spec, worst_runtime):
        self.spec = spec
        self.worst_runtime = worst_runtime

    @staticmethod
    def connect():
        import psycopg
        from oracle.oracle import _db_connect_kwargs
        conn = psycopg.connect(**_db_connect_kwargs(), autocommit=True, connect_timeout=5)
        try:
            conn.execute("LOAD 'pg_hint_plan'")
            conn.execute('SET pg_hint_plan.enable_hint = on')
            conn.execute('SET join_collapse_limit = 1')
            conn.execute("SET client_encoding = 'UTF8'")
        except BaseException:
            conn.close()
            raise
        return conn

    def __call__(self, plan, timeout):
        import psycopg
        from oracle.oracle import _decode_query
        query = _decode_query(self.spec, plan)
        # Connection or extension failures abort the run, without fabricated scores.
        with self.connect() as conn:
            conn.execute("SELECT set_config('statement_timeout', %s, false)",
                         (str(max(1, math.ceil(timeout * 1000))),))
            start = time.perf_counter()
            try:
                conn.execute(query)
            except psycopg.errors.QueryCanceled:
                return {'runtime': timeout, 'censored': 1, 'status': 'timeout',
                        'elapsed_seconds': time.perf_counter() - start}
            except psycopg.Error as exc:
                # A lost database or invalid SQL is an infrastructure/setup failure.
                if exc.sqlstate not in {'53200', '54000', '54001'}:
                    raise
                return {'runtime': min(self.worst_runtime, timeout * 2),
                        'censored': 1, 'status': 'query_resource_error',
                        'sqlstate': exc.sqlstate, 'error': str(exc),
                        'elapsed_seconds': time.perf_counter() - start}
            elapsed = time.perf_counter() - start
            return {'runtime': elapsed, 'censored': 0, 'status': 'completed',
                    'elapsed_seconds': elapsed}


class Engine:
    def __init__(self, method, initial, seed, batch_size, timeout, latent_bound):
        import numpy as np
        import torch
        self.method = method
        self.batch_size = batch_size
        self.timeout = timeout
        self.state = None
        self.study = None
        z = np.asarray([row['z'] for row in initial])
        self.bounds = list(zip(np.minimum(-latent_bound, z.min(axis=0)).tolist(),
                               np.maximum(latent_bound, z.max(axis=0)).tolist()))
        if method == 'lolbo':
            from lolbo.lolbo import LOLBOState
            self.state = LOLBOState(
                objective=SimpleNamespace(objective_function=SimpleNamespace(timeout=timeout)),
                train_x=[row['plan'] for row in initial],
                train_y=torch.tensor([-row['runtime'] for row in initial]).float().view(-1, 1),
                train_z=torch.tensor(z).float(),
                censoring=torch.tensor([row['censored'] for row in initial]).float().view(-1, 1),
                censored_obs_is_max=True, k=100, num_update_epochs=2,
                init_n_epochs=80, learning_rte=0.01, bsz=batch_size,
                acq_func='ts', timeout_strategy='ours', timeout_percentile=0.1,
                gp_stdev_multiplier=2, thompson_rejection=False, verbose=False,
            )
        else:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = (optuna.samplers.RandomSampler(seed=seed) if method == 'random'
                       else optuna.samplers.TPESampler(seed=seed))
            self.study = optuna.create_study(direction='minimize', sampler=sampler)
            distributions = {f'z_{i}': optuna.distributions.FloatDistribution(lo, hi)
                             for i, (lo, hi) in enumerate(self.bounds)}
            for row in initial:
                self.study.add_trial(optuna.trial.create_trial(
                    params={f'z_{i}': value for i, value in enumerate(row['z'])},
                    distributions=distributions, value=row['runtime'],
                    user_attrs={'source': 'bao_import', 'censored': row['censored']},
                ))

    def ask(self, remaining):
        count = min(self.batch_size, remaining)
        if self.state is not None:
            state = self.state
            state.bsz = count
            state.update_surrogate_model()
            state.generate_candidates()
            state.set_db_objective_timeout()
            return [{'z': z, 'timeout': float(timeout), 'trial': None}
                    for z, timeout in zip(state.z_next.detach().cpu().tolist(), state.timeouts_next)]
        proposals = []
        for _ in range(count):
            trial = self.study.ask()
            z = [trial.suggest_float(f'z_{i}', lo, hi) for i, (lo, hi) in enumerate(self.bounds)]
            proposals.append({'z': z, 'timeout': self.timeout, 'trial': trial.number})
        return proposals

    def tell(self, proposals, rows):
        if self.state is not None:
            import torch
            self.state.update_next(
                z_next_=torch.tensor([p['z'] for p in proposals]).float(),
                y_next_=torch.tensor([-row['runtime'] for row in rows]).float(),
                x_next_=[p['plan'] for p in proposals],
                censoring_next_=torch.tensor([row['censored'] for row in rows]).float(),
                acquisition=True,
            )
            if self.state.tr_state.restart_triggered:
                self.state.initialize_tr_state()
            self.state.new_best_found = False
        else:
            for proposal, row in zip(proposals, rows):
                self.study.tell(proposal['trial'], row['runtime'])


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def checkpoint(path, payload):
    import torch
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save({**payload, 'rng': rng_state()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def evaluate_pending(directory, proposals, initial, completed, oracle):
    """Recover a partial batch using one durable result per charged proposal."""
    rows = []
    journal = JsonJournal(directory)
    cache = {tuple(row['plan']): row for row in initial + completed}
    for offset, proposal in enumerate(proposals):
        number = len(completed) + offset
        result_path = f'proposal_{number:04d}.json'
        pending_path = f'proposal_{number:04d}.pending.json'
        if journal.exists(result_path):
            row = journal.read(result_path)
            if row['proposal'] != proposal:
                raise RuntimeError('Saved proposal does not match checkpoint')
        else:
            attempts_path = f'proposal_{number:04d}.database_attempts.json'
            attempts = journal.read(attempts_path) if journal.exists(attempts_path) else []
            if journal.exists(pending_path):
                pending = journal.read(pending_path)
                recovery = attempts[-1].get('storage_recovery', {}) if attempts else {}
                reviewed_outage = (recovery.get('replay_authorized') is True
                                   and recovery.get('active_database_calls') == 0
                                   and recovery.get('original_result_unavailable') is True
                                   and bool(recovery.get('evidence_report')))
                if (pending != proposal or not attempts
                        or (attempts[-1].get('status') != 'io_error' and not reviewed_outage)
                        or any(a.get('proposal') != proposal for a in attempts)):
                    raise RuntimeError(f'Unresolved database call: {pending_path}; refusing to repeat it')
            key = tuple(proposal['plan'])
            if key in cache:
                cached = cache[key]
                result = {'runtime': cached['runtime'], 'censored': cached['censored'],
                          'status': 'cache_hit', 'cached_status': cached['status'],
                          'database_call': False}
            else:
                attempt = {'proposal': proposal, 'status': 'in_flight', 'started_at': time.time()}
                attempts.append(attempt)
                journal.update({pending_path: proposal, attempts_path: attempts})
                started = time.perf_counter()
                try:
                    result = {**oracle(proposal['plan'], proposal['timeout']), 'database_call': True}
                except Exception as exc:
                    # Other interruptions require a separately verified recovery record.
                    attempt.update(status='io_error' if getattr(exc, 'sqlstate', None) == '58030' else 'error',
                                   sqlstate=getattr(exc, 'sqlstate', None), error=str(exc),
                                   elapsed_seconds=time.perf_counter() - started, finished_at=time.time())
                    journal.write(attempts_path, attempts)
                    raise
                attempt.update(status='returned', result_status=result['status'],
                               elapsed_seconds=time.perf_counter() - started, finished_at=time.time())
                result.update(database_attempts=len(attempts), database_attempt_history=attempts,
                              successful_attempt_elapsed_seconds=result.get('elapsed_seconds'),
                              elapsed_seconds=sum(a.get('elapsed_seconds') or 0 for a in attempts),
                              database_elapsed_unknown_attempts=sum(a.get('elapsed_seconds') is None for a in attempts))
            if not math.isfinite(result['runtime']) or result['runtime'] <= 0:
                raise ValueError('Oracle returned an invalid runtime')
            row = {**result, 'plan': proposal['plan'], 'z': proposal['z'],
                   'proposal': proposal, 'proposal_index': number}
            values = {result_path: row}
            if attempts:
                values[attempts_path] = attempts
            journal.update(values, delete=(pending_path,))
        rows.append(row)
        cache[tuple(row['plan'])] = row
    return rows


def write_history(directory, meta, initial, completed, total, status, error=None):
    journal = JsonJournal(directory)
    rows = initial + completed
    observations = []
    for index, row in enumerate(rows):
        observations.append({
            'config': {f'z_{i}': value for i, value in enumerate(row['z'])},
            'objectives': [row['runtime']], 'constraints': None,
            'trial_state': int(row['censored']), 'elapsed_time': row.get('elapsed_seconds', 0),
            'extra_info': {**row, 'source': 'bao_import' if index < len(initial) else meta['method'],
                           'oracle': 'postgres_runtime', 'charged_index': index},
        })
    calls = sum(row.get('database_attempts', int(row.get('database_call', False))) for row in completed)
    uncensored = [row['runtime'] for row in rows if not row['censored']]
    summary = {**meta, 'status': status, 'budget_total': total,
               'initial_observations': len(initial), 'proposals': len(completed),
               'charged_observations': len(rows), 'database_calls': calls,
               'cache_hits': sum(row['status'] == 'cache_hit' for row in completed),
               'best_uncensored_runtime': min(uncensored) if uncensored else None,
               'unresolved_database_calls': sum(
                   not journal.exists(name.replace('.pending.json', '.json'))
                   for name in journal.names('proposal_*.pending.json')),
               'evidence_storage': 'sqlite_per_task_v1',
               'error': error}
    migrations_path = directory / 'runner_migrations.json'
    if migrations_path.exists():
        summary['runner_migrations'] = json.loads(migrations_path.read_text())
    atomic_json(directory / 'history.json', {
        'task_id': meta['task_id'], 'num_objectives': 1, 'num_constraints': 0,
        'meta_info': summary, 'observations': observations,
    })
    atomic_json(directory / 'status.json', summary)
    return summary


def run_loop(directory, meta, initial, engine, backend, oracle, total, stop_after_batches=None,
             resume_runner_sha256=None):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'checkpoint.pt'
    journal = JsonJournal(directory)
    if state_path.exists():
        saved = torch.load(state_path, weights_only=False)
        previous = saved['meta']
        runner_migration = (previous != meta and bool(resume_runner_sha256)
                            and previous.get('runner_sha256') == resume_runner_sha256
                            and bool(meta.get('runner_sha256'))
                            and {**previous, 'runner_sha256': meta['runner_sha256']} == meta)
        if (previous != meta and not runner_migration) or saved['total'] != total:
            raise ValueError('Resume settings/assets differ from the saved run')
        if runner_migration:
            migrations_path = directory / 'runner_migrations.json'
            migrations = json.loads(migrations_path.read_text()) if migrations_path.exists() else []
            if not any(m['from_sha256'] == resume_runner_sha256 and m['to_sha256'] == meta['runner_sha256']
                       for m in migrations):
                migrations.append({'from_sha256': resume_runner_sha256,
                                   'to_sha256': meta['runner_sha256'], 'recorded_at': time.time(),
                                   'scope': 'runner code only; all run settings and assets match'})
                atomic_json(migrations_path, migrations)
        engine, initial = saved['engine'], saved['initial']
        completed, pending = saved['completed'], saved['pending']
        restore_rng(saved['rng'])
    else:
        if journal.names('proposal_*') or (directory / 'history.json').exists():
            raise RuntimeError('Existing results without a checkpoint; refusing to overwrite')
        completed, pending = [], None
    if total < len(initial):
        raise ValueError('Budget cannot be smaller than the Bao initialization')
    journal.migrate()

    def save():
        checkpoint(state_path, {'meta': meta, 'total': total, 'engine': engine,
                               'initial': initial, 'completed': completed, 'pending': pending})

    save()
    batches = 0
    try:
        while len(initial) + len(completed) < total:
            remaining = total - len(initial) - len(completed)
            if pending is None:
                pending = engine.ask(remaining)
                if not pending or len(pending) > remaining:
                    raise RuntimeError('Optimizer returned an invalid batch size')
                decoded = backend.decode([proposal['z'] for proposal in pending])
                if len(decoded) != len(pending):
                    raise RuntimeError('Decoder batch size mismatch')
                pending = [{**proposal, **plan} for proposal, plan in zip(pending, decoded)]
                save()
            new_rows = evaluate_pending(directory, pending, initial, completed, oracle)
            engine.tell(pending, new_rows)
            completed.extend(new_rows)
            pending = None
            save()
            write_history(directory, meta, initial, completed, total, 'running')
            batches += 1
            if (stop_after_batches is not None and batches >= stop_after_batches
                    and len(initial) + len(completed) < total):
                return write_history(directory, meta, initial, completed, total, 'paused')
        return write_history(directory, meta, initial, completed, total, 'complete')
    except BaseException as exc:
        # Include durable results from an unfinished batch in the readable report.
        visible = list(completed)
        while journal.exists(f'proposal_{len(visible):04d}.json'):
            visible.append(journal.read(f'proposal_{len(visible):04d}.json'))
        write_history(directory, meta, initial, visible, total, 'failed', str(exc))
        raise


def asset_manifest(root, checkpoint_path, task, args):
    sources = {}
    for folder in ('lolbo', 'vae', 'oracle', 'codec', 'workload'):
        for path in sorted((root / folder).rglob('*.py')):
            sources[str(path.relative_to(root))] = fingerprint(path)
    sources['your_tasks/your_objective_functions.py'] = fingerprint(
        root / 'your_tasks/your_objective_functions.py')
    return {
        'task_id': task, 'method': args.method, 'split': args.split,
        'variant': 'frozen_vae_latent_turbo' if args.method == 'lolbo' else args.method,
        'seed': args.seed, 'batch_size': args.batch_size, 'update_e2e': False,
        'device': args.device,
        'timeout_seconds_random_tpe': args.timeout, 'latent_bound_random_tpe': args.latent_bound,
        'budget_unit': 'Bao observations plus proposals, including cache hits',
        'vae_checkpoint': str(checkpoint_path), 'vae_sha256': fingerprint(checkpoint_path),
        'bao_sha256': fingerprint(root / 'initialization_data/ceb_3k_bao_initialization.csv'),
        'sql_sha256': fingerprint(sql_path(root, task)),
        'train_split_sha256': fingerprint(TRAIN_FILE), 'test_split_sha256': fingerprint(TEST_FILE),
        'runner_sha256': fingerprint(__file__), 'native_sources_sha256': sources,
        'packages': {name: importlib.metadata.version(name)
                     for name in ('torch', 'gpytorch', 'botorch', 'optuna', 'psycopg')},
        'database': {key: os.environ.get(key) for key in ('DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER')},
        'session_settings': {'pg_hint_plan.enable_hint': 'on', 'join_collapse_limit': 1},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('random', 'tpe', 'lolbo'), required=True)
    parser.add_argument('--task-id', help='One ID from the selected authoritative split')
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--qp-root', type=Path, default=QP_ROOT)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=int, default=TOTAL_BUDGET)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--timeout', type=float, default=100.0)
    parser.add_argument('--latent-bound', type=float, default=4.0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stop-after-batches', type=int,
                        help='Save and exit at a batch boundary for a resumability check')
    parser.add_argument('--preflight', action='store_true', help='Check assets; no database queries or GPU use')
    parser.add_argument('--resume-runner-sha256',
                        help='Explicitly allow migration from this saved runner hash; all other provenance must match')
    args = parser.parse_args()
    if args.batch_size < 1 or args.budget < INITIAL_OBSERVATIONS:
        parser.error('Require batch-size >= 1 and budget >= 49')
    if args.stop_after_batches is not None and args.stop_after_batches < 1:
        parser.error('stop-after-batches must be positive')
    if (not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.latent_bound) or args.latent_bound <= 0):
        parser.error('Timeout and latent bound must be positive')
    train, test = load_split()
    selected = train if args.split == 'train' else test
    if args.task_id and args.task_id not in selected:
        parser.error('task-id must be in the selected authoritative split')
    root = args.qp_root.resolve()
    ckpt = validate_assets(root, args.checkpoint)
    initializations = load_initializations(root / 'initialization_data/ceb_3k_bao_initialization.csv',
                                          train + test if args.preflight else selected)
    for task in train + test:
        sql_path(root, task)
    if args.preflight:
        native_path(root)
        from lolbo.lolbo import LOLBOState
        from vae.model import VAEModule
        model = VAEModule.load_from_checkpoint(str(ckpt), map_location='cpu')
        encoded_rows = validate_initial_encodings(model, initializations)
        print(json.dumps({'train': len(train), 'test': len(test), 'exact_sql_files': len(train + test),
                          'split': args.split, 'bao_rows_checked': encoded_rows,
                          'vae_checkpoint': str(ckpt), 'vae_sha256': fingerprint(ckpt),
                          'budget': args.budget, 'proposals': args.budget - INITIAL_OBSERVATIONS,
                          'batch_size': args.batch_size, 'update_e2e': False,
                          'native_gp_class': LOLBOState.__name__,
                          'vae_parameters': sum(p.numel() for p in model.parameters()),
                          'database_calls': 0, 'status': 'encoding_preflight_passed',
                          'pending': ['live database and optimizer integration check']}))
        return
    if args.task_id is None or args.output is None:
        parser.error('Execution requires --task-id and --output; use --preflight to inspect assets')
    if args.method == 'lolbo' and not args.device.startswith('cuda'):
        parser.error('Native frozen-VAE GP optimizer requires --device cuda:N')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    native_path(root)
    for key, default in {'DB_HOST': '127.0.0.1', 'DB_PORT': '5432',
                         'DB_USER': 'imdb', 'DB_NAME': 'imdb'}.items():
        os.environ.setdefault(key, default)
    meta = asset_manifest(root, ckpt, args.task_id, args)
    output = args.output / args.task_id / args.method / f'seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    # A process holds the task lock for its lifetime. Database calls are serialized
    # across these baselines using the endpoint lock below.
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        endpoint = ':'.join(os.environ[key] for key in ('DB_HOST', 'DB_PORT', 'DB_NAME'))
        lock_id = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
        with Path(f'/tmp/ceb_baseline_db_{lock_id}.lock').open('w') as db_lock:
            fcntl.flock(db_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with RuntimeOracle.connect():
                pass
            initial = initializations[args.task_id]
            backend = FrozenBackend(root, ckpt, args.task_id, initial, args.device)
            engine = None
            if not (output / 'checkpoint.pt').exists():
                vectors = backend.encode([row['plan'] for row in initial])
                initial = [{**row, 'z': z} for row, z in zip(initial, vectors)]
                engine = Engine(args.method, initial, args.seed, args.batch_size, args.timeout, args.latent_bound)
            oracle = RuntimeOracle(backend.spec, backend.worst_runtime)
            print(json.dumps(run_loop(output, meta, initial, engine, backend, oracle,
                                      args.budget, args.stop_after_batches,
                                      resume_runner_sha256=args.resume_runner_sha256)))


if __name__ == '__main__':
    main()
