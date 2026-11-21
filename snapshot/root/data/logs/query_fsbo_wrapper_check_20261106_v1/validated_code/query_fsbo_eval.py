#!/usr/bin/env python3
"""Frozen query FSBO + peptide batch Thompson wrapper; no VAE updates/recenter."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np

TOOLS = Path(__file__).resolve().parent
SOURCE = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
TRANSFER = Path('/root/data/logs/query_transfer_train800_20261106_v1')
POOL = Path('/root/data/logs/query_fsbo_pilot_20261106_v1/active_pool.json')
PEPTIDE_CPU = Path('/root/data/logs/peptide_bo_train800_20261027_v5b/upstream_cpu')
FORMAT = 'query-fsbo-frozen-ts-v1'
LIVE_CLIENTS = []
sys.path.insert(0, str(SOURCE / 'code'))
import run_query_ceb_baselines as query
from query_vae_client import QueryVAEBackend
from peptide_transfer_models import send, receive


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    query.atomic_json(path, value)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def peptide_turbo(recipe):
    # A private namespace prevents the peptide and query lolbo packages colliding.
    name = '_query_fsbo_peptide_bo'
    path = str(Path(recipe['peptide_cpu']) / 'lolbo/utils/bo_utils')
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [path]
        sys.modules[name] = package
    elif sys.modules[name].__path__ != [path]:
        raise ValueError('Peptide wrapper implementation changed in this process')
    return importlib.import_module(name + '.turbo')


def normalized(x, recipe):
    low, high = np.asarray(recipe['lower']), np.asarray(recipe['upper'])
    return ((np.asarray(x)-low)/np.where(high > low, high-low, 1.)).astype(np.float32)


def distinct_completed(rows):
    distinct = {}
    for row in rows:
        if row['censored']:
            continue
        if not np.isfinite(row['runtime']) or row['runtime'] <= 0:
            raise ValueError('Invalid measured query runtime')
        distinct.setdefault(tuple(row['plan']), row)
    if not distinct:
        raise ValueError('FSBO requires at least one uncensored query observation')
    return list(distinct.values())


class ModelClient:
    def __init__(self, recipe_path, recipe):
        self.last_sample = None
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        self.child = subprocess.Popen([recipe['model_python'], '-u', str(TOOLS/'query_fsbo_model.py'),
            '--recipe', str(recipe_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        LIVE_CLIENTS.append(self)

    def request(self, op, **values):
        send(self.child.stdin, dict(values, op=op))
        reply = receive(self.child.stdout)
        if reply is None or reply.get('error'):
            raise RuntimeError('Query FSBO model: ' + (reply.get('error', 'EOF') if reply else 'EOF'))
        return reply['result']

    def close(self):
        try:
            if self.child.poll() is None:
                send(self.child.stdin, dict(op='close'))
        finally:
            self.child.stdin.close()
            self.child.stdout.close()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.terminate()
                self.child.wait(timeout=10)
            LIVE_CLIENTS.remove(self)


class BridgePosterior:
    def __init__(self, client, x, recipe):
        self.client, self.x, self.recipe = client, x, recipe

    def rsample(self, sample_shape):
        import torch
        count = int(sample_shape[0])
        result = self.client.request('sample', candidates=normalized(self.x.numpy(), self.recipe),
            count=count, seed=int(torch.randint(0, 2**31-1, (1,)).item()))
        draws = np.asarray(result['draws'])
        if draws.shape != (count, len(self.x)) or not np.isfinite(draws).all():
            raise ValueError('Malformed query FSBO posterior')
        self.client.last_sample = result['info']
        return torch.as_tensor(draws, dtype=self.x.dtype).unsqueeze(-1)


class BridgeModel:
    def __init__(self, client, recipe):
        self.client, self.recipe = client, recipe

    def posterior(self, x, observation_noise=False):
        return BridgePosterior(self.client, x, self.recipe)


class FSBOEngine:
    """Checkpointable target state; model IPC is never serialized into a run."""
    def __init__(self, recipe_path, recipe, initial, vae):
        self.recipe_path, self.recipe = str(recipe_path), recipe
        self.rows, self.vae = list(initial), vae
        self.task, self.fallback = vae.task, vae.fallback
        self.batch_size, self.rounds = recipe['batch_size'], 0
        self.client = None
        self.turbo = peptide_turbo(recipe).TurboState(dim=64, batch_size=self.batch_size,
            best_value=max(-r['runtime'] for r in distinct_completed(initial)))

    def __getstate__(self):
        return dict(self.__dict__, client=None, vae=None)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.vae = QueryVAEBackend(self.recipe['vae_pool'], self.task, self.fallback,
                                   seed=self.recipe['seed'])

    def ask(self, remaining):
        import torch
        memory_guard(self.recipe)
        observed = distinct_completed(self.rows)
        if self.client is None:
            self.client = ModelClient(Path(self.recipe_path), self.recipe)
        fit = self.client.request('fit',
            x=normalized([r['z'] for r in observed], self.recipe),
            score=np.asarray([-r['runtime'] for r in observed]), seed=self.recipe['seed']+self.rounds)
        x = torch.tensor([r['z'] for r in observed], dtype=torch.float32)
        y = torch.tensor([-r['runtime'] for r in observed], dtype=torch.float32).view(-1, 1)
        count = min(self.batch_size, remaining)
        with torch.no_grad():
            z = peptide_turbo(self.recipe).generate_batch(self.turbo, BridgeModel(self.client, self.recipe),
                x, y, count, n_candidates=self.recipe['n_candidates'], device=torch.device('cpu'))
        if z.shape != (count, 64) or not torch.isfinite(z).all():
            raise ValueError('Malformed query Thompson acquisition batch')
        return [dict(z=vector, timeout=self.recipe['timeout_seconds'], trial=None,
                     fit=fit if i == 0 else None,
                     posterior=self.client.last_sample if i == 0 else None)
                for i, vector in enumerate(z.tolist())]

    def tell(self, proposals, rows):
        import torch
        # Store measured plans at their canonical frozen encoder means, never
        # the raw latent that happened to decode to them. No VAE training occurs.
        known = {tuple(r['plan']): r['z'] for r in self.rows}
        fresh = list(dict.fromkeys(tuple(r['plan']) for r in rows if tuple(r['plan']) not in known))
        if fresh:
            encoded = np.asarray(self.vae.encode([list(p) for p in fresh]), dtype=np.float32)
            if encoded.shape != (len(fresh), 64) or not np.isfinite(encoded).all():
                raise ValueError('Invalid canonical query encoding')
            known.update(zip(fresh, encoded.tolist()))
        for row in rows:
            row['canonical_z'] = known[tuple(row['plan'])]
            self.rows.append(dict(row, z=row['canonical_z']))
        scores = [-r['runtime'] for r in rows if not r['censored']]
        # Timeouts remain recorded bounds, never exact targets or incumbents.
        if not scores:
            scores = [self.turbo.best_value]
        turbo = peptide_turbo(self.recipe)
        turbo.update_state(self.turbo, torch.tensor(scores).view(-1, 1), None)
        if self.turbo.restart_triggered:
            self.turbo = turbo.TurboState(dim=64, batch_size=self.batch_size,
                best_value=max(-r['runtime'] for r in distinct_completed(self.rows)))
        self.rounds += 1


def canonical_context(native, task):
    sys.path.insert(0, str(native))
    from codec.codec import AliasesCodec
    from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
    from oracle.oracle import plan_has_crossjoin
    sql = query.sql_path(native, task)
    definition = get_workload_set('CEB_3K').extract_query(sql.read_text(), task)
    spec = WorkloadSpec.from_definition(definition, OracleCodec.Aliases)
    codec = AliasesCodec(spec.all_tables)

    def canonical(plan):
        tree = codec.decode(spec.query_tables, list(plan))
        result = codec.encode(tree)
        if not tree.equal(codec.decode(spec.query_tables, list(result))):
            raise ValueError('Query canonicalization changed the executed plan')
        return list(result)

    return spec, canonical, lambda plan: plan_has_crossjoin(spec, list(plan))


class CanonicalBackend:
    def __init__(self, vae, canonical):
        self.vae, self.canonical = vae, canonical

    def decode(self, vectors):
        import torch
        seed = int(torch.randint(0, 2**31-1, (1,)).item())
        return [dict(row, plan=self.canonical(row['plan']))
                for row in self.vae.decode(vectors, seed=seed)]


def prepare(args):
    source_plan = json.loads((SOURCE/'plan.json').read_text())
    manifest = json.loads((TRANSFER/'corpus/manifest.json').read_text())
    status = json.loads((TRANSFER/'fsbo/status.json').read_text())
    if status['status'] != 'complete' or not status['reload_verified']:
        raise ValueError('Query FSBO source training has not completed')
    train, test = query.load_split(SOURCE/'train800.txt', SOURCE/'test200.txt')
    tasks = [args.component_task] if args.component_task else test
    if args.component_task and args.component_task not in train:
        raise ValueError('Bounded component checks must use a training query')
    if not (1 <= args.batch_size <= 128 and args.n_candidates >= args.batch_size and args.proposals > 0):
        raise ValueError('Invalid query acquisition budget')
    native = SOURCE/'native'
    initial = query.load_initializations(native/'initialization_data/ceb_3k_bao_initialization.csv', tasks)
    task_data = {}
    for task in tasks:
        spec, canonical, crossjoin = canonical_context(native, task)
        rows = [dict(r, plan=canonical(r['plan'])) for r in initial[task]]
        uncensored = distinct_completed(rows)
        fallback = min((r for r in uncensored if not crossjoin(r['plan'])), key=lambda r:r['runtime'])['plan']
        task_data[task] = dict(initial=rows, fallback=fallback,
                              sql_sha256=digest(query.sql_path(native, task)))
    settings = dict(format=FORMAT, domain='ceb_query_runtime', method='fsbo',
        purpose='component_check' if args.component_task else 'heldout_test200',
        source=str(SOURCE), native=str(native), transfer=str(TRANSFER), seed=123,
        train=train, test=test, allowed_tasks=tasks, initialization=49,
        proposals=args.proposals, batch_size=args.batch_size, n_candidates=args.n_candidates,
        timeout_seconds=100., timeout_policy='fixed_cap_censored_excluded_from_exact_FSBO',
        representation='frozen_author_VAE_posterior_mean_64',
        lower=manifest['lower'], upper=manifest['upper'],
        acquisition='native_peptide_batch_thompson_sampling_without_replacement',
        trust_region='peptide_TuRBO_incumbent_center_and_width_updates',
        update_e2e=False, e2e_updates=0, recenter=False, recenter_calls=0,
        stall_policy='ordinary_trust_region_updates_only',
        target_policy='first_distinct_uncensored_canonical_plan', target_max_updates=10000,
        fsbo_repo='/root/data/cys/baseline/repos/FSBO',
        fsbo_weights=str(TRANSFER/'fsbo/weights'), fsbo_weights_sha256=status['weights_sha256'],
        model_python='/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python',
        query_python='/root/data/cys/envs/bolt_clean_venv/bin/python',
        peptide_cpu=str(PEPTIDE_CPU), vae_pool=json.loads(POOL.read_text())['vae_pool'],
        workers=json.loads(POOL.read_text())['active_workers'])
    paths = [TOOLS/n for n in ['query_fsbo_eval.py','query_fsbo_model.py','query_vae_client.py',
                               'fsbo_target_refit.py','peptide_transfer_models.py','peptide_transfer.py']]
    paths += [SOURCE/'code'/n for n in ['run_query_ceb_baselines.py','query_transfer_common.py','run_artifact_store.py']]
    paths += [PEPTIDE_CPU/'lolbo/utils/bo_utils'/n for n in ['turbo.py','constrained_max_posterior_sampling.py']]
    paths += [Path(settings['fsbo_repo'])/n for n in ['fsbo_modules.py','fsbo_utils.py']]
    paths += [native/n for n in ['codec/codec.py','oracle/oracle.py','workload/workloads.py',
                                'vae/model.py','vae/CEB_64_author.ckpt',
                                'initialization_data/ceb_3k_bao_initialization.csv']]
    paths += [TRANSFER/'corpus/manifest.json', TRANSFER/'fsbo/weights',
              SOURCE/'train800.txt', SOURCE/'test200.txt']
    settings['pinned_files'] = {str(p):digest(p) for p in paths}
    if (settings['fsbo_weights_sha256'] != digest(settings['fsbo_weights'])
            or digest(SOURCE/'train800.txt') != source_plan['train_sha256']
            or digest(SOURCE/'test200.txt') != source_plan['test_sha256']
            or manifest['vae_sha256'] != digest(native/'vae/CEB_64_author.ckpt')
            or status['identity']['corpus_sha256'] != digest(TRANSFER/'corpus/manifest.json')):
        raise ValueError('Query checkpoint/corpus provenance failed')
    args.campaign.mkdir(parents=True, exist_ok=True)
    if (args.campaign/'recipe.json').exists():
        raise FileExistsError('Use a new campaign directory; existing recipes are immutable')
    atomic_json(args.campaign/'tasks.json', task_data)
    settings['tasks_sha256'] = digest(args.campaign/'tasks.json')
    atomic_json(args.campaign/'recipe.json', settings)
    atomic_json(args.campaign/'worker_plan.json', dict(launched=False, workers=[dict(
        name=w['name'], host=w['host'], cpus=w['cpus'], db_port=w['db_port'],
        command=[settings['query_python'], '-u', str(TOOLS/'query_fsbo_eval.py'),
                 'run-worker', '--campaign', str(args.campaign), '--worker', w['name']])
        for w in settings['workers']]))
    print(json.dumps(dict(prepared=str(args.campaign), tasks=len(tasks), launched=False)))


def verify(campaign):
    recipe = json.loads((campaign/'recipe.json').read_text())
    if (recipe['format'] != FORMAT or recipe['domain'] != 'ceb_query_runtime'
            or recipe['method'] != 'fsbo' or recipe['update_e2e'] or recipe['recenter']
            or recipe['e2e_updates'] != 0 or recipe['recenter_calls'] != 0):
        raise ValueError('Expected frozen query FSBO without VAE updates or recentering')
    for path, expected in recipe['pinned_files'].items():
        if digest(path) != expected:
            raise ValueError('Pinned query FSBO asset changed: ' + path)
    if digest(campaign/'tasks.json') != recipe['tasks_sha256']:
        raise ValueError('Query Bao initialization changed')
    if (len(recipe['train']) != 800 or len(recipe['test']) != 200
            or set(recipe['train']) & set(recipe['test'])
            or not set(recipe['allowed_tasks']) <= set(recipe['train'] if recipe['purpose']=='component_check' else recipe['test'])):
        raise ValueError('Invalid query FSBO task split')
    return recipe


def memory_guard(recipe):
    root = Path('/sys/fs/cgroup')
    maximum = (root/'memory.max').read_text().strip()
    if maximum != 'max' and int(maximum)-int((root/'memory.current').read_text()) < 32*2**30:
        raise RuntimeError('Query worker admission requires 32 GiB memory headroom')


def run_task(campaign, recipe, task, worker, stop_after_batches=None):
    import torch
    torch.set_num_threads(1)
    if task not in recipe['allowed_tasks']:
        raise ValueError('Query task is outside this campaign split')
    os.sched_setaffinity(0, {worker['cpus'][0]})
    os.environ.update(DB_HOST='127.0.0.1', DB_PORT=str(worker['db_port']), DB_NAME='imdb',
                      DB_USER='imdb', DB_PASSWORD='imdb')
    memory_guard(recipe)
    data = json.loads((campaign/'tasks.json').read_text())[task]
    spec, canonical, crossjoin = canonical_context(Path(recipe['native']), task)
    if digest(query.sql_path(Path(recipe['native']), task)) != data['sql_sha256']:
        raise ValueError('Query SQL changed')
    vae = QueryVAEBackend(recipe['vae_pool'], task, data['fallback'], seed=recipe['seed'])
    backend = CanonicalBackend(vae, canonical)
    initial, engine = data['initial'], None
    directory = campaign/'runs'/task
    with lock(campaign/'locks'/(task+'.lock')):
        if not (directory/'checkpoint.pt').exists():
            plans = list(dict.fromkeys(tuple(r['plan']) for r in initial))
            features = np.asarray(vae.encode([list(p) for p in plans]))
            if features.shape != (len(plans), 64) or not np.isfinite(features).all():
                raise ValueError('Invalid Bao query encoder means')
            known = dict(zip(plans, features.tolist()))
            initial = [dict(r, z=known[tuple(r['plan'])]) for r in initial]
            engine = FSBOEngine(campaign/'recipe.json', recipe, initial, vae)
        meta = dict(task_id=task, method='fsbo', split='train_component' if recipe['purpose']=='component_check' else 'test',
            seed=recipe['seed'], recipe_sha256=digest(campaign/'recipe.json'),
            runner_sha256=digest(__file__), vae_sha256=recipe['pinned_files'][str(Path(recipe['native'])/'vae/CEB_64_author.ckpt')],
            update_e2e=False, e2e_updates=0, recenter=False, recenter_calls=0,
            fsbo_weights_sha256=recipe['fsbo_weights_sha256'],
            acquisition=recipe['acquisition'], target_policy=recipe['target_policy'])
        # Load the private TuRBO module before unpickling checkpointed state.
        peptide_turbo(recipe)
        oracle = query.RuntimeOracle(spec, max(r['runtime'] for r in initial))
        previous_clients = list(LIVE_CLIENTS)
        try:
            return query.run_loop(directory, meta, initial, engine, backend, oracle,
                                  recipe['initialization']+recipe['proposals'], stop_after_batches)
        finally:
            for client in list(LIVE_CLIENTS):
                if client not in previous_clients:
                    client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--campaign', type=Path, required=True)
    prep.add_argument('--component-task')
    prep.add_argument('--proposals', type=int, default=1000)
    prep.add_argument('--batch-size', type=int, default=10)
    prep.add_argument('--n-candidates', type=int, default=5000)
    for name in ['run-task','run-worker']:
        run = commands.add_parser(name)
        run.add_argument('--campaign', type=Path, required=True)
        run.add_argument('--worker', required=True)
        run.add_argument('--stop-after-batches', type=int)
        if name == 'run-task':
            run.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    recipe = verify(args.campaign)
    worker = next(w for w in recipe['workers'] if w['name'] == args.worker)
    # Same endpoint lock convention as the existing query baselines.
    endpoint = '127.0.0.1:%s:imdb' % worker['db_port']
    lock_id = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    with lock(Path('/tmp')/('ceb_baseline_db_'+lock_id+'.lock')):
        tasks = [args.task] if args.command=='run-task' else recipe['allowed_tasks']
        for task in tasks:
            if (args.campaign/'STOP').exists():
                break
            status_path = args.campaign/'runs'/task/'status.json'
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status['status'] == 'complete':
                    continue
                if status['status'] == 'failed':
                    raise RuntimeError('Inspect failed query before resuming: '+task)
            try:
                result = run_task(args.campaign, recipe, task, worker, args.stop_after_batches)
            except BlockingIOError:
                continue
            print(json.dumps(result), flush=True)
            if args.stop_after_batches:
                break


if __name__ == '__main__':
    # Keep serialized engine classes importable when invoked as a script.
    sys.modules['query_fsbo_eval'] = sys.modules[__name__]
    FSBOEngine.__module__ = 'query_fsbo_eval'
    main()
