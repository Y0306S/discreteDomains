"""Independent checkpoint audit and real-history target adaptation checks."""

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import resource
import sqlite3
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
TRAINING = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
CASE = Path('/root/data/logs/peptide_transfer_test200_20261028_v1/fsbo')
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, '/root/data/cys/baseline/repos/FSBO')

import gpytorch
import numpy as np
import torch
import peptide_transfer as transfer
from fsbo_modules import DeepKernelGP
from peptide_transfer_models import FSBOBackend


class AuditUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # The CPU posterior environment predates upstream's Python 3.10 type
        # syntax. Only inspect the saved state; never run/rewrite its trust region.
        if module == 'lolbo.utils.bo_utils.turbo' and name == 'TurboState':
            return SimpleNamespace
        return super().find_class(module, name)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = ROOT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(str(temporary), str(path))


def source_audit():
    status = json.loads((TRAINING / 'fsbo/status.json').read_text())
    manifest = json.loads((TRAINING / 'corpus/manifest.json').read_text())
    state = torch.load(str(TRAINING / 'fsbo/training_state.pt'), map_location='cpu')
    recipe = json.loads((CASE / 'recipe.json').read_text())
    checks = dict(checkpoint_hash=digest(TRAINING / 'fsbo/weights') == status['weights_sha256'],
        corpus_hash=digest(TRAINING / 'corpus/manifest.json') == status['identity']['corpus_sha256'],
        training_loop_hash=digest(TOOLS.parent / 'examples/benchmark/domains/train_taskset_transfer.py') ==
            status['identity']['shared_training_loop_sha256'],
        exporter_hash=digest(TOOLS / 'peptide_transfer.py') == status['identity']['trainer_sha256'],
        official_implementation_hash=digest('/root/data/cys/baseline/repos/FSBO/fsbo_modules.py') ==
            status['identity']['config']['implementation_sha256'],
        training_state_identity=state['identity'] == status['identity'],
        budget=state['step'] == status['step'] == 100000,
        selected_step=state['best_step'] == status['best_step'] == 94900,
        disjoint_source_validation=not set(manifest['fsbo_train_tasks']) & set(manifest['fsbo_validation_tasks']),
        source_split_matches=set(manifest['source_tasks']) == set(recipe['train']),
        no_test_leakage=not set(manifest['source_tasks']) & set(recipe['test']))
    backend = FSBOBackend(recipe)
    initial = DeepKernelGP(256, os.devnull, 123, hidden_size=[128, 128], kernel='rbf',
        ard=False, checkpoint=str(TRAINING / 'fsbo'), epochs=10000)
    for name, model in [('trained', backend.model), ('initialized', initial)]:
        for module in (model.model, model.feature_extractor, model.likelihood):
            module.eval()
        checks[name + '_finite_parameters'] = all(torch.isfinite(p).all().item() for p in model.parameters())
    checks['feature_extractor_learned'] = any(not torch.equal(p, initial.feature_extractor.state_dict()[k])
        for k, p in backend.model.feature_extractor.state_dict().items())
    checks['gp_learned'] = any(not torch.equal(p, initial.model.state_dict()[k])
        for k, p in backend.model.model.state_dict().items())
    optimizer_steps = [float(s['step']) for s in state['optimizer']['state'].values()]
    checks['all_parameters_updated'] = all(s == 100000 for s in optimizer_steps)
    losses = {'trained': [], 'initialized': []}
    rng = np.random.RandomState(status['identity']['seed'] + 1)
    coverage = {}
    with sqlite3.connect('file:' + str(TRAINING / 'corpus/corpus.sqlite') + '?mode=ro', uri=True) as db:
        checks['corpus_inventory'] = set(r[0] for r in db.execute('SELECT id FROM tasks')) == set(manifest['source_tasks'])
        for task in manifest['source_tasks']:
            data = transfer.read_task(db, task)
            if data['payload_sha256'] != manifest['task_hashes'][task]:
                raise ValueError('Source payload differs from pinned manifest: ' + task)
            if task in state['seen']:
                seen = state['seen'][task]
                if len(seen) != len(data['y']) or not seen.any():
                    raise ValueError('Missing training coverage: ' + task)
                coverage[task] = dict(seen=int(seen.sum()), total=len(seen))
        for task in manifest['fsbo_validation_tasks']:
            data = transfer.read_task(db, task)
            x = transfer.normalize(data['X'], manifest)
            y = np.asarray(data['y'], dtype=float)
            y = (y - y.min()) / (float(np.ptp(y)) or 1.)
            order = rng.permutation(len(x))
            split = min(50, max(1, len(order) // 2))
            s, q = order[:split], order[split:split + 50]
            sx, sy, qx, qy = [torch.as_tensor(v, dtype=torch.float32) for v in (x[s], y[s], x[q], y[q])]
            with torch.no_grad():
                for name, model in [('trained', backend.model), ('initialized', initial)]:
                    model.model.set_train_data(model.feature_extractor(sx), sy, strict=False)
                    pred = model.likelihood(model.model(model.feature_extractor(qx)))
                    losses[name].append(float(-pred.log_prob(qy) / len(q)))
    score = float(np.mean(losses['trained']))
    baseline = float(np.mean(losses['initialized']))
    checks['all_train_tasks_sampled'] = len(coverage) == 640
    checks['validation_reproduced'] = abs(score - status['best_validation_loss']) < 0.00001
    checks['validation_better_than_initialization'] = score < baseline
    result = dict(passed=all(checks.values()), checks=checks, source_tasks=800,
        training_tasks=640, validation_tasks=160, optimizer_updates=state['step'],
        selected_step=state['best_step'], validation_nll=score, initialized_nll=baseline,
        validation_task_wins=sum(a < b for a, b in zip(losses['trained'], losses['initialized'])),
        observations_sampled=sum(r['seen'] for r in coverage.values()),
        source_observations=sum(r['total'] for r in coverage.values()),
        weights_sha256=digest(TRAINING / 'fsbo/weights'), coverage=coverage,
        torch=torch.__version__, gpytorch=gpytorch.__version__, at=time.time())
    write('training_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'coverage'}), flush=True)
    if not result['passed']:
        raise RuntimeError('Training audit failed')


def target_audit():
    recipe = json.loads((CASE / 'recipe.json').read_text())
    settings = dict(maximum_updates=1000, patience=16, min_delta=0.0001,
        likelihood='deterministic_seeded_likelihood')
    recipe['fsbo_target_epochs'] = settings['maximum_updates']
    recipe['fsbo_target_stopping'] = settings
    sys.path.insert(0, recipe['cpu_upstream'])
    histories = [json.loads(p.read_text()) for p in (CASE / 'runs').glob('*/history.json')]
    history = max(histories, key=lambda h: len(h['observations']))
    with sqlite3.connect('file:' + str(CASE / 'runs' / history['task'] / 'runtime.sqlite') + '?mode=ro', uri=True) as db:
        saved = AuditUnpickler(io.BytesIO(db.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0])).load()
        pending = db.execute("SELECT value FROM meta WHERE key='pending'").fetchone()
        assert pending is None or pickle.loads(pending[0]) is None
    x = transfer.normalize(saved['features'], recipe)
    score = np.asarray([r['score'] for r in saved['rows']])
    reports, draws, states = [], [], []
    for repeat in range(2):
        backend = FSBOBackend(recipe)
        count = [0]
        start = time.monotonic()
        original_forward = backend.model.mll.forward
        def tracked_forward(*inputs, **kwargs):
            output = original_forward(*inputs, **kwargs)
            count[0] += 1
            if count[0] == 1 or count[0] % 100 == 0:
                value = dict(task=history['task'], repeat=repeat, evaluations=count[0],
                    loss=float(-output.detach()), seconds=time.monotonic() - start)
                write('fit_progress.json', value)
                print(json.dumps(value), flush=True)
            return output
        backend.model.mll.forward = tracked_forward
        info = backend.fit(x, score, recipe['seed'] + saved['progress']['rounds'])
        backend.model.mll.forward = original_forward
        states.append(copy.deepcopy(backend.model.state_dict()))
        sample = backend.sample(x[:16], 50, 937)['draws']
        snapshot = backend.snapshot()
        backend.restore(snapshot)
        restored = backend.sample(x[:16], 50, 937)['draws']
        assert np.array_equal(sample, restored)
        reports.append(info)
        draws.append(sample)
        print(json.dumps(dict(task=history['task'], repeat=repeat, fit=info)), flush=True)
    from fsbo_target_refit import fit_target
    backend = FSBOBackend(recipe)
    initial = copy.deepcopy(backend.model.state_dict())
    tx = torch.as_tensor(x[:100])
    ty = torch.as_tensor((score[:100] - score[:100].min()) / (float(np.ptp(score[:100])) or 1.), dtype=torch.float32)
    plateau = fit_target(backend.model, tx, ty, 100, lr=0.)
    plateau_restored = all(torch.equal(v, backend.model.state_dict()[k]) for k, v in initial.items())
    capped = fit_target(backend.model, tx, ty, 3, patience=16)
    checks = dict(real_history=True, observations_above_stochastic_threshold=len(x) > 800,
        repeatable_weights=all(torch.equal(v, states[1][k]) for k, v in states[0].items()),
        repeatable_samples=np.array_equal(draws[0], draws[1]),
        best_weights_restore=all(abs(f['best_loss'] - f['restored_loss']) < 0.00001 for f in reports),
        bounded_adaptation=all(f['stop_reason'] in ('best_loss_patience', 'maximum_updates') and
            f['optimizer_updates'] <= 1000 for f in reports),
        real_loss_improved=all(f['best_loss'] < f['initial_loss'] - 0.01 for f in reports),
        joint_feature_and_gp=all(f['joint_feature_and_gp'] for f in reports),
        feature_extractor_adapted=any(not torch.equal(v, backend.initial[k]) for k, v in states[0].items() if k.startswith('feature_extractor.')),
        plateau_stops=plateau['optimizer_updates'] == 16 and plateau['best_step'] == 0 and plateau_restored,
        final_update_measured=capped['optimizer_updates'] == 3 and capped['likelihood_evaluations'] == 4,
        snapshot_reload=True, finite_samples=all(np.isfinite(v).all() for v in draws))
    old = [r['fit'] for r in saved['rows'] if r.get('fit')][-1]
    result = dict(passed=all(checks.values()), checks=checks, task=history['task'],
        observations=len(x), fit=reports, previous_fit=old, plateau=plateau, capped=capped, settings=settings,
        code_sha256={str(TOOLS / n): digest(TOOLS / n) for n in
            ('peptide_transfer_models.py', 'peptide_transfer_eval.py', 'peptide_transfer_policy.py', 'fsbo_target_refit.py')},
        unchanged_recipe_sha256=digest(CASE / 'recipe.json'), at=time.time())
    write('target_validation.json', result)
    if not result['passed']:
        raise RuntimeError('Target refitting validation failed')


def publish():
    training = json.loads((ROOT / 'training_audit.json').read_text())
    validation = json.loads((ROOT / 'target_validation.json').read_text())
    if not training['passed'] or not validation['passed']:
        raise RuntimeError('Both independent audits must pass')
    scale = json.loads((ROOT / 'scale_validation.json').read_text())
    component_path = ROOT / 'component_final_report/fsbo_report.json'
    component = json.loads(component_path.read_text())
    if not scale['passed'] or not component['passed'] or not component['production_scale_passed']:
        raise RuntimeError('Production-scale/component checks must pass')
    recipe = json.loads((CASE / 'recipe.json').read_text())
    previous = {}
    for name in ('peptide_transfer_eval.py', 'peptide_transfer_models.py'):
        path = ROOT / 'prior_source' / name
        expected = recipe['code'][str(TOOLS / name)]
        if digest(path) != expected:
            raise RuntimeError('Original source snapshot does not match its recipe')
        previous[str(path)] = expected
    amended = {}
    for path, expected in validation['code_sha256'].items():
        snapshot = ROOT / 'amended_source' / Path(path).name
        if digest(snapshot) != expected:
            raise RuntimeError('Amended source snapshot differs from validated code')
        amended[str(snapshot)] = expected
    amendment = dict(format='fsbo-target-fit-amendment-v1', recipe_sha256=digest(CASE / 'recipe.json'),
        settings=validation['settings'],
        code={name: dict(previous_sha256=recipe['code'].get(name), sha256=sha)
              for name, sha in validation['code_sha256'].items()},
        validation=dict(path=str(ROOT / 'target_validation.json'), sha256=digest(ROOT / 'target_validation.json')),
        training_audit=dict(path=str(ROOT / 'training_audit.json'), sha256=digest(ROOT / 'training_audit.json')),
        component_validation=dict(path=str(component_path), sha256=digest(component_path)),
        scale_validation=dict(path=str(ROOT / 'scale_validation.json'), sha256=digest(ROOT / 'scale_validation.json')),
        original_source_snapshots=previous,
        amended_source_snapshots=amended,
        transition='Retain all measured evaluations; replace only subsequent joint target fits',
        completed_evaluations_retained=True, created_at=time.time())
    path = CASE / 'target_fit_amendment.json'
    if path.exists():
        raise RuntimeError('Amendment already exists')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(amendment, indent=2))
    os.replace(str(temporary), str(path))
    from peptide_transfer_policy import effective_recipe
    effective_recipe(CASE / 'recipe.json')
    print(json.dumps(dict(published=str(path), original_recipe_preserved=True)), flush=True)


def scale_audit():
    recipe = json.loads((CASE / 'recipe.json').read_text())
    manifest = json.loads((TRAINING / 'corpus/manifest.json').read_text())
    with sqlite3.connect('file:' + str(TRAINING / 'corpus/corpus.sqlite') + '?mode=ro', uri=True) as db:
        task = db.execute('SELECT id FROM tasks ORDER BY observations DESC LIMIT 1').fetchone()[0]
        data = transfer.read_task(db, task)
    recipe['fsbo_target_epochs'] = 3
    backend = FSBOBackend(recipe)
    info = backend.fit(transfer.normalize(data['X'], manifest), data['y'], 123)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    value = dict(passed=peak < 6.0 and info['optimizer_updates'] == 3,
        task=task, fit=info, peak_rss_gib=peak, reserved_worker_gib=6.0,
        note='Bounded dense-likelihood scale check, not a convergence or quality test')
    write('scale_validation.json', value)
    print(json.dumps(value), flush=True)
    if not value['passed']:
        raise RuntimeError('Production-scale memory check failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('source', 'target', 'publish', 'scale'))
    parser.add_argument('--cpu', type=int, default=18)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    {'source': source_audit, 'target': target_audit, 'publish': publish, 'scale': scale_audit}[args.action]()
