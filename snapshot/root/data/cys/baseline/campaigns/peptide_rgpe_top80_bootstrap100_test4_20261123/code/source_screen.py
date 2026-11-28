"""One-time mean-only source selection on the 100 initialization observations."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = '/root/data/cys/llmoptimizer_benchmark/tools'
sys.path[:0] = [TOOLS, '/root/data/cys/baseline/repos/transfer-hpo-framework']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(tmp), str(path))


def ranking_loss(prediction, target):
    """Same ordered-pair strict comparisons (including ties) as upstream RGPE."""
    p = np.asarray(prediction).reshape(-1)
    y = np.asarray(target).reshape(-1)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid source screening predictions')
    return int(np.count_nonzero((p[:, None] < p[None, :]) ^
                                (y[:, None] < y[None, :])))


def select_indices(predictions, target, count=80):
    losses = np.array([ranking_loss(p, target) for p in predictions])
    # Stable original source inventory order breaks equal-loss ties.
    order = np.lexsort((np.arange(len(losses)), losses))
    return order[:count], losses


class MeanSource:
    """Prediction state excludes the large Cholesky matrix and training covariance."""
    def __init__(self, state):
        self.state = state

    def predict(self, x):
        d = self.state
        mu = d['kernel'](np.asarray(x, dtype=np.float64), d['train']).dot(d['alpha'])
        mu = mu * d['scale'] + d['offset']
        return np.asarray(mu).reshape(-1, 1), None


def main():
    from peptide_transfer_models import SourceBank
    recipe = json.loads((ROOT/'rgpe/recipe.json').read_text())
    tasks = recipe['allowed_tasks']
    bank = SourceBank(recipe)
    inputs = {t: np.load(str(ROOT/'prefilter'/t/'initialization.npz')) for t in tasks}
    combined = np.concatenate([inputs[t]['x'] for t in tasks])
    if combined.shape != (400, 256):
        raise ValueError('Expected four disjoint test-task initializations of 100 points')
    light = ROOT/'prefilter/mean_models'
    light.mkdir(exist_ok=True)
    predictions = np.empty((800, len(combined)), dtype=np.float64)
    timings = []
    started = time.monotonic()
    validation = []
    for i, task in enumerate(bank.tasks):
        tick = time.monotonic()
        model = bank[i]  # Verifies serialized SHA, source history and corpus identity.
        gp = model.gp
        state = dict(train=gp.X_train_, alpha=gp.alpha_, kernel=gp.kernel_,
                     scale=float(model.std_y_), offset=float(model.mean_y_),
                     task=task, model_sha256=recipe['rgpe_model_hashes'][task])
        compact = MeanSource(state)
        mu = compact.predict(combined)[0].reshape(-1)
        if not np.isfinite(mu).all():
            raise ValueError('Nonfinite mean prediction: ' + task)
        if i < 3:
            # Independent SMAC mean-only path; neither path requests variance.
            reference = model._predict(combined, cov_return_type=None)[0].reshape(-1)
            error = float(np.max(np.abs(mu-reference)))
            if not np.allclose(mu, reference, rtol=1e-11, atol=1e-10):
                raise ValueError('Compact source predictor differs from SMAC')
            validation.append(dict(task=task, max_error=error))
        predictions[i] = mu
        path = light/(task+'.pkl')
        with path.with_suffix('.tmp').open('wb') as stream:
            pickle.dump(state, stream, protocol=4)
        os.replace(str(path.with_suffix('.tmp')), str(path))
        del compact, model, gp, state
        timings.append(time.monotonic()-tick)
        if (i+1) % 25 == 0 or i == 0:
            atomic(ROOT/'prefilter/progress.json', dict(completed=i+1, total=800,
                   seconds=time.monotonic()-started, last_task=task))
            print(json.dumps(dict(completed=i+1, seconds=time.monotonic()-started)), flush=True)
    for j, target_task in enumerate(tasks):
        target_dir = ROOT/'prefilter'/target_task
        mu = predictions[:, j*100:(j+1)*100]
        chosen, losses = select_indices(mu, inputs[target_task]['mic'], 80)
        # Keep inventory order for RGPE seed/model mapping; ranking order is recorded too.
        selected = [bank.tasks[i] for i in sorted(chosen.tolist())]
        np.savez(str(target_dir/'source_predictions.npz'), means=mu, losses=losses)
        atomic(target_dir/'selection.json', dict(format='rgpe-initial-top80-v1',
            task=target_task, observations=100, candidate_sources=800, retained_sources=80,
            policy='ordered_pair_ranking_loss_mean_only', tie_break='original_inventory_index',
            fixed_for_all_rounds=True, initialization_sha256=digest(target_dir/'initialization.npz'),
            recipe_sha256=digest(ROOT/'rgpe/recipe.json'),
            selected_tasks=selected, ranked_selected_tasks=[bank.tasks[i] for i in chosen],
            sources=[dict(task=t, loss=int(losses[i]), selected=i in set(chosen.tolist()),
                          model_sha256=recipe['rgpe_model_hashes'][t]) for i,t in enumerate(bank.tasks)],
            light_model_hashes={t:digest(light/(t+'.pkl')) for t in selected},
            mean_only_validation=validation, total_shared_screen_seconds=time.monotonic()-started))
    atomic(ROOT/'prefilter/complete.json', dict(passed=True, tasks=tasks, sources=800,
        selected_per_task=80, mean_only_validation=validation,
        total_seconds=time.monotonic()-started, per_source_seconds=timings))


if __name__ == '__main__':
    main()
