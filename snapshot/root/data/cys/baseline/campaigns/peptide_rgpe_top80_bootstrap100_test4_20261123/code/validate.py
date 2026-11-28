"""Check ranking semantics and bootstrap parity against full SMAC GP predictors."""
import copy
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from source_screen import ROOT, MeanSource, atomic, ranking_loss, select_indices
from peptide_transfer_models import SourceBank
from rgpe.methods.rgpe import compute_ranking_loss, compute_rank_weights
from rgpe.utils import get_gaussian_process
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.epm.util_funcs import get_types


def main():
    started = time.monotonic()
    rng = np.random.RandomState(73)
    for n in (3, 8, 100):
        for _ in range(5):
            y, p = rng.randint(0, 7, size=(2, n)).astype(float)
            assert ranking_loss(p, y) == compute_ranking_loss(p[None, :], y, False)[0]
    selected, losses = select_indices(np.tile(np.arange(100), (800, 1)), np.arange(100))
    assert np.array_equal(selected, np.arange(80)) and np.all(losses == 0)
    recipe = json.loads((ROOT/'rgpe/recipe.json').read_text())
    bank = SourceBank(recipe)
    full, light = [], []
    for i in range(3):
        full.append(bank[i])
        with (ROOT/'prefilter/mean_models'/(bank.tasks[i]+'.pkl')).open('rb') as stream:
            light.append(MeanSource(pickle.load(stream)))
    initial = np.load(str(ROOT/'prefilter/pep_104/initialization.npz'))
    x, y = initial['x'][:20], initial['mic'][:20]
    y = (y-y.mean())/y.std()
    cs = ConfigurationSpace()
    cs.add_hyperparameters([UniformFloatHyperparameter('z%03d'%i, 0., 1.) for i in range(256)])
    types, bounds = get_types(cs, None)
    target = get_gaussian_process(cs, types, bounds, np.random.RandomState(61), None)
    target._train(x, y, do_optimize=False)
    values = []
    for models in (full, light):
        values.append(compute_rank_weights(x, y, models, copy.deepcopy(target), 100,
                      'bootstrap', 'probabilistic-ld', 4100, np.random.RandomState(97)))
    assert np.array_equal(values[0][0], values[1][0])
    assert np.array_equal(values[0][1], values[1][1])
    atomic(ROOT/'evidence/validation.json', dict(passed=True,
        ranking_loss_matches_upstream_with_ties=True, stable_top80_tie_break=True,
        bootstrap_full_vs_mean_only_weights_equal=True,
        bootstrap_drop_probabilities_equal=True, actual_source_GPs=3,
        bootstrap_samples=100, weights=values[0][0].tolist(),
        seconds=time.monotonic()-started))
    print((ROOT/'evidence/validation.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
