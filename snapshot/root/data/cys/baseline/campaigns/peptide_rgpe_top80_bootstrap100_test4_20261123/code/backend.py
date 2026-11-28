"""Fixed top-80 sources with upstream bootstrap RGPE using 100 resamples."""
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np

from source_screen import ROOT, MeanSource, digest
from peptide_transfer_models import RGPEBackend, SourceBank, checked_xy


class Backend(RGPEBackend):
    def __init__(self, recipe, task):
        self.recipe, self.task, self.ensemble = recipe, task, None
        if (recipe['rgpe_sampling_mode'], recipe['rgpe_rank_samples']) != ('bootstrap', 100):
            raise ValueError('This pilot requires bootstrap with 100 resamples')
        path = ROOT/'prefilter'/task/'selection.json'
        self.selection = json.loads(path.read_text())
        initial_path = path.with_name('initialization.npz')
        if (self.selection['recipe_sha256'] != digest(ROOT/'rgpe/recipe.json')
                or self.selection['initialization_sha256'] != digest(initial_path)
                or self.selection['task'] != task):
            raise ValueError('Source screen identity changed')
        self.initial = np.load(str(initial_path))
        tasks = self.selection['selected_tasks']
        if len(tasks) != 80 or len(set(tasks)) != 80 or not set(tasks) <= set(recipe['train']):
            raise ValueError('Invalid fixed source selection')
        subset = dict(recipe, rgpe_sources=tasks)
        self.bank = SourceBank(subset)
        self.mean_models = []
        for name in tasks:
            path = ROOT/'prefilter/mean_models'/(name+'.pkl')
            if digest(path) != self.selection['light_model_hashes'][name]:
                raise ValueError('Mean predictor changed: '+name)
            with path.open('rb') as stream:
                value = pickle.load(stream)
            if value['task'] != name or value['model_sha256'] != recipe['rgpe_model_hashes'][name]:
                raise ValueError('Mean predictor source mismatch')
            self.mean_models.append(MeanSource(value))
        self.selection_sha256 = digest(ROOT/'prefilter'/task/'selection.json')

    def fit(self, x, score, seed):
        from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
        from smac.epm.util_funcs import get_types
        from rgpe.utils import get_gaussian_process
        from rgpe.methods.rgpe import compute_rank_weights
        from peptide_rgpe_exact import exact_objective
        x, score = checked_xy(x, score)
        if (len(x) < 100 or not np.array_equal(x[:100], self.initial['x'])
                or not np.array_equal(-score[:100], self.initial['mic'])):
            raise ValueError('Target initialization differs from the fixed source screen')
        self.bank.target = None
        mic, rng = -score, np.random.RandomState(seed)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([UniformFloatHyperparameter('z%03d'%i, 0., 1.) for i in range(256)])
        types, bounds = get_types(cs, None)
        started = time.monotonic()
        target = get_gaussian_process(cs, types, bounds, rng, None)
        with exact_objective(target):
            target.train(x.astype(np.float64), mic)
        fit_seconds = time.monotonic()-started
        mean, scale = float(mic.mean()), float(mic.std()) or 1.
        tick = time.monotonic()
        weights, drop = compute_rank_weights(x, (mic-mean)/scale,
            self.mean_models, target, self.recipe['rgpe_rank_samples'],
            self.recipe['rgpe_sampling_mode'], 'probabilistic-ld',
            self.recipe['initialization']+self.recipe['proposals'], rng)
        rank_seconds = time.monotonic()-tick
        if len(weights) != 81 or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.):
            raise ValueError('Invalid bootstrap RGPE weights')
        self.bank.target = target
        self.ensemble = SimpleNamespace(model_list_=self.bank, weights_=weights,
            Y_mean_=mean, Y_std_=scale, variance_mode='average')
        self.fit_info = dict(method='rgpe', target_observations=len(x), source_models=80,
            weights=weights.tolist(), source_tasks=self.bank.tasks,
            nonzero_weights=int(np.count_nonzero(weights)), target_weight=float(weights[-1]),
            ranking='bootstrap', sampling_mode='bootstrap', num_posterior_samples=100,
            weight_dilution='probabilistic-ld', dilution_horizon=self.recipe['initialization']+self.recipe['proposals'],
            source_selection_sha256=self.selection_sha256,
            target_fit_seconds=fit_seconds, bootstrap_seconds=rank_seconds,
            seconds=time.monotonic()-started, source_prediction='mean_only',
            target_optimizer_restarts=target.n_opt_restarts)
        return self.fit_info
