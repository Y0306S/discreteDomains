#!/usr/bin/env python3
"""Isolated, persistent FSBO/RGPE posterior workers for peptide evaluation."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import struct
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import peptide_transfer as transfer


def receive(stream):
    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise EOFError('Truncated model request header')
    size = struct.unpack('!Q', header)[0]
    payload = bytearray()
    while len(payload) < size:
        chunk = stream.read(min(size - len(payload), 2**20))
        if not chunk:
            raise EOFError('Truncated model request payload')
        payload.extend(chunk)
    return pickle.loads(payload)


def send(stream, value):
    payload = pickle.dumps(value, protocol=4)
    stream.write(struct.pack('!Q', len(payload)))
    stream.write(payload)
    stream.flush()


def checked_xy(x, y):
    x, y = np.asarray(x), np.asarray(y).reshape(-1)
    if (x.shape != (len(y), 256) or len(y) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Invalid target observations')
    return x, y


class FSBOBackend:
    def __init__(self, recipe):
        sys.path.insert(0, str(Path('/root/data/cys/baseline/repos/FSBO')))
        import torch
        from fsbo_modules import DeepKernelGP
        self.torch = torch
        torch.set_num_threads(1)
        self.recipe = recipe
        self.model = DeepKernelGP(256, os.devnull, recipe['seed'],
            hidden_size=[128, 128], kernel='rbf', ard=False,
            checkpoint=str(Path(recipe['fsbo_weights']).parent),
            epochs=recipe['fsbo_target_epochs'], lr=0.001)
        weights = torch.load(recipe['fsbo_weights'], map_location='cpu')
        self.model.model.load_state_dict(weights['gp'], strict=True)
        self.model.likelihood.load_state_dict(weights['likelihood'], strict=True)
        self.model.feature_extractor.load_state_dict(weights['net'], strict=True)
        self.initial = copy.deepcopy(self.model.state_dict())
        self.offset, self.scale, self.fit_info = None, None, None

    def fit(self, x, score, seed):
        torch = self.torch
        x, score = checked_xy(x, score)
        torch.manual_seed(seed)
        self.offset = float(score.min())
        self.scale = float(score.max() - self.offset) or 1.0
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor((score - self.offset) / self.scale, dtype=torch.float32)
        self.model.load_state_dict(self.initial, strict=True)
        from fsbo_target_refit import fit_target
        settings = self.recipe.get('fsbo_target_stopping', {})
        fit = fit_target(self.model, self.x, self.y, self.recipe['fsbo_target_epochs'],
            patience=settings.get('patience', self.recipe.get('fsbo_target_patience', 16)),
            min_delta=settings.get('min_delta', 0.0001),
            lr=self.recipe['fsbo_target_learning_rate'], seed=seed)
        self.fit_info = dict(fit, method='fsbo', target_observations=len(score),
            checkpoint_sha256=self.recipe['fsbo_weights_sha256'], label_scaling='target_minmax',
            stopping_code_sha256=hashlib.sha256(Path(__file__).with_name('fsbo_target_refit.py').read_bytes()).hexdigest(),
            amendment_sha256=self.recipe.get('target_fit_amendment_sha256'))
        return self.fit_info

    def sample(self, candidates, count, seed):
        torch = self.torch
        torch.manual_seed(seed)
        with torch.no_grad():
            x = torch.as_tensor(candidates, dtype=torch.float32)
            posterior = self.model.model(self.model.feature_extractor(x))
            samples = posterior.rsample(torch.Size([count])).cpu().numpy()
        return dict(draws=samples * self.scale + self.offset,
                    info=dict(method='fsbo', candidates=len(candidates), samples=count,
                              correlated=True, higher_is_better=True))

    def snapshot(self):
        stream = __import__('io').BytesIO()
        self.torch.save(dict(model=self.model.state_dict(), x=self.x, y=self.y,
            offset=self.offset, scale=self.scale, fit_info=self.fit_info), stream)
        return stream.getvalue()

    def restore(self, payload):
        value = self.torch.load(__import__('io').BytesIO(payload), map_location='cpu')
        self.model.model.train()
        self.model.load_state_dict(value['model'], strict=True)
        self.x, self.y = value['x'], value['y']
        self.offset, self.scale, self.fit_info = value['offset'], value['scale'], value['fit_info']
        self.model.model.eval()
        self.model.likelihood.eval()
        self.model.feature_extractor.eval()
        with self.torch.no_grad():
            self.model.model.set_train_data(self.model.feature_extractor(self.x), self.y, strict=False)
        return self.fit_info


class SourceBank:
    """Load one source GP at a time, with a recipe-pinned inventory."""

    def __init__(self, recipe):
        self.recipe = recipe
        self.tasks = recipe['rgpe_sources']
        self.target = None

    def __len__(self):
        return len(self.tasks) + (self.target is not None)

    def __getitem__(self, index):
        if index == len(self.tasks) and self.target is not None:
            return self.target
        task = self.tasks[index]
        with sqlite3.connect('file:' + self.recipe['rgpe_bank'] + '?mode=ro', uri=True, timeout=120) as db:
            row = db.execute('SELECT metadata,sha256,payload FROM models WHERE task=?', (task,)).fetchone()
        if row is None or row[1] != self.recipe['rgpe_model_hashes'][task]:
            raise ValueError('Source model inventory changed: ' + task)
        metadata = json.loads(row[0])
        if (hashlib.sha256(row[2]).hexdigest() != row[1] or not metadata['reload_verified']
                or metadata['source_payload_sha256'] != self.recipe['source_payload_hashes'][task]
                or metadata['corpus_sha256'] != self.recipe['corpus_sha256']):
            raise ValueError('Source GP provenance failed: ' + task)
        return pickle.loads(row[2])


def target_ranking_loss(x, y, samples, model, seed):
    """Upstream correct joint LOOCV, without the N x N x D fold stack."""
    from rgpe.utils import get_gaussian_process
    from peptide_rgpe_posterior import sample_rank_posterior
    losses = np.zeros(samples, dtype=np.int64)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        fold = get_gaussian_process(model.configspace, model.types, model.bounds,
                                    model.rng, model.kernel)
        fold._train(x[keep], y[keep], do_optimize=False)
        draws = sample_rank_posterior(fold, x, samples, seed)
        losses += np.sum((draws[:, i:i+1] < draws) ^ (y[i] < y)[None, :], axis=1)
        del fold, draws
    return losses


class RGPEBackend:
    def __init__(self, recipe):
        sys.path.insert(0, str(transfer.RGPE_REPO))
        import torch
        torch.set_num_threads(1)
        self.recipe, self.bank, self.ensemble = recipe, SourceBank(recipe), None

    def fit(self, x, score, seed):
        from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
        from smac.epm.util_funcs import get_types
        from rgpe.utils import get_gaussian_process
        from rgpe.methods import rgpe as upstream
        from peptide_rgpe_exact import exact_objective
        from peptide_rgpe_posterior import sample_rank_posterior
        self.bank.target = None
        x, score = checked_xy(x, score)
        mic, rng = -score, np.random.RandomState(seed)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([UniformFloatHyperparameter('z%03d' % i, 0., 1.) for i in range(256)])
        types, bounds = get_types(cs, None)
        target = get_gaussian_process(cs, types, bounds, rng, None)
        started = time.monotonic()
        with exact_objective(target):
            target.train(x.astype(np.float64), mic)
        mean, scale = float(mic.mean()), float(mic.std()) or 1.0
        # Only this isolated process uses the corrected sampler and streamed LOOCV.
        upstream.sample_sobol = sample_rank_posterior
        upstream.compute_target_model_ranking_loss = target_ranking_loss
        weights, drop = upstream.compute_rank_weights(x, (mic - mean) / scale,
            self.bank, target, self.recipe['rgpe_rank_samples'], 'correct',
            'probabilistic-ld', self.recipe['initialization'] + self.recipe['proposals'], rng)
        self.bank.target = target
        self.ensemble = SimpleNamespace(model_list_=self.bank, weights_=weights,
            Y_mean_=mean, Y_std_=scale, variance_mode='average')
        # The bank includes the target only after source ranking has finished.
        self.fit_info = dict(method='rgpe', target_observations=len(mic),
            source_models=len(self.recipe['rgpe_sources']), weights=weights.tolist(),
            nonzero_weights=int(np.count_nonzero(weights)), target_weight=float(weights[-1]),
            ranking='correct_joint_LOOCV', weight_dilution='probabilistic-ld',
            seconds=time.monotonic() - started, target_optimizer_restarts=target.n_opt_restarts)
        return self.fit_info

    def sample(self, candidates, count, seed):
        from peptide_rgpe_posterior import merge_posterior, sample_posterior
        if self.ensemble is None:
            raise ValueError('RGPE target is not fitted')
        mean, covariance, info = merge_posterior(self.ensemble, candidates, source_output='standardized')
        draws, sampling = sample_posterior(mean, covariance, count, seed)
        info.update(sampling, method='rgpe', posterior='merged_independent_GP',
                    retained_source_covariances=1, higher_is_better=True)
        return dict(draws=draws, info=info)

    def snapshot(self):
        target, gp = self.bank.target, self.bank.target.gp
        # Factorizations are reproducible caches, not learned parameters. Keep the
        # fitted kernel and normalized data, then rebuild caches without optimization.
        try:
            target.gp = None
            model = pickle.dumps(target, protocol=4)
        finally:
            target.gp = gp
        return pickle.dumps(dict(target_state=dict(model=model,
            x=gp.X_train_, normalized_y=gp.y_train_), weights=self.ensemble.weights_,
            mean=self.ensemble.Y_mean_, scale=self.ensemble.Y_std_, fit_info=self.fit_info), protocol=4)

    def restore(self, payload):
        value = pickle.loads(payload)
        if 'target_state' in value:
            compact = value['target_state']
            target = pickle.loads(compact['model'])
            target.gp = target._get_gp()
            target.gp.fit(compact['x'], compact['normalized_y'])
            self.bank.target = target
        else:
            self.bank.target = value['target']
        self.ensemble = SimpleNamespace(model_list_=self.bank, weights_=value['weights'],
            Y_mean_=value['mean'], Y_std_=value['scale'], variance_mode='average')
        self.fit_info = value['fit_info']
        return self.fit_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Model workers require CPU-only visibility')
    from peptide_transfer_policy import effective_recipe
    recipe = effective_recipe(args.recipe)
    if recipe['method'] not in ('fsbo', 'rgpe'):
        parser.error('Unknown transfer model')
    wire = sys.stdout.buffer
    # Third-party constructors print status; keep the framed output channel binary-only.
    sys.stdout = sys.stderr
    backend = None
    while True:
        request = receive(sys.stdin.buffer)
        if request is None:
            break
        try:
            if request['op'] == 'close':
                send(wire, dict(ok=True))
                break
            if backend is None:
                backend = (FSBOBackend if recipe['method'] == 'fsbo' else RGPEBackend)(recipe)
            if request['op'] == 'fit':
                if isinstance(backend, RGPEBackend):
                    backend.bank.target = None
                result = backend.fit(request['x'], request['score'], request['seed'])
            elif request['op'] == 'sample':
                result = backend.sample(request['candidates'], request['count'], request['seed'])
            elif request['op'] == 'snapshot':
                result = backend.snapshot()
            elif request['op'] == 'restore':
                result = backend.restore(request['payload'])
            else:
                raise ValueError('Unknown model operation')
            send(wire, dict(result=result))
        except Exception:
            send(wire, dict(error=traceback.format_exc()))


if __name__ == '__main__':
    main()
