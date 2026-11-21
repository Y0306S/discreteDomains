#!/usr/bin/env python3
"""Isolated query FSBO fit/posterior process, using the peptide target refit."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


class QueryFSBOModel:
    def __init__(self, recipe):
        import torch
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            raise ValueError('Query FSBO model workers must be CPU-only')
        sys.path.insert(0, recipe['fsbo_repo'])
        from fsbo_modules import DeepKernelGP
        self.torch, self.recipe = torch, recipe
        self.model = DeepKernelGP(64, os.devnull, recipe['seed'], hidden_size=[128, 128],
            kernel='rbf', ard=False, checkpoint=str(Path(recipe['fsbo_weights']).parent),
            epochs=recipe['target_max_updates'], lr=0.001)
        weights = torch.load(recipe['fsbo_weights'], map_location='cpu')
        for name, key in [('model', 'gp'), ('likelihood', 'likelihood'), ('feature_extractor', 'net')]:
            getattr(self.model, name).load_state_dict(weights[key], strict=True)
        self.initial = copy.deepcopy(self.model.state_dict())
        self.fitted = False

    def fit(self, x, score, seed):
        from fsbo_target_refit import fit_target
        torch = self.torch
        x, score = np.asarray(x, dtype=np.float32), np.asarray(score, dtype=np.float32)
        if (score.ndim != 1 or len(score) < 1 or x.shape != (len(score), 64)
                or not np.isfinite(x).all() or not np.isfinite(score).all()):
            raise ValueError('Invalid query target observations')
        self.offset, self.scale = float(score.min()), float(np.ptp(score)) or 1.
        self.model.load_state_dict(self.initial, strict=True)
        self.fitted = False
        torch.manual_seed(seed)
        info = fit_target(self.model, torch.as_tensor(x),
            torch.as_tensor((score-self.offset)/self.scale), self.recipe['target_max_updates'],
            patience=16, min_delta=0.0001, lr=0.001, seed=seed)
        self.fitted = True
        return dict({k: v for k, v in info.items() if k != 'trace'},
            target_observations=len(score), method='fsbo', dimension=64,
            checkpoint_sha256=self.recipe['fsbo_weights_sha256'], label_scaling='target_minmax',
            target_policy='first_distinct_uncensored_canonical_plan')

    def sample(self, candidates, count, seed):
        torch = self.torch
        x = np.asarray(candidates, dtype=np.float32)
        if (not self.fitted or x.ndim != 2 or x.shape[1] != 64
                or not 1 <= count <= len(x) or not np.isfinite(x).all()):
            raise ValueError('Invalid query posterior request')
        # Same joint posterior rsample as the first peptide FSBO wrapper.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            posterior = self.model.model(self.model.feature_extractor(torch.as_tensor(x)))
            draws = posterior.rsample(torch.Size([count])).cpu().numpy()
        if not np.isfinite(draws).all():
            raise ValueError('Nonfinite query FSBO posterior')
        return dict(draws=draws*self.scale+self.offset,
                    info=dict(correlated=True, higher_is_better=True, candidates=len(x), samples=count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    for path, expected in recipe['pinned_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned query FSBO file changed: ' + path)
    # Reuse the existing peptide binary framing, without importing its models.
    from peptide_transfer_models import receive, send
    model = QueryFSBOModel(recipe)
    while True:
        request = receive(sys.stdin.buffer)
        if request is None or request['op'] == 'close':
            break
        try:
            op = request.pop('op')
            if op not in ('fit', 'sample'):
                raise ValueError('Unknown query model operation')
            send(sys.stdout.buffer, dict(result=getattr(model, op)(**request)))
        except Exception as exc:
            send(sys.stdout.buffer, dict(error=type(exc).__name__ + ': ' + str(exc)))


if __name__ == '__main__':
    main()
