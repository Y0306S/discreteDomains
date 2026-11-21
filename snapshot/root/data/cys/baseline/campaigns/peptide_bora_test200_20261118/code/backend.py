"""CPU-only frozen author VAE, APEX oracle, and released EI numerical search."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
from common import ROOT, config, valid

sys.path.insert(0, str(ROOT/'vendor'))
from bora.acquisition import ExpectedImprovement


@contextmanager
def cwd(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class Backend:
    def __init__(self, cfg=None):
        import torch
        self.cfg = cfg or config()
        self.native = Path(self.cfg['cpu_upstream'])
        sys.path.insert(0, str(self.native))
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        from uniref_vae.load_vae import load_vae
        with cwd(self.native):
            self.vae, self.data = load_vae(self.cfg['vae'], max_string_length=30)
        self.vae.requires_grad_(False)
        self.cache = {}
        self._oracle = None

    def encode(self, sequences):
        import torch
        from uniref_vae.data import collate_fn
        missing = list(dict.fromkeys(s for s in sequences if s not in self.cache))
        with torch.inference_mode():
            for start in range(0, len(missing), 128):
                part = missing[start:start+128]
                encoded = [self.data.encode(s).unsqueeze(0) for s in self.data.tokenize_sequence(part)]
                mu, _ = self.vae.encode(collate_fn(encoded).cpu())
                vectors = mu.reshape(-1, 256).cpu().numpy().astype(np.float64)
                if not np.isfinite(vectors).all():
                    raise ValueError('Nonfinite author VAE encoding')
                self.cache.update(zip(part, vectors))
        return np.stack([self.cache[s] for s in sequences])

    def decode(self, vectors):
        import torch
        with torch.inference_mode():
            sample = self.vae.sample(z=torch.as_tensor(vectors, dtype=torch.float32).reshape(-1, 2, 128))
            return [self.data.decode(sample[i]) for i in range(sample.size(-2))]

    def score(self, sequences):
        import torch
        if self._oracle is None:
            with cwd(self.native):
                from apex_oracle.APEX_predict import apex_wrapper
            self._oracle = apex_wrapper
        with cwd(self.native), torch.inference_mode():
            values = self._oracle(sequences)
        values = np.asarray(values)[:, 0].astype(float)
        if len(values) != len(sequences) or not np.isfinite(values).all():
            raise RuntimeError('Invalid APEX result; no observations committed')
        return values


def edit_pool(reference, anchors, count, excluded, rng):
    """Bounded insert/delete/substitute candidates; exact constraints decide admission."""
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    result, seen = [], set(excluded)
    for _ in range(max(10000, count*200)):
        source = reference if rng.rand() < .5 else anchors[rng.randint(len(anchors))]
        seq = list(source)
        for _ in range(rng.randint(1, max(2, int(len(reference)*.25)+1))):
            action = rng.choice(['substitute', 'insert', 'delete'], p=[.8, .1, .1])
            if action == 'insert' and len(seq) < 30:
                seq.insert(rng.randint(len(seq)+1), alphabet[rng.randint(20)])
            elif action == 'delete' and len(seq) > 1:
                seq.pop(rng.randint(len(seq)))
            else:
                seq[rng.randint(len(seq))] = alphabet[rng.randint(20)]
        sequence = ''.join(seq)
        if sequence not in seen and valid(sequence, reference):
            seen.add(sequence)
            result.append(sequence)
            if len(result) == count:
                return result
    raise RuntimeError('Unable to form a distinct feasible candidate pool')


class LatentSpace:
    constraint = None
    _allow_duplicate_points = False
    _discretization_steps = None

    def __init__(self, x, y, rng):
        self.x, self.y, self.rng = x, y, rng
        # Bounds use the shared initialization only, frozen before optimization.
        self.bounds = np.array([[-1., 1.]]*x.shape[1])
        self.sampled = []

    def __len__(self):
        return len(self.y)

    @property
    def empty(self):
        return not len(self)

    def _target_max(self):
        return float(np.max(self.y))

    def random_points(self, count):
        points = self.rng.uniform(-1, 1, size=(count, self.x.shape[1]))
        self.sampled.append(points)
        return points

    def __contains__(self, point):
        return bool(np.any(np.all(np.isclose(self.x, point, rtol=0, atol=1e-12), axis=1)))


def bo_candidates(backend, gp, x, y, low, scale, reference, observed, rng, cfg, count):
    """Released EI/random/L-BFGS-B, then exact decoded-sequence admission.

    A valid reference-edit pool supplements infeasible decodes. Every admitted
    sequence is canonically re-encoded and ranked by the same fitted GP/EI.
    """
    begun = time.monotonic()
    acq = ExpectedImprovement(xi=cfg['acquisition']['xi'], random_state=rng)
    space = LatentSpace(x, y, rng)
    vectors = acq.suggest(gp, space, n_random=cfg['acquisition']['n_random'],
        n_l_bfgs_b=cfg['acquisition']['n_l_bfgs_b'], fit_gp=False, n=count)
    random_points = space.sampled[0]
    f = acq._get_acq(gp, constraint=None)
    # Bound prediction batches so 4k observations do not create enormous temporaries.
    scores = np.concatenate([f(a) for a in np.array_split(random_points, max(1, len(random_points)//256))])
    top = random_points[np.argsort(scores)[:cfg['acquisition']['decoded_candidates']]]
    latent = np.concatenate([np.asarray(vectors).reshape(-1, x.shape[1]), top])
    decoded = backend.decode(low+((latent+1)/2)*scale)
    excluded = set(observed)
    feasible = list(dict.fromkeys(s for s in decoded if s not in excluded and valid(s, reference)))
    top_ids = np.argsort(-y)[:min(20, len(y))]
    anchors = [observed[i] for i in top_ids]
    supplement = edit_pool(reference, anchors, cfg['acquisition']['valid_edit_supplement'],
        excluded | set(feasible), rng)
    candidates = feasible+supplement
    z = 2*(backend.encode(candidates)-low)/scale-1
    values = f(z)
    ids = np.argsort(values)[:count]
    backend.cache = {s: backend.cache[s] for s in set(observed+candidates) if s in backend.cache}
    return [candidates[i] for i in ids], dict(seconds=time.monotonic()-begun,
        decoded=len(decoded), feasible_decoded=len(feasible), feasible_edit_candidates=len(supplement),
        chosen_sources=['decoded_latent' if i < len(feasible) else 'EI_ranked_valid_edits' for i in ids])
