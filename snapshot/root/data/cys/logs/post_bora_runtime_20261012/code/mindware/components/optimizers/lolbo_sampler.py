"""Optuna sampler exposing LOL-BO's trust-region local GP-BO.

Wrapping the ported algorithm as an ``optuna.samplers.BaseSampler`` means LOL-BO
reuses exactly the same ask/tell, config-space round-trip, failure handling,
history recording and trajectory bookkeeping as the TPE and CMA-ES baselines, so
the only thing that differs between those runs is the proposal mechanism.

Encoding: the search space is mapped to [0, 1]^d -- one dim per float/int
hyperparameter (log-scaled where the distribution is), and one-hot dims per
categorical. One-hot rather than an index keeps the RBF kernel from inventing an
ordering over unordered choices. Decoding rounds ints and takes the argmax over
each categorical's block.
"""

import numpy as np
import optuna
import torch
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace

from mindware.components.optimizers.lolbo_turbo_port import (
    TurboState,
    generate_batch,
    make_surrogate,
    update_state,
    update_surr_model,
)


class _Codec:
    """Bidirectional map between an Optuna search space and [0, 1]^d."""

    def __init__(self, search_space):
        self.names = list(search_space.keys())
        self.dists = search_space
        self.slices = {}
        self.cat_blocks = []      # (start, stop) of each one-hot block
        d = 0
        for name in self.names:
            dist = search_space[name]
            is_cat = isinstance(dist, CategoricalDistribution)
            width = len(dist.choices) if is_cat else 1
            self.slices[name] = slice(d, d + width)
            if is_cat:
                self.cat_blocks.append((d, d + width))
            d += width
        self.dim = d

    def encode(self, params):
        v = np.zeros(self.dim, dtype=np.float64)
        for name in self.names:
            dist, sl = self.dists[name], self.slices[name]
            val = params[name]
            if isinstance(dist, CategoricalDistribution):
                v[sl][list(dist.choices).index(val)] = 1.0
            else:
                v[sl.start] = self._unit(dist, val)
        return v

    def decode(self, v):
        out = {}
        for name in self.names:
            dist, sl = self.dists[name], self.slices[name]
            if isinstance(dist, CategoricalDistribution):
                out[name] = dist.choices[int(np.argmax(v[sl]))]
            else:
                out[name] = self._value(dist, float(np.clip(v[sl.start], 0.0, 1.0)))
        return out

    @staticmethod
    def _unit(dist, val):
        lo, hi = dist.low, dist.high
        if getattr(dist, "log", False):
            if val <= 0 or lo <= 0:
                return 0.0
            return (np.log(val) - np.log(lo)) / max(np.log(hi) - np.log(lo), 1e-12)
        return (val - lo) / max(hi - lo, 1e-12)

    @staticmethod
    def _value(dist, u):
        lo, hi = dist.low, dist.high
        if getattr(dist, "log", False) and lo > 0:
            val = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
        else:
            val = float(lo + u * (hi - lo))
        val = float(np.clip(val, lo, hi))
        if isinstance(dist, IntDistribution):
            step = dist.step or 1
            val = int(round((val - lo) / step)) * step + lo
            return int(np.clip(val, lo, hi))
        step = getattr(dist, "step", None)
        if step:
            val = round((val - lo) / step) * step + lo
        return float(np.clip(val, lo, hi))


class LOLBOTurboSampler(BaseSampler):
    """LOL-BO (minus the VAE) as an Optuna sampler.

    Defaults mirror ``LOLBOState``: DKL GP surrogate, Thompson sampling inside an
    adaptive trust region, ``learning_rte=0.01``, ``num_update_epochs=2``,
    ``init_n_epochs=20``, ``bsz=10``.
    """

    def __init__(self, seed=1, n_init=10, bsz=10, learning_rte=0.01,
                 num_update_epochs=2, init_n_epochs=20, device=None,
                 n_candidates=None, **_ignored):
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.bsz = int(bsz)
        self.learning_rte = float(learning_rte)
        self.num_update_epochs = int(num_update_epochs)
        self.init_n_epochs = int(init_n_epochs)
        self.n_candidates = n_candidates
        self._device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._rs = RandomSampler(seed=self.seed)
        self._space = IntersectionSearchSpace()
        self._codec = None
        self._tr_state = None
        self._model = None
        self._mll = None
        self._n_seen = 0
        self._pending = []          # decoded candidates not yet handed out
        self._initial_fit_done = False

    # -- Optuna plumbing ---------------------------------------------------- #
    def infer_relative_search_space(self, study, trial):
        space = {}
        for name, dist in self._space.calculate(study).items():
            if isinstance(dist, (FloatDistribution, IntDistribution, CategoricalDistribution)):
                space[name] = dist
        return space

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._rs.sample_independent(study, trial, param_name, param_distribution)

    def reseed_rng(self):
        self._rs.reseed_rng()

    # -- the actual optimiser ---------------------------------------------- #
    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}

        completed = [
            t for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and np.isfinite(t.value)
            and all(n in t.params for n in search_space)
        ]
        if len(completed) < self.n_init:
            return {}                       # warm-up: independent random sampling

        if self._codec is None or set(self._codec.names) != set(search_space):
            self._codec = _Codec(search_space)
            self._tr_state = None
            self._model = None
            self._pending = []
            self._n_seen = 0

        # hand out an already-generated batch member before refitting
        if self._pending:
            return self._pending.pop(0)

        X = np.stack([self._codec.encode(t.params) for t in completed])
        # study direction is minimise; LOL-BO/TuRBO maximise
        Y = np.array([-float(t.value) for t in completed])

        train_z = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(Y, dtype=torch.float32)

        if self._model is None:
            torch.manual_seed(self.seed)
            self._model, self._mll = make_surrogate(train_z, self._device)
            self._tr_state = TurboState(
                dim=self._codec.dim, batch_size=self.bsz,
                best_value=float(train_y.max()),
            )
            self._initial_fit_done = False

        # trust-region update from whatever arrived since the last proposal
        if self._n_seen and len(completed) > self._n_seen:
            new_y = train_y[self._n_seen:]
            if len(new_y):
                self._tr_state = update_state(self._tr_state, new_y)
                if self._tr_state.restart_triggered:
                    self._tr_state = TurboState(
                        dim=self._codec.dim, batch_size=self.bsz,
                        best_value=float(train_y.max()),
                    )
        self._n_seen = len(completed)

        n_epochs = self.num_update_epochs if self._initial_fit_done else self.init_n_epochs
        self._model = update_surr_model(
            self._model, self._mll, self._learning_rate(), train_z, train_y,
            n_epochs, self._device,
        )
        self._initial_fit_done = True

        with torch.no_grad():
            cand = generate_batch(
                self._tr_state, self._model,
                train_z.to(self._device), train_y.to(self._device),
                batch_size=self.bsz, n_candidates=self.n_candidates,
                device=self._device, cat_blocks=self._codec.cat_blocks,
            )
        cand = cand.detach().cpu().numpy()
        self._pending = [self._codec.decode(np.clip(row, 0.0, 1.0)) for row in cand]
        return self._pending.pop(0) if self._pending else {}

    def _learning_rate(self):
        return self.learning_rte
