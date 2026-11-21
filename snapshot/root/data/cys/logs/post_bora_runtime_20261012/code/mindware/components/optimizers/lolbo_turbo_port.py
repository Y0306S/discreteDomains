"""LOL-BO's trust-region local GP-BO, ported to run inside the benchmark env.

Provenance
----------
Ported from the official LOL-BO implementation (Maus et al., NeurIPS 2022),
https://github.com/nataliemaus/lolbo, local checkout at
``/root/data/cys/baseline/repos/lolbo`` (commit 0876147). Ported files:

  lolbo/utils/bo_utils/turbo.py   -> TurboState / update_state / generate_batch
  lolbo/utils/bo_utils/ppgpr.py   -> GPModelDKL
  lolbo/utils/bo_utils/base.py    -> DenseNetwork / Swish
  lolbo/utils/utils.py            -> update_surr_model

Why a port rather than the upstream package: upstream pins torch 1.11 / gpytorch
1.6 (env ``bl_lolbo_exact``), which cannot be imported into this py3.11 /
gpytorch 1.15 harness. The port keeps upstream's algorithm and default
hyperparameters but uses the modern gpytorch API, and drops the hard ``.cuda()``
calls so it runs on CPU too.

Scope: this is the ML/DL-HPO reduction of LOL-BO that ``pt4opt-baselines.md``
specifies -- "no VAE; feed the standardised hyperparameter vector straight to the
GP surrogate and pick the next batch inside an adaptive trust region centred on
the current best, i.e. LOL-BO minus the structure-representation layer, a
TuRBO-style local GP-BO". The latent-space half of LOL-BO is not used here.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass

import gpytorch
import torch
from botorch.generation import MaxPosteriorSampling
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.models import ApproximateGP
from gpytorch.mlls import PredictiveLogLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# lolbo/utils/bo_utils/base.py
# --------------------------------------------------------------------------- #
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, swish):
        if swish:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("swish", Swish()),
            ]))
        else:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("norm", torch.nn.BatchNorm1d(output_dim)),
                ("relu", torch.nn.ReLU(True)),
            ]))


class DenseNetwork(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dims, swish=True):
        prev_dims = [input_dim] + list(hidden_dims[:-1])
        layers = OrderedDict([
            (f"hidden{i + 1}", _LinearBlock(prev_dim, cur_dim, swish=swish))
            for i, (prev_dim, cur_dim) in enumerate(zip(prev_dims, hidden_dims))
        ])
        self.output_dim = hidden_dims[-1]
        super().__init__(layers)


# --------------------------------------------------------------------------- #
# lolbo/utils/bo_utils/ppgpr.py -- approximate GP with a deep kernel
# --------------------------------------------------------------------------- #
class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256)):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1), hidden_dims=hidden_dims,
        ).to(inducing_points.device)
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1  # must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x),
        )

    def __call__(self, x, *args, **kwargs):
        return super().__call__(self.feature_extractor(x), *args, **kwargs)

    def posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs):
        self.eval()
        self.likelihood.eval()
        return GPyTorchPosterior(distribution=self.likelihood(self(X)))


def make_surrogate(train_z, device):
    """lolbo.LOLBOState.initialize_surrogate_model, device-agnostic."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    n_pts = min(train_z.shape[0], 1024)
    model = GPModelDKL(train_z[:n_pts, :].to(device), likelihood=likelihood).to(device)
    mll = PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.size(-2))
    return model.eval().to(device), mll


def update_surr_model(model, mll, learning_rte, train_z, train_y, n_epochs, device):
    """lolbo/utils/utils.py::update_surr_model, device-agnostic."""
    model = model.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": learning_rte}], lr=learning_rte,
    )
    train_bsz = min(len(train_y), 128)
    loader = DataLoader(
        TensorDataset(train_z.to(device), train_y.to(device)),
        batch_size=train_bsz, shuffle=True,
    )
    for _ in range(n_epochs):
        for inputs, scores in loader:
            optimizer.zero_grad()
            loss = -mll(model(inputs.to(device)), scores.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model.eval()


# --------------------------------------------------------------------------- #
# lolbo/utils/bo_utils/turbo.py
# --------------------------------------------------------------------------- #
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:      # expand
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:    # shrink
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(state, model, X, Y, batch_size, n_candidates=None,
                   dtype=torch.float32, device=torch.device("cpu"), unit_cube=True,
                   cat_blocks=None):
    """Thompson-sampling candidate generation inside the trust region.

    Upstream also offers acqf="ei"; it falls back to "ts" on any exception and
    LOLBOState's default is "ts", so only the "ts" path is ported.

    ``unit_cube`` is a REQUIRED adaptation for the ML/DL-HPO reduction, not a
    stylistic choice. Upstream sets ``weights = ones * 8`` ("less than 4 stdevs on
    either side max"), which is calibrated for LOL-BO's *unbounded* VAE latent
    space where z is roughly unit-normal. Here the search space is the normalised
    hyperparameter cube [0, 1]^d, so that formula gives a trust region of
    center +/- 8*0.8/2 = +/- 3.2 -- wider than the whole cube. Every candidate
    then clips to the boundary and the "local" search degrades to something worse
    than uniform sampling (measured: worse than RandomSampler). With
    ``unit_cube=True`` we use the canonical TuRBO region for a unit cube,
    center +/- length/2 clamped to [0, 1], keeping upstream's TurboState schedule
    (length 0.8 initial, 0.5**7 min, 1.6 max) untouched.
    """
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()
    if unit_cube:
        weights = torch.ones_like(x_center)
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    else:
        weights = torch.ones_like(x_center) * 8  # upstream: unbounded latent space
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0

    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, max(dim - 1, 1), size=(len(ind),), device=device)] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Second required adaptation: one-hot categorical blocks need a Hamming-style
    # resample, not a continuous perturbation. In a one-hot block the incumbent
    # coordinate is 1.0 and the rest are 0.0, so a trust region of width
    # length/2 <= 0.4 keeps the incumbent coord in [0.6, 1.0] and the others in
    # [0, 0.4] -- argmax can never flip, and the categorical is frozen for the
    # whole run at whatever the init design happened to pick (measured: two of
    # three seeds locked onto a penalised category and never escaped). So for each
    # candidate we either keep the centre's category or draw a fresh one uniformly,
    # with the same prob_perturb that governs the continuous dims, and write a
    # clean 0/1 one-hot so the GP sees vectors of the same form it was trained on.
    for start, stop in (cat_blocks or []):
        width = stop - start
        if width <= 1:
            continue
        flip = torch.rand(n_candidates, device=device) <= prob_perturb
        n_flip = int(flip.sum())
        if n_flip == 0:
            continue
        choice = torch.randint(0, width, size=(n_flip,), device=device)
        block = X_cand[flip, start:stop]
        block.zero_()
        block[torch.arange(n_flip, device=device), choice] = 1.0
        X_cand[flip, start:stop] = block

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    return thompson_sampling(X_cand, num_samples=batch_size)
