"""Measure likelihood variability with fixed GP parameters and observations."""

import json
from pathlib import Path

import gpytorch
import numpy as np
import torch


class FixedGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


torch.set_num_threads(1)
torch.manual_seed(123)
x = torch.linspace(0, 1, 850).reshape(-1, 1)
y = torch.sin(x[:, 0] * 5)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = FixedGP(x, y, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
values = {}
with torch.no_grad():
    for name, limit in [('production_default', gpytorch.settings.max_cholesky_size.value()),
                        ('deterministic_cholesky_check', 1000)]:
        with gpytorch.settings.max_cholesky_size(limit):
            loss = [float(-mll(model(x), y)) for _ in range(8)]
        values[name] = dict(losses=loss, range=float(np.ptp(loss)), std=float(np.std(loss)))
result = dict(observations=850, parameters_and_observations_fixed=True, optimizer_updates=0,
              torch=torch.__version__, gpytorch=gpytorch.__version__, checks=values,
              production_recipe_modified=False)
Path(__file__).with_name('likelihood_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
