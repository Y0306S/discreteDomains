import torch
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.generation.sampling import SamplingStrategy
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model
from torch import Tensor

class MaxPosteriorSampling(SamplingStrategy):
    """Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(self, model: Model, constraint_models: list=None, objective: MCAcquisitionObjective | None=None, replacement: bool=True, constrained: bool=False) -> None:
        """Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The objective. Typically, the AcquisitionObjective under which
                the samples are evaluated. If a ScalarizedObjective, samples from the
                scalarized posterior are used. Defaults to `IdentityMCObjective()`.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        if objective is None:
            objective = IdentityMCObjective()
        self.objective = objective
        self.replacement = replacement
        self.constraint_models = constraint_models
        self.constrained = constrained

    def forward(self, X: Tensor, num_samples: int=1, observation_noise: bool=False, max_constr_val: int=0) -> Tensor:
        """Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(X, observation_noise=observation_noise)
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        if self.constrained:
            if samples.shape[-1] > 1:
                constraint_samples = samples[:, :, 1:]
                samples = samples[:, :, 0].unsqueeze(-1)
            elif self.constraint_models is not None:
                all_constraint_samples = []
                for constr_model in self.constraint_models:
                    constraint_posterior = constr_model.posterior(X, observation_noise=observation_noise)
                    constr_samples = constraint_posterior.rsample(sample_shape=torch.Size([num_samples]))
                    all_constraint_samples.append(constr_samples)
                constraint_samples = torch.cat(all_constraint_samples, dim=-1)
            valid_samples = constraint_samples <= max_constr_val
            if valid_samples.shape[-1] > 1:
                valid_samples = torch.all(valid_samples, dim=-1).unsqueeze(-1)
            if valid_samples.sum() == 0:
                constraint_samples = constraint_samples.sum(dim=-1)
                min_idxs = torch.argmin(constraint_samples, dim=-1)
                min_violators = X[min_idxs, :]
                return min_violators
            samples = torch.where(valid_samples, samples, -torch.inf * torch.ones(samples.shape).cpu())
        obj = self.objective(samples, X=X)
        if self.replacement:
            idcs = torch.argmax(obj, dim=-1)
        else:
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                n_b = sub_idcs.size(-1)
                idcs = torch.stack([_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)], dim=-1)
            else:
                raise NotImplementedError('MaxPosteriorSampling without replacement for more than a single batch dimension is not yet implemented.')
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        return torch.gather(Xe, -2, idcs)
