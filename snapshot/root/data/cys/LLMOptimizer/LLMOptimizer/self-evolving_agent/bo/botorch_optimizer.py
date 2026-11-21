from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BOResult:
    config: dict[str, Any]
    vector: np.ndarray
    source: str


class BoTorchOptimizer:
    """Minimal BoTorch wrapper with a deterministic random fallback path."""

    def __init__(
        self,
        adapter,
        seed: int = 1,
        use_random_fallback: bool = False,
    ):
        self.adapter = adapter
        self.rng = np.random.default_rng(seed)
        self.use_random_fallback = use_random_fallback
        if not self.use_random_fallback:
            self._check_dependencies()

    @staticmethod
    def _check_dependencies() -> None:
        try:
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "BoTorch and GPyTorch are required for this pipeline. "
                "Install them in the hpobench environment before running BO."
            ) from exc

    def _random_candidate(self) -> np.ndarray:
        return self.adapter.random_vector(self.rng)

    def suggest(
        self,
        observed_vectors: list[np.ndarray],
        observed_losses: list[float],
        candidate_budget: int = 1,
    ) -> list[BOResult]:
        if self.use_random_fallback or len(observed_vectors) < 2:
            results = []
            for _ in range(candidate_budget):
                vector = self._random_candidate()
                results.append(
                    BOResult(
                        config=self.adapter.vector_to_config(vector),
                        vector=vector,
                        source="random",
                    )
                )
            return results

        import torch
        from botorch.acquisition import qExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        train_x = torch.tensor(np.asarray(observed_vectors), dtype=torch.double)
        train_y = torch.tensor(np.asarray(observed_losses), dtype=torch.double).unsqueeze(-1)
        train_y = -train_y

        # Standardize targets manually for stable small-data behavior.
        y_mean = train_y.mean()
        y_std = train_y.std().clamp_min(1e-8)
        train_y_norm = (train_y - y_mean) / y_std

        model = SingleTaskGP(train_x, train_y_norm)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = train_y_norm.max()
        bounds = torch.stack(
            [
                torch.zeros(train_x.shape[-1], dtype=torch.double),
                torch.ones(train_x.shape[-1], dtype=torch.double),
            ]
        )
        acqf = qExpectedImprovement(model=model, best_f=best_f)
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=candidate_budget,
            num_restarts=10,
            raw_samples=128,
        )

        results = []
        for row in candidate.detach().cpu().numpy():
            results.append(
                BOResult(
                    config=self.adapter.vector_to_config(row),
                    vector=np.asarray(row, dtype=np.float64),
                    source="bo",
                )
            )
        return results
