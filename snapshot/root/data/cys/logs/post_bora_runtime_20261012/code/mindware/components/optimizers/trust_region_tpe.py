"""TPE with TuRBO-style trust region in latent space (VAE + trust-region TPE)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import FrozenTrial, TrialState

from mindware.components.optimizers.trust_region import (
    TrustRegionState,
    reseed_trust_region_from_observations,
    trust_region_bounds,
    update_trust_region_minimize,
    z_params_from_config,
)

_Z_PREFIX = "z_"
_BAD_VALUE = 1e10


class TrustRegionTPESampler(BaseSampler):
    """
    Optuna TPE sampler constrained to a local trust region around the incumbent.

    Samples all z_* parameters via ``sample_relative`` (one TPE fit per trial),
    not 256× ``sample_independent`` calls.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        n_startup_trials: int = 10,
        latent_bound: float = 4.0,
        tr_length: float = 0.8,
        warn_independent_sampling: bool = False,
        tpe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        tpe_kwargs = dict(tpe_kwargs or {})
        tpe_kwargs.setdefault("seed", int(seed))
        tpe_kwargs.setdefault("n_startup_trials", int(n_startup_trials))
        tpe_kwargs.setdefault("multivariate", True)
        tpe_kwargs.setdefault("warn_independent_sampling", warn_independent_sampling)
        if "warn_independent_sampling" in tpe_kwargs:
            tpe_kwargs.pop("warn_independent_sampling", None)
        self._tpe = TPESampler(**tpe_kwargs)
        self._rng = np.random.RandomState(int(seed))
        self.n_startup_trials = int(n_startup_trials)
        self.latent_bound = float(latent_bound)
        self.initial_tr_length = float(tr_length)
        self.tr_state: Optional[TrustRegionState] = None
        self._dim: Optional[int] = None
        self._bounds_low: Optional[np.ndarray] = None
        self._bounds_high: Optional[np.ndarray] = None

    def _z_dim(self) -> int:
        if self._dim is not None:
            return int(self._dim)
        return 0

    def _completed_feasible(self, study: "optuna.study.Study") -> list[FrozenTrial]:
        return [
            t
            for t in study.trials
            if t.state == TrialState.COMPLETE
            and t.value is not None
            and float(t.value) < _BAD_VALUE
            and np.isfinite(float(t.value))
        ]

    def _z_distributions(self, trial: FrozenTrial) -> Dict[str, FloatDistribution]:
        out: Dict[str, FloatDistribution] = {}
        for name, dist in trial.distributions.items():
            if not str(name).startswith(_Z_PREFIX):
                continue
            if isinstance(dist, FloatDistribution):
                out[str(name)] = dist
        return out

    def _init_dims_from_z(self, z_dists: Dict[str, FloatDistribution]) -> None:
        if not z_dists:
            return
        max_idx = max(int(str(name).split("_", 1)[1]) for name in z_dists)
        dim = max_idx + 1
        if self._dim is None:
            self._init_tr_state(dim)
        bound = float(self.latent_bound)
        for dist in z_dists.values():
            bound = max(bound, float(abs(dist.low)), float(abs(dist.high)))
        self._bounds_low = np.full(dim, -bound, dtype=np.float64)
        self._bounds_high = np.full(dim, bound, dtype=np.float64)

    def _ensure_bounds(self, param_distribution: FloatDistribution) -> None:
        if self._bounds_low is not None:
            return
        bound = float(max(abs(param_distribution.low), abs(param_distribution.high), self.latent_bound))
        dim = self._z_dim()
        if dim <= 0:
            dim = 1
        self._dim = dim
        self._bounds_low = np.full(dim, -bound, dtype=np.float64)
        self._bounds_high = np.full(dim, bound, dtype=np.float64)

    def _init_tr_state(self, dim: int) -> None:
        self._dim = int(dim)
        self.tr_state = TrustRegionState(dim=int(dim), length=self.initial_tr_length)

    def _prepare_tr_state(self, study: "optuna.study.Study") -> None:
        if self.tr_state is None:
            self._init_tr_state(max(1, self._z_dim()))
        if self.tr_state.center is None or self.tr_state.restart_triggered:
            self._reset_center_from_study(study)
            if self.tr_state.restart_triggered:
                self.tr_state.reset_length(self.initial_tr_length)

    def _tr_bounds_for_idx(self, idx: int) -> tuple[float, float]:
        if self.tr_state is None or self._bounds_low is None or self._bounds_high is None:
            bound = float(self.latent_bound)
            return -bound, bound
        center = self.tr_state.center
        if center is None:
            center = np.zeros(self._z_dim(), dtype=np.float64)
        tr_lb, tr_ub = trust_region_bounds(
            center,
            self._bounds_low,
            self._bounds_high,
            self.tr_state.length,
        )
        low = float(tr_lb[idx])
        high = float(tr_ub[idx])
        if high <= low:
            low, high = float(self._bounds_low[idx]), float(self._bounds_high[idx])
        return low, high

    def reseed_from_observations(self, observations: Sequence[Dict[str, Any]]) -> None:
        dim = self._z_dim()
        if dim <= 0:
            return
        if self.tr_state is None:
            self._init_tr_state(dim)
        reseed_trust_region_from_observations(
            self.tr_state,
            list(observations),
            initial_length=self.initial_tr_length,
            bad_value=_BAD_VALUE,
        )

    def infer_relative_search_space(
        self, study: "optuna.study.Study", trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        z_dists = self._z_distributions(trial)
        if not z_dists:
            return {}
        self._init_dims_from_z(z_dists)
        if len(self._completed_feasible(study)) < self.n_startup_trials:
            return dict(z_dists)
        self._prepare_tr_state(study)
        narrowed: Dict[str, BaseDistribution] = {}
        for name, dist in z_dists.items():
            idx = int(str(name).split("_", 1)[1])
            low, high = self._tr_bounds_for_idx(idx)
            narrowed[name] = FloatDistribution(low=low, high=high, log=dist.log)
        return narrowed

    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        if not search_space:
            return {}
        if len(self._completed_feasible(study)) < self.n_startup_trials:
            out: Dict[str, Any] = {}
            for name, dist in search_space.items():
                if isinstance(dist, FloatDistribution):
                    out[name] = float(self._rng.uniform(float(dist.low), float(dist.high)))
            return out
        return dict(self._tpe.sample_relative(study, trial, search_space))

    def sample_independent(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if str(param_name).startswith(_Z_PREFIX):
            if isinstance(param_distribution, FloatDistribution):
                return float(
                    self._rng.uniform(float(param_distribution.low), float(param_distribution.high))
                )
            return self._tpe.sample_independent(study, trial, param_name, param_distribution)
        return self._tpe.sample_independent(study, trial, param_name, param_distribution)

    def after_trial(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._tpe.after_trial(study, trial, state, values)
        if state != TrialState.COMPLETE or not values:
            return
        val = float(values[0])
        if not np.isfinite(val) or val >= _BAD_VALUE:
            return
        if self.tr_state is None:
            z_dim = sum(1 for k in trial.params if str(k).startswith(_Z_PREFIX))
            if z_dim <= 0:
                return
            self._init_tr_state(z_dim)
        z = z_params_from_config(trial.params, self.tr_state.dim)
        if self.tr_state.center is None:
            self.tr_state.center = z.copy()
            self.tr_state.best_value = val
        update_trust_region_minimize(self.tr_state, val)
        if val <= self.tr_state.best_value:
            self.tr_state.center = z.copy()

    def _reset_center_from_study(self, study: "optuna.study.Study") -> None:
        if self.tr_state is None:
            return
        best_trial: Optional[FrozenTrial] = None
        best_val = float("inf")
        for t in study.trials:
            if t.state != TrialState.COMPLETE or t.value is None:
                continue
            val = float(t.value)
            if val >= _BAD_VALUE or not np.isfinite(val):
                continue
            if val < best_val:
                best_val = val
                best_trial = t
        if best_trial is None:
            self.tr_state.center = np.zeros(self.tr_state.dim, dtype=np.float64)
            return
        self.tr_state.center = z_params_from_config(best_trial.params, self.tr_state.dim)
        self.tr_state.best_value = best_val


def build_trust_region_tpe_sampler(
    cs: Any,
    seed: int,
    n_startup_trials: int,
    latent_bound: Optional[float] = None,
    tr_length: float = 0.8,
) -> TrustRegionTPESampler:
    """Build sampler and register z parameter names on a throwaway study attr slot."""
    bound = float(latent_bound or 4.0)
    z_names: list[str] = []
    for hp in cs.get_hyperparameters():
        name = str(hp.name)
        if name.startswith(_Z_PREFIX):
            z_names.append(name)
            bound = max(bound, float(abs(hp.lower)), float(abs(hp.upper)))
    sampler = TrustRegionTPESampler(
        seed=int(seed),
        n_startup_trials=int(n_startup_trials),
        latent_bound=bound,
        tr_length=float(tr_length),
    )
    if z_names:
        sampler._init_tr_state(len(z_names))
        sampler._bounds_low = np.full(len(z_names), -bound, dtype=np.float64)
        sampler._bounds_high = np.full(len(z_names), bound, dtype=np.float64)
    return sampler
