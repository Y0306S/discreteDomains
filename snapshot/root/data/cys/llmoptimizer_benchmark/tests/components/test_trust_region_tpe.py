"""Unit tests for trust-region TPE sampler."""

from __future__ import annotations

import optuna
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from mindware.components.optimizers.trust_region import (
    reseed_trust_region_from_observations,
    trust_region_bounds,
    update_trust_region_minimize,
    TrustRegionState,
)
from mindware.components.optimizers.trust_region_tpe import build_trust_region_tpe_sampler
from mindware.domains.base import latent_config_space


def test_trust_region_bounds_matches_bolt_scaling():
    center = [0.0, 1.0]
    lb, ub = (-4.0, -4.0)
    hb = (4.0, 4.0)
    tr_lb, tr_ub = trust_region_bounds(center, lb, hb, length=0.8)
    # width=8, half=3.2
    assert tr_lb[0] == -3.2
    assert tr_ub[0] == 3.2
    assert tr_lb[1] == -2.2
    assert tr_ub[1] == 4.0  # clipped to global ub


def test_trust_region_shrinks_on_failures():
    state = TrustRegionState(dim=4, length=0.8, failure_tolerance=2)
    update_trust_region_minimize(state, 10.0)
    state.center = [0.0, 0.0, 0.0, 0.0]
    update_trust_region_minimize(state, 11.0)
    update_trust_region_minimize(state, 12.0)
    assert state.length == 0.4


def test_trust_region_sampler_narrows_after_startup():
    cs = latent_config_space(dim=4, bound=4.0)
    sampler = build_trust_region_tpe_sampler(cs, seed=1, n_startup_trials=3)

    def objective(trial: optuna.Trial) -> float:
        vals = []
        for hp in cs.get_hyperparameters():
            vals.append(trial.suggest_float(hp.name, float(hp.lower), float(hp.upper)))
        # synthetic bowl; optimum at origin
        return float(sum(v * v for v in vals))

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=8)

    assert sampler.tr_state is not None
    assert sampler.tr_state.length <= 0.8
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(completed) == 8


def test_reseed_from_observations():
    state = TrustRegionState(dim=2, length=0.8, failure_tolerance=1)
    obs = [
        {"config": {"z_0": 1.0, "z_1": 0.0}, "objectives": [5.0]},
        {"config": {"z_0": 0.5, "z_1": 0.0}, "objectives": [4.0]},
        {"config": {"z_0": 0.0, "z_1": 0.0}, "objectives": [1.0]},
    ]
    reseed_trust_region_from_observations(state, obs, initial_length=0.8)
    assert state.best_value == 1.0
    assert state.center is not None
    assert float(state.center[0]) == 0.0
