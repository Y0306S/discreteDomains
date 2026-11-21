"""Optuna feedback for bounded ML evaluations, without synthetic objective scores."""

import copy
import math
import os

import optuna
from openbox.utils.constants import SUCCESS, TIMEOUT


def runtime_constraints(trial):
    return (1.0 if trial.user_attrs.get("evaluation_status") == "TIMEOUT" else 0.0,)


def tpe_sampler_params(params):
    result = dict(params)
    callback = result.get("constraints_func")
    if callback is not None and callback is not runtime_constraints:
        raise ValueError("ML timeout feedback requires the runtime constraint callback")
    result["constraints_func"] = runtime_constraints
    return result


def observation_feedback(observation, timeout_seconds):
    value = float(observation.objectives[0]) if observation.objectives else math.inf
    if observation.trial_state == SUCCESS and math.isfinite(value):
        status, state, objective = "SUCCESS", optuna.trial.TrialState.COMPLETE, value
    elif observation.trial_state == TIMEOUT:
        status, state, objective = "TIMEOUT", optuna.trial.TrialState.PRUNED, None
    else:
        status, state, objective = "FAILED", optuna.trial.TrialState.FAIL, None
    return objective, state, {
        "evaluation_status": status,
        "evaluation_seconds": float(observation.elapsed_time or 0.0),
        "evaluation_limit_seconds": timeout_seconds,
        "runtime_policy_id": os.environ.get("ML_RUNTIME_POLICY_ID"),
    }


def upgrade_runtime_study(study):
    """Preserve an in-memory sampler's RNG and trial IDs when enabling constraints."""
    if not isinstance(study.sampler, optuna.samplers.TPESampler):
        return study
    callback = getattr(study.sampler, "_constraints_func", None)
    if callback is runtime_constraints:
        return study
    if callback is not None:
        raise RuntimeError("Cannot replace an existing TPE constraint policy")
    if not isinstance(study._storage, optuna.storages.InMemoryStorage):
        raise RuntimeError("Runtime constraint migration requires an in-memory ML study")
    sampler = copy.deepcopy(study.sampler)
    # Optuna has no public setter. Keep the sampler state; validate this adapter
    # against the installed Optuna version in the timeout regression tests.
    sampler._constraints_func = runtime_constraints
    updated = optuna.create_study(study_name=study.study_name, directions=study.directions,
                                  sampler=sampler, pruner=copy.deepcopy(study.pruner))
    for trial in study.get_trials(deepcopy=True):
        if trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            trial.system_attrs["constraints"] = runtime_constraints(trial)
        updated.add_trial(trial)
    if [(t.number, t._trial_id) for t in updated.trials] != [(t.number, t._trial_id) for t in study.trials]:
        raise RuntimeError("Runtime constraint migration changed pending trial IDs")
    for key, value in study.user_attrs.items():
        updated.set_user_attr(key, value)
    return updated
