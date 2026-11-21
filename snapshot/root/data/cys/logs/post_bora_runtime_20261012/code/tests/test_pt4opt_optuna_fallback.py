"""Regression tests for the Optuna ask/tell fallback path.

Bug 1: when Optuna's suggestions could not form a valid Configuration, the code
returned a *randomly sampled* configuration but still told the original trial the
resulting objective. The sampler then learned
"(params the trial suggested) -> (score of a different config)".

Bug 2: that fallback silently consumed a full evaluation and a trial slot with no
record, converting optimizer budget into random search.

The trigger in practice is **forbidden clauses** (see liblinear_svc.py, which has
12 of them). Optuna suggests each hyperparameter independently and cannot honour a
joint constraint, so it produces forbidden combinations regularly.

Note that bounded retry alone is NOT sufficient: a FAIL trial is excluded from
TPE's model, so the sampler never learns a region is infeasible, and univariate TPE
will keep proposing the same forbidden pair indefinitely. Hence the repair path.
"""

import optuna
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from mindware.components.optimizers.pt4opt_optuna_optimizer import (
    OptunaConfigSpaceMixin,
    PT4OptOptunaOptimizer,
)
from mindware.components.optimizers.utils import MyConfiguration

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _forbidden_space():
    """Mirrors the liblinear_svc pattern: penalty=l1 together with loss=hinge is
    forbidden. Defaults must be a legal combination or ConfigSpace rejects the
    clause at construction time."""
    cs = ConfigurationSpace(seed=0)
    alg = CategoricalHyperparameter("algorithm", ["lsvc"])
    pen = CategoricalHyperparameter("lsvc:penalty", ["l1", "l2"], default_value="l2")
    loss = CategoricalHyperparameter("lsvc:loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
    cs.add_hyperparameters([alg, pen, loss])
    cs.add_forbidden_clause(
        ForbiddenAndConjunction(ForbiddenEqualsClause(pen, "l1"), ForbiddenEqualsClause(loss, "hinge"))
    )
    return cs


def _deep_conditional_space():
    cs = ConfigurationSpace(seed=0)
    alg = CategoricalHyperparameter("algorithm", ["svm"])
    ker = CategoricalHyperparameter("svm:kernel", ["rbf"])
    gam = UniformFloatHyperparameter("svm:gamma", 1e-4, 1.0, log=True)
    cs.add_hyperparameters([alg, ker, gam])
    cs.add_condition(EqualsCondition(ker, alg, "svm"))
    cs.add_condition(EqualsCondition(gam, ker, "rbf"))
    return cs


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    debug = info = warning


class _Harness(OptunaConfigSpaceMixin):
    """Minimal stand-in exposing only what the code under test touches."""

    max_suggest_retries = PT4OptOptunaOptimizer.max_suggest_retries
    supports_external_trial_injection = True

    _safe_tell = PT4OptOptunaOptimizer._safe_tell
    _enqueue_config = PT4OptOptunaOptimizer._enqueue_config
    _ask_valid_trial = PT4OptOptunaOptimizer._ask_valid_trial

    def __init__(self, config_space, study, algorithm_choices):
        self.config_space = config_space
        self.study = study
        self.logger = _Logger()
        self._algorithm_hp = config_space["algorithm"]
        self._algorithm_choices = list(algorithm_choices)
        self._single_algorithm = self._algorithm_choices[0] if len(self._algorithm_choices) == 1 else None
        self.invalid_suggest_cnt = 0
        self.fallback_cnt = 0
        self.external_injected_cnt = 0


def _harness(space_fn, choices, seed=1):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    return _Harness(space_fn(), study, choices), study


def test_deep_conditional_space_never_falls_back():
    h, study = _harness(_deep_conditional_space, ["svm"])
    for _ in range(10):
        trial = study.ask()
        cfg = h._try_suggest_config_from_trial(trial)
        assert cfg is not None
        assert "svm:gamma" in cfg.get_dictionary(), "innermost conditional HP was never assigned"
        study.tell(trial, 0.0)
    assert not h.logger.warnings


def test_invalid_suggestion_returns_none_not_a_substitute():
    h, study = _harness(_forbidden_space, ["lsvc"])
    h._build_config_from_trial = lambda trial: (_ for _ in ()).throw(ValueError("boom"))
    assert h._try_suggest_config_from_trial(study.ask()) is None
    assert any("invalid ConfigSpace configuration" in w for w in h.logger.warnings)


def test_infrastructure_errors_propagate_and_are_not_mistaken_for_invalid_configs():
    """A missing sampler backend (e.g. ImportError: No module named 'cmaes') must
    NOT be swallowed as "invalid configuration". Doing so silently degrades the
    optimizer into random sampling -- observed as 24/25 repairs before this was
    narrowed from a bare `except Exception`."""
    h, study = _harness(_forbidden_space, ["lsvc"])

    h._build_config_from_trial = lambda trial: (_ for _ in ()).throw(ImportError("No module named 'cmaes'"))
    try:
        h._try_suggest_config_from_trial(study.ask())
    except ImportError:
        pass
    else:
        raise AssertionError("ImportError was swallowed and reported as an invalid config")

    h._build_config_from_trial = lambda trial: (_ for _ in ()).throw(RuntimeError("sampler exploded"))
    try:
        h._try_suggest_config_from_trial(study.ask())
    except RuntimeError:
        pass
    else:
        raise AssertionError("RuntimeError was swallowed and reported as an invalid config")


def test_backcompat_shim_still_substitutes():
    """The hybrid LLM/TPE subclass calls _suggest_config_from_trial and relies on
    getting a configuration unconditionally."""
    h, study = _harness(_forbidden_space, ["lsvc"])
    h._build_config_from_trial = lambda trial: (_ for _ in ()).throw(ValueError("boom"))
    assert isinstance(h._suggest_config_from_trial(study.ask()), MyConfiguration)


def test_forbidden_clause_actually_triggers_the_fallback():
    """Guards the premise: if Optuna stopped producing forbidden combos, the repair
    path would be dead code and this suite would silently stop testing anything."""
    h, study = _harness(_forbidden_space, ["lsvc"])
    invalid = 0
    for _ in range(40):
        trial = study.ask()
        if h._try_suggest_config_from_trial(trial) is None:
            invalid += 1
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, 0.0)
    assert invalid > 0, "forbidden combination was never produced; premise broken"


def test_every_evaluation_is_attributed_to_a_matching_trial():
    """Bug 1 + bug 2 together: across many iterations every configuration handed
    back for evaluation is owned by a trial whose params equal it, and no evaluation
    is spent without attribution."""
    h, study = _harness(_forbidden_space, ["lsvc"])

    attributed = repaired = 0
    for _ in range(40):
        trial, cfg, was_repaired = h._ask_valid_trial()
        assert cfg is not None, "no configuration to evaluate"
        repaired += bool(was_repaired)

        if trial is not None:
            cfg_dict = cfg.get_dictionary()
            for k, v in trial.params.items():
                assert cfg_dict.get(k) == v, f"trial param {k}={v!r} != config {cfg_dict.get(k)!r}"
            h._safe_tell(trial, 0.2 if cfg_dict["lsvc:penalty"] == "l1" else 0.8)
            attributed += 1

    assert attributed == 40, f"only {attributed}/40 evaluations were attributed"
    assert repaired > 0, "repair path never exercised; forbidden combos not reached"

    # No completed trial may sit in the forbidden corner.
    forbidden = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.params.get("lsvc:penalty") == "l1"
        and t.params.get("lsvc:loss") == "hinge"
    ]
    assert forbidden == [], f"{len(forbidden)} forbidden configs entered the model"


def test_retry_alone_would_degenerate():
    """Documents *why* the repair path exists: with FAIL-only handling, TPE locks
    onto the forbidden corner after n_startup_trials and never escapes."""
    h, study = _harness(_forbidden_space, ["lsvc"])
    tail_invalid = 0
    for i in range(30):
        trial = study.ask()
        cfg = h._try_suggest_config_from_trial(trial)
        if cfg is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            if i >= 15:
                tail_invalid += 1
        else:
            study.tell(trial, 0.0)
    # Absorbing state: essentially every late trial is the same invalid suggestion.
    assert tail_invalid >= 10, (
        f"expected TPE to lock onto the forbidden corner, got {tail_invalid}/15 invalid in the tail"
    )


def test_enqueue_config_round_trips():
    h, study = _harness(_forbidden_space, ["lsvc"])
    raw = h.config_space.sample_configuration()
    cfg = MyConfiguration(raw[0] if isinstance(raw, list) else raw)

    params = h._config_to_optuna_params(cfg)
    # single-choice 'algorithm' is never asked of the trial, so it is not a param
    assert "algorithm" not in params
    assert "lsvc:penalty" in params

    trial = h._enqueue_config(cfg)
    assert trial is not None
    cfg_dict = cfg.get_dictionary()
    for k, v in trial.params.items():
        assert cfg_dict.get(k) == v
    assert h.external_injected_cnt == 1


def test_cmaes_opts_out_of_injection():
    from mindware.components.optimizers.pt4opt_cmaes_optimizer import CMAESOptimizer
    from mindware.components.optimizers.pt4opt_tpe_optimizer import TPEOptimizer

    assert PT4OptOptunaOptimizer.supports_external_trial_injection is True
    assert TPEOptimizer.supports_external_trial_injection is True
    assert CMAESOptimizer.supports_external_trial_injection is False


def test_cmaes_repair_is_unattributed_not_mispaired():
    """With injection disabled the repaired config must come back with trial=None --
    recorded in the trajectory but never told to a foreign trial."""
    h, study = _harness(_forbidden_space, ["lsvc"])
    h.supports_external_trial_injection = False
    h._build_config_from_trial = lambda trial: (_ for _ in ()).throw(ValueError("boom"))

    trial, cfg, was_repaired = h._ask_valid_trial()
    assert was_repaired is True
    assert cfg is not None
    assert trial is None, "CMA-ES must not receive an injected trial"
    assert h.fallback_cnt == 1
    assert h.external_injected_cnt == 0
