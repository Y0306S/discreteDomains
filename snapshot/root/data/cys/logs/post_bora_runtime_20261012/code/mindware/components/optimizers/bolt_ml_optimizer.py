"""BOLT learned starts followed by the existing ML LOLBO sampler."""

import math
from dataclasses import asdict
from pathlib import Path

import optuna
from ConfigSpace import Configuration
from openbox.utils.constants import SUCCESS, TIMEOUT

from mindware.components.optimizers.bolt_ml_initializer import BOLTMLInitializer
from mindware.components.optimizers.lolbo_sampler import LOLBOTurboSampler
from mindware.components.optimizers.online_upstream import source_digest
from mindware.components.optimizers.pt4opt_optuna_optimizer import OptunaConfigSpaceMixin


class BOLTMLOptimizer(OptunaConfigSpaceMixin, BOLTMLInitializer):
    optimizer_label = "bolt_ml"
    initialization_evaluations = 10
    max_suggest_retries = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._algorithm_hp = self._get_hyperparameter("algorithm")
        self._algorithm_choices = self._get_choices(self._algorithm_hp)
        self._single_algorithm = self.fixed_algorithm
        sampler = LOLBOTurboSampler(seed=self.seed, device=self.params.get("lolbo_device"))
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.asset_identity.update(
            scope="10 learned initialization evaluations + 90 ML LOLBO evaluations",
            bolt_protocol="bolt-ml-lolbo-v1", initialization_evaluations=10,
            bo_evaluations=90, bo_backend="LOLBOTurboSampler", optuna_version=optuna.__version__,
            bo_settings={"n_init": sampler.n_init, "bsz": sampler.bsz,
                         "learning_rte": sampler.learning_rte,
                         "num_update_epochs": sampler.num_update_epochs,
                         "init_n_epochs": sampler.init_n_epochs,
                         "n_candidates": sampler.n_candidates},
            bo_sources={name: source_digest(Path(__file__).with_name(name)) for name in (
                "lolbo_sampler.py", "lolbo_turbo_port.py", "pt4opt_optuna_optimizer.py")})
        self._expected_bolt_identity = dict(self.asset_identity)

    def _validate_budget(self):
        if self.evaluation_num_limit != 100:
            raise ValueError("BOLT-ML requires 100 actual evaluations: 10 learned starts + 90 BO")
        if self.per_run_time_limit != 2400 or self.time_limit is not None:
            raise ValueError("BOLT-ML requires 2400 seconds per evaluation and no overall timeout")
        if not self.resolver.default_on_error:
            raise ValueError("BOLT-ML requires the agreed ConfigSpace default-repair policy")

    def bo_state(self):
        sampler = self.study.sampler
        return {"observations": sum(t.state == optuna.trial.TrialState.COMPLETE
                                    for t in self.study.get_trials(deepcopy=False)),
                "gp_fitted_observations": sampler._n_seen,
                "gp_initialized": sampler._model is not None,
                "queued_candidates": list(sampler._pending),
                "trust_region": asdict(sampler._tr_state) if sampler._tr_state else None}

    def _suggest(self):
        if self.trial_cnt < self.initialization_evaluations:
            proposal = super()._suggest()
            return {**proposal, "phase": "learned_initialization", "trial_id": None,
                    "supplier": "bolt_model"}

        for retry in range(self.max_suggest_retries):
            trial = self.study.ask()
            config = self._try_suggest_config_from_trial(trial)
            if config is not None:
                sampler = self.study.sampler
                supplier = "lolbo" if sampler._model is not None else "lolbo_startup_random"
                proposal = {"config": dict(config), "phase": "bo", "trial_id": trial._trial_id,
                            "supplier": supplier, "note": "BOLT BO via " + supplier}
                self._record_decision({"stage": "bo_proposal", "evaluation_id": self.trial_cnt,
                                       "optuna_trial": trial.number, "config": dict(config),
                                       "supplier": supplier, "invalid_retries": retry,
                                       "bo_state": self.bo_state()})
                return proposal
            trial.set_user_attr("rejection_reason", "invalid_configspace_combination")
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            self._record_decision({"stage": "invalid_bo_proposal", "evaluation_id": self.trial_cnt,
                                   "optuna_trial": trial.number, "params": trial.params})

        # Match the ML LOLBO baseline's recovery after four invalid suggestions.
        config = dict(self._sample_fallback_configuration())
        self._record_decision({"stage": "bo_recovery", "evaluation_id": self.trial_cnt,
                               "config": config, "invalid_retries": self.max_suggest_retries})
        return {"config": config, "phase": "bo", "trial_id": None,
                "supplier": "lolbo_invalid_random_fallback",
                "note": "BOLT BO: sampled valid config after four invalid LOLBO proposals"}

    def _evaluate(self, proposal):
        result = super()._evaluate(proposal)
        proposal["observed_state"] = int(result[2])
        return result

    def _after_observation(self, proposal, perf):
        trial = (optuna.trial.Trial(self.study, proposal["trial_id"])
                 if proposal["trial_id"] is not None else None)
        observed = proposal["observed_state"]
        state = (optuna.trial.TrialState.COMPLETE if observed == SUCCESS and math.isfinite(perf)
                 else optuna.trial.TrialState.PRUNED if observed == TIMEOUT
                 else optuna.trial.TrialState.FAIL)
        config = Configuration(self.config_space, values=proposal["config"])
        attrs = {"evaluation_id": self.trial_cnt - 1, "phase": proposal["phase"],
                 "supplier": proposal["supplier"], "evaluation_state": observed,
                 "evaluation_limit_seconds": self.per_run_time_limit}
        self._tell_config_result(config, -perf if math.isfinite(perf) else None,
                                 trial, state=state, attrs=attrs)
        self._record_decision({"stage": "bo_observation", **attrs,
                               "config": proposal["config"], "optuna_state": state.name,
                               "objective": -perf if math.isfinite(perf) else None})

    def _checkpoint_restore(self, extra):
        super()._checkpoint_restore(extra)
        if self.asset_identity != self._expected_bolt_identity:
            raise RuntimeError("BOLT checkpoint uses different model/backend settings; start a fresh run")
        records = [t for t in self.study.get_trials(deepcopy=False)
                   if "evaluation_id" in t.user_attrs]
        if sorted(t.user_attrs["evaluation_id"] for t in records) != list(range(self.trial_cnt)):
            raise RuntimeError("BOLT study and evaluation ledger disagree")
