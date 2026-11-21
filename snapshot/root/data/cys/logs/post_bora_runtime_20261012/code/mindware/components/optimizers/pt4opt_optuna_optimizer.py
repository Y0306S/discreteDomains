import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
from openbox.utils.constants import SUCCESS

try:
    from ConfigSpace.hyperparameters import OrdinalHyperparameter
except ImportError:
    OrdinalHyperparameter = ()

try:
    from ConfigSpace.exceptions import ForbiddenValueError
except ImportError:  # pragma: no cover - older/newer ConfigSpace layouts
    ForbiddenValueError = ValueError

# Exceptions that genuinely mean "these values are not a valid Configuration"
# (forbidden clause violated, illegal value, inactive/missing hyperparameter).
# Anything else -- ImportError from a missing sampler backend, AttributeError,
# RuntimeError -- is an infrastructure fault and MUST propagate rather than be
# silently reinterpreted as an invalid config, which would degrade the optimizer
# into random sampling without anyone noticing.
INVALID_CONFIG_ERRORS = (ForbiddenValueError, ValueError, KeyError, TypeError)

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.utils import MyConfiguration
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase


class OptunaConfigSpaceMixin:
    def _build_config_from_trial(self, trial) -> MyConfiguration:
        """Assemble a Configuration from this trial's suggestions. Raises on failure."""
        algorithm = self._suggest_algorithm(trial)
        values: Dict[str, Any] = {}
        if self._algorithm_hp is not None and algorithm is not None:
            values["algorithm"] = algorithm

        hps = self.config_space.get_hyperparameters()
        # Iterate to a fixpoint. In practice one pass suffices at any conditional
        # depth, because get_hyperparameters() is topologically ordered and `values`
        # is mutated in place, so a child sees its parent assigned within the same
        # pass. This loop is therefore defensive only -- it costs nothing (the
        # `changed` check breaks immediately) and removes the dependence on that
        # ordering guarantee. The real trigger for the fallback path below is
        # forbidden clauses, which Optuna cannot honour because it suggests each
        # hyperparameter independently.
        for _ in range(len(hps) + 1):
            changed = False
            for hp in hps:
                name = hp.name
                if name in values or name == "algorithm":
                    continue
                if not self._is_relevant_to_algorithm(name, algorithm):
                    continue
                if not self._conditions_are_satisfied(hp, values):
                    continue
                values[name] = self._suggest_value(trial, hp)
                changed = True
            if not changed:
                break

        return MyConfiguration(Configuration(self.config_space, values=values))

    def _try_suggest_config_from_trial(self, trial):
        """Return None when Optuna's suggestions cannot form a valid Configuration.

        Returning None (rather than a randomly sampled substitute) lets the caller
        void the trial. Substituting silently would pair the params recorded on
        this trial with a score produced by a *different* configuration, which
        feeds the sampler a bogus coordinate.
        """
        try:
            return self._build_config_from_trial(trial)
        except INVALID_CONFIG_ERRORS as e:
            self.logger.warning("Optuna generated an invalid ConfigSpace configuration: %s", e)
            return None

    def _suggest_config_from_trial(self, trial) -> MyConfiguration:
        # Back-compat shim: subclasses (e.g. the hybrid LLM/TPE optimizer) call this
        # and expect a configuration unconditionally. New code should prefer
        # _try_suggest_config_from_trial so the trial can be voided on failure.
        cfg = self._try_suggest_config_from_trial(trial)
        return cfg if cfg is not None else self._sample_fallback_configuration()

    def _trial_matches_config(self, trial, config):
        values = config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config)
        if set(trial.params) != set(self._config_to_optuna_params(config)):
            return False
        try:
            # ConfigSpace normalizes float precision. Compare configurations after
            # that normalization rather than rejecting harmless float round-off.
            replay = Configuration(self.config_space, values={**values, **trial.params})
        except INVALID_CONFIG_ERRORS:
            return False
        return self._config_signature(replay) == self._config_signature(config)

    def _tell_config_result(self, config, objective, trial=None, *, state=None, attrs=None):
        state = state or (optuna.trial.TrialState.FAIL if objective is None
                          else optuna.trial.TrialState.COMPLETE)
        attrs = dict(attrs or {})
        if trial is not None and not self._trial_matches_config(trial, config):
            trial.set_user_attr("rejection_reason", "evaluated_configuration_mismatch")
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            trial = None
        if trial is not None:
            for key, value in attrs.items():
                trial.set_user_attr(key, value)
            return self.study.tell(trial, objective if state == optuna.trial.TrialState.COMPLETE else None,
                                   state=state)

        # A FixedTrial derives the same distributions as ask/suggest without
        # consuming an unrelated WAITING trial from the study's enqueue queue.
        fixed = optuna.trial.FixedTrial(self._config_to_optuna_params(config))
        replay = self._try_suggest_config_from_trial(fixed)
        if (replay is None or self._config_signature(replay) != self._config_signature(config)
                or not self._trial_matches_config(fixed, config)):
            raise RuntimeError("Actual evaluated configuration cannot be recorded exactly in Optuna")
        system_attrs = {}
        callback = getattr(self.study.sampler, "_constraints_func", None)
        if callback is not None:
            provisional = optuna.trial.create_trial(params=fixed.params, distributions=fixed.distributions,
                state=state, value=objective if state == optuna.trial.TrialState.COMPLETE else None,
                user_attrs=attrs)
            system_attrs["constraints"] = tuple(callback(provisional))
        record = optuna.trial.create_trial(params=fixed.params, distributions=fixed.distributions,
            state=state, value=objective if state == optuna.trial.TrialState.COMPLETE else None,
            user_attrs=attrs, system_attrs=system_attrs)
        self.study.add_trial(record)
        return record

    def _suggest_algorithm(self, trial):
        hp = self._algorithm_hp
        if hp is None:
            return None
        choices = self._get_choices(hp)
        if len(choices) == 1:
            return choices[0]
        return trial.suggest_categorical("algorithm", choices)

    def _suggest_value(self, trial, hp):
        name = hp.name
        if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            return self._fixed_value(hp)
        if isinstance(hp, UniformIntegerHyperparameter):
            low = int(hp.lower)
            high = int(hp.upper)
            log = bool(getattr(hp, "log", False))
            step = int(getattr(hp, "q", None) or 1)
            if log or step == 1:
                return trial.suggest_int(name, low, high, log=log)
            return trial.suggest_int(name, low, high, step=step)
        if isinstance(hp, UniformFloatHyperparameter):
            low = float(hp.lower)
            high = float(hp.upper)
            log = bool(getattr(hp, "log", False))
            step = getattr(hp, "q", None)
            if step is not None and not log:
                return trial.suggest_float(name, low, high, step=float(step))
            return trial.suggest_float(name, low, high, log=log)
        if isinstance(hp, CategoricalHyperparameter):
            return trial.suggest_categorical(name, list(hp.choices))
        if OrdinalHyperparameter and isinstance(hp, OrdinalHyperparameter):
            return trial.suggest_categorical(name, list(hp.sequence))
        return self._default_value(hp)

    def _conditions_are_satisfied(self, hp, values: Dict[str, Any]) -> bool:
        try:
            conditions = self.config_space.get_parent_conditions_of(hp.name)
        except Exception:
            conditions = []
        for condition in conditions:
            if not self._condition_is_satisfied(condition, values):
                return False
        return True

    def _condition_is_satisfied(self, condition, values: Dict[str, Any]) -> bool:
        if hasattr(condition, "satisfied_by_value"):
            try:
                return bool(condition.satisfied_by_value(values))
            except Exception:
                pass

        components = getattr(condition, "components", None)
        if components:
            class_name = condition.__class__.__name__.lower()
            if "or" in class_name:
                return any(self._condition_is_satisfied(comp, values) for comp in components)
            return all(self._condition_is_satisfied(comp, values) for comp in components)

        parent = getattr(condition, "parent", None)
        parent_name = getattr(parent, "name", None)
        if parent_name is None or parent_name not in values:
            return False

        actual = values[parent_name]
        class_name = condition.__class__.__name__
        if "InCondition" in class_name:
            return actual in getattr(condition, "values", [])
        if "NotEqualsCondition" in class_name:
            return actual != getattr(condition, "value", None)
        if "EqualsCondition" in class_name:
            return actual == getattr(condition, "value", None)
        if "GreaterThanCondition" in class_name:
            return actual > getattr(condition, "value", None)
        if "LessThanCondition" in class_name:
            return actual < getattr(condition, "value", None)
        return False

    def _is_relevant_to_algorithm(self, hp_name: str, algorithm) -> bool:
        if algorithm is None or ":" not in hp_name:
            return True
        return hp_name.startswith(f"{algorithm}:")

    def _sample_fallback_configuration(self) -> MyConfiguration:
        sampled = self.config_space.sample_configuration()
        if isinstance(sampled, list):
            sampled = sampled[0]
        if isinstance(sampled, MyConfiguration):
            return sampled
        return MyConfiguration(sampled)

    def _config_to_optuna_params(self, cfg) -> Dict[str, Any]:
        """Project a Configuration onto the params Optuna would have suggested.

        Constants/unparametrized HPs are skipped because _suggest_value never asks
        the trial for them, so they are not part of the trial's param space.
        """
        cfg_dict = cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)
        algorithm = cfg_dict.get("algorithm", self._single_algorithm)
        params: Dict[str, Any] = {}
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
                continue
            if name != "algorithm" and not self._is_relevant_to_algorithm(name, algorithm):
                continue
            if name == "algorithm" and len(self._algorithm_choices) <= 1:
                continue
            if name in cfg_dict:
                params[name] = cfg_dict[name]
        return params

    @staticmethod
    def _config_signature(cfg):
        cfg_dict = cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)
        return tuple(sorted((str(k), str(v)) for k, v in cfg_dict.items()))

    def _get_hyperparameter(self, name):
        try:
            return self.config_space[name]
        except Exception:
            try:
                return self.config_space.get_hyperparameter(name)
            except Exception:
                return None

    def _get_choices(self, hp) -> List[Any]:
        if hp is None:
            return []
        if isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            return [self._fixed_value(hp)]
        if hasattr(hp, "choices"):
            return list(hp.choices)
        if hasattr(hp, "sequence"):
            return list(hp.sequence)
        return [self._default_value(hp)]

    def _get_single_algorithm(self):
        if self._algorithm_hp is None:
            return None
        choices = self._algorithm_choices
        if len(choices) == 1:
            return choices[0]
        if isinstance(self._algorithm_hp, (Constant, UnParametrizedHyperparameter)):
            return self._fixed_value(self._algorithm_hp)
        return None

    @staticmethod
    def _fixed_value(hp):
        return getattr(hp, "value", getattr(hp, "default_value", None))

    @staticmethod
    def _default_value(hp):
        return getattr(hp, "default_value", getattr(hp, "value", None))

    def _update_optuna_observation_history(self, obs):
        selected_algo = self._selected_algorithm(obs.config)
        if self.bo_mode == "bo":
            self._update_history_object(self.KD.bo_history, obs, self.config_space)
        elif self.bo_mode == "bo_mab":
            history = self.KD.bo_history_mab.get(selected_algo)
            if history is not None:
                self._update_history_object(history, obs, history.config_space)

    def _update_history_object(self, history, obs, config_space):
        ori_config = obs.config
        new_obs = deepcopy(obs)
        values = ori_config.get_dictionary().copy()
        try:
            new_config = Configuration(config_space, values=values)
        except Exception:
            hp_names = {hp.name for hp in config_space.get_hyperparameters()}
            filtered_values = {k: v for k, v in values.items() if k in hp_names}
            new_config = Configuration(config_space, values=filtered_values)
        new_obs.config = MyConfiguration(new_config, extra_info=getattr(ori_config, "extra_info", {}))
        history.update_observation(new_obs)

    def _selected_algorithm(self, cfg):
        if hasattr(cfg, "get_dictionary"):
            cfg_dict = cfg.get_dictionary()
        else:
            cfg_dict = dict(cfg)
        return cfg_dict.get("algorithm", self._single_algorithm)


class PT4OptOptunaOptimizer(OptunaConfigSpaceMixin, BaseOptimizer):
    """Shared pt4opt lifecycle wrapper for Optuna ask/tell optimizers."""

    optimizer_label = "optuna"

    # How many times to re-ask when Optuna's suggestion cannot be turned into a
    # valid Configuration. Sampling is stochastic, so a retry usually succeeds and
    # costs nothing (no evaluation is spent on a voided trial).
    max_suggest_retries = 4

    # Whether an evaluation of a configuration the sampler did NOT propose may be
    # injected back into the study. Safe for TPE (it just gains one KDE
    # observation); disabled for CMA-ES, whose sampler rebuilds its internal
    # optimizer state from completed trials and is perturbed by foreign points.
    supports_external_trial_injection = True

    def __init__(
        self,
        evaluator,
        config_space,
        name,
        eval_type,
        time_limit=None,
        evaluation_limit=None,
        per_run_time_limit=300,
        per_run_mem_limit=1024,
        inner_iter_num_per_iter=1,
        timestamp=None,
        output_dir="./",
        seed=1,
        n_jobs=1,
        topk=50,
        optimizer_params=None,
        task_id="GenTrans",
        logging_enabled=True,
        failure_penalty=1e12,
        **sampler_params,
    ):
        super().__init__(
            evaluator=evaluator,
            config_space=config_space,
            name=name,
            eval_type=eval_type,
            task_id=task_id,
            time_limit=time_limit,
            evaluation_limit=evaluation_limit,
            per_run_time_limit=per_run_time_limit,
            per_run_mem_limit=per_run_mem_limit,
            inner_iter_num_per_iter=inner_iter_num_per_iter,
            timestamp=timestamp,
            output_dir=output_dir,
            seed=seed,
            topk=topk,
        )

        self.optimizer_params = dict(optimizer_params or {})
        self.optimizer_params.update(sampler_params)
        self.optimizer_params.pop("rand_prob", None)
        self.optimizer_params.pop("acq_optimizer_rand_prob", None)
        self.failure_penalty = float(failure_penalty)

        if n_jobs != 1:
            self.logger.warning(
                "%s currently uses sequential Optuna ask/tell; n_jobs=%s will be ignored.",
                self.__class__.__name__,
                n_jobs,
            )
        self.n_jobs = 1

        self.trial_cnt = 0
        self.exp_output = {}

        # Contamination bookkeeping: invalid_suggest_cnt counts voided (FAIL) trials
        # whose params could not form a Configuration; fallback_cnt counts
        # evaluations actually spent on a randomly sampled configuration after all
        # retries failed. fallback_cnt / n_evals is the contamination rate.
        self.invalid_suggest_cnt = 0
        self.fallback_cnt = 0
        self.external_injected_cnt = 0

        self.KD = KnowledgeDatabase.instance()
        self.configs = self.KD.configs
        self.perfs = self.KD.perfs

        self._algorithm_hp = self._get_hyperparameter("algorithm")
        self._algorithm_choices = self._get_choices(self._algorithm_hp)
        self._single_algorithm = self._get_single_algorithm()
        self.bo_mode = "bo_mab" if self._single_algorithm is not None else "bo"

        self.study = optuna.create_study(
            direction="minimize",
            sampler=self._build_sampler(),
            study_name=f"{self.optimizer_label}-{task_id}-{int(self.timestamp)}",
        )

        self.logging_enabled = logging_enabled
        if self.logging_enabled:
            self.traj_rows: List[Dict[str, Any]] = []
            self.traj_csv_path = os.path.join(
                self.output_dir, f"{int(self.timestamp)}_opt_{self.optimizer_label}_traj.csv"
            )
            self.start_wall_time = time.time()
            self.outer_iter_idx = 0

        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            try:
                self.config_num_threshold = int(len(set(self.config_space.sample_configuration(5000))))
            except Exception:
                self.config_num_threshold = 1500
        self.logger.debug("The maximum trial number in HPO is: %d" % self.config_num_threshold)
        self.maximum_config_num = min(1500, self.config_num_threshold)

    def _build_sampler(self):
        raise NotImplementedError

    def _safe_tell(self, trial, values=None, state=None):
        try:
            if state is not None:
                self.study.tell(trial, state=state)
            else:
                self.study.tell(trial, values)
        except Exception as e:
            self.logger.warning("Failed to tell Optuna trial %s: %s", getattr(trial, "number", None), e)

    def _enqueue_config(self, config):
        """Get a trial whose params are exactly `config`, via enqueue_trial + ask.

        Using enqueue_trial lets the study derive the parameter distributions itself,
        avoiding a hand-written ConfigSpace -> Optuna distribution mapping. Returns
        None if the params do not round-trip, so the caller never tells a trial whose
        params disagree with the configuration actually evaluated.
        """
        if not self.supports_external_trial_injection:
            return None
        params = self._config_to_optuna_params(config)
        if not params:
            return None
        try:
            self.study.enqueue_trial(params)
            trial = self.study.ask()
            replay = self._try_suggest_config_from_trial(trial)
            if replay is None or self._config_signature(replay) != self._config_signature(config):
                self.logger.warning("Enqueued config did not round-trip; voiding that trial.")
                self._safe_tell(trial, state=optuna.trial.TrialState.FAIL)
                return None
            self.external_injected_cnt += 1
            return trial
        except Exception as e:
            self.logger.warning("Failed to enqueue config into the study: %s", e)
            return None

    def _ask_valid_trial(self, deadline=None):
        """Obtain (trial, config, repaired) where trial.params always match config.

        Retries a bounded number of times on invalid suggestions, then *repairs*
        rather than retrying forever.

        Why repair instead of relying on retries: a FAIL trial is excluded from
        TPE's model, so the sampler never learns that an infeasible region is
        infeasible. Because univariate TPE models each parameter independently, it
        can combine two marginally-good values into a forbidden combination and
        then propose that same point indefinitely -- an absorbing state where every
        retry returns the identical invalid suggestion. Observed in practice: after
        n_startup_trials, 100% of suggestions were the same forbidden pair.

        So on exhaustion we sample a valid configuration and enqueue it, which both
        keeps the evaluation attributed to a real trial (no wasted budget) and feeds
        the sampler an observation from the feasible region.
        """
        for _ in range(self.max_suggest_retries):
            if deadline is not None and time.time() > deadline:
                break
            trial = self.study.ask()
            config = self._try_suggest_config_from_trial(trial)
            if config is not None:
                return trial, config, False
            # Void it: never let this trial receive a real objective.
            self._safe_tell(trial, state=optuna.trial.TrialState.FAIL)
            self.invalid_suggest_cnt += 1

        self.fallback_cnt += 1
        config = self._sample_fallback_configuration()
        trial = self._enqueue_config(config)
        self.logger.warning(
            "Repaired to a sampled configuration after %d invalid Optuna suggestions (attributed=%s).",
            self.max_suggest_retries,
            trial is not None,
        )
        return trial, config, True

    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                self.logger.info("Early stop triggered, terminating optimization.")
                break
            self.iterate()
        return np.max(self.perfs) if self.perfs else -np.inf

    def iterate(self, budget=MAX_INT) -> Tuple[float, float, Any, List[Dict]]:
        if self.optimizer_label == "tpe":
            from mindware.components.optimizers.ml_evaluation_feedback import upgrade_runtime_study
            self.study = upgrade_runtime_study(self.study)
        _start_time = time.time()
        newly_observed_rows = []

        if len(self.configs) == 0 and hasattr(self, "init_hpo_iter_num") and self.init_hpo_iter_num is not None:
            inner_iter_num = self.init_hpo_iter_num
            print("initial hpo trial num is set to %d" % inner_iter_num)
        else:
            inner_iter_num = self.inner_iter_num_per_iter

        for i in range(inner_iter_num):
            if len(self.configs) >= self.maximum_config_num or self.early_stopped_flag or self.timeout_flag:
                break
            if time.time() - _start_time > budget:
                self.logger.warning("Time limit exceeded for this iteration!")
                break

            iter_start_time = time.time()

            from mindware.components.optimizers import ml_checkpoint
            saved = ml_checkpoint.pending(self)
            if saved is None:
                trial, config, fallback_used = self._ask_valid_trial(deadline=_start_time + budget)
                iter_idx = trial.number if trial is not None else len(self.study.trials)
                saved = {"config": config, "trial_id": trial._trial_id if trial else None,
                         "iter_idx": iter_idx, "fallback_used": fallback_used}
            else:
                config, iter_idx, fallback_used = saved["config"], saved["iter_idx"], saved["fallback_used"]
                trial = optuna.trial.Trial(self.study, saved["trial_id"]) if saved["trial_id"] is not None else None
            ml_checkpoint.prepare(self, saved)
            obs = wrapper_evaluater(config, self.evaluator, self.per_run_time_limit)

            raw_objective = float(obs.objectives[0]) if obs.objectives else np.inf
            success = obs.trial_state == SUCCESS and np.isfinite(raw_objective)
            feedback = {}
            # trial is None only when the config could not be attributed (CMA-ES, which
            # opts out of injection, or a failed round-trip). In that case the result is
            # still recorded in the trajectory, just not fed to the sampler -- never
            # told to some other trial whose params disagree with `config`.
            if self.optimizer_label == "tpe":
                from mindware.components.optimizers.ml_evaluation_feedback import observation_feedback
                objective, state, feedback = observation_feedback(obs, self.per_run_time_limit)
                self._tell_config_result(config, objective, trial, state=state, attrs=feedback)
            elif trial is not None:
                if success:
                    self._safe_tell(trial, raw_objective)
                else:
                    # Failed/timeout/non-finite eval: report FAIL state instead of a fixed
                    # numeric penalty. A constant sentinel (e.g. 1e12) can collide with or be
                    # outdone by unbounded metrics such as a large MSE, which would corrupt the
                    # sampler's ordering. A FAIL trial is simply ignored by TPE/CMA-ES.
                    self._safe_tell(trial, state=optuna.trial.TrialState.FAIL)

            iter_time = time.time() - iter_start_time
            eval_time = obs.elapsed_time if obs.elapsed_time is not None else 0.0
            smac_time = iter_time - eval_time
            perf = -raw_objective if success else float("-inf")

            if success:
                self.exp_output[time.time()] = (config, -raw_objective)
                self.update_saver([config], [raw_objective])
                self.update_history(obs, trail_id=f"{iter_idx}")

            note = f"candidate {i + 1}/{inner_iter_num} via {self.optimizer_label}"
            if fallback_used:
                note += " [fallback=True random config, not sampler-proposed]"

            row_data = {
                "perf": perf,
                "config": config,
                "eval_time": eval_time,
                "smac_time": smac_time,
                "iter_time": iter_time,
                "note": note,
                "fallback": fallback_used,
            }
            newly_observed_rows.append(row_data)

            if self.logging_enabled:
                self._record_event(
                    stage="iterate",
                    iter_idx=iter_idx,
                    cfg=config,
                    perf=perf,
                    eval_time=eval_time,
                    smac_time=smac_time,
                    iter_time=iter_time,
                    note=note,
                    fallback=fallback_used,
                    feedback=feedback,
                )

        if self.logging_enabled:
            self.outer_iter_idx += 1

        if self.time_limit is not None and time.time() - self.timestamp > self.time_limit or (
            hasattr(self, "evaluation_num_limit")
            and self.evaluation_num_limit is not None
            and len(self.perfs) >= self.evaluation_num_limit
        ):
            self.timeout_flag = True

        iteration_cost = time.time() - _start_time
        if self.perfs:
            incumbent_idx = int(np.argmax(self.perfs))
            incumbent = self.configs[incumbent_idx]
            self.incumbent_config = incumbent.get_dictionary().copy()
            self.incumbent_perf = float(self.perfs[incumbent_idx])

        return self.incumbent_perf, iteration_cost, self.incumbent_config, newly_observed_rows

    def update_history(self, obs, **extra_infos):
        self._update_optuna_observation_history(obs)
        self.KD.update_history(obs, source=self.bo_mode, **extra_infos)

    def _record_event(
        self, stage, iter_idx, cfg, perf, eval_time=0.0, smac_time=0.0, iter_time=0.0, note="", fallback=False,
        feedback=None,
    ):
        if not self.logging_enabled:
            return

        wall_elapsed_sec = time.time() - self.start_wall_time
        algo = self._selected_algorithm(cfg) or "default"

        from mindware.components.optimizers.resume_traj import (
            cfg_as_json,
            observations_jsonl_path,
            persist_jsonl_row,
            persist_study_pickle,
            study_pickle_path,
        )

        config_json = cfg_as_json(cfg)
        row = {
            "ts": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "iter_idx": iter_idx,
            "perf": perf,
            "best_so_far": max(self.perfs) if self.perfs else float("-inf"),
            "algorithm": algo,
            "smac_time": smac_time,
            "eval_time": eval_time,
            "iter_time": iter_time,
            "wall_elapsed_sec": wall_elapsed_sec,
            "note": note,
            "fallback": bool(fallback),
            "config_json": config_json,
            **(feedback or {}),
        }
        self.traj_rows.append(row)

        try:
            columns = [
                "ts",
                "stage",
                "iter_idx",
                "perf",
                "best_so_far",
                "algorithm",
                "smac_time",
                "eval_time",
                "iter_time",
                "wall_elapsed_sec",
                "note",
                "fallback",
                "config_json",
            ]
            columns += [key for key in ("evaluation_status", "evaluation_seconds", "evaluation_limit_seconds", "runtime_policy_id")
                        if any(key in rec for rec in self.traj_rows)]
            for rec in self.traj_rows:
                rec.setdefault("config_json", "")
            df = pd.DataFrame(self.traj_rows)
            tmp_path = self.traj_csv_path + ".tmp"
            df[columns].to_csv(tmp_path, index=False)
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp_path, self.traj_csv_path)
        except IOError as e:
            self.logger.warning(f"Failed to save trajectory to {self.traj_csv_path}. Error: {e}")

        try:
            persist_jsonl_row(
                observations_jsonl_path(self.traj_csv_path),
                {
                    "iter_idx": iter_idx,
                    "perf": perf,
                    "success": bool(np.isfinite(perf) and perf > -1e18),
                    "algorithm": algo,
                    "config": json.loads(config_json) if config_json.startswith("{") else None,
                    **(feedback or {}),
                },
            )
        except Exception as exc:
            self.logger.warning("Failed to append TPE observation jsonl: %s", exc)
        try:
            persist_study_pickle(study_pickle_path(self.traj_csv_path), self.study)
        except Exception as exc:
            self.logger.warning("Failed to pickle Optuna study: %s", exc)

    def restore_from_traj(
        self,
        traj_path: str,
        history_json: Optional[str] = None,
        observations_jsonl: Optional[str] = None,
    ) -> int:
        """Replay saved trials into this study and continue from the last row.

        Returns the number of trajectory rows restored (the next trial index).
        """
        from openbox.utils.constants import FAILED, SUCCESS
        from openbox.utils.history import Observation

        from mindware.components.optimizers.resume_traj import (
            copy_traj_bundle,
            load_history_json_configs,
            load_jsonl_configs,
            load_traj_rows,
            observations_jsonl_path,
            persist_study_pickle,
            resolve_row_configs,
            rewrite_observations_jsonl,
            rewrite_traj_csv,
            row_success_perf,
            study_pickle_path,
            try_load_study_pickle,
        )

        rows = load_traj_rows(traj_path)
        if not rows:
            return 0

        algo = self._single_algorithm
        jsonl_path = observations_jsonl or observations_jsonl_path(traj_path)
        jsonl_cfgs = load_jsonl_configs(jsonl_path, default_algorithm=algo)
        hist_cfgs = load_history_json_configs(history_json, default_algorithm=algo)
        recovered, n_topk = resolve_row_configs(
            rows,
            traj_path=traj_path,
            default_algorithm=algo,
            jsonl_cfgs=jsonl_cfgs,
            hist_cfgs=hist_cfgs,
        )

        copy_traj_bundle(traj_path, self.traj_csv_path)
        self.traj_rows = [dict(r) for r in rows]
        try:
            rewrite_traj_csv(self.traj_csv_path, self.traj_rows)
        except Exception as exc:
            self.logger.warning("Resume: failed to persist recovered config_json: %s", exc)

        pickled = try_load_study_pickle(study_pickle_path(traj_path))
        replay_study = pickled is None
        if pickled is not None:
            self.study = pickled
            self.logger.info(
                "Restored Optuna study pickle with %d trials from %s",
                len(self.study.trials),
                study_pickle_path(traj_path),
            )

        restored_cfg = 0
        for i, row in enumerate(self.traj_rows):
            success, perf = row_success_perf(row)
            cfg_dict = recovered[i] if i < len(recovered) else None
            if cfg_dict is None:
                continue
            try:
                cfg = self.KD._dict_to_cfg(cfg_dict)
            except Exception as exc:
                self.logger.warning("Resume: skip trial %s, invalid config: %s", i, exc)
                continue

            raw_obj = -float(perf) if success else None
            if replay_study:
                trial = self._enqueue_config(cfg)
                if trial is not None:
                    if raw_obj is None:
                        self._safe_tell(trial, state=optuna.trial.TrialState.FAIL)
                    else:
                        self._safe_tell(trial, raw_obj)

            if success:
                obs = Observation(
                    config=cfg,
                    objectives=[float(raw_obj)],
                    trial_state=SUCCESS,
                    elapsed_time=float(row.get("eval_time") or 0.0),
                )
                self.update_saver([cfg], [raw_obj])
                try:
                    self.update_history(obs, trail_id=str(row.get("iter_idx", i)))
                except Exception as exc:
                    self.logger.warning("Resume: history update failed at trial %s: %s", i, exc)
                    if cfg not in self.configs:
                        self.configs.append(cfg)
                        self.perfs.append(perf)
                restored_cfg += 1
            else:
                restored_cfg += 1

        if restored_cfg == 0:
            self.logger.warning(
                "Resume: %d traj rows at %s have no recoverable configs; "
                "TPE surrogate will restart while the traj continues.",
                len(rows),
                traj_path,
            )
        else:
            self.logger.info(
                "Resume: restored %d/%d TPE trials (study replay=%s, topk_fill=%d) from %s",
                restored_cfg,
                len(rows),
                replay_study,
                n_topk,
                traj_path,
            )

        self.trial_cnt = len(rows)
        try:
            persist_study_pickle(study_pickle_path(self.traj_csv_path), self.study)
        except Exception:
            pass
        try:
            rewrite_observations_jsonl(
                observations_jsonl_path(self.traj_csv_path),
                self.traj_rows,
                recovered,
                default_algorithm=algo,
            )
        except Exception as exc:
            self.logger.warning("Resume: failed to rewrite observations jsonl: %s", exc)
        if self.perfs:
            idx = int(np.argmax(self.perfs))
            self.incumbent_config = self.configs[idx].get_dictionary().copy()
            self.incumbent_perf = float(self.perfs[idx])
        return len(rows)

    def get_opt_trajectory(self):
        best = float(max(self.perfs)) if self.perfs else float("-inf")
        traj_rows = getattr(self, "traj_rows", [])
        n_evals = len(self.perfs)
        fallback_cnt = int(getattr(self, "fallback_cnt", 0))
        return {
            "n_evals": n_evals,
            "final_best": best,
            "csv_path": getattr(self, "traj_csv_path", None),
            "detail_perfs": ",".join([str(p) for p in self.perfs]),
            # Contamination accounting: fallback_cnt evaluations were spent on
            # randomly sampled configurations the sampler never proposed, so they can
            # be excluded in downstream analysis.
            "fallback_cnt": fallback_cnt,
            "invalid_suggest_cnt": int(getattr(self, "invalid_suggest_cnt", 0)),
            "external_injected_cnt": int(getattr(self, "external_injected_cnt", 0)),
            "fallback_rate": (fallback_cnt / n_evals) if n_evals else 0.0,
            "history": [
                {
                    "iter_idx": r["iter_idx"],
                    "perf": r["perf"],
                    "best_so_far": r["best_so_far"],
                    "algorithm": r["algorithm"],
                    "smac_time": r.get("smac_time", 0.0),
                    "eval_time": r.get("eval_time", 0.0),
                    "iter_time": r.get("iter_time", 0.0),
                    "wall_elapsed_sec": r["wall_elapsed_sec"],
                    "fallback": bool(r.get("fallback", False)),
                }
                for r in traj_rows
            ][:500],
        }
