"""LLM hybrid loop for one TaskSet CASH task (RGPE or FSBO min-max).

Schedule (identical to the ML ``PT4OptLLMRGPEOptimizer._choose_source`` campaign
``HPUCB_rgpe_p07p03_retry5_llm8_cool20``):

* trials 1-3   ``llm_warm_start``: the three warm-start presets (no LLM call);
* trials 4-8   forced ``llm`` turns;
* trial 9+     PUCB selector (LLM prior 0.7 -> 0.3, q=arm_best, c2=1.4) between
               ``llm`` and ``rgpe``; during trials 10-20 a trial that follows an
               LLM-selected trial is forced to RGPE (cooldown);
* LLM turns use the ``history_tpe5`` retry policy: up to five proposals, each
  duplicate is replayed as a reminder; after five duplicates RGPE supplies the
  configuration (arm attempted = llm, evaluation credit = rgpe). An unparseable
  reply falls back to a random unseen configuration credited to the LLM arm.

``TaskSetLLMFSBO`` retains this schedule and replaces the RGPE proposal arm,
fallback and evaluation credit with FSBO. Prompt retrieval is independent.

Durable state (resume-safe, one evaluation in flight at most):
  history.json   OpenBox-style observations (same layout as the RGPE baseline)
  pending.json   the proposal being evaluated
  state.pkl      selector + identity after every committed evaluation
  trials/trial_NNNN.json  prompts, raw replies, retrieval audit, RGPE evidence
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from mindware.components.optimizers.hybrid_selector import selector_from_params
from mindware.domains.dlhpo.taskset_llm import experience as E
from mindware.domains.dlhpo.taskset_llm import p5_prompt as P
from mindware.domains.dlhpo.taskset_llm import spec as S
from mindware.domains.dlhpo.taskset_llm.inference import InferenceServiceError
from mindware.domains.history import history_exists, load_openbox_history, observation, write_rotating_openbox_history

HYBRID = dict(hybrid_selector="pucb", llm_prob=0.7, pucb_p1=0.7, pucb_c2=1.4, pucb_q="arm_best", bo_arm="rgpe")
N_WARM_START, LLM_PHASE_END, COOLDOWN_START, COOLDOWN_END = 3, 8, 10, 20
MAX_LLM_ATTEMPTS = 5
FAILED_OBJECTIVE = 1e10


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode())


def rows_from_observations(observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for i, obs in enumerate(observations):
        loss = float(obs["objectives"][0])
        ok = obs["trial_state"] == 0 and math.isfinite(loss) and loss < FAILED_OBJECTIVE
        rows.append({"trial": i + 1, "config": dict(obs["config"]), "loss": loss if ok else None, "successful": ok,
                     "source": (obs.get("extra_info") or {}).get("source"),
                     "selected_arm": (obs.get("extra_info") or {}).get("selected_arm")})
    return rows


class TaskSetLLMRGPE:
    def __init__(
        self,
        *,
        task_id: str,
        cs: ConfigurationSpace,
        evaluate: Callable[[Dict[str, Any]], Any],
        run_dir: Path,
        llm: Any,
        presets: Sequence[Dict[str, Any]],
        specs: Dict[str, Dict[str, Any]],
        rgpe_suggest: Optional[Callable] = None,
        bo_suggest: Optional[Callable] = None,
        bo_arm: str = "rgpe",
        bank: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        source_predict: Optional[Callable[[List[str], List[Dict[str, Any]]], Dict[str, List[float]]]] = None,
        identity: Optional[Dict[str, Any]] = None,
        budget: int = 100,
        seed: int = 1,
        prompt_options: Optional[Dict[str, Any]] = None,
    ):
        if len(presets) < N_WARM_START:
            raise ValueError(f"{task_id}: {N_WARM_START} warm-start presets are required, got {len(presets)}")
        self.task_id, self.cs, self.evaluate = task_id, cs, evaluate
        self.run_dir = Path(run_dir)
        self.llm, self.specs = llm, specs
        self.presets = [S.coerce_config(p, cs) for p in presets[:N_WARM_START]]
        if bo_arm not in {"rgpe", "fsbo"}:
            raise ValueError("Supported TaskSet surrogate arms: rgpe, fsbo")
        self.bo_arm = bo_arm
        self.prompt_options = dict(prompt_options or {})
        if set(self.prompt_options) - {"protocol_text", "parameter_notes"}:
            raise ValueError("Only protocol_text and parameter_notes may override prompt context")
        self.bo_suggest = bo_suggest or rgpe_suggest
        self.bank, self.source_predict = bank, source_predict
        hybrid = dict(HYBRID, bo_arm=bo_arm)
        self.budget, self.seed = int(budget), int(seed)
        self.identity = dict(identity or {}, task_id=task_id, budget=self.budget, seed=self.seed,
                             method=f"taskset_llm_{bo_arm}", hybrid=hybrid, n_warm_start=N_WARM_START,
                             llm_phase_end=LLM_PHASE_END, cooldown=[COOLDOWN_START, COOLDOWN_END],
                             max_llm_attempts=MAX_LLM_ATTEMPTS)
        if self.prompt_options:
            self.identity["prompt_context"] = self.prompt_options
        self.history_path = self.run_dir / "history.json"
        self.pending_path = self.run_dir / "pending.json"
        self.state_path = self.run_dir / "state.pkl"
        self.observations: List[Dict[str, Any]] = []
        self.selector = selector_from_params(hybrid, rng=np.random.RandomState(self.seed))
        self._restore()

    # ------------------------------------------------------------- state
    def _restore(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not history_exists(self.history_path):
            if self.state_path.exists():
                raise RuntimeError("state.pkl exists without history.json")
            return
        history = load_openbox_history(self.history_path)
        if history["meta_info"] != self.identity:
            raise ValueError("Resume identity changed (adapter, assets, budget or schedule)")
        state = pickle.loads(self.state_path.read_bytes())
        if state["identity"] != self.identity or state["n_observations"] != len(history["observations"]):
            raise ValueError("state.pkl does not match history.json")
        self.observations = list(history["observations"])
        self.selector = state["selector"]

    def _commit(self) -> None:
        write_rotating_openbox_history(self.history_path, task_id=self.task_id,
                                       observations=self.observations, meta=self.identity)
        _atomic_write(self.state_path, pickle.dumps({"identity": self.identity, "selector": self.selector,
                                                     "n_observations": len(self.observations)}))

    def rows(self) -> List[Dict[str, Any]]:
        return rows_from_observations(self.observations)

    # ----------------------------------------------------------- schedule
    def choose_source(self, trial: int) -> Tuple[str, Dict[str, Any]]:
        details: Dict[str, Any] = {"round": trial, "cooldown_active": False, "pucb_proposed_arm": None,
                                   "cooldown_window": LLM_PHASE_END + 1 <= trial <= COOLDOWN_END}
        if trial <= N_WARM_START:
            details.update(selection_reason="preset_warm_start", selected_arm="llm_warm_start")
            return "llm_warm_start", details
        if trial <= LLM_PHASE_END:
            details.update(selection_reason="initial_llm_phase", selected_arm="llm")
            return "llm", details
        m, M = trial - (LLM_PHASE_END + 1), max(1, self.budget - (LLM_PHASE_END + 1))
        arm = self.selector.choose(m, M, higher_is_better=True)
        previous = self.observations[-1].get("extra_info", {}) if self.observations else {}
        cooldown = COOLDOWN_START <= trial <= COOLDOWN_END and previous.get("selected_arm") == "llm"
        details.update(pucb_proposed_arm=arm, cooldown_active=cooldown, schedule_index=m, schedule_denominator=M,
                       selection_reason="llm_cooldown" if cooldown else "pucb",
                       selector_snapshot=self.selector.snapshot())
        arm = self.bo_arm if cooldown else arm
        details["selected_arm"] = arm
        return arm, details

    # ---------------------------------------------------------- proposals
    def _seen(self) -> set:
        return {S.signature(o["config"]) for o in self.observations}

    def _random_unseen(self, trial: int) -> Dict[str, Any]:
        self.cs.seed(self.seed * 100003 + trial)
        seen = self._seen()
        for _ in range(1000):
            config = dict(self.cs.sample_configuration().get_dictionary())
            if S.signature(config) not in seen:
                return config
        raise RuntimeError("No unseen random configuration")

    def _retrieval(self, trial: int, rows: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
        if not (self.bank and self.source_predict and E.retrieval_window(trial)):
            return [], {"active": False}
        cards, stats = self.bank
        audit = E.rerank(self.task_id, rows, cards, stats, self.specs, self.source_predict)
        audit["active"] = True
        return E.retrieval_block(audit, cards), audit

    def _llm_turn(self, trial: int, log: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        rows = self.rows()
        retrieval_lines, audit = self._retrieval(trial, rows)
        system, user, meta = P.build_p5_prompt(self.task_id, self.specs, self.cs, rows, trial=trial,
                                               seed=self.seed, retrieval_lines=retrieval_lines, **self.prompt_options)
        log.update(system_prompt=system, user_prompt=user, prompt_meta=meta, retrieval=audit, attempts=[])
        seen = self._seen()
        reminders: List[Dict[str, Any]] = []
        for index in range(MAX_LLM_ATTEMPTS):
            prompt = P.retry_prompt(user, reminders)
            attempt: Dict[str, Any] = {"attempt": index, "user_prompt": prompt if reminders else None}
            started = time.time()
            raw = ""
            try:
                raw = self.llm.complete(system, prompt)
                config, notes, payload = S.parse_reply(raw, self.cs)
            except InferenceServiceError:
                # An unavailable shared service must not contaminate benchmark results.
                raise
            except Exception as exc:  # parse failure -> random, credited to the LLM arm
                attempt.update(raw_response=raw, error=f"{type(exc).__name__}: {exc}", seconds=time.time() - started)
                log["attempts"].append(attempt)
                return self._random_unseen(trial), {"configuration_source": "random",
                                                    "note": "random fallback after llm error"}
            duplicate = S.signature(config) in seen
            attempt.update(raw_response=raw, parsed_payload=payload, config=config, repairs=notes,
                           duplicate=duplicate, seconds=time.time() - started)
            log["attempts"].append(attempt)
            if not duplicate:
                return config, {"configuration_source": "llm", "llm_attempts": index + 1,
                                "repairs": notes, "experience_branch": meta["experience_branch"]}
            reminders.append(P.retry_reminder(rows, config))
        return None, {"llm_retry_exhausted": True, "llm_attempts": MAX_LLM_ATTEMPTS,
                      "experience_branch": meta["experience_branch"]}

    def propose(self, trial: int) -> Dict[str, Any]:
        source, details = self.choose_source(trial)
        log: Dict[str, Any] = {"trial": trial, "selection": details}
        started = time.time()
        meta: Dict[str, Any] = {}
        if source == "llm_warm_start":
            config, notes = self.presets[trial - 1]
            meta = {"configuration_source": "preset", "repairs": notes}
        else:
            config = None
            if source == "llm":
                config, meta = self._llm_turn(trial, log)
            if config is None:  # RGPE arm, or LLM exhausted five duplicate proposals
                bo_config, diagnostic = self.bo_suggest(self.observations, trial)
                config = dict(bo_config)
                meta_bo = {self.bo_arm: {k: v for k, v in diagnostic.items() if k != "candidates"}}
                if S.signature(config) in self._seen():
                    raise RuntimeError(f"{self.bo_arm.upper()} returned a previously evaluated configuration")
                meta.update(meta_bo, configuration_source=self.bo_arm)
                source = self.bo_arm
        Configuration(self.cs, values=config).is_valid_configuration()
        log.update(config=config, source=source, meta=meta, proposal_seconds=time.time() - started)
        _atomic_json(self.run_dir / "trials" / f"trial_{trial:04d}.json", log)
        return {"identity": self.identity, "trial": trial, "config": config, "source": source,
                "selected_arm": details["selected_arm"], "selection": details, "meta": meta,
                "selector_after_selection": pickle.dumps(self.selector).hex(),
                "proposal_seconds": log["proposal_seconds"]}

    # ---------------------------------------------------------------- run
    def _record(self, pending: Dict[str, Any], result: Any, seconds: float) -> None:
        loss = float(result.objective)
        ok = bool(result.feasible and math.isfinite(loss))
        perf = -loss if ok else float("-inf")
        best = [-r["loss"] for r in self.rows() if r["successful"]]
        prev_best = max(best) if best else float("-inf")
        source, selected = pending["source"], pending["selected_arm"]
        if source in self.selector.arms:
            arm_best = self.selector.arm_best.get(source)
            self.selector.update(source, perf, higher_is_better=True,
                                 beat_incumbent=bool(ok and perf > prev_best),
                                 beat_arm_best=bool(ok and (arm_best is None or perf > float(arm_best))))
            if selected != source:
                self.selector.record_attempt(selected)
        extra = {"source": source, "selected_arm": selected, "selection": pending["selection"],
                 "meta": pending["meta"], "proposal_seconds": pending["proposal_seconds"],
                 "info": dict(getattr(result, "info", {}) or {})}
        self.observations.append(observation(pending["config"], loss if ok else FAILED_OBJECTIVE,
                                             trial_state=0 if ok else 1, elapsed_time=seconds, extra=extra))
        self._commit()

    def step(self) -> Dict[str, Any]:
        trial = len(self.observations) + 1
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text())
            if (pending["identity"] == self.identity and pending["trial"] == trial - 1
                    and self.observations and pending["config"] == self.observations[-1]["config"]):
                # Evaluation was committed before interruption, but pending was not removed.
                self.pending_path.unlink()
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text())
            if pending["identity"] != self.identity or pending["trial"] != trial:
                raise ValueError("pending.json does not belong to the next trial of this run")
            if pending.get("selector_after_selection"):
                self.selector = pickle.loads(bytes.fromhex(pending["selector_after_selection"]))
        else:
            pending = self.propose(trial)
            _atomic_json(self.pending_path, pending)
        started = time.time()
        result = self.evaluate(pending["config"])
        self._record(pending, result, time.time() - started)
        self.pending_path.unlink()
        return {"trial": trial, "source": pending["source"], "selected_arm": pending["selected_arm"],
                "objective": self.observations[-1]["objectives"][0],
                "successful": self.observations[-1]["trial_state"] == 0}

    def run(self) -> List[Dict[str, Any]]:
        events = []
        while len(self.observations) < self.budget:
            events.append(self.step())
            print(json.dumps(dict(task=self.task_id, **events[-1])), flush=True)
        _atomic_json(self.run_dir / "progress.json", {"identity": self.identity, "status": "complete",
                                                       "attempts": len(self.observations), "updated_at": time.time()})
        return events


class TaskSetLLMFSBO(TaskSetLLMRGPE):
    """Same ML selection/cooldown/retry policy with the FSBO min-max BO arm."""

    def __init__(self, **kwargs):
        super().__init__(bo_arm="fsbo", **kwargs)
