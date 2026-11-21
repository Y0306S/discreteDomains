"""SemanticOpt's five learned starts and two-candidate, success-weighted EI."""

import copy
import json
import math
from pathlib import Path

from scipy.stats import norm

from examples.benchmark.semanticopt_ml_sft import SYSTEM
from mindware.components.optimizers.online_llm_optimizer import OnlineLLMOptimizer, QWEN_BASE
from mindware.components.optimizers.semanticopt_protocol import build_messages, finite_number
from mindware.components.optimizers.strict_recommendation import decode_object
from mindware.components.optimizers.online_upstream import source_digest


ADAPTER = "/root/data/cys/baseline/artifacts/semanticopt_ml_train80_lora/model/final_adapter"


def initialization_payload(raw):
    """Recover one labelled initialization object without mixing separate replies."""
    keys = {"initial_configuration_" + str(i) for i in range(1, 6)}
    decoder = json.JSONDecoder()
    envelopes = []
    cursor = 0
    while cursor < len(raw):
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(raw[start:])
        except ValueError:
            cursor = start + 1
            continue
        cursor = start + length
        if isinstance(value, dict) and keys.intersection(value):
            # Recheck duplicate keys and non-finite values with the strict parser.
            envelopes.append(decode_object(raw[start:cursor]))
    if len(envelopes) != 1:
        raise ValueError("Expected one unambiguous initialization object")
    return envelopes[0]


def weighted_ei(mean, std, failure_probability, best):
    improvement = best - mean
    if std == 0:
        return (1 - failure_probability) * max(0.0, improvement)
    z = improvement / std
    return (1 - failure_probability) * (improvement * norm.cdf(z) + std * norm.pdf(z))


class SemanticOptOptimizer(OnlineLLMOptimizer):
    optimizer_label = "semanticopt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.llm_client is None:
            from examples.benchmark.hf_qwen_client import HuggingFaceQwenClient
            self.llm_client = HuggingFaceQwenClient(
                model_path=QWEN_BASE, adapter_path=ADAPTER,
                temperature=self.params.get("temperature", 1.0), max_new_tokens=2048,
                enable_thinking=False)
        self.semantic_initial, self.semantic_history = [], []
        self.task_type = self.params["task_type"]
        self.asset_identity.update(
            adapter=ADAPTER, adapter_sha256=source_digest(Path(ADAPTER) / "adapter_model.safetensors"),
            initialization_count=5, candidate_batch_size=2,
            training_data_adaptation="525 train80 ML trajectories",
            prediction_fallback="one shared default forecast, then EI among usable candidates",
            initialization_parser="one labelled JSON envelope, optional surrounding prose",
            inference="five learned starts, then success-probability-weighted EI over two candidates")

    def _generate(self, messages, stage):
        raw = self.llm_client.complete_messages(messages, use_s1b_template=False)
        self._record_decision({"stage": stage, "messages": copy.deepcopy(messages), "raw_text": raw})
        try:
            return decode_object(raw), raw
        except (ValueError, TypeError):
            return {}, raw

    def _messages(self):
        return build_messages(SYSTEM, self.task_context["dataset"], self.config_space,
                              self.fixed_algorithm, self.task_type, self.evaluation_num_limit,
                              self.semantic_history[:5], self.semantic_history[5:])

    def _resolve(self, raw):
        config = dict(self.resolver.resolve(raw))
        self._record_decision(self.resolver.last_record)
        return config

    def _stats(self, payload, had_failure):
        mean = finite_number(payload.get("mean"), "mean")
        std = finite_number(payload.get("std"), "std", lower=0)
        failure = finite_number(payload.get("failure_probability"), "failure_probability", 0, 1) \
            if had_failure else 0.0
        return {"mean": mean, "std": std, **({"failure_probability": failure} if had_failure else {})}

    def _request_prediction(self, messages, config, had_failure, stage, raw=None):
        refresh_messages = copy.deepcopy(messages)
        if raw is not None:
            refresh_messages.append({"role": "assistant", "content": raw})
        refresh_messages.append({"role": "user", "content":
            "Predict the objective distribution for this exact configuration. Do not change it. "
            "Return next_configuration, mean, std" +
            (", failure_probability" if had_failure else "") + ".\n" + json.dumps(config)})
        payload, _ = self._generate(refresh_messages, stage)
        try:
            supplied = payload.get("next_configuration")
            if isinstance(supplied, dict):
                supplied = dict(supplied)
                supplied.setdefault("algorithm", self.fixed_algorithm)
            if supplied != config:
                raise ValueError("Prediction refresh changed the configuration")
            return self._stats(payload, had_failure)
        except (ValueError, TypeError):
            return None

    def _candidate(self, messages):
        payload, raw = self._generate(messages, "candidate")
        original = payload.get("next_configuration")
        config = self._resolve(json.dumps(original) if isinstance(original, dict) else raw)
        # Fixed identifier insertion does not change the proposed tunable configuration.
        actual_original = dict(original) if isinstance(original, dict) else None
        if actual_original is not None:
            actual_original.setdefault("algorithm", self.fixed_algorithm)
        had_failure = any(row["status"] == "failure" for row in self.semantic_history)
        try:
            stats = self._stats(payload, had_failure)
            needs_refresh = actual_original != config
        except (ValueError, TypeError):
            stats, needs_refresh = None, True
        refreshed = False
        if needs_refresh:
            stats = self._request_prediction(messages, config, had_failure, "prediction_refresh", raw)
            refreshed = True
            if stats is None:
                return {"config": dict(self.config_space.get_default_configuration()), "prediction": None,
                        "prediction_refreshed": True, "note": "labelled default: predictions unavailable"}
        return {"config": config, "prediction": stats, "prediction_refreshed": refreshed,
                "note": "SemanticOpt success-weighted EI"}

    def _suggest(self):
        if self.trial_cnt < 5:
            if not self.semantic_initial:
                messages = self._messages()
                messages[0]["content"] = (
                    "You are a black-box optimization assistant for tabular machine learning. "
                    "Return exactly one JSON object with keys initial_configuration_1 through "
                    "initial_configuration_5, each containing one valid ConfigSpace configuration.")
                messages[-1]["content"] += "\nProvide 5 initial configurations that explore promising regions."
                payload, raw = self._generate(messages, "five_initializations")
                recovery = {"stage": "initialization_parsing", "route": "direct_json"}
                if not payload:
                    try:
                        payload = initialization_payload(raw)
                        recovery["route"] = "embedded_json"
                    except (ValueError, TypeError) as exc:
                        recovery.update(route="labelled_defaults", error=str(exc))
                recovery["recovered_keys"] = sorted(payload)
                self._record_decision(recovery)
                for index in range(1, 6):
                    key = "initial_configuration_" + str(index)
                    value = payload.get(key)
                    present = isinstance(value, (dict, str)) and bool(value)
                    # Prose inside a labelled slot goes through the existing GPT extractor.
                    # Never pass the entire five-slot reply to a single-config extractor.
                    text = value if isinstance(value, str) else json.dumps(value) if present else "{}"
                    config = self._resolve(text)
                    self.semantic_initial.append({"config": config, "prediction": None,
                        "note": "SemanticOpt model initialization" if present
                        else "labelled default: missing initial configuration"})
            return self.semantic_initial[self.trial_cnt]
        candidates = [self._candidate(self._messages()) for _ in range(2)]
        missing = [c for c in candidates if c["prediction"] is None]
        if missing:
            default = dict(self.config_space.get_default_configuration())
            had_failure = any(row["status"] == "failure" for row in self.semantic_history)
            # Both failed candidates use the same default and share this one request.
            stats = self._request_prediction(self._messages(), default, had_failure,
                                             "default_prediction_refresh")
            for candidate in missing:
                candidate.update(config=dict(default), prediction=copy.deepcopy(stats),
                                 default_prediction_refreshed=True)
                if stats is not None:
                    candidate["note"] = "labelled default: refreshed predictions, success-weighted EI"
        usable = [c for c in candidates if c["prediction"] is not None]
        if not usable:
            selected = candidates[0]
        else:
            best = min((row["objective"] for row in self.semantic_history if row["status"] == "success"),
                       default=None)
            for candidate in usable:
                p = candidate["prediction"]
                candidate["acquisition"] = weighted_ei(p["mean"], p["std"],
                    p.get("failure_probability", 0), best) if best is not None else None
            selected = max(usable, key=lambda c: c["acquisition"]) if best is not None else usable[0]
        self._record_decision({"stage": "candidate_selection", "candidates": copy.deepcopy(candidates),
                              "selected_config": selected["config"]})
        return selected

    def _after_observation(self, proposal, perf):
        success = math.isfinite(perf)
        response = dict(next_configuration=proposal["config"], **(proposal.get("prediction") or {}))
        self.semantic_history.append({"configuration": proposal["config"], "proposal": response,
                                      "status": "success" if success else "failure",
                                      **({"objective": -perf} if success else {})})
