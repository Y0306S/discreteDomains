"""Training-source GPs queried only at observations earned by the current run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
from pathlib import Path

import numpy as np

from mindware.components.experience.bank_gates import hp_stem
from mindware.components.experience.surrogate_landscape import (
    _cfg_get, configs_to_array, get_algo_cs,
)


SCHEMA = "source_gp.v1"
UTILITY = "higher_is_better; regression=-mse"
ML_TRAIN_80_FIXED_SHA256 = "ff6bdb76fe4b52b439fc7b7c5bbf380f4a4db5debb7d56cee8ed01a260a821b4"
FANOVA_ADVICE_POLICY = {
    "importance_kind": "main_effect", "min_importance": 0.05,
    "min_parameters": 1, "max_parameters": None, "require_region": True,
    "require_full_fanova": True,
    "bad_region_policy": "subtract_good_region",
}


class SourceGPError(RuntimeError):
    """An RS artifact or observation violates the declared protocol."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "ConfigSpace", "openbox")
    }}


def manifest_tasks(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise SourceGPError(f"Expected a task list: {path}")
    pairs = set()
    for row in rows:
        row = row.get("metadata", row)
        ds, algo = row.get("dataset"), row.get("algorithm")
        if not ds or not algo:
            raise SourceGPError(f"Missing dataset/algorithm in {path}")
        pairs.add((str(ds).lower(), str(algo).lower()))
    return pairs


def source_key(card):
    prob = card["problem"]
    return str(prob["dataset_id"]).lower(), str(prob["algorithm_id"]).lower()


def space_digest(cs):
    from ConfigSpace.read_and_write import json as cs_json
    return json_digest(json.loads(cs_json.write(cs)))


def configuration(cs, raw, algorithm):
    from ConfigSpace import Configuration
    from ConfigSpace.hyperparameters import Constant, UnParametrizedHyperparameter
    values = {}
    for hp in cs.get_hyperparameters():
        # AdaBoost has its own "algorithm" HP, separate from the CASH algorithm selector.
        value = raw.get(f"{algorithm}:{hp.name}")
        if value is None:
            value = _cfg_get(raw, hp.name, algorithm)
        if value is None and isinstance(hp, (Constant, UnParametrizedHyperparameter)):
            value = hp.default_value
        if value is not None:
            values[hp.name] = value
    try:
        # Configuration validates missing active parameters, bounds, and forbiddens.
        return Configuration(cs, values=values, allow_inactive_with_values=True)
    except Exception as exc:
        raise SourceGPError(f"Invalid {algorithm} observation: {exc}") from exc


def canonical_score(score, higher_is_better):
    score = float(score)
    if not math.isfinite(score) or not isinstance(higher_is_better, bool):
        raise SourceGPError("Expected a finite score with an explicit metric direction")
    return score if higher_is_better else -score


def _valid_region(region):
    if isinstance(region, list):
        return bool(region) and all(isinstance(v, (str, bool, int, float)) and
                                   (not isinstance(v, float) or math.isfinite(v)) for v in region)
    if isinstance(region, dict) and set(region) == {"low", "high"}:
        low, high = region["low"], region["high"]
        return (isinstance(low, (int, float)) and isinstance(high, (int, float))
                and math.isfinite(low) and math.isfinite(high) and low <= high)
    return False


def subtract_good_region(bad, good):
    """Return the bad set minus the good set, preserving open numeric boundaries."""
    if not _valid_region(bad):
        return []
    if isinstance(bad, list):
        if good is not None and not isinstance(good, list):
            raise SourceGPError("Good and bad regions have incompatible types")
        remaining = [v for v in bad if v not in (good or [])]
        return [remaining] if remaining else []
    low, high = bad["low"], bad["high"]
    if good is None:
        return [{"low": low, "high": high, "low_inclusive": True, "high_inclusive": True}]
    if not isinstance(good, dict):
        raise SourceGPError("Good and bad regions have incompatible types")
    if high < good["low"] or low > good["high"]:
        return subtract_good_region(bad, None)
    regions = []
    if low < good["low"]:
        regions.append({"low": low, "high": good["low"],
                        "low_inclusive": True, "high_inclusive": False})
    if high > good["high"]:
        regions.append({"low": good["high"], "high": high,
                        "low_inclusive": False, "high_inclusive": True})
    return regions


def format_region(region):
    if isinstance(region, list):
        return json.dumps(region, sort_keys=True, allow_nan=False)
    left = "[" if region.get("low_inclusive", True) else "("
    right = "]" if region.get("high_inclusive", True) else ")"
    return f"{left}{region['low']}, {region['high']}{right}"


def fanova_advice(card):
    """Select all important parameters; one usable parameter is sufficient."""
    knowledge = card.get("knowledge") or {}
    fanova = knowledge.get("fanova") or {}
    decision = {"engine": fanova.get("engine"), "reasons": [], "parameter_decisions": []}
    decision["importance_1d"] = fanova.get("importance_1d") or {}
    decision["importance_total"] = fanova.get("importance_total") or {}
    decision["gate_config"] = dict(FANOVA_ADVICE_POLICY)
    if fanova.get("engine") != "fanova_full_cs":
        decision["reasons"].append("not_full_configspace_fanova")
    factors = {hp_stem(f["name"]): f for f in
               (knowledge.get("factors_all") or knowledge.get("factors") or [])}
    approved = []
    for raw_name, raw_importance in sorted(decision["importance_1d"].items()):
        name = hp_stem(raw_name)
        try:
            importance = float(raw_importance)
        except (TypeError, ValueError):
            importance = float("nan")
        parameter = {"name": name, "importance": importance if math.isfinite(importance) else None}
        decision["parameter_decisions"].append(parameter)
        if not math.isfinite(importance) or importance < FANOVA_ADVICE_POLICY["min_importance"]:
            parameter["omission_reason"] = "below_importance_threshold_or_invalid"
            continue
        fac = factors.get(name, {})
        regions = {key: fac[key] for key in ("good_region", "bad_region")
                   if _valid_region(fac.get(key))}
        if regions:
            approved.append({"name": name, "importance": importance, **regions,
                             "avoid_regions": subtract_good_region(regions.get("bad_region"),
                                                                   regions.get("good_region"))})
            parameter["selected"] = True
        else:
            parameter["omission_reason"] = "missing_valid_range"
    approved.sort(key=lambda item: (-item["importance"], item["name"]))
    decision["approved_parameters"] = approved
    decision["keep_hps"] = [item["name"] for item in approved]
    decision["n_keep"] = len(approved)
    if not approved:
        decision["reasons"].append("no_important_parameters_with_ranges")
    decision["usable_for_prompt"] = not decision["reasons"] and bool(approved)
    if not decision["usable_for_prompt"]:
        return "", decision
    prob = card["problem"]
    lines = [f"Similar task: {prob['dataset_id']} ({prob['algorithm_id']})"]
    for item in approved:
        parts = []
        if "good_region" in item:
            parts.append("prefer " + format_region(item["good_region"]))
        if item["avoid_regions"]:
            parts.append("avoid " + " or ".join(format_region(r) for r in item["avoid_regions"]))
        lines.append(f"- {item['name']}: {'; '.join(parts)}")
    return "\n".join(lines), decision


def rs_position(current_trial, start=4, stop=9, source="llm"):
    active = int(start) <= int(current_trial) < int(stop)
    return {"ranker": "source_gp", "global_evaluation": int(current_trial),
            "window_start": int(start), "window_stop_exclusive": int(stop),
            "rs_position": int(current_trial) - int(start) + 1 if active else None,
            "window_active": active, "source": source, "selected_ids": [],
            "status": "pending" if active and source == "llm" else
                      ("tpe_slot_no_prompt" if active and source == "tpe" else "outside_window")}


def successful_observations(rows):
    out = []
    for row in rows:
        score = float(row["perf"])
        if math.isfinite(score):
            cfg = row["config_json"]
            out.append({"trial": int(row["iter_idx"]) + 1, "score": score,
                        "source": row["stage"],
                        "config": json.loads(cfg) if isinstance(cfg, str) else dict(cfg)})
    return out


def _build_source_model(task):
    index, card, output = task
    from mindware.components.optimizers.ml_checkpoint import atomic_pickle
    from openbox.surrogate.base.build_gp import create_gp_model
    from openbox.utils.util_funcs import get_types

    ds, algo = source_key(card)
    prob = card["problem"]
    expected = ("bal_acc", True) if prob["task_type"] == "classification" else ("mse", False)
    if (prob.get("metric"), prob.get("higher_is_better")) != expected:
        raise SourceGPError(f"Unsupported source metric: {ds}/{algo}")
    cs = get_algo_cs(algo, prob["task_type"])
    trials = card["retrieval"]["trials"]
    configs, values, rejected = [], [], []
    for trial in trials:
        try:
            value = canonical_score(trial["score"], prob["higher_is_better"])
            config = configuration(cs, trial["config"], algo)
        except (SourceGPError, ValueError, TypeError) as exc:
            rejected.append({"trial": trial.get("trial"), "reason": str(exc)})
            continue
        configs.append(config)
        values.append(value)
    record = {"id": card["id"], "dataset": ds, "algorithm": algo,
              "task_type": prob["task_type"], "metric": prob["metric"],
              "original_higher_is_better": prob["higher_is_better"],
              "space_sha256": space_digest(cs), "trials_sha256": json_digest(trials),
              "n_training_observations": len(values), "rejected_observations": rejected,
              "fanova": fanova_advice(card)[1]}
    if len(values) < 3:
        record.update(status="excluded", reason="fewer_than_three_valid_training_observations")
    else:
        types, bounds = get_types(cs)
        model = create_gp_model(model_type="gp", config_space=cs, types=types,
                                bounds=bounds, rng=np.random.RandomState(42))
        model.train(configs_to_array(configs, cs), np.asarray(values).reshape(-1, 1))
        file = f"model_{index:04d}.pkl"
        atomic_pickle(output / file, model)
        record.update(status="ready", file=file, sha256=digest(output / file))
    return record


def build_artifact(bank_path, train_manifest, test_manifest, output, *, expected_train_sha256=None, workers=1):
    from mindware.components.experience.retrieve import ExperienceBank
    from mindware.components.optimizers.ml_checkpoint import atomic_json

    if expected_train_sha256 is not None and digest(train_manifest) != expected_train_sha256:
        raise SourceGPError("Training manifest does not match the approved fixed training split")
    bank = ExperienceBank.load(Path(bank_path))
    train, test = manifest_tasks(train_manifest), manifest_tasks(test_manifest)
    keys = [source_key(card) for card in bank.experiences]
    if train & test or set(keys) & test or not set(keys) <= train:
        raise SourceGPError("Source/test contamination or unapproved training source")
    if len(set(keys)) != len(keys):
        raise SourceGPError("Duplicate source task in bank")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = {"schema": SCHEMA, "utility": UTILITY, "environment": environment(),
               "bank_sha256": digest(bank_path), "seed": 42, "backend": "openbox_gp",
               "code_sha256": digest(__file__), "sources": []}
    for label, path in (("train", train_manifest), ("test", test_manifest)):
        name = label + Path(path).suffix
        (output / name).write_bytes(Path(path).read_bytes())
        payload[label] = {"file": name, "sha256": digest(path),
                          "original_path": str(Path(path).resolve())}
    tasks = [(index, card, output) for index, card in enumerate(bank.experiences)]

    def collect(records):
        for record in records:
            payload["sources"].append(record)
            if len(payload["sources"]) % 25 == 0:
                print(f"source GPs: {len(payload['sources'])}/{len(tasks)}", flush=True)

    payload["build_workers"] = max(1, int(workers))
    if payload["build_workers"] == 1:
        collect(map(_build_source_model, tasks))
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=payload["build_workers"], mp_context=get_context("spawn")) as pool:
            collect(pool.map(_build_source_model, tasks))
    atomic_json(output / "manifest.json", payload)
    return payload


class SourceGPBank:
    def __init__(self, directory, bank_path, bank):
        self.directory = Path(directory)
        self.models = {}
        self.spaces = {}
        self.bank = bank
        try:
            manifest = self.directory / "manifest.json"
            self.manifest = json.loads(manifest.read_text())
            self.manifest_sha256 = digest(manifest)
            data = self.manifest
            if data["schema"] != SCHEMA or data["utility"] != UTILITY:
                raise SourceGPError("Incompatible source GP schema or score convention")
            if data["environment"] != environment() or data["code_sha256"] != digest(__file__):
                raise SourceGPError("Source GP code/environment differs from the build")
            if data["bank_sha256"] != digest(bank_path):
                raise SourceGPError("Source GP artifact belongs to a different bank")
            manifests = {}
            for label in ("train", "test"):
                path = self.directory / data[label]["file"]
                if digest(path) != data[label]["sha256"]:
                    raise SourceGPError(f"Changed {label} task manifest")
                manifests[label] = manifest_tasks(path)
            train, test = manifests["train"], manifests["test"]
            self.test_tasks = test
            self.records = {r["id"]: r for r in data["sources"]}
            if len(self.records) != len(data["sources"]) or set(self.records) != {c["id"] for c in bank.experiences}:
                raise SourceGPError("Source GP entries do not match the complete bank")
            keys = [source_key(c) for c in bank.experiences]
            if train & test or set(keys) & test or not set(keys) <= train or len(set(keys)) != len(keys):
                raise SourceGPError("Source/test contamination or duplicate training source")
            for card in bank.experiences:
                record = self.records[card["id"]]
                if (record["dataset"], record["algorithm"]) != source_key(card):
                    raise SourceGPError("Source GP task identity mismatch")
                if record["trials_sha256"] != json_digest(card["retrieval"]["trials"]):
                    raise SourceGPError("Source GP training history mismatch")
                if record["status"] == "ready":
                    self._verify_model(record)
                elif record["status"] != "excluded" or not record.get("reason"):
                    raise SourceGPError("Unexplained source GP exclusion")
        except SourceGPError:
            raise
        except Exception as exc:
            raise SourceGPError(f"Cannot load source GP artifact {directory}: {exc}") from exc

    def _verify_model(self, record):
        if digest(self.directory / record["file"]) != record["sha256"]:
            raise SourceGPError(f"Changed source model: {record['id']}")

    def _space(self, algorithm, task_type):
        key = (algorithm, task_type)
        if key not in self.spaces:
            cs = get_algo_cs(algorithm, task_type)
            self.spaces[key] = cs, space_digest(cs)
        return self.spaces[key]

    def preflight(self, algorithm, task_type):
        _, fingerprint = self._space(algorithm, task_type)
        records = [r for r in self.records.values()
                   if r["algorithm"] == algorithm and r["task_type"] == task_type]
        if not records:
            raise SourceGPError(f"No source models for {task_type}/{algorithm}")
        for record in records:
            if record["space_sha256"] != fingerprint:
                raise SourceGPError(f"Source configuration space changed: {record['id']}")
            if record["status"] == "ready":
                self._model(record)

    def _model(self, record):
        if record["id"] not in self.models:
            try:
                self._verify_model(record)
                self.models[record["id"]] = pickle.loads((self.directory / record["file"]).read_bytes())
            except Exception as exc:
                raise SourceGPError(f"Cannot load model {record['id']}: {exc}") from exc
        return self.models[record["id"]]

    def rank(self, dataset, algorithm, task_type, observations, k=3, floor=0.5, seed=1):
        from mindware.components.experience.retrieve import RetrievalHit, spearman
        audit = {"artifact_sha256": self.manifest_sha256, "utility": UTILITY,
                 "observations": observations, "n_observations": len(observations),
                 "spearman_floor": floor, "candidates": [], "selected_ids": [],
                 "tie_break_seed": int(seed), "tie_break_method": "sha256_task_seed_source"}
        if len(observations) < 3:
            audit["status"] = "insufficient_successful_observations"
            return "", [], audit
        scores = [canonical_score(t["score"], True) for t in observations]
        if len(set(scores)) < 2:
            audit["status"] = "constant_target_scores"
            return "", [], audit
        cs, fingerprint = self._space(algorithm, task_type)
        configs = [configuration(cs, t["config"], algorithm) for t in observations]
        audit["n_distinct_configurations"] = len({json_digest(c.get_dictionary()) for c in configs})
        X = configs_to_array(configs, cs)
        candidates = []
        for card in self.bank.experiences:
            record = self.records[card["id"]]
            if record["algorithm"] != algorithm or record["task_type"] != task_type:
                continue
            if record["dataset"] == str(dataset).lower():
                continue
            if record["space_sha256"] != fingerprint:
                raise SourceGPError(f"Incompatible source configuration space: {record['id']}")
            candidate = {"id": card["id"], "dataset": record["dataset"],
                         "algorithm": algorithm, "spearman": None, "predictions": [],
                         "model_sha256": record.get("sha256"),
                         "tie_break_key": json_digest([int(seed), str(dataset).lower(),
                                                       algorithm, task_type, card["id"]])}
            if record["status"] == "excluded":
                candidate["omission_reason"] = record["reason"]
            else:
                try:
                    predictions, _ = self._model(record).predict(X)
                    predictions = np.asarray(predictions).reshape(-1)
                    if len(predictions) != len(scores) or not np.all(np.isfinite(predictions)):
                        raise ValueError("Invalid source prediction vector")
                except Exception as exc:
                    raise SourceGPError(f"Source prediction failed: {record['id']}: {exc}") from exc
                candidate["predictions"] = predictions.tolist()
                if len(set(predictions)) < 2:
                    candidate["omission_reason"] = "constant_source_predictions"
                else:
                    candidate["spearman"] = float(spearman(predictions.tolist(), scores))
            candidates.append((candidate, card))
        candidates.sort(key=lambda pair: (-(pair[0]["spearman"] if pair[0]["spearman"] is not None else -2),
                                          pair[0]["tie_break_key"], pair[0]["id"]))
        hits, blocks = [], []
        for rank, (candidate, card) in enumerate(candidates, 1):
            candidate["rank"] = rank
            block, decision = fanova_advice(card)
            candidate["fanova"] = decision
            if "omission_reason" not in candidate:
                if candidate["spearman"] < floor - 1e-12:
                    candidate["omission_reason"] = "below_similarity_floor"
                elif not block:
                    candidate["omission_reason"] = "no_usable_fanova_advice"
                elif len(hits) >= k:
                    candidate["omission_reason"] = "selection_full"
                else:
                    candidate["selected"] = True
                    hits.append(RetrievalHit(experience=card, score=candidate["spearman"],
                                             spearman=candidate["spearman"],
                                             features={"source_gp": True, "rank": rank}))
                    blocks.append(block)
            audit["candidates"].append(candidate)
        audit["selected_ids"] = [h.exp_id for h in hits]
        audit["status"] = "selected" if hits else "no_eligible_advice"
        guidance = (
            "Source-task advice: Similarity is tentative and based on the current run's observed "
            "configurations. fANOVA identifies important source parameters; these ranges summarize "
            "historical source results and may not transfer to this task. Use them as suggestions, "
            "weigh the current task's results, and choose the next configuration from the full search "
            "space. Conflicting sources are alternatives. Good ranges are preserved; avoid ranges "
            "exclude every value in the same source's good range. Square brackets include endpoints; "
            "parentheses exclude endpoints."
        )
        return "\n\n".join([guidance, *blocks]) if blocks else "", hits, audit


def attach_source_gp(bank, bank_path, directory, *, expected_train_sha256=ML_TRAIN_80_FIXED_SHA256):
    if not directory:
        raise SourceGPError("source_gp requires --source-gp-cache")
    bank.source_gp = SourceGPBank(directory, bank_path, bank)
    if bank.source_gp.manifest["train"]["sha256"] != expected_train_sha256:
        bank.source_gp = None
        raise SourceGPError("Source GPs were not built from the approved fixed training split")
    return bank
