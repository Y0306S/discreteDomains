#!/usr/bin/env python3
"""Fit immutable source GPs and run the upstream RGPE(TAF) policy."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration
from smac.epm.util_funcs import get_types
from smac.configspace import convert_configurations_to_array
from rgpe.methods.rgpe import RGPE, compute_rank_weights, copula_transform, get_gaussian_process
from rgpe.methods.taf import TAF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.transfer_hpo.protocol import acquisition_diagnostics, atomic_json, sha256_file, validate_dump
from mindware.components.optimizers.transfer_hpo.schema import cs_from_spec, schema_fingerprint


def configs_from_rows(cs, rows, algorithm):
    """Reject a bad batch instead of shifting labels or original candidate indices."""
    out = []
    for index, row in enumerate(rows):
        values = dict(row)
        values.setdefault("algorithm", algorithm)
        try:
            config = Configuration(cs, values=values)
            cs.check_configuration(config)
        except Exception as exc:
            raise ValueError("Invalid configuration at original row %d: %s" % (index, exc)) from exc
        out.append(config)
    return out


class StrictRGPE(RGPE):
    """Keep upstream ranking failures visible instead of switching to target-only BO."""

    def _train(self, X, Y):
        if self.normalization == "mean/var":
            self.Y_mean_ = float(Y.mean())
            self.Y_std_ = float(Y.std()) or 1.0
            scaled = (Y.flatten() - self.Y_mean_) / self.Y_std_
        else:
            self.Y_mean_, self.Y_std_ = 0.0, 1.0
            scaled = copula_transform(Y) if self.normalization == "Copula" else Y
        target = get_gaussian_process(bounds=self.bounds, types=self.types,
                                      configspace=self.configspace, rng=self.rng, kernel=None)
        self.target_model = target.train(X, scaled)
        self.model_list_ = self.base_models + [target]
        if len(X) < 3:
            self.weights_ = np.ones(len(self.model_list_)) / len(self.model_list_)
            dropped = np.full(len(self.base_models), np.nan)
        else:
            self.weights_, dropped = compute_rank_weights(
                train_x=X, train_y=scaled, base_models=self.base_models, target_model=target,
                num_samples=self.num_posterior_samples, sampling_mode=self.sampling_mode,
                weight_dilution_strategy=self.weight_dilution_strategy,
                number_of_function_evaluations=self.number_of_function_evaluations,
                rng=self.rng, alpha=self.alpha,
            )
        if not np.isfinite(self.weights_).all() or not np.isclose(self.weights_.sum(), 1.0):
            raise RuntimeError("Invalid RGPE ranking weights")
        self.weights_over_time.append(self.weights_.copy())
        self.p_drop_over_time.append(dropped)
        return self


def make_model(cs, data, seed, normalization="None", n_trials=100, samples=1000):
    types, bounds = get_types(cs, None)
    return StrictRGPE(training_data=data, num_posterior_samples=samples,
                      weight_dilution_strategy="probabilistic-ld", number_of_function_evaluations=n_trials,
                      sampling_mode="bootstrap", variance_mode="average", normalization=normalization,
                      configspace=cs, types=types, bounds=bounds, seed=seed)


def load_dump(path):
    dump = json.loads(Path(path).read_text())
    task_ids = validate_dump(dump)
    cs = cs_from_spec(dump["configspace"])
    if schema_fingerprint(cs) != dump["schema_fingerprint"]:
        raise ValueError("ConfigSpace round-trip changed the source schema")
    return dump, task_ids, cs


def cmd_fit(args):
    dump, task_ids, cs = load_dump(args.dump)
    cache = Path(args.cache)
    if cache.exists() or cache.with_suffix(".json").exists():
        raise ValueError("Source cache already exists; select a new cache path")
    data = {}
    for index, task in enumerate(task_ids):
        record = dump["tasks"][task]
        configs = configs_from_rows(cs, record["configs"], dump["algorithm"])
        y = np.asarray(record["y_minimize"], dtype=float)
        if len(configs) != len(y) or len(y) < 2 or not np.isfinite(y).all():
            raise ValueError("Invalid source labels for %s" % task)
        data[index] = {"configurations": configs, "y": y}
    model = make_model(cs, data, args.seed, args.normalization)
    identity = {"method": "rgpe_taf", "format_version": 2,
                "group": dump["group"], "schema_fingerprint": dump["schema_fingerprint"],
                "dump_sha256": sha256_file(args.dump), "source_tasks": task_ids,
                "seed": args.seed, "normalization": args.normalization}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump({"identity": identity, "base_models": model.base_models,
                     "rng": model.rng.get_state()}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(str(temporary), str(cache))
    atomic_json(cache.with_suffix(".json"), dict(identity, status="complete", cache_sha256=sha256_file(cache)))
    print(json.dumps({"cache": str(cache), "n_source_gps": len(task_ids)}))
    return 0


def cmd_suggest(args):
    dump, task_ids, cs = load_dump(args.dump)
    req = json.loads(Path(args.request).read_text())
    cache = Path(args.cache)
    meta = json.loads(cache.with_suffix(".json").read_text())
    if (meta.get("status") != "complete" or meta["dump_sha256"] != sha256_file(args.dump)
            or meta["source_tasks"] != task_ids or meta["schema_fingerprint"] != req.get("schema_fingerprint")
            or meta["cache_sha256"] != sha256_file(cache)):
        raise ValueError("RGPE cache, target schema, or source provenance mismatch")
    # This cache is a local artifact produced by fit-sources, never an external checkpoint.
    with cache.open("rb") as stream:
        payload = pickle.load(stream)
    if payload["identity"] != {k: v for k, v in meta.items() if k not in ("status", "cache_sha256")}:
        raise ValueError("RGPE cache metadata mismatch")
    model = make_model(cs, {}, int(req.get("seed", meta["seed"])), meta["normalization"],
                       int(req["n_trials"]), int(req.get("num_posterior_samples", 1000)))
    model.base_models = payload["base_models"]
    model.rng.set_state(payload["rng"])
    observed = configs_from_rows(cs, req["target_configs"], dump["algorithm"])
    y = np.asarray(req["y_minimize"], dtype=float)
    if not len(observed) or len(observed) != len(y) or not np.isfinite(y).all():
        raise ValueError("Target configurations and finite objectives must align")
    pending = configs_from_rows(cs, req["pending_configs"], dump["algorithm"])
    if not pending:
        raise ValueError("Empty candidate pool")
    x = convert_configurations_to_array(observed)
    model.train(x, y)
    acquisition = TAF(model)
    acquisition.update(model=model, X=x)
    scores = acquisition._compute(convert_configurations_to_array(pending)).reshape(-1)
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite RGPE acquisition values")
    atomic_json(args.out, {"index": int(np.argmax(scores)), "n_sources": len(task_ids),
                           "weights": model.weights_.tolist(), "cache_sha256": meta["cache_sha256"],
                           "source_weights": [{"task": task, "weight": float(weight)}
                                              for task, weight in zip(task_ids, model.weights_[:-1])],
                           "target_weight": float(model.weights_[-1]),
                           "acquisition": acquisition_diagnostics("TAF", scores)})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-sources")
    fit.add_argument("--dump", required=True)
    fit.add_argument("--cache", required=True)
    fit.add_argument("--seed", type=int, default=1)
    fit.add_argument("--normalization", choices=["None", "Copula", "mean/var"], default="None")
    suggest = sub.add_parser("suggest")
    suggest.add_argument("--dump", required=True)
    suggest.add_argument("--cache", required=True)
    suggest.add_argument("--request", required=True)
    suggest.add_argument("--out", required=True)
    args = parser.parse_args()
    return cmd_fit(args) if args.command == "fit-sources" else cmd_suggest(args)


if __name__ == "__main__":
    sys.exit(main())
