#!/usr/bin/env python3
"""Pin and check QueryOpt/peptide BO environments without launching optimization.

The three import profiles run in separate CPU-only interpreters because QueryOpt
and peptides both provide a package named lolbo, and original FSBO needs an older
GPyTorch version. No package installation or oracle calls are performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "query-peptide-bo-environment-v1"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_files(expected):
    failures = [str(path) for path, value in expected.items()
                if not Path(path).is_file() or digest(path) != value]
    if failures:
        raise ValueError("Pinned files missing or changed: " + ", ".join(failures[:10]))


def cpu_environment():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled")
    env.pop("PYTHONPATH", None)
    return env


def sampler_probe(dimensions):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    result = {}
    for name, sampler in (("random", optuna.samplers.RandomSampler(seed=0)),
                          ("tpe", optuna.samplers.TPESampler(seed=0))):
        study = optuna.create_study(direction="minimize", sampler=sampler)
        # Twelve synthetic trials exercise TPE past its ten startup observations.
        for _ in range(12):
            trial = study.ask()
            values = [trial.suggest_float("z_%d" % i, -1.0, 1.0)
                      for i in range(dimensions)]
            study.tell(trial, sum(value * value for value in values))
        if len(study.trials) != 12 or any(t.state.name != "COMPLETE" for t in study.trials):
            raise ValueError("Sampler ask/tell did not complete")
        result[name] = {"dimensions": dimensions, "synthetic_trials": 12,
                        "oracle_calls": 0, "passed": True}
    return result


def profile_probe(profile, root):
    """Called only in an isolated child process, with all GPUs hidden."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Environment probes must hide all GPUs")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import gpytorch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    packages = ["torch", "gpytorch", "numpy", "scipy", "scikit-learn"]
    modules = ["torch", "gpytorch", "numpy", "scipy", "sklearn"]
    checks = {}
    if profile == "fsbo":
        from fsbo_modules import MLP, ExactGPLayer
        from fsbo_utils import EI
        packages += ["xgboost", "tensorboard"]
        modules += ["fsbo_modules", "fsbo_utils", "xgboost", "tensorboard"]
        for dimensions in (64, 256):
            features = MLP(dimensions, hidden_size=[128, 128]).eval()
            with torch.no_grad():
                x = features(torch.randn(6, dimensions))
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = ExactGPLayer(x[:4], torch.arange(4).float(), likelihood,
                                     {"kernel": "rbf", "ard": False}, 128).eval()
                posterior = model(x[4:])
                ei = EI(3.0, posterior.mean.numpy(), posterior.stddev.numpy())
                if not np.isfinite(ei).all() or np.any(ei < 0):
                    raise ValueError("Original FSBO posterior/EI failed")
            checks[str(dimensions)] = {"untrained_exact_gp_and_ei": True}
    else:
        packages += ["botorch", "linear-operator", "optuna", "lightning", "pandas"]
        modules += ["botorch", "linear_operator", "optuna", "lightning", "pandas",
                    "lolbo.lolbo", "lolbo.utils.bo_utils.turbo"]
        if profile == "query":
            packages += ["psycopg", "networkx", "sqlglot"]
            modules += ["psycopg", "networkx", "sqlglot", "vae.model", "oracle.oracle",
                        "your_tasks.your_objective_functions"]
            dimensions, hidden = 64, (64, 64)
        else:
            packages += ["python-Levenshtein"]
            modules += ["Levenshtein", "lolbo.info_transformer_vae_objective",
                        "apex_oracle.APEX_predict", "uniref_vae.load_vae"]
            dimensions, hidden = 256, (256, 256)
        if profile == "peptide":
            original_load = torch.load

            def load_on_cpu(*args, **kwargs):
                kwargs["map_location"] = "cpu"
                return original_load(*args, **kwargs)

            # APEX imports CUDA-stored weights eagerly. Only this probe remaps them.
            with patch.object(torch, "load", side_effect=load_on_cpu):
                for name in modules:
                    importlib.import_module(name)
            checks["apex_weight_load"] = "map_location=cpu in probe only"
        else:
            for name in modules:
                importlib.import_module(name)
        from lolbo.utils.bo_utils.ppgpr import GPModelDKL
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        x = torch.randn(6, dimensions)
        model = GPModelDKL(x[:4], likelihood, hidden_dims=hidden).eval()
        with torch.no_grad():
            posterior = model(x[4:])
            if not (torch.isfinite(posterior.mean).all()
                    and torch.isfinite(posterior.variance).all()
                    and (posterior.variance >= 0).all()):
                raise ValueError("Native deep-kernel GP forward pass failed")
        checks["untrained_native_gp_forward"] = True
        checks["samplers"] = sampler_probe(dimensions)
    imported = {name: importlib.import_module(name) for name in modules}
    own_names = ["fsbo_modules", "fsbo_utils"] if profile == "fsbo" else ["lolbo.lolbo"]
    for name in own_names:
        if root.resolve() not in Path(imported[name].__file__).resolve().parents:
            raise ValueError("Imported another domain's module: " + name)
    module_paths = {name: str(Path(module.__file__).resolve())
                    for name, module in imported.items() if getattr(module, "__file__", None)}
    sitecustomize = sys.modules.get("sitecustomize")
    if sitecustomize is not None:
        module_paths["sitecustomize"] = str(Path(sitecustomize.__file__).resolve())
    return dict(profile=profile, passed=True, python=sys.version, executable=sys.executable,
                prefix=sys.prefix, base_prefix=sys.base_prefix, sys_path=sys.path,
                packages={name: importlib.metadata.version(name) for name in packages},
                module_paths=module_paths, module_hashes={path: digest(path) for path in module_paths.values()},
                checks=checks, gpu_used=False, optimizer_updates=0, oracle_calls=0)


def run_probe(script, profile, python, root):
    result = subprocess.run([str(python), str(script), "--probe", profile, "--root", str(root)],
                            cwd=root, env=cpu_environment(), capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(profile + " environment check failed:\n" + result.stderr[-4000:])
    return json.loads(result.stdout)


def collect(config, script):
    query = Path(config["query_campaign"])
    peptide = Path(config["peptide_campaign"])
    plan, recipe = read(query / "plan.json"), read(peptide / "recipe.json")
    for name in ("initial_observations", "proposals", "budget", "batch_size", "seed"):
        if config["query"][name] != plan[name]:
            raise ValueError("Query setup differs from the approved campaign: " + name)
    for name in ("initialization", "proposals", "batch_size", "similarity"):
        if config["peptide"][name] != recipe[name]:
            raise ValueError("Peptide setup differs from the approved campaign: " + name)
    if config["peptide"]["budget"] != recipe["initialization"] + recipe["proposals"]:
        raise ValueError("Peptide total must include initialization")
    pins = {str(query / name): value for name, value in plan["files_sha256"].items()}
    pins.update(recipe["code"])
    pins.update(recipe["assets"])
    pins.update({str(query / "train800.txt"): plan["train_sha256"],
                 str(query / "test200.txt"): plan["test_sha256"],
                 plan["checkpoint"]: plan["vae_sha256"]})
    for path in (query / "plan.json", peptide / "recipe.json", script):
        pins[str(path)] = digest(path)
    fsbo_root = Path(config["fsbo_repository"])
    for path in fsbo_root.glob("*.py"):
        pins[str(path)] = digest(path)
    check_files(pins)
    train, test = (query / "train800.txt").read_text().split(), (query / "test200.txt").read_text().split()
    if len(set(train)) != 800 or len(set(test)) != 200 or set(train) & set(test):
        raise ValueError("Query split is not train800/test200")
    if (len(set(recipe["train"])) != 900 or len(set(recipe["test"])) != 100
            or set(recipe["train"]) & set(recipe["test"])):
        raise ValueError("Peptide split is not train900/test100")
    profiles = {}
    for name, python, root in (("query", config["runtime_python"], query / "native"),
                               ("peptide", config["runtime_python"], Path(recipe["upstream"])),
                               ("fsbo", config["fsbo_python"], fsbo_root)):
        profiles[name] = run_probe(script, name, python, root)
        pins.update(profiles[name]["module_hashes"])
    statuses = []
    for path in (query / "outputs").glob("*/lolbo/seed0/status.json"):
        status = read(path)
        if status["task_id"] not in train:
            raise ValueError("Unexpected query task in training outputs")
        statuses.append(status)
    complete = [s for s in statuses if s.get("status") == "complete"
                and s.get("proposals") == 1000 and s.get("charged_observations") == 1049]
    hold = peptide / "launch_hold.json"
    readiness = {}
    for domain in ("query", "peptide"):
        for method in config["methods"]:
            blockers = []
            adapter = (domain == "query" and method != "fsbo") or (domain == "peptide" and method == "lolbo")
            if not adapter:
                blockers.append("Target-run adapter not integrated into the audited runner")
            if method == "fsbo":
                blockers += ["Export and validate eligible training sources",
                             "Train domain-specific original FSBO and verify reload"]
            if domain == "peptide":
                blockers += config["peptide"]["unresolved"][:2]
                if method in ("random", "tpe"):
                    blockers.append(config["peptide"]["unresolved"][2])
            blockers.append("Launch-time resource allocation and live component check")
            readiness[domain + "/" + method] = dict(environment_verified=True, adapter_present=adapter,
                                                     production_ready=False, remaining=blockers)
    return dict(format=FORMAT, created_at=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, config=config, profiles=profiles, pinned_files=pins,
                source_progress=dict(query_train800_statuses=dict(Counter(s.get("status") for s in statuses)),
                                     complete_at_check=len(complete),
                                     note="Completion does not establish uncensored source eligibility; export is still required"),
                peptide_launch_hold=read(hold) if hold.exists() else None,
                peptide_blocked_tasks=recipe["blocked_tasks"], readiness=readiness,
                environment_checks_passed=True, production_ready=False, launched=False,
                resource_assignment=None, oracle_calls=0, optimizer_updates=0,
                query_preflight_command=[config["runtime_python"], str(query / "code/run_query_ceb_baselines.py"),
                    "--method", "random", "--preflight", "--qp-root", str(query / "native"),
                    "--checkpoint", plan["checkpoint"], "--budget", "1049", "--batch-size", "10"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/query_peptide_bo/setup.json")
    parser.add_argument("--output", type=Path, help="New setup directory; existing directories are refused")
    parser.add_argument("--verify", type=Path, help="Recheck a prepared setup without writing to it")
    parser.add_argument("--probe", choices=("query", "peptide", "fsbo"), help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(profile_probe(args.probe, args.root)))
        return
    script = Path(__file__).resolve()
    if args.verify:
        if args.output:
            parser.error("--output and --verify are mutually exclusive")
        manifest = read(args.verify / "manifest.json")
        if manifest.get("format") != FORMAT:
            raise ValueError("Unknown setup format")
        check_files(manifest["pinned_files"])
        for name, expected in manifest["profiles"].items():
            root = {"query": Path(manifest["config"]["query_campaign"]) / "native",
                    "peptide": Path(read(Path(manifest["config"]["peptide_campaign"]) / "recipe.json")["upstream"]),
                    "fsbo": Path(manifest["config"]["fsbo_repository"])}[name]
            current = run_probe(script, name, expected["executable"], root)
            for key in ("python", "packages", "module_paths", "module_hashes"):
                if current[key] != expected[key]:
                    raise ValueError("Environment changed: " + name + "/" + key)
        print(json.dumps(dict(verified=str(args.verify.resolve()), environment_checks_passed=True,
                              production_ready=False, launched=False, oracle_calls=0)))
        return
    if not args.output:
        parser.error("Provide --output to prepare or --verify to check an existing setup")
    if args.output.exists():
        parser.error("Output already exists; use --verify or select a new directory")
    config = read(args.config)
    if config.get("format") != FORMAT or config["launch"]["enabled"]:
        raise ValueError("Expected a preparation-only configuration")
    report = collect(config, script)
    report["pinned_files"][str(args.config.resolve())] = digest(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(script, args.output / script.name)
    report["pinned_files"][str((args.output / script.name).resolve())] = digest(script)
    for name, profile in report["profiles"].items():
        requirements = args.output / (name + ".requirements.txt")
        requirements.write_text("# Recorded installed versions; do not install into active environments.\n" +
                                "\n".join(k + "==" + v for k, v in sorted(profile["packages"].items())) + "\n")
        report["pinned_files"][str(requirements.resolve())] = digest(requirements)
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(prepared=str(args.output.resolve()), profiles=list(report["profiles"]),
                          pinned_files=len(report["pinned_files"]), environment_checks_passed=True,
                          production_ready=False, launched=False, oracle_calls=0), indent=2))


if __name__ == "__main__":
    main()
