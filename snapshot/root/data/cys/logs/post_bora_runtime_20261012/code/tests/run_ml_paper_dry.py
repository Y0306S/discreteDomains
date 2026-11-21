"""Real-Qwen, real-ML checks for the four paper adapters, separate from campaigns."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ConfigSpace import Configuration

from examples.benchmark.validate_semanticopt_ml import setup_task
from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.bora_optimizer import BORAOptimizer
from mindware.components.optimizers.bolt_ml_initializer import BOLTMLInitializer
from mindware.components.optimizers.bolt_ml_optimizer import BOLTMLOptimizer
from mindware.components.optimizers.semanticopt_optimizer import SemanticOptOptimizer
from mindware.components.optimizers.tosfit_optimizer import ToSFiTOptimizer
from mindware.components.optimizers.tosfit_runtime import ToSFiTRuntime
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.transfer_hpo.protocol import allowed_task_ids
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase


def fingerprint(model):
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in model.named_parameters():
            digest.update(name.encode())
            for block in value.detach().reshape(-1).split(4 * 1024 * 1024):
                if not torch.isfinite(block).all():
                    raise ValueError("Non-finite model weights")
                digest.update(block.cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write(path, value):
    ml_checkpoint.atomic_json(path, value)


def verify_saved_weights(model, path):
    from safetensors import safe_open
    path = Path(path)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    state = model.state_dict()
    count, ignored = 0, []
    ignored_patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", []) or []
    for shard in sorted(set(index.values())):
        with safe_open(path / shard, framework="pt", device="cpu") as saved:
            for name in saved.keys():
                if name not in state and any(re.search(pattern, name) for pattern in ignored_patterns):
                    ignored.append(name)
                    continue
                expected = saved.get_tensor(name)
                if name not in state or not torch.equal(state[name].detach().cpu(), expected.to(state[name].dtype)):
                    raise ValueError("Loaded full-model tensor mismatch: " + name)
                count += 1
    missing = sorted(set(dict(model.named_parameters())) - set(index))
    if missing:
        raise ValueError("Loaded model contains parameters absent from checkpoint: " + str(missing))
    return {"model_path": str(path), "verified_tensors": count, "checkpoint_tensors": len(index),
            "architecture_ignored_tensors": ignored, "ignored_patterns": ignored_patterns}


class AuditedRuntime(ToSFiTRuntime):
    def __init__(self, directory):
        super().__init__("cuda:0", "cuda:1")
        self.directory, self.initial_hash = directory, None

    def _ensure_loaded(self):
        super()._ensure_loaded()
        if self.initial_hash is None:
            self.initial_hash = fingerprint(self.reference)
            if fingerprint(self.model) != self.initial_hash:
                raise ValueError("Fresh ToSFiT policy differs from its clean reference")
            write(self.directory / "fresh_model.json", {"policy_and_reference_sha256": self.initial_hash})

    def propose(self, prompt, step):
        print("ToSFiT proposal", step + 1, "batch", self.recipe["batch_size"], flush=True)
        sample = super().propose(prompt, step)
        if sample["gradient_step"]:
            policy, reference = fingerprint(self.model), fingerprint(self.reference)
            evidence = {"policy_sha256": policy, "reference_sha256": reference,
                        "policy_changed": policy != self.initial_hash,
                        "reference_unchanged": reference == self.initial_hash}
            write(self.directory / "gradient_update.json", evidence)
            if not evidence["policy_changed"] or not evidence["reference_unchanged"]:
                raise ValueError("ToSFiT update/frozen-reference check failed")
        return sample


def run(args):
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not args.verify and not args.resume:
        sources = [Path(__file__), ROOT / "examples/benchmark/hf_qwen_client.py"] + list(
            (ROOT / "mindware/components/optimizers").glob("*ml*.py")) + [
            ROOT / "mindware/components/optimizers" / name for name in (
                "bora_optimizer.py", "tosfit_optimizer.py", "tosfit_runtime.py",
                "semanticopt_optimizer.py", "strict_recommendation.py", "isolated_evaluator.py",
                "lolbo_sampler.py", "lolbo_turbo_port.py", "pt4opt_optuna_optimizer.py")]
        hashes = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        write(directory / "source_hashes.json", hashes)
    task_type, dataset, algorithm = args.task.split("/")
    if args.task not in allowed_task_ids():
        raise ValueError("Development dry run must be a train80 task")
    task = {"task_type": task_type, "dataset": dataset, "algorithm": algorithm,
            "metric": "bal_acc" if task_type == "cls" else "mse"}
    cs, evaluator, profile, provenance = setup_task(task, directory, 1)
    KnowledgeDatabase._instance = None
    KnowledgeDatabase.instance(task_type=CLASSIFICATION if task_type == "cls" else REGRESSION,
        metric=task["metric"], task_id=args.task, datanode=evaluator.data_node,
        config_space=cs, include_algorithms=[algorithm])
    os.environ.update(ML_SESSION_DIR=str(directory / "session"),
                      ML_RUN_ID="paper-dry-20261005:" + args.method + ":" + args.task,
                      ML_RUNTIME_POLICY_ID="2400s-no-overall-timeout")
    count = {"bora": 8, "semanticopt": 8, "bolt": 10, "bolt_ml": 100, "tosfit": 100}[args.method]
    if args.evaluations is not None:
        if not 1 <= args.evaluations <= 100:
            raise ValueError("Dry-run evaluation budget must be between 1 and 100")
        count = args.evaluations
    kwargs = dict(evaluator=evaluator, config_space=cs, evaluation_limit=count,
        per_run_time_limit=2400, time_limit=None, task_id=args.task,
        output_dir=str(directory), task_type=task_type, seed=1, topk=100,
        recommendation_default_fallback=True, extractor_max_tokens=2048,
        extractor_timeout=60, extractor_max_calls=100, dataset_profile=profile,
        generator_max_tokens=4096)
    if args.bora_gpu_memory_gib is not None:
        kwargs["gpu_memory_limit_gib"] = args.bora_gpu_memory_gib
    if args.method == "tosfit":
        runtime = AuditedRuntime(directory)
        kwargs["tosfit_runtime"] = runtime
    cls = {"bora": BORAOptimizer, "tosfit": ToSFiTOptimizer,
           "semanticopt": SemanticOptOptimizer, "bolt": BOLTMLInitializer,
           "bolt_ml": BOLTMLOptimizer}[args.method]
    opt = cls(**kwargs)
    restored = ml_checkpoint.attach(opt)
    if args.verify:
        state = json.loads((directory / "before_reload.json").read_text())
        if opt.trial_cnt != state["evaluations"] or ml_checkpoint.pending(opt) is not None:
            raise ValueError("Ledger did not restore exactly")
        evidence = {"evaluations": opt.trial_cnt, "new_evaluations": 0,
                    "ledger_restored": opt.traj_rows == state["rows"]}
        if args.method == "tosfit":
            evidence.update(policy_reloaded=fingerprint(runtime.model) == state["policy_sha256"],
                            reference_reloaded=fingerprint(runtime.reference) == state["reference_sha256"],
                            gp_observations=runtime.gp.n_observations)
            write(directory / "full_model_verification.json",
                  verify_saved_weights(runtime.reference, opt.asset_identity["base_model"]))
        elif args.method in ("bora", "bolt", "bolt_ml"):
            opt.llm_client._ensure_loaded()
            write(directory / "full_model_verification.json",
                  verify_saved_weights(opt.llm_client._model, opt.llm_client.model_path))
        if args.method == "bolt_ml":
            evidence.update(bo_state_restored=opt.bo_state() == state["bo_state"],
                            gp_reloaded=fingerprint(opt.study.sampler._model) == state["gp_sha256"]
                            if state["gp_sha256"] else opt.study.sampler._model is None)
        if not all(v for k, v in evidence.items() if k.endswith(("restored", "reloaded"))):
            raise ValueError("Reload check failed")
        write(directory / "reload_verified.json", evidence)
        print("RELOAD VERIFIED", json.dumps(evidence), flush=True)
        return
    if restored and not args.resume:
        raise ValueError("Initial dry run requires a fresh directory")
    if args.resume and not restored:
        raise ValueError("Resume requested without committed evaluations")
    if not restored:
        write(directory / "identity.json", {"method": args.method, "task": args.task,
        "assets": opt.asset_identity, "provenance": provenance, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"), "evaluation_timeout_seconds": 2400,
        "scope": "development dry run, not benchmark results", "torch": torch.__version__})
    started = time.monotonic()
    previous_elapsed = 0.0
    diagnostic_actions = (["ASSISTANT_SELECTS"] if args.bora_selection_only else
                          ["BO", "ASSISTANT_SUGGESTS", "BO", "ASSISTANT_SELECTS"] if args.bora_actions else [])
    action_index = 0
    if args.bora_actions and (args.method != "bora" or args.resume):
        raise ValueError("Action coverage requires a fresh BORA diagnostic")
    if args.bora_selection_only and (args.method != "bora" or not args.resume):
        raise ValueError("Selection recheck requires resuming a saved BORA diagnostic")
    if args.bora_selection_only:
        write(directory / "selection_recheck.json", {
            "resumed_evaluations": restored, "code_root": str(ROOT),
            "preserved_parent": str(directory.with_name(directory.name + "_before_selection_fix")),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("mindware/components/optimizers/bora_ml_runtime.py", "tests/run_ml_paper_dry.py")},
            "scope": "append live selection check after float-identity and distinct-candidate fixes"})
    if restored:
        control_record = json.loads((directory / "default_control.json").read_text())
        previous_elapsed = json.loads((directory / "progress.json").read_text())["elapsed_seconds"]
    else:
        control = wrapper_evaluater(Configuration(cs, values=dict(cs.get_default_configuration())),
                                    evaluator, 2400, force_isolation=True)
        control_record = {"configuration": dict(cs.get_default_configuration()),
            "objective": control.objectives, "trial_state": control.trial_state,
            "elapsed_seconds": control.elapsed_time, "scope": "separate default comparison"}
        write(directory / "default_control.json", control_record)
    while opt.trial_cnt < count:
        expected = None
        if diagnostic_actions and opt.trial_cnt >= 5 and not opt.queue:
            if action_index == len(diagnostic_actions):
                break
            expected = diagnostic_actions[action_index]
            # Change only diagnostic state; exercise the released get_action and actual Qwen calls.
            action_type = type(opt.policy.current_action)
            opt.policy.current_action = action_type.BO
            opt.policy._skip_update = False
            opt.policy._failures = opt.policy._plateau_duration if expected != "BO" else -1
            opt.policy._lower_unc_threshold, opt.policy._upper_unc_threshold = .3, .5
            opt.policy._uncertainty = .8 if expected == "ASSISTANT_SUGGESTS" else .4
            opt._assistant()._model_info["max_tokens"] = 4096
            opt._record_decision({"stage": "diagnostic_action_fixture", "expected_action": expected,
                "scope": "forced policy state and summary trigger, real proposals and evaluations"})
            action_index += 1
        opt.iterate()
        row = opt.traj_rows[-1]
        if expected is not None and row["note"] != expected:
            raise ValueError("BORA did not execute requested live action: " + str((expected, row["note"])))
        if expected == "ASSISTANT_SELECTS":
            assert len(opt.selection_candidates) == 5, "Expected five distinct offered candidates"
        print(args.method, args.task, opt.trial_cnt, row["evaluation_status"], row["perf"], row["note"], flush=True)
        write(directory / "progress.json", {"evaluations": opt.trial_cnt,
              "latest": row, "elapsed_seconds": previous_elapsed + time.monotonic() - started})
        if args.method == "tosfit" and runtime.policy_version >= 1:
            break
        if args.stop_after is not None and opt.trial_cnt >= args.stop_after:
            break
    state = {"evaluations": opt.trial_cnt, "rows": opt.traj_rows}
    if diagnostic_actions:
        actions = {r["note"] for r in opt.traj_rows}
        assert {"BO", "ASSISTANT_SUGGESTS", "ASSISTANT_SELECTS"} <= actions, actions
        assert not opt.queue, "Diagnostic must finish evaluating the selected batch"
        summaries = [r for r in opt.decision_records if r.get("stage") == "bora_history_summary"]
        assert len(summaries) >= 2, "Both intervention prompts must exercise history summaries"
        write(directory / "action_coverage.json", {"actions": sorted(actions), "summary_calls": len(summaries),
            "scope": "forced action state; real Qwen, official assistant, GP and evaluator",
            "policy_trust_updates": opt.policy._trust_model.iteration_rolling_trusts})
    if args.method == "bolt_ml":
        state.update(bo_state=opt.bo_state(), gp_sha256=fingerprint(opt.study.sampler._model)
                     if opt.study.sampler._model is not None else None)
    if args.method == "tosfit":
        state.update(policy_sha256=fingerprint(runtime.model), reference_sha256=fingerprint(runtime.reference))
    elif getattr(opt.llm_client, "adapter_verification", None):
        write(directory / "adapter_verification.json", opt.llm_client.adapter_verification)
    write(directory / "before_reload.json", state)
    write(directory / "result.json", {"method": args.method, "task": args.task,
        "successful": sum(r["evaluation_status"] == "SUCCESS" for r in opt.traj_rows),
        "evaluations": opt.trial_cnt, "best_score": opt.incumbent_perf,
        "default_score": -control_record["objective"][0],
        "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        "resumed_from_evaluation": restored,
        "actions": [r["note"] for r in opt.traj_rows],
        "extractor_calls": opt.extractor_calls,
        "online_updates": runtime.policy_version if args.method == "tosfit" else None,
        "diagnostic_stop_after": args.stop_after,
        "gpu_memory": [{"peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30}
                       for i in range(torch.cuda.device_count())],
        "passed_execution": all(r["evaluation_status"] == "SUCCESS" for r in opt.traj_rows),
        "full_bolt_handoff_verified": (opt.trial_cnt > 10 and opt.study.sampler._model is not None)
        if args.method == "bolt_ml" else False if args.method == "bolt" else None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bora", "tosfit", "semanticopt", "bolt", "bolt_ml"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--evaluations", type=int)
    parser.add_argument("--bora-actions", action="store_true")
    parser.add_argument("--bora-selection-only", action="store_true")
    parser.add_argument("--bora-gpu-memory-gib", type=float)
    args = parser.parse_args()
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    try:
        run(args)
    except Exception as exc:
        write(args.out / ("reload_error.json" if args.verify else "error.json"),
              {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise
