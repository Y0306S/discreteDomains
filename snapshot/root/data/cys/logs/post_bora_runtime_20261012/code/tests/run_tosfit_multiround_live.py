"""Approved short ToSFiT integration check with real Qwen, GPT, ML fits and restarts."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from mindware import CASH
from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.strict_recommendation import RECOMMENDATION_POLICY
from mindware.components.optimizers.transfer_hpo.protocol import allowed_task_ids
from mindware.components.utils.constants import CLASSIFICATION
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS
from run_online_llm_live import (
    AuditedToSFiT, benchmark_args, default_repair_verified, fingerprint, memory, write,
)
import run_toy_categorical_optimizers as runner


LIMITS = {"warmup_rounds": 2, "batch_size": 4, "fine_tuning_rounds": 3,
          "ml_calls": 9, "gpt_calls": 6, "wall_seconds": 1800}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def calls(directory):
    path = directory / "ml_calls.json"
    return read(path)["calls"] if path.exists() else 0


class MultiRoundRuntime(AuditedToSFiT):
    def __init__(self, directory):
        super().__init__(directory)
        self.recipe.update(batch_size=LIMITS["batch_size"],
                           n_marginal_likelihood_warmup_steps=LIMITS["warmup_rounds"])
        self.propose_calls = 0
        self.batch_tokens, self.backward_count = [], 0

    def _score(self, model, tokens, feature=False):
        if feature:
            self.batch_tokens.append(tokens[0].tolist())
        if model is self.model and model.training and torch.is_grad_enabled():
            check(tokens[0].tolist() == self.batch_tokens[self.backward_count],
                  "Fine-tuning received different tokens from Qwen generation")
            self.backward_count += 1
        return super()._score(model, tokens, feature)

    def propose(self, prompt, step):
        self._ensure_loaded()
        before = fingerprint(self.model)
        self.batch_tokens, self.backward_count = [], 0
        sample = super().propose(prompt, step)
        self.propose_calls += 1
        after = fingerprint(self.model)
        reference = fingerprint(self.reference)
        changed = before["sha256"] != after["sha256"]
        check(changed == sample["gradient_step"], "Unexpected zero update or warmup weight change")
        check(after["finite"] and reference["finite"], "Non-finite model weights")
        check(reference == self.baseline, "Reference weights changed")
        check(sample["tokens"] == self.batch_tokens[0], "Proposed tokens do not match generation")
        expected_backward = LIMITS["batch_size"] if sample["gradient_step"] else 0
        check(self.backward_count == expected_backward, "Incorrect gradient minibatch count")
        write(self.directory / "generations" / ("step-%03d" % step) / "round_audit.json", {
            "bo_step": step, "policy_version": self.policy_version,
            "gradient_step": sample["gradient_step"], "before": before, "after": after,
            "reference": reference, "changed": changed,
            "original_token_minibatches_verified": self.backward_count})
        return sample


def make_optimizer(directory):
    check(torch.cuda.device_count() == 2, "Expose exactly the two approved GPUs")
    for device in range(2):
        available, total = torch.cuda.mem_get_info(device)
        check(available >= 35 * 2**30, "Insufficient GPU headroom")
        torch.cuda.set_per_process_memory_fraction(32 * 2**30 / total, device)
    args = benchmark_args("tosfit", directory)
    args.n_trials, args.extractor_max_calls = LIMITS["ml_calls"], LIMITS["gpt_calls"]
    _, train, _, dataset_path, profile = runner.load_real_data(args)
    runtime = MultiRoundRuntime(directory)
    params = runner.build_optimizer_params(args, "tosfit", args.algorithm, profile)
    params["tosfit_runtime"] = runtime
    os.environ["ML_SESSION_DIR"] = str(directory / "session")
    identity = {"task": "cls/car/lightgbm", "limits": LIMITS, "seed": args.seed,
                "recommendation_policy": RECOMMENDATION_POLICY,
                "diagnostic_recipe": runtime.recipe, "protocol": "multiround-live-v1"}
    os.environ["ML_RUN_ID"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    CONFIGS.ORIGINAL_DATA_DIR = str(directory / "history")
    CONFIGS.HISTORY_RUN_SCOPE = str(directory / "history" / "scope")
    KnowledgeDatabase._instance = None
    cash = CASH(include_algorithms=args.algorithm, task_type=CLASSIFICATION, metric=args.metric,
                data_node=train, evaluation=args.evaluation,
                resampling_params={"test_size": args.valid_size, "shuffle": False},
                optimizer="tosfit", time_limit=None, amount_of_resource=LIMITS["ml_calls"],
                per_run_time_limit=120, inner_iter_num_per_iter=1,
                output_dir=str(directory / "outputs"), seed=args.seed, n_jobs=1,
                task_id="car_lightgbm_tosfit_multiround", topk=LIMITS["ml_calls"],
                optimizer_params=params)
    opt = cash.optimizer
    opt.asset_identity.update(recipe=runtime.recipe, diagnostic_only=True, diagnostic_limits=LIMITS)
    if not (directory / "settings.json").exists():
        write(directory / "settings.json", dict(identity, dataset_path=str(dataset_path),
              assets=opt.asset_identity, context=opt.task_context, gpu_limit_gib=32))
    ml_checkpoint.attach(opt)
    return opt, runtime


def run_phase(directory, phase):
    def evaluate(*args, **kwargs):
        count = calls(directory)
        check(count < LIMITS["ml_calls"], "ML-call allowance exhausted")
        write(directory / "ml_calls.json", {"calls": count + 1})
        return wrapper_evaluater(*args, **kwargs)

    with patch("mindware.components.optimizers.online_llm_optimizer.wrapper_evaluater", side_effect=evaluate):
        opt, runtime = make_optimizer(directory)

        def iterate():
            before = opt.trial_cnt
            opt.iterate()
            check(opt.trial_cnt == before + 1, "No progress within diagnostic allowance")
            row = opt.traj_rows[-1]
            write(directory / "live_progress.json", {
                "phase": phase, "bo_steps": opt.bo_step, "policy_version": runtime.policy_version,
                "ml_calls": calls(directory), "gpt_calls": opt.extractor_client.calls,
                "gp_observations": runtime.gp.n_observations, "latest": row})
            print(json.dumps({"phase": phase, "bo_steps": opt.bo_step,
                              "policy_version": runtime.policy_version, "score": row["perf"],
                              "note": row["note"]}), flush=True)
            check(row["oracle_called"] and math.isfinite(row["perf"]),
                  "Recommendation rejected or ML fit failed; inspect the recorded decision")

        if phase == "initial":
            check(opt.trial_cnt == 0, "Initial phase requires a fresh session")
            opt.resolver.resolve('{"max_depth": -1, "learning_rate": 0.05}')
            record = dict(opt.resolver.last_record)
            check(default_repair_verified(record, opt.config_space), "Live default-repair probe failed")
            check(record["config"]["lightgbm:learning_rate"] == 0.05, "Valid value changed")
            write(directory / "extractor_probe.json", record)
            opt.extractor_calls = opt.extractor_client.calls
            while opt.bo_step < LIMITS["warmup_rounds"]:
                iterate()
            proposal = opt._suggest()
            proposal["suggest_time"] = 0.0
            check(proposal["config"] is not None, "First fine-tuned recommendation was rejected")
            check(runtime.policy_version == 1, "Expected first fine-tuning round")
            ml_checkpoint.prepare(opt, proposal)
            result = {"passed": True, "policy": fingerprint(runtime.model),
                      "reference": fingerprint(runtime.reference), "policy_version": 1,
                      "ml_calls": calls(directory), "gpt_calls": opt.extractor_client.calls,
                      "pending_config": proposal["config"], "pending_features": proposal["sample"]["features"],
                      "gp_observations": runtime.gp.n_observations, "pending": True}
        elif phase == "continue":
            saved = read(directory / "initial.json")
            check(fingerprint(runtime.model) == saved["policy"], "Pending policy did not reload exactly")
            check(fingerprint(runtime.reference) == saved["reference"], "Reference did not reload exactly")
            check(calls(directory) == saved["ml_calls"], "Reload invoked the ML evaluator")
            check(opt.extractor_client.calls == saved["gpt_calls"], "Reload called GPT")
            check(runtime.gp.n_observations == saved["gp_observations"], "GP observations changed on reload")
            check(ml_checkpoint.pending(opt) is not None, "Pending recommendation was lost")
            iterate()
            check(runtime.policy_version == 1 and runtime.propose_calls == 0, "Resume repeated fine-tuning")
            check(opt.extractor_client.calls == saved["gpt_calls"], "Pending evaluation called GPT again")
            check(json.loads(opt.traj_rows[-1]["config_json"]) == saved["pending_config"],
                  "Resume evaluated a different configuration")
            while runtime.policy_version < LIMITS["fine_tuning_rounds"]:
                iterate()
            check(opt.bo_step == 5 and runtime.gp.n_observations == 5, "Incomplete round feedback")
            feature = torch.tensor([saved["pending_features"]], dtype=torch.float64)
            result = {"passed": True, "pending_resume_without_retraining": True,
                      "policy": fingerprint(runtime.model), "reference": fingerprint(runtime.reference),
                      "policy_version": runtime.policy_version, "bo_steps": opt.bo_step,
                      "gp_observations": runtime.gp.n_observations,
                      "gp_prediction": runtime.gp.posterior_mean(feature).tolist(),
                      "ml_calls": calls(directory), "gpt_calls": opt.extractor_client.calls,
                      "memory": memory()}
        else:
            saved = read(directory / "continue.json")
            check(fingerprint(runtime.model) == saved["policy"], "Final policy did not reload exactly")
            check(fingerprint(runtime.reference) == saved["reference"], "Final reference mismatch")
            check(runtime.policy_version == 3 and runtime.propose_calls == 0, "Final reload trained the model")
            check(ml_checkpoint.pending(opt) is None, "Unexpected pending final evaluation")
            check(calls(directory) == saved["ml_calls"], "Final reload invoked ML")
            check(opt.extractor_client.calls == saved["gpt_calls"], "Final reload invoked GPT")
            check(runtime.gp.n_observations == saved["gp_observations"], "Final GP observation mismatch")
            feature = torch.tensor([read(directory / "initial.json")["pending_features"]], dtype=torch.float64)
            check(runtime.gp.posterior_mean(feature).tolist() == saved["gp_prediction"],
                  "GP predictions changed after checkpoint reload")
            result = {"passed": True, "policy_version": runtime.policy_version,
                      "ml_calls": calls(directory), "gpt_calls": opt.extractor_client.calls,
                      "new_ml_calls": 0, "new_gpt_calls": 0, "new_gradient_rounds": 0}
        write(directory / (phase + ".json"), result)
        print("Completed phase: " + phase, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=("initial", "continue", "verify"))
    args = parser.parse_args()
    check("cls/car/lightgbm" in allowed_task_ids(), "Task is outside train80")
    if args.phase:
        try:
            run_phase(args.out, args.phase)
            return 0
        except Exception as exc:
            from mindware.modules.configs import AVAILABLE_LLMs
            message = str(exc)
            key = AVAILABLE_LLMs["gpt-4o-mini"].get("api_key")
            if key:
                message = message.replace(key, "<redacted>")
            write(args.out / (args.phase + ".json"), {
                "passed": False, "error_type": type(exc).__name__, "error": message})
            print(type(exc).__name__ + ": " + message, flush=True)
            return 1
    check(not args.out.exists() or not any(args.out.iterdir()), "Use a fresh diagnostic directory")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"passed": False, "limits": LIMITS, "live_qwen": True, "live_gpt": True, "phases": {}}
    write(args.out / "report.json", report)
    try:
        for phase in ("initial", "continue", "verify"):
            remaining = LIMITS["wall_seconds"] - (time.monotonic() - started)
            check(remaining > 0, "Diagnostic wall-time allowance exhausted")
            completed = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                        "--out", str(args.out.resolve()), "--phase", phase], timeout=remaining)
            path = args.out / (phase + ".json")
            report["phases"][phase] = read(path) if path.exists() else {"passed": False}
            write(args.out / "report.json", report)
            check(completed.returncode == 0 and report["phases"][phase]["passed"], "Phase failed: " + phase)
        report["passed"] = True
    except (AssertionError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    report["elapsed_seconds"] = time.monotonic() - started
    report["ml_calls"] = calls(args.out)
    usage = args.out / "session" / "extractor_usage.json"
    report["gpt_calls"] = read(usage)["calls"] if usage.exists() else 0
    write(args.out / "report.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
