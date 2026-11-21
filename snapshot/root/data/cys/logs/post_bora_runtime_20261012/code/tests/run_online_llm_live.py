"""Bounded real Qwen/GPT/CASH integration tests. Invocations make live API calls."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/benchmark"))

import torch

from mindware import CASH
from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.online_llm_optimizer import QWEN_BASE
from mindware.components.optimizers.strict_recommendation import (
    GPTConfigurationExtractor, RecommendationError, StrictRecommendationResolver, RECOMMENDATION_POLICY,
)
from mindware.components.optimizers.tosfit_runtime import ToSFiTRuntime
from mindware.components.optimizers.transfer_hpo.protocol import allowed_task_ids
from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs
from mindware.components.utils.constants import CLASSIFICATION
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS
from hf_qwen_client import HuggingFaceQwenClient
import run_toy_categorical_optimizers as runner


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    ml_checkpoint.atomic_json(path, clean(value))


def fingerprint(model):
    digest, finite, count = hashlib.sha256(), True, 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            flattened = parameter.detach().reshape(-1)
            for block in flattened.split(4 * 1024 * 1024):
                finite = finite and bool(torch.isfinite(block).all())
                digest.update(block.cpu().view(torch.uint8).numpy().tobytes())
            count += parameter.numel()
    return {"sha256": digest.hexdigest(), "finite": finite, "parameters": count}


def memory():
    return [{"device": i, "allocated_gib": torch.cuda.memory_allocated(i) / 2**30,
             "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
             "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30}
            for i in range(torch.cuda.device_count())]


class AuditedQwen(HuggingFaceQwenClient):
    def __init__(self, directory):
        super().__init__(model_path=QWEN_BASE, temperature=0.7, max_new_tokens=512,
                         enable_thinking=False)
        self.directory, self.initial = directory, None

    def _ensure_loaded(self):
        super()._ensure_loaded()
        if self.initial is None:
            self.initial = fingerprint(self._model)
            write(self.directory / "base_model.json", self.initial)


class AuditedToSFiT(ToSFiTRuntime):
    def __init__(self, directory):
        super().__init__("cuda:0", "cuda:1")
        self.directory, self.baseline = directory, None
        self.audit_step, self.sample_number = 0, 0

    def _ensure_loaded(self):
        super()._ensure_loaded()
        if self.baseline is None:
            policy, reference = fingerprint(self.model), fingerprint(self.reference)
            if policy != reference or not policy["finite"]:
                raise RuntimeError("Fresh policy and reference weights do not match")
            self.baseline = reference
            write(self.directory / "base_models.json", {"policy": policy, "reference": reference})

    def _score(self, model, tokens, feature=False):
        if feature:
            folder = self.directory / "generations" / ("step-%03d" % self.audit_step)
            write(folder / ("sample-%02d.json" % self.sample_number), {
                "original_tokens": tokens[0].tolist(),
                "raw_text": self.tokenizer.decode(tokens[0], skip_special_tokens=True)})
            self.sample_number += 1
        return super()._score(model, tokens, feature)

    def propose(self, prompt, step):
        self.audit_step, self.sample_number = step, 0
        print("ToSFiT BO step %d: generating batch of %d" %
              (step, self.recipe["batch_size"]), flush=True)
        started = time.time()
        sample = super().propose(prompt, step)
        audit = {"bo_step": step, "policy_version": self.policy_version,
                 "seconds": time.time() - started, "memory": memory(),
                 "generation_batch_size": sample["generation_batch_size"]}
        if sample["gradient_step"]:
            policy, reference = fingerprint(self.model), fingerprint(self.reference)
            audit.update(policy=policy, reference=reference,
                         policy_changed=policy["sha256"] != self.baseline["sha256"],
                         reference_unchanged=reference == self.baseline)
            write(self.directory / "gradient_update.json", audit)
            if not policy["finite"] or not audit["policy_changed"] or not audit["reference_unchanged"]:
                raise RuntimeError("Actual Qwen9B weight-change/frozen-reference check failed")
        write(self.directory / "generations" / ("step-%03d" % step) / "summary.json", audit)
        return sample


def extractor_probe(directory, selected_cases=None, budget_path=None, max_calls=3, recommendation_file=None):
    cs = build_hpo_cs("cls", "lightgbm")
    extractor = GPTConfigurationExtractor(max_tokens=512, timeout=60, max_calls=max_calls,
                                          budget_path=budget_path or directory / "api_usage.json")
    resolver = StrictRecommendationResolver(cs, extractor)
    cases = [("valid_json", json.dumps(dict(cs.get_default_configuration())), True),
             ("incomplete_prose", "I recommend max_depth 7 and learning_rate 0.05.", True),
             ("explicit_invalid", '{"max_depth": -1}', True),
             ("categorical_json", '{"algorithm": "lightgbm", "max_depth": 7}', True)]
    if recommendation_file is not None:
        cases.append(("recorded_qwen", json.loads(recommendation_file.read_text())["raw_text"], True))
    records = []
    for name, text, expected in cases:
        if selected_cases and name not in selected_cases:
            continue
        before = extractor.calls
        try:
            config = dict(resolver.resolve(text))
            accepted, error = True, None
        except RecommendationError as exc:
            accepted, error, config = False, str(exc), None
        extraction_succeeded = "extraction" in resolver.last_record
        passed = accepted == expected and (name == "valid_json" or extraction_succeeded)
        if name == "explicit_invalid":
            passed = passed and default_repair_verified(resolver.last_record, cs)
        record = dict(resolver.last_record, case=name, expected_accepted=expected,
                      passed=passed, error=error, config=config,
                      api_calls=extractor.calls - before)
        records.append(record)
        write(directory / "extractor.json", records)
        print("Extractor %s: accepted=%s, API calls=%d" % (name, accepted, record["api_calls"]), flush=True)
    passed = all(r["passed"] and (r["case"] != "valid_json" or r["api_calls"] == 0) for r in records)
    return {"passed": passed, "api_calls": extractor.calls, "cases": records,
            "recommendation_policy": RECOMMENDATION_POLICY,
            "recommendation_file": str(recommendation_file) if recommendation_file else None}


def default_repair_verified(record, cs):
    name = "lightgbm:max_depth"
    return (record.get("accepted") and record.get("assignments", {}).get(name) == -1
            and record.get("config", {}).get(name) == cs[name].default_value
            and record.get("repairs", {}).get(name) == {
                "original": -1, "replacement": cs[name].default_value, "reason": "out_of_range"})


def benchmark_args(method, directory):
    old = sys.argv
    sys.argv = ["run", "--optimizer", method, "--dataset", "car", "--task-type", "cls",
                "--algorithm", "lightgbm", "--metric", "bal_acc", "--n-trials",
                "8" if method == "bora" else "33", "--time-limit", "none",
                "--per-run-time-limit", "120", "--llm-max-tokens", "512",
                "--extractor-max-tokens", "512", "--extractor-timeout", "60",
                "--extractor-max-calls", "10" if method == "bora" else "33",
                "--tosfit-device", "cuda:0", "--tosfit-reference-device", "cuda:1",
                "--skip-test-eval", "--max-rounds", "1", "--dataset-retries", "1",
                "--output-dir", str(directory / "outputs"), "--history-dir", str(directory / "history")]
    try:
        return runner.parse_args()
    finally:
        sys.argv = old


def optimizer_probe(method, directory, resume_only=False):
    expected_devices = 1 if method == "bora" else 2
    if torch.cuda.device_count() != expected_devices:
        raise RuntimeError("Expose exactly %d selected GPUs for this probe" % expected_devices)
    for device in range(expected_devices):
        available, total = torch.cuda.mem_get_info(device)
        # Reload needs no training workspace; the full allocation cap must still be free.
        required_headroom_gib = 32 if resume_only else 35
        if available < required_headroom_gib * 2**30:
            raise RuntimeError("Insufficient free GPU headroom for bounded probe")
        torch.cuda.set_per_process_memory_fraction(32 * 2**30 / total, device)
    args = benchmark_args(method, directory)
    _, train, _, dataset_path, profile = runner.load_real_data(args)
    params = runner.build_optimizer_params(args, method, args.algorithm, profile)
    runtime = AuditedToSFiT(directory) if method == "tosfit" else None
    client = AuditedQwen(directory) if method == "bora" else None
    if runtime:
        params["tosfit_runtime"] = runtime
    if client:
        params["llm_client"] = client
    session_dir = directory / "session"
    os.environ["ML_SESSION_DIR"] = str(session_dir)
    identity = {"method": method, "task": "cls/car/lightgbm", "attempt_limit": args.n_trials,
                "seed": args.seed, "dataset_path": str(dataset_path), "protocol": "live-probe-v2",
                "recommendation_policy": RECOMMENDATION_POLICY}
    os.environ["ML_RUN_ID"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    CONFIGS.ORIGINAL_DATA_DIR = str(directory / "history")
    CONFIGS.HISTORY_RUN_SCOPE = str(directory / "history" / "scope")
    KnowledgeDatabase._instance = None
    cash = CASH(include_algorithms=args.algorithm, task_type=CLASSIFICATION, metric=args.metric,
                data_node=train, evaluation=args.evaluation,
                resampling_params={"test_size": args.valid_size, "shuffle": False},
                optimizer=method, time_limit=None, amount_of_resource=args.n_trials,
                per_run_time_limit=120, inner_iter_num_per_iter=1,
                output_dir=str(directory / "outputs"), seed=args.seed, task_id="car_lightgbm_" + method,
                topk=args.n_trials, optimizer_params=params)
    opt = cash.optimizer
    if (session_dir / "checkpoint.pkl").exists() and not resume_only:
        raise RuntimeError("Use a new output directory or --resume-only")
    restored = ml_checkpoint.attach(opt)
    if resume_only:
        if restored == 0:
            raise RuntimeError("No committed observations to resume")
        report = {"passed": True, "restored_attempts": restored, "memory": memory(),
                  "new_ml_evaluations": 0, "new_api_calls": 0}
        if runtime:
            expected = json.loads((directory / "gradient_update.json").read_text())
            current = fingerprint(runtime.model)
            reference = fingerprint(runtime.reference)
            report.update(policy=current, reference=reference, policy_version=runtime.policy_version)
            report["passed"] = current == expected["policy"] and reference == expected["reference"]
        else:
            report["passed"] = opt.initialized and len(opt.configs) == restored
        return report
    write(directory / "settings.json", dict(identity, assets=opt.asset_identity, context=opt.task_context,
                                            gpu_limit_gib=32, profile=profile))
    preflight_path = directory / "extractor_preflight.json"
    if method == "bora" and not preflight_path.exists():
        # These calls share BORA's persistent allowance, including interrupted attempts.
        records = []
        for text, expected in (("I recommend max_depth 7 and learning_rate 0.05.", True),
                               ('{"max_depth": -1}', True)):
            try:
                opt.resolver.resolve(text)
                accepted = True
            except RecommendationError:
                accepted = False
            record = dict(opt.resolver.last_record)
            record["passed"] = accepted == expected and "extraction" in record
            if text == '{"max_depth": -1}':
                record["passed"] = record["passed"] and default_repair_verified(record, opt.config_space)
            records.append(record)
            write(preflight_path, records)
    forced_selection = False
    while not opt._budget_exhausted():
        if method == "bora" and opt.trial_cnt == 6 and not opt.queue:
            # Exercise selection without misrepresenting a short run as a benchmark score.
            opt.policy._failures = opt.policy.plateau_duration
            opt.policy._uncertainty = (opt.policy._lower_unc_threshold + opt.policy._upper_unc_threshold) / 2
            forced_selection = True
            write(directory / "forced_branch.json", {"after_attempts": 6, "branch": "ASSISTANT_SELECTS",
                                                       "reason": "integration branch coverage"})
        before = opt.trial_cnt
        opt.iterate()
        if opt.trial_cnt == before:
            raise RuntimeError("Optimizer made no progress within its allowance")
        latest = opt.traj_rows[-1]
        print("%s attempt %d/%d score=%s note=%s" % (
            method, opt.trial_cnt, args.n_trials, latest["perf"], latest["note"]), flush=True)
        write(directory / "live_progress.json", {"attempts": opt.trial_cnt, "latest": latest,
                                                 "memory": memory(), "bo_step": getattr(opt, "bo_step", None)})
        if runtime and runtime.policy_version >= 1:
            break
    scores = [row["perf"] for row in opt.traj_rows]
    routes = [record.get("route") for record in opt.decision_records]
    report = {"passed": all(math.isfinite(score) for score in scores),
              "attempts": opt.trial_cnt, "oracle_calls": sum(row["oracle_called"] for row in opt.traj_rows),
              "successful_evaluations": sum(math.isfinite(score) for score in scores),
              "best_score": opt.incumbent_perf, "routes": routes,
              "api_calls": opt.extractor_calls, "memory": memory(),
              "schema_fingerprint": opt.schema_fingerprint, "forced_selection": forced_selection,
              "notes": [row["note"] for row in opt.traj_rows]}
    if runtime:
        report.update(policy_version=runtime.policy_version, bo_steps=opt.bo_step,
                      gp_observations=runtime.gp.n_observations if runtime.gp else 0)
        report["passed"] = report["passed"] and runtime.policy_version >= 1
    else:
        final = fingerprint(client._model)
        report.update(model_unchanged=final == client.initial,
                      accepted_qwen_proposals=sum(r.get("accepted", False) and "config" in r
                                                 for r in opt.decision_records),
                      extractor_preflight_passed=all(r["passed"] for r in json.loads(preflight_path.read_text())),
                      selection_accepted=any(r.get("accepted") and "indices" in r for r in opt.decision_records))
        report["passed"] = (report["passed"] and report["model_unchanged"] and report["selection_accepted"]
                            and report["accepted_qwen_proposals"] > 0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("extractor", "bora", "tosfit"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume-only", action="store_true")
    parser.add_argument("--extractor-case", action="append",
                        choices=("valid_json", "incomplete_prose", "explicit_invalid", "categorical_json", "recorded_qwen"))
    parser.add_argument("--recommendation-file", type=Path)
    parser.add_argument("--extractor-budget-path", type=Path)
    parser.add_argument("--extractor-max-calls", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if "cls/car/lightgbm" not in allowed_task_ids():
        raise RuntimeError("Live-test task is outside train80")
    started = time.time()
    def timeout_signal(signum, frame):
        raise TimeoutError("Live probe reached its process wall-time limit")
    signal.signal(signal.SIGTERM, timeout_signal)
    try:
        report = extractor_probe(args.out, args.extractor_case, args.extractor_budget_path,
                                 args.extractor_max_calls, args.recommendation_file) if args.method == "extractor" else optimizer_probe(
            args.method, args.out, args.resume_only)
    except Exception as exc:
        message = str(exc)
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            message = message.replace(api_key, "<redacted>")
        report = {"passed": False, "error_type": type(exc).__name__, "error": message,
                  "traceback": traceback.format_exc().replace(api_key, "<redacted>") if api_key else traceback.format_exc()}
    report.update(method=args.method, elapsed_seconds=time.time() - started, resume_only=args.resume_only)
    write(args.out / ("resume_report.json" if args.resume_only else "report.json"), report)
    print(json.dumps(clean(report), indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
