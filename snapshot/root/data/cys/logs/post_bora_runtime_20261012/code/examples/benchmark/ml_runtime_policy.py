"""Load the optimizer-independent ML evaluation resource policy."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def optional_seconds(value):
    if value.lower() == "none":
        return None
    try:
        seconds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a positive number of seconds or 'none'") from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("Use a positive number of seconds or 'none'")
    return seconds


def load_policy(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    policy = json.loads(raw)
    if policy.get("workload") != "ml" or not policy.get("policy_id"):
        raise ValueError("A named ML runtime policy is required")
    for name in ("per_evaluation_seconds", "attempts_per_search"):
        if type(policy.get(name)) is not int or policy[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    overall = policy.get("overall_search_seconds")
    if overall is not None and (type(overall) is not int or overall <= 0):
        raise ValueError("overall_search_seconds must be null or a positive integer")
    if policy.get("failed_attempts_consume_budget") is not True:
        raise ValueError("This runner counts failed attempts toward the trial budget")
    if policy.get("evaluation_start_method") != "forkserver":
        raise ValueError("The ML policy requires forkserver timeout enforcement")
    if policy.get("optuna_timeout_state", "PRUNED") != "PRUNED" or policy.get("optuna_error_state", "FAIL") != "FAIL":
        raise ValueError("ML timeout feedback uses PRUNED for timeouts and FAIL for errors")
    return {**policy, "source": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def apply_policy(args):
    if not args.runtime_policy:
        args.runtime_policy_details = None
        return
    policy = load_policy(args.runtime_policy)
    args.per_run_time_limit = policy["per_evaluation_seconds"]
    args.time_limit = policy["overall_search_seconds"]
    args.n_trials = policy["attempts_per_search"]
    args.runtime_policy_details = policy
    os.environ["ML_ISOLATED_EVALUATION"] = "1"
    os.environ["ML_RUNTIME_POLICY_ID"] = policy["policy_id"]
    os.environ["ML_PREVIOUS_RUNTIME_POLICY_ID"] = policy.get("resume_from_policy_id", "")
    print(f"[Runtime policy] {policy['policy_id']}: per-evaluation={args.per_run_time_limit}s, "
          f"overall={args.time_limit}, attempts={args.n_trials}, isolation=forkserver")
