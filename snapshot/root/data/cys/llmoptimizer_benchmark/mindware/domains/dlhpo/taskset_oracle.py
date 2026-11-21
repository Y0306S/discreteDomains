"""Score one TaskSet CASH config via the existing train_inner eval wrapper.

Default BO protocol: 1k steps, eval every 1k, one replica, minimum normalized loss.
Paper protocol (Metz et al. 2020 §3.3): 10k steps, eval every 200, 5 replicas.
Inner TF1 training always runs in the ``taskset_eval`` conda env; this module
is called from ``hpo_benchmark`` (Optuna / OpenBox).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mindware.domains.base import EvaluationResult
from mindware.domains.paths import (
    TASKSET_BASELINE_CACHE,
    TASKSET_EVAL_DIR,
    TASKSET_ROOT,
    domain_bench,
)

_WRAPPER = None
_BASELINE: Optional[Dict[str, Any]] = None
_SEED_LOCK = threading.Lock()
_SEED_COUNTER = 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return value


def _load_wrapper():
    global _WRAPPER
    if _WRAPPER is not None:
        return _WRAPPER
    eval_dir = Path(TASKSET_EVAL_DIR)
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import run_taskset_dl_eval_optimized as wrapper  # type: ignore
    runtime_root = os.environ.get("TASKSET_RUNTIME_ROOT")
    if runtime_root:
        root = Path(runtime_root).resolve()
        if not (root / "task_set/train_inner.py").is_file():
            raise FileNotFoundError(f"Invalid TASKSET_RUNTIME_ROOT: {root}")
        wrapper.TASK_SET_ROOT = root
        wrapper.TASK_CONFIGS_DIR = root / "task_set/tasks/configs"
        wrapper.OPTIMIZER_CONFIGS_PATH = root / "task_set/optimizers/configs"
    from taskset_paper_scoring import (  # type: ignore
        PAPER_EVAL_EVERY_N,
        PAPER_NUM_REPLICAS,
        PAPER_TRAINING_STEPS,
        aggregate_replica_scores,
        get_task_best_val,
        load_baseline_cache,
        normalize_valid_outer_value,
        score_replica_curve,
    )

    _WRAPPER = {
        "mod": wrapper,
        "PAPER_TRAINING_STEPS": PAPER_TRAINING_STEPS,
        "PAPER_EVAL_EVERY_N": PAPER_EVAL_EVERY_N,
        "PAPER_NUM_REPLICAS": PAPER_NUM_REPLICAS,
        "aggregate_replica_scores": aggregate_replica_scores,
        "get_task_best_val": get_task_best_val,
        "load_baseline_cache": load_baseline_cache,
        "normalize_valid_outer_value": normalize_valid_outer_value,
        "score_replica_curve": score_replica_curve,
    }
    return _WRAPPER


def _baseline_from_gcs(task_id: str) -> Optional[Dict[str, Any]]:
    npz_dir = TASKSET_ROOT / "gcs_cache" / task_id
    if not npz_dir.is_dir():
        return None
    sys.path.insert(0, str(TASKSET_EVAL_DIR))
    from build_baselines_from_gcs_cache import best_val_for_task  # type: ignore

    return best_val_for_task(npz_dir)


def load_task_baseline(task_id: str, cache_path: Optional[Path] = None) -> Tuple[float, Dict[str, Any]]:
    global _BASELINE
    wrapper = _load_wrapper()
    path = Path(cache_path or TASKSET_BASELINE_CACHE)
    if _BASELINE is None:
        _BASELINE = wrapper["load_baseline_cache"](path)
    best = wrapper["get_task_best_val"](task_id, _BASELINE)
    if best is not None:
        return float(best), {"source": str(path)}
    gcs = _baseline_from_gcs(task_id)
    if gcs is None:
        raise FileNotFoundError(
            f"No paper best_val for {task_id!r}. Expected {path} or "
            f"{TASKSET_ROOT / 'gcs_cache' / task_id}"
        )
    _BASELINE[task_id] = gcs
    best = wrapper["get_task_best_val"](task_id, _BASELINE)
    if best is None:
        raise FileNotFoundError(f"GCS baseline for {task_id!r} has no finite best_val")
    return float(best), {"source": "gcs_cache", "entry": gcs}


def bo_protocol() -> Dict[str, int]:
    """Match taskset_corpus_train_1000s_100t/_queue/env.sh."""
    return {"training_steps": 1000, "eval_every_n": 1000, "num_replicas": 1}


def paper_protocol() -> Dict[str, int]:
    w = _load_wrapper()
    return {
        "training_steps": int(w["PAPER_TRAINING_STEPS"]),
        "eval_every_n": int(w["PAPER_EVAL_EVERY_N"]),
        "num_replicas": int(w["PAPER_NUM_REPLICAS"]),
    }


def smoke_protocol() -> Dict[str, int]:
    return {"training_steps": 400, "eval_every_n": 200, "num_replicas": 1}


# Which scalar becomes EvaluationResult.objective.
#
# "mean_normalized_valid_outer" is the paper §3.3 default: the mean of the
# normalized VALID_OUTER curve. Note that normalization sets 1.0 == validation
# loss at initialization, so the step-0 point contributes a constant 1.0. Over
# the paper's 10k/200 protocol that is 1 of 51 points and negligible; on a short
# curve it dominates (with 2 points the objective cannot go below ~0.5).
#
# "min_normalized_valid_outer" is the paper's own sanctioned alternative
# (§3.3 "similar to taking the minimum"; Appendix A.4 measures a negligible
# change, Pearson rho = 0.911). It has no constant floor, so it keeps the full
# [0, 1] range on short curves -- prefer it whenever training_steps is small.
OBJECTIVE_METRICS = ("mean_normalized_valid_outer", "min_normalized_valid_outer")
DEFAULT_OBJECTIVE_METRIC = "min_normalized_valid_outer"
PAPER_OBJECTIVE_METRIC = "mean_normalized_valid_outer"


def _next_opt_seed(task_id: str, family: str, hps: Dict[str, Any]) -> int:
    global _SEED_COUNTER
    with _SEED_LOCK:
        _SEED_COUNTER += 1
        n = _SEED_COUNTER
    blob = json.dumps(
        {"task": task_id, "family": family, "hps": hps, "n": n, "pid": os.getpid()},
        sort_keys=True,
        default=str,
    )
    return int(hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8], 16) % 900_000_000


def evaluate_taskset_config(
    *,
    task_id: str,
    optimizer_family: str,
    hyperparameters: Dict[str, Any],
    work_dir: Optional[Path] = None,
    baseline_cache: Optional[Path] = None,
    training_steps: Optional[int] = None,
    eval_every_n: Optional[int] = None,
    num_replicas: Optional[int] = None,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    smoke: bool = False,
    warm_timeout: int = 3600,
    replica_timeout: int = 2400,
    debug: bool = False,
) -> EvaluationResult:
    """Register {family, hps} and score with train_inner + paper §3.3 metric."""
    payload = {
        "optimizer_family": str(optimizer_family),
        "hyperparameters": _jsonable(hyperparameters),
    }
    if objective_metric not in OBJECTIVE_METRICS:
        raise ValueError(
            f"objective_metric must be one of {OBJECTIVE_METRICS}, got {objective_metric!r}"
        )
    wrapper = _load_wrapper()
    proto = smoke_protocol() if smoke else bo_protocol()
    steps = int(training_steps if training_steps is not None else proto["training_steps"])
    every = int(eval_every_n if eval_every_n is not None else proto["eval_every_n"])
    nrep = int(num_replicas if num_replicas is not None else proto["num_replicas"])
    # train_inner evals when `step % eval_every_n == 0 or step == training_steps`,
    # starting at step 0. If eval_every_n > training_steps the only eval point is
    # step 0, so value == init_val and EVERY config normalizes to exactly 1.0 --
    # a constant objective, and BO gets no signal at all. Fail loudly instead.
    if every > steps:
        raise ValueError(
            f"eval_every_n ({every}) > training_steps ({steps}): the curve would hold only the "
            "step-0 eval, making the normalized objective a constant 1.0 for every configuration. "
            f"Use eval_every_n <= training_steps (e.g. --eval-every-n {steps})."
        )
    evaluation_info = {
        "oracle": "train_inner",
        "protocol": {"training_steps": steps, "eval_every_n": every, "num_replicas": nrep},
        "objective_metric": objective_metric,
    }
    try:
        best_val, base_info = load_task_baseline(task_id, baseline_cache)
    except Exception as exc:
        return EvaluationResult(
            objective=1e10,
            artifact=payload,
            feasible=False,
            info={**evaluation_info, "error": f"baseline:{exc}"},
        )

    root = Path(work_dir or (domain_bench("taskset") / task_id / "_oracle"))
    custom_dir = root / "custom_opts"
    train_dir = root / "train_inner"
    custom_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    opt_seed = _next_opt_seed(task_id, str(optimizer_family), payload["hyperparameters"])
    mod = wrapper["mod"]
    opt_name, opt_path = mod.create_optimizer_config(
        str(optimizer_family), payload["hyperparameters"], opt_seed, custom_dir
    )
    try:
        mod.register_custom_optimizer(opt_path, opt_seed, debug=debug, force=True)
    except Exception as exc:
        return EvaluationResult(
            objective=1e10,
            artifact=payload,
            feasible=False,
            info={**evaluation_info, "error": f"register:{exc}", "opt_name": opt_name},
        )

    replica_results = mod.run_replicas_parallel(
        task_id,
        opt_name,
        train_dir,
        steps,
        every,
        nrep,
        warm_timeout,
        replica_timeout,
        debug=debug,
        debug_dir=root / "debug" if debug else None,
        show_progress=True,
    )
    scored = []
    details = []
    for rep in replica_results:
        if rep.get("status") != "ok":
            details.append(rep)
            continue
        scores = wrapper["score_replica_curve"](rep["curves"], best_val)
        # Paper Appendix A.4's min-aggregation variant. Computed per replica so
        # aggregate_replica_scores still averages over random initializations
        # (paper §3.3). score_replica_curve already returns every input we need,
        # so the shared scoring module is untouched.
        scores["min_normalized_valid_outer"] = wrapper["normalize_valid_outer_value"](
            scores["min_valid_outer"], scores["init_valid_outer"], best_val
        )
        scored.append(scores)
        details.append({"replica": rep.get("replica"), "status": "ok", **scores})

    if not scored:
        err = details[0].get("error") if details else "all_failed"
        return EvaluationResult(
            objective=1e10,
            artifact=payload,
            feasible=False,
            info={
                **evaluation_info,
                "error": err,
                "opt_name": opt_name,
                "replica_details": details,
            },
        )

    agg = wrapper["aggregate_replica_scores"](scored)
    obj = float(agg.get(objective_metric, float("nan")))
    feasible = math.isfinite(obj)
    return EvaluationResult(
        objective=obj if feasible else 1e10,
        artifact=payload,
        feasible=feasible,
        info={
            **evaluation_info,
            "opt_name": opt_name,
            "best_val_reference": best_val,
            "baseline": base_info,
            # Both aggregations are always recorded, so a run scored one way can be
            # re-analysed the other way without re-training.
            "mean_normalized_valid_outer": agg.get("mean_normalized_valid_outer"),
            "min_normalized_valid_outer": agg.get("min_normalized_valid_outer"),
            # Paper p5: "order optimizers based on validation loss and report test
            # loss." The objective above is validation-only; test loss is reported.
            "mean_final_test_loss": agg.get("final_test_loss"),
            "mean_init_valid_outer": agg.get("init_valid_outer"),
            "mean_min_valid_outer": agg.get("min_valid_outer"),
            "replicas_ok": len(scored),
            "replicas_total": nrep,
            "replica_details": details,
        },
    )
