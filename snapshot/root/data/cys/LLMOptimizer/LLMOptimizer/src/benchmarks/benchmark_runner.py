"""
benchmark_runner.py
-------------------
Shared logic for running the warm-start HPO agent against any HPOBench
ML benchmark running locally on the host machine.
"""

# Set XGBoost/OpenMP thread limits to prevent over-subscription
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"
os.environ["XGBOOST_NUM_THREADS"] = "16"

import warnings
import json
import os
import glob
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.agents.hpo_agent import HPOAgent, FailedRecommendation
from src.config.model_registry import calculate_cost


# ── Benchmark imports (Local HPOBench) ────────────────────────────────────────
try:
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
    from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
    from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
    from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
    from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
except ImportError as e:
    raise ImportError(f"HPOBench missing or local dependencies not installed. Error: {e}")

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"

BENCHMARK_MAP = {
    "xgboost": XGBoostBenchmark,
    "rf":      RandomForestBenchmark,
    "svm":     SVMBenchmark,
    "lr":      LRBenchmark,
    "nn":      NNBenchmark,
}

RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results_2")
LEGACY_RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results")


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _load_reusable_default_baseline(model_name: str, task_id: int, llm_model: str):
    """
    Reuse the newest valid default baseline artifact for the same
    (model_name, task_id, llm_model).
    """
    safe_llm_model = str(llm_model).replace("/", "_").replace(":", "_")
    safe_model_name = str(model_name).upper().replace("/", "_").replace(":", "_")
    candidates = []
    for root in (RESULTS_DIR, LEGACY_RESULTS_DIR):
        pattern = os.path.join(root, safe_model_name, str(task_id), safe_llm_model, "hpo_*.json")
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None

    # Newest first based on file modification time.
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)

    for path in candidates:
        try:
            with open(path, "r") as f:
                artifact = json.load(f)
        except Exception:
            # Ignore malformed artifacts and keep scanning older files.
            continue

        if artifact.get("benchmark") not in (None, safe_model_name):
            continue
        artifact_task_id = artifact.get("task_id")
        if artifact_task_id is not None and str(artifact_task_id) != str(task_id):
            continue

        default = artifact.get("default_baseline")
        if not isinstance(default, dict):
            continue

        config = default.get("config")
        if not isinstance(config, dict):
            continue

        loss = _safe_float(default.get("loss"))
        accuracy = _safe_float(default.get("accuracy"))
        cost = _safe_float(default.get("cost"))
        if loss is None and accuracy is None:
            continue
        if accuracy is None and loss is not None:
            accuracy = (1 - loss) * 100

        return {
            "config": config,
            "loss": loss,
            "accuracy": accuracy,
            "cost": cost,
            "source_path": path,
            "source_benchmark": artifact.get("benchmark"),
        }

    return None


def _normalize_reused_default_config(cs, cached_config: dict) -> dict:
    """
    Keep output JSON format stable by preserving the current benchmark's default
    config structure and only overriding matching keys from a cached baseline.
    """
    normalized = dict(cs.get_default_configuration())
    if not isinstance(cached_config, dict):
        return normalized

    cs_dict = dict(cs)
    for key, value in cached_config.items():
        if key not in cs_dict:
            continue
        hp = cs_dict[key]
        try:
            hp_type = type(hp).__name__
            if "Integer" in hp_type:
                normalized[key] = int(float(value))
            elif "Float" in hp_type:
                normalized[key] = float(value)
            else:
                normalized[key] = value
        except Exception:
            # Ignore invalid cached values and keep the current default.
            continue
    return normalized


def _extract_config_info(cs) -> list:
    """
    Pull the name, type, range/choices, and default for every tunable
    hyperparameter out of a ConfigSpace object.
    """
    params = []
    for name, hp in dict(cs).items():
        hp_type = type(hp).__name__
        entry   = {"name": name}

        if "Integer" in hp_type:
            entry["type"]      = "int"
            entry["range"]     = [int(hp.lower), int(hp.upper)]
            entry["log_scale"] = bool(getattr(hp, "log", False))
            entry["default"]   = int(hp.default_value)

        elif "Float" in hp_type:
            entry["type"]      = "float"
            entry["range"]     = [float(hp.lower), float(hp.upper)]
            entry["log_scale"] = bool(getattr(hp, "log", False))
            entry["default"]   = float(hp.default_value)

        elif "Categorical" in hp_type:
            choices = list(hp.choices)
            default = hp.default_value
            entry["type"]    = "categorical"
            entry["choices"] = choices
            entry["default"] = default

        elif "Ordinal" in hp_type:
            entry["type"]     = "ordinal"
            entry["sequence"] = list(hp.sequence)
            entry["default"]  = hp.default_value
        else:
            continue

        params.append(entry)
    return params


def _extract_fidelity_info(fs) -> list:
    """Extract fidelity parameter names and max values."""
    result = []
    for f in fs.values():
        if hasattr(f, "upper") and f.upper is not None:
            max_value = int(f.upper)
        elif hasattr(f, "sequence"):
            max_value = int(f.sequence[-1])
        else:
            max_value = int(getattr(f, "default_value", 100))
        result.append({
            "name":  f.name,
            "max_value": max_value,
        })
    return result


def _format_config_for_prompt(config_params: list) -> str:
    """
    Render tunable parameters only — fidelity is excluded from the LLM prompt.
    """
    lines = [
        "Use ONLY the parameter names listed below — do not invent new names.",
        "",
        "CONFIG PARAMETERS (return these in warm_start_config):",
    ]
    for p in config_params:
        if p["type"] in ("int", "float"):
            scale = "  [log scale]" if p.get("log_scale") else ""
            lines.append(
                f"  {p['name']}: {p['type']} in [{p['range'][0]}, {p['range'][1]}]{scale}"
                f"  (default = {p['default']})"
            )
        elif p["type"] == "categorical":
            lines.append(
                f"  {p['name']}: one of {p['choices']}"
                f"  (default = {p['default']})"
            )
        elif p["type"] == "ordinal":
            lines.append(
                f"  {p['name']}: one of {p['sequence']}"
                f"  (default = {p['default']})"
            )

    return "\n".join(lines)


def _print_dataset_profile(meta: dict):
    """Print a formatted dataset profile block to the terminal."""
    fb = meta.get("feature_breakdown", {})
    cd = meta.get("class_distribution", {})

    class_str = ", ".join(f"'{k}': {v:.2f}%" for k, v in cd.items()) if cd else "N/A"

    print(f"{'─'*65}")
    print(f"  DATASET PROFILE")
    print(f"{'─'*65}")
    print(f"  Samples       : {meta['rows']:,}")
    print(f"  Features      : {meta['features']}  "
          f"({fb.get('numeric',0)} numeric, "
          f"{fb.get('categorical_or_string',0)} categorical, "
          f"{fb.get('boolean',0)} boolean)")
    if meta.get("task_type") == "classification":
        print(f"  Classes       : {meta.get('num_classes', '?')}  ({class_str})")
    print(f"  Sparsity      : {meta['sparsity'] * 100:.1f}%")
    print(f"  Task source   : {meta.get('task_type_source', 'inferred')}")
    print(f"{'─'*65}")


def _evaluate_config(benchmark, config_dict: dict, fidelity_config: dict) -> tuple:
    import traceback as _tb
    try:
        result = benchmark.objective_function(
            configuration=config_dict,
            fidelity=fidelity_config,
            rng=1,
        )
        return result["function_value"], result.get("cost", 0.0)
    except Exception as e:
        print(f"   [WARN] Evaluation failed: {e}")
        _tb.print_exc()
        return None, None


def _config_signature(config_dict: dict) -> str:
    """Stable fingerprint for deduplicating configs inside a single run."""
    return json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)


def _coerce_recommended_config(cs, warm_start_config):
    """Convert a recommendation into a concrete config or mark it invalid."""
    config_dict = dict(cs.get_default_configuration())
    cs_dict = dict(cs)
    recommended_params = []
    param_errors = []
    invalid = False

    for param in warm_start_config:
        k = param.name if hasattr(param, "name") else param["name"]
        v = param.value if hasattr(param, "value") else param["value"]
        recommended_params.append({"name": k, "value": v})

        hp = cs_dict.get(k)
        if hp is None:
            invalid = True
            param_errors.append(f"Parameter '{k}' not in ConfigSpace")
            continue

        try:
            hp_type = type(hp).__name__
            if "Integer" in hp_type:
                config_dict[k] = int(float(v))
            elif "Float" in hp_type:
                config_dict[k] = float(v)
            else:
                config_dict[k] = v
        except Exception as e:
            invalid = True
            param_errors.append(f"Parameter '{k}' failed validation: {e}")

    return config_dict, recommended_params, param_errors, invalid


# ── Main entry point ──────────────────────────────────────────────────────────

def run_benchmark(
    model_name: str,
    task_id: int,
    api_key: str,
    llm_model: str = "gpt-5-mini",
    base_url: str = "https://api.openai-hk.com/v1",
    model_key: str = None,
    cot_prompt: str = "",
    prompt_variant: str = None,
) -> dict:
    if model_name not in BENCHMARK_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(BENCHMARK_MAP)}")

    timestamp = datetime.now().isoformat(timespec="seconds")
    expected_benchmark = model_name.upper()

    benchmark = BENCHMARK_MAP[model_name](task_id=task_id, rng=1)
    agent = HPOAgent(
        api_key=api_key,
        model=llm_model,
        base_url=base_url,
    )

    cs              = benchmark.get_configuration_space(seed=1)
    fidelity_space  = benchmark.get_fidelity_space()
    config_params   = _extract_config_info(cs)
    fidelity_params = _extract_fidelity_info(fidelity_space)
    config_space_str = _format_config_for_prompt(config_params)

    # ── Profiling ─────────────────────────────────────────────────────────
    agent.analyze(benchmark_obj=benchmark,
                  intent=f"HPO warm-start for {model_name.upper()}",
                  task_id=task_id)
    meta = agent.current_profile["meta"]

    def _fresh_benchmark():
        return BENCHMARK_MAP[model_name](task_id=task_id, rng=1)

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Model  : {model_name.upper()}")
    print(f"  LLM    : {llm_model}")
    print(f"  Task   : {meta['task_type'].title()}  (ID: {task_id})")
    print(f"  Metric : {meta['metric_name']}  [{meta['metric_direction']}]")
    print(f"{'='*65}")
    _print_dataset_profile(meta)
    print(f"  ConfigSpace: {len(config_params)} tunable parameters")

    fidelity_config = {}
    for f in fidelity_space.values():
        if hasattr(f, "upper") and f.upper is not None:
            fidelity_config[f.name] = f.upper
        elif hasattr(f, "sequence"):
            fidelity_config[f.name] = f.sequence[-1]
        else:
            fidelity_config[f.name] = getattr(f, "default_value", None)

    # ── Evaluate baseline (or reuse from previous artifact) ──────────────
    reused_baseline = _load_reusable_default_baseline(
        model_name=model_name,
        task_id=task_id,
        llm_model=llm_model,
    )
    if reused_baseline is not None:
        default_dict = _normalize_reused_default_config(cs, reused_baseline["config"])
        default_loss = reused_baseline["loss"]
        default_acc = reused_baseline["accuracy"]
        default_cost = reused_baseline["cost"]
    else:
        print(f"\n>> Evaluating default configuration (baseline)...")
        default_dict = dict(cs.get_default_configuration())
        default_loss, default_cost = _evaluate_config(_fresh_benchmark(), default_dict, fidelity_config)
        default_acc = (1 - default_loss) * 100 if default_loss is not None else None

    # ── Evaluate 3 recommendations (batch call) ───────────────────────────
    evaluation_results = []
    per_run_tokens = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print(f"\n>> Querying LLM with 3 separate n=1 calls for 3 warm-start candidates...")

    # 3 separate n=1 calls to get individual reasoning for each config
    recs, batch_tokens, prompt = agent.recommend_batch(
        model_name=model_name,
        config_space_info=config_space_str,
        n=3,
        previous_results=[],  # No feedback between candidates (independent)
        cot_prompt=cot_prompt,
    )
    per_run_tokens.append(batch_tokens)
    for k in total_tokens:
        total_tokens[k] += batch_tokens[k]

    print(
        f"  Tokens: {batch_tokens['prompt_tokens']} in / "
        f"{batch_tokens['completion_tokens']} out / "
        f"{batch_tokens['total_tokens']} total"
    )
    batch_cost_usd = calculate_cost(model_key, batch_tokens["prompt_tokens"], batch_tokens["completion_tokens"]) if model_key else None
    if batch_cost_usd is not None:
        print(f"  API Cost: ${batch_cost_usd:.6f}")

    # Evaluate each recommendation
    seen_config_signatures = {}
    for idx, rec in enumerate(recs, start=1):
        print(f"\n{'='*40}")
        print(f"  RUN {idx}/3")
        print(f"{'='*40}")

        # Initialize result dict with common fields
        result = {
            "config_num": idx,
            "raw_response": None,
            "reasoning_trace": None,
            "recommended_config": None,
            "evaluated_config": None,
            "loss": None,
            "accuracy": None,
            "cost": None,
            "skipped": False,
            "error_reason": None,
            "is_duplicate_config": False,
            "duplicate_of_config_num": None,
            "config_signature": None,
        }

        if isinstance(rec, FailedRecommendation):
            print(f"  SKIPPED — LLM returned invalid configuration.")
            print(f"  Reason : {rec.error_reason}")
            result["skipped"] = True
            result["error_reason"] = rec.error_reason
            result["raw_response"] = rec.raw_response
            evaluation_results.append(result)
            continue

        # Successful LLM response
        trace = rec.reasoning_trace
        result["reasoning_trace"] = trace
        result["raw_response"] = rec.raw_response
        print(f"  Rationale: {trace}\n")

        config_dict, recommended_params, param_errors, invalid_config = _coerce_recommended_config(
            cs=cs,
            warm_start_config=rec.warm_start_config,
        )
        for param in recommended_params:
            print(f"   {param['name']} = {param['value']}")

        result["recommended_config"] = recommended_params
        result["evaluated_config"] = None if invalid_config else {k: v for k, v in config_dict.items()}
        config_signature = _config_signature(config_dict)
        result["config_signature"] = config_signature
        if config_signature in seen_config_signatures:
            result["is_duplicate_config"] = True
            result["duplicate_of_config_num"] = seen_config_signatures[config_signature]
            print(f"   [DUPLICATE] matches Config {result['duplicate_of_config_num']}")
        else:
            seen_config_signatures[config_signature] = idx

        if invalid_config:
            print("  INVALID CONFIGURATION: skipping evaluation.")
            result["skipped"] = True
            result["error_reason"] = "; ".join(param_errors)
            evaluation_results.append(result)
            continue

        print(f"\n  Evaluating...")
        ws_loss, ws_cost = _evaluate_config(_fresh_benchmark(), config_dict, fidelity_config)

        if ws_loss is not None:
            ws_acc = (1 - ws_loss) * 100
            print(f"  Result: Accuracy = {ws_acc:.2f}% | Error = {ws_loss:.5f} | Time = {ws_cost:.4f}s")
            result["loss"] = ws_loss
            result["accuracy"] = ws_acc
            result["cost"] = ws_cost
        else:
            print("  VALIDATION FAILED for this config.")
            result["skipped"] = True
            result["error_reason"] = "Config validation/evaluation failed"
        
        evaluation_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────
    W = 65
    print(f"\n{'━'*W}")
    print(f"  {'FINAL PERFORMANCE SUMMARY':^{W-4}}")
    print(f"{'━'*W}")
    print(f"  {'Configuration':<20} | {'Bal. Acc':>9} | {'Error':>7} | {'Time (s)':>8}")
    print(f"  {'─'*20}-+-{'─'*9}-+-{'─'*7}-+-{'─'*8}")

    if default_acc is not None:
        print(f"  {'Default Config':<20} | {default_acc:>8.2f}% | {default_loss:.5f} | {default_cost:>8.4f}")

    best_loss = default_loss if default_loss is not None else float('inf')
    best_config_idx = "Default"

    for res in evaluation_results:
        if res.get("skipped"):
            reason_short = res["error_reason"][:40]
            print(f"  Config {res['config_num']:<13} | {'—':>9} | {'—':>7} | {reason_short}")
        else:
            print(f"  Config {res['config_num']:<13} | {res['accuracy']:>8.2f}% | {res['loss']:.5f} | {res['cost']:>8.4f}")
            if res['loss'] < best_loss:
                best_loss = res['loss']
                best_config_idx = f"Config {res['config_num']}"

    print(f"{'━'*W}")
    total_cost_usd = calculate_cost(
        model_key, total_tokens["prompt_tokens"], total_tokens["completion_tokens"]
    ) if model_key else None
    if total_cost_usd is not None:
        print(
            f"  BEST: {best_config_idx}   |   TOKENS: {total_tokens['total_tokens']} "
            f"(in: {total_tokens['prompt_tokens']}, out: {total_tokens['completion_tokens']}) "
            f"| COST: ${total_cost_usd:.6f}"
        )
    else:
        print(
            f"  BEST: {best_config_idx}   |   TOKENS: {total_tokens['total_tokens']} "
            f"(in: {total_tokens['prompt_tokens']}, out: {total_tokens['completion_tokens']})"
        )
    print(f"{'━'*W}")

    # ── Build full artifact ───────────────────────────────────────────────
    default_baseline = {
        "config": default_dict,
        "loss": default_loss,
        "accuracy": default_acc,
        "cost": default_cost,
        "source_path": reused_baseline["source_path"] if reused_baseline else None,
        "source_benchmark": reused_baseline["source_benchmark"] if reused_baseline else expected_benchmark,
    }
    artifact = {
        "schema_version": 2,
        "artifact_type": "ml_hpo",
        "benchmark": expected_benchmark,
        "task_id": task_id,
        "model": llm_model,
        "timestamp": timestamp,
        "prompt": prompt,
        "default_baseline": default_baseline,
        "recommendations": evaluation_results,
        "best_performer": best_config_idx,
        "token_usage": dict(total_tokens),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = str(model_name).upper().replace("/", "_").replace(":", "_")
    safe_llm_model = str(llm_model).replace("/", "_").replace(":", "_")
    out_dir = os.path.join(RESULTS_DIR, safe_model_name, str(task_id), safe_llm_model)
    os.makedirs(out_dir, exist_ok=True)
    if artifact["benchmark"] != safe_model_name or artifact.get("task_id") != task_id:
        raise RuntimeError(
            f"Refusing to save artifact for benchmark={artifact['benchmark']} task_id={artifact.get('task_id')} "
            f"under path benchmark={safe_model_name} task_id={task_id}"
        )
    log_filename = os.path.join(
        out_dir,
        f"hpo_{ts_file}.json"
    )
    with open(log_filename, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"  Saved: {log_filename}\n")

    return artifact
