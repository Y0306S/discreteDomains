"""
nas_runner.py
-------------
NAS benchmark runner using LOCAL benchmarks instead of containers.

This version imports from hpobench.benchmarks.nas.* (local) instead of
hpobench.container.benchmarks.nas.* (containers) to avoid SIF capability issues.

Data files required:
- NASBench-201: ~/.local/share/hpobench/nasbench_201/*.json (extracted from nasbench_201_data_v1.3.zip)
"""

import warnings
import json
import os
import textwrap
import sys
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to import local NASBench-201 benchmarks (only need data files, no extra packages)
try:
    from hpobench.benchmarks.nas.nasbench_201 import (
        Cifar10ValidNasBench201Benchmark,
        Cifar100NasBench201Benchmark,
        ImageNetNasBench201Benchmark
    )
    NASBENCH_201_AVAILABLE = True
except ImportError as e:
    NASBENCH_201_AVAILABLE = False
    NASBENCH_201_ERROR = str(e)
    Cifar10ValidNasBench201Benchmark = None
    Cifar100NasBench201Benchmark = None
    ImageNetNasBench201Benchmark = None

# Try to import NASBench-101 (requires tensorflow==1.15.0, nasbench, tabular_benchmarks from GitHub)
try:
    from hpobench.benchmarks.nas.nasbench_101 import NASCifar10ABenchmark
    NASBENCH_101_AVAILABLE = True
except ImportError as e:
    NASBENCH_101_AVAILABLE = False
    NASBENCH_101_ERROR = str(e)
    NASCifar10ABenchmark = None

# Try to import Tabular benchmarks (requires tensorflow==1.15.0, tabular_benchmarks from GitHub)
try:
    from hpobench.benchmarks.nas.tabular_benchmarks import (
        SliceLocalizationBenchmark,
        ProteinStructureBenchmark,
        NavalPropulsionBenchmark,
        ParkinsonsTelemonitoringBenchmark
    )
    TABULAR_AVAILABLE = True
except ImportError as e:
    TABULAR_AVAILABLE = False
    TABULAR_ERROR = str(e)
    SliceLocalizationBenchmark = None
    ProteinStructureBenchmark = None
    NavalPropulsionBenchmark = None
    ParkinsonsTelemonitoringBenchmark = None

from src.agents.nas_agent import TabularNASAgent, FailedRecommendation
from src.config.model_registry import calculate_cost

# Build the benchmark map based on what's available
NAS_BENCHMARK_MAP = {}

# NASBench-201 benchmarks (work with just data files)
if NASBENCH_201_AVAILABLE:
    NAS_BENCHMARK_MAP.update({
        "Cifar10ValidNasBench201": Cifar10ValidNasBench201Benchmark,
        "Cifar100NasBench201": Cifar100NasBench201Benchmark,
        "ImageNetNasBench201": ImageNetNasBench201Benchmark,
    })

# NASBench-101 benchmarks (need extra dependencies)
if NASBENCH_101_AVAILABLE:
    NAS_BENCHMARK_MAP.update({
        "NASCifar10A": NASCifar10ABenchmark,
    })

# Tabular benchmarks (need extra dependencies)
if TABULAR_AVAILABLE:
    NAS_BENCHMARK_MAP.update({
        "SliceLocalization": SliceLocalizationBenchmark,
        "ProteinStructure": ProteinStructureBenchmark,
        "NavalPropulsion": NavalPropulsionBenchmark,
        "ParkinsonsTelemonitoring": ParkinsonsTelemonitoringBenchmark,
    })

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"
RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results_2")

_CIFAR10_PROFILE = (
    "CIFAR-10 contains 60,000 color images of size 32x32 across 10 classes. "
    "It is split into 50,000 training and 10,000 test images. "
    "Classes include airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
    "The dataset is a standard benchmark for evaluating feature extraction and architecture efficiency."
)

_CIFAR100_PROFILE = (
    "CIFAR-100 contains 60,000 color images of size 32x32 across 100 fine-grained classes. "
    "It is split into 50,000 training and 10,000 test images. "
    "Compared with CIFAR-10, the task is harder due to higher class granularity and requires stronger representation quality."
)

_IMAGENET16_PROFILE = (
    "ImageNet-16-120 is a downsampled ImageNet subset with 120 classes and 16x16 images. "
    "It is more diverse than CIFAR and stresses generalization under larger class variety."
)

_FCNET_PROFILE = (
    "The model is a two-layer fully connected neural network (MLP) for regression on UCI tabular datasets. "
    "You must tune topology and optimization dynamics: n_units_1, n_units_2, activation_fn_1, "
    "activation_fn_2, dropout_1, dropout_2, init_lr, lr_schedule, and batch_size."
)

NAS_BENCHMARK_DESCRIPTIONS = {
    "NASCifar10A": {
        "task_description": (
            "NAS-Bench-101 search space. A candidate cell is defined by a 7x7 upper-triangular adjacency matrix "
            "and a length-7 operation list. Node 0 is input, node 6 is output, and nodes 1-5 are intermediate nodes. "
            "The graph must be a DAG and contain at most 9 edges."
        ),
        "targets_and_parameters": (
            "Target: minimize validation error (or loss) under the benchmark objective. "
            "You must decide both topology (connections) and node operations. "
            "For operations, intermediate nodes can only be: conv3x3-bn-relu, conv1x1-bn-relu, maxpool3x3."
        ),
        "dataset_profile": _CIFAR10_PROFILE,
    },
    "Cifar10ValidNasBench201": {
        "task_description": (
            "NAS-Bench-201 search space. The cell topology is fixed: a fully-connected 4-node DAG with edges "
            "(0,1), (0,2), (1,2), (0,3), (1,3), (2,3)."
        ),
        "targets_and_parameters": (
            "Target: minimize validation error (or loss). "
            "Only operation choices on the fixed edges are tunable. "
            "Use operations from the provided configuration space (for example none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3)."
        ),
        "dataset_profile": _CIFAR10_PROFILE,
    },
    "Cifar100NasBench201": {
        "task_description": (
            "NAS-Bench-201 fixed-topology 4-node DAG cell with 6 directed edges. "
            "Architecture search is operation selection over a pre-defined graph."
        ),
        "targets_and_parameters": (
            "Target: minimize validation error (or loss). "
            "Tune edge operations only; topology is fixed by NAS-Bench-201."
        ),
        "dataset_profile": _CIFAR100_PROFILE,
    },
    "ImageNetNasBench201": {
        "task_description": (
            "NAS-Bench-201 fixed-topology 4-node DAG cell with fully connected forward edges. "
            "Search focuses on selecting edge operations over a fixed structure."
        ),
        "targets_and_parameters": (
            "Target: minimize validation error (or loss). "
            "Choose operations for each fixed edge in the cell."
        ),
        "dataset_profile": _IMAGENET16_PROFILE,
    },
    "SliceLocalization": {
        "task_description": "FCNet tabular NAS benchmark using a two-layer MLP for regression.",
        "targets_and_parameters": "Target: minimize validation Mean Squared Error (MSE) using architecture/training hyperparameters from the search space.",
        "dataset_profile": (
            "Slice Localization (UCI): tabular regression dataset for predicting relative slice position from image-derived features. "
            + _FCNET_PROFILE
        ),
    },
    "ProteinStructure": {
        "task_description": "FCNet tabular NAS benchmark using a two-layer MLP for regression.",
        "targets_and_parameters": "Target: minimize validation Mean Squared Error (MSE) with robust generalization on biochemical feature inputs.",
        "dataset_profile": (
            "Protein Structure (UCI): tabular regression benchmark for protein tertiary structure related targets. "
            + _FCNET_PROFILE
        ),
    },
    "NavalPropulsion": {
        "task_description": "FCNet tabular NAS benchmark using a two-layer MLP for regression.",
        "targets_and_parameters": "Target: minimize validation Mean Squared Error (MSE) for engine condition prediction.",
        "dataset_profile": (
            "Naval Propulsion (UCI): tabular regression benchmark for ship propulsion system condition monitoring. "
            + _FCNET_PROFILE
        ),
    },
    "ParkinsonsTelemonitoring": {
        "task_description": "FCNet tabular NAS benchmark using a two-layer MLP for regression.",
        "targets_and_parameters": "Target: minimize validation Mean Squared Error (MSE) for clinical score prediction.",
        "dataset_profile": (
            "Parkinsons Telemonitoring (UCI): tabular regression benchmark based on biomedical voice measurements. "
            + _FCNET_PROFILE
        ),
    },
}


def _loss_to_accuracy(benchmark_name: str, loss):
    """
    Convert loss to accuracy percentage.
    Only applies to classification benchmarks.
    Regression benchmarks (FCNet) return None since MSE is not normalized.
    """
    if loss is None:
        return None
    loss = float(loss)
    if benchmark_name == "NASCifar10A":
        return (1.0 - loss) * 100.0
    if benchmark_name in {"Cifar10ValidNasBench201", "Cifar100NasBench201", "ImageNetNasBench201"}:
        return 100.0 - loss
    # Regression benchmarks (FCNet tabular) - no accuracy conversion
    # MSE values are raw and dataset-dependent, not normalized
    return None


def _extract_config_info(cs) -> list:
    params = []
    for name, hp in dict(cs).items():
        hp_type = type(hp).__name__
        entry = {"name": name}
        if "Integer" in hp_type:
            entry.update({"type": 'int', 'range': [int(hp.lower), int(hp.upper)], "default": int(hp.default_value)})
        elif "Float" in hp_type:
            entry.update({"type": 'float', 'range': [float(hp.lower), float(hp.upper)], "default": float(hp.default_value)})
        elif "Categorical" in hp_type:
            entry.update({"type": 'categorical', 'choices': list(hp.choices), "default": hp.default_value})
        elif "Ordinal" in hp_type:
            entry.update({"type": "ordinal", "sequence": list(hp.sequence), "default": hp.default_value})
        else:
            continue
        params.append(entry)
    return params


def _format_config_for_prompt(config_params: list) -> str:
    lines = ["CONFIG PARAMETERS:"]
    for p in config_params:
        if p['type'] in ('int', 'float'):
            lines.append(f"  {p['name']}: {p['type']} in [{p['range'][0]}, {p['range'][1]}] (default = {p['default']})")
        elif p['type'] in ('categorical', 'ordinal'):
            lines.append(f"  {p['name']}: one of {p.get('choices', p.get('sequence'))} (default = {p['default']})")
    return "\n".join(lines)


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


def _get_prompt_sections(benchmark_name: str):
    desc = NAS_BENCHMARK_DESCRIPTIONS.get(
        benchmark_name,
        {
            "task_description": "Neural architecture search benchmark.",
            "targets_and_parameters": "Target: optimize benchmark objective using only the provided search space.",
            "dataset_profile": "Dataset profile is not explicitly defined for this benchmark.",
        },
    )
    return (
        desc["task_description"],
        desc["targets_and_parameters"],
        desc["dataset_profile"],
    )


def _print_text_block(title: str, text: str, width: int = 65):
    print(f"{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    for paragraph in str(text).split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            print("  ")
            continue
        for line in textwrap.wrap(paragraph, width=max(20, width - 4)):
            print(f"  {line}")
    print(f"{'─' * width}")


def run_nas_benchmark(
    benchmark_name: str,
    api_key: str,
    task_id: int = None,
    llm_model: str = "gpt-5-mini",
    base_url: str = "https://api.openai-hk.com/v1",
    model_key: str = None,
):
    if benchmark_name not in NAS_BENCHMARK_MAP:
        error_msg = f"Unknown benchmark: {benchmark_name}.\n"
        error_msg += f"Available benchmarks: {list(NAS_BENCHMARK_MAP)}\n\n"
        if benchmark_name in ["NASCifar10A", "NASCifar10B", "NASCifar10C"]:
            error_msg += "Note: NASBench-101 benchmarks require tensorflow==1.15.0 and nasbench package from GitHub.\n"
            error_msg += "These are not available due to network restrictions.\n"
        elif benchmark_name in ["SliceLocalization", "ProteinStructure", "NavalPropulsion", "ParkinsonsTelemonitoring"]:
            error_msg += "Note: Tabular benchmarks require tensorflow==1.15.0 and tabular_benchmarks package from GitHub.\n"
            error_msg += "These are not available due to network restrictions.\n"
        raise ValueError(error_msg)

    timestamp = datetime.now().isoformat(timespec="seconds")

    BenchmarkClass = NAS_BENCHMARK_MAP[benchmark_name]

    def _fresh_benchmark():
        if task_id is not None:
            return BenchmarkClass(task_id=task_id, rng=1)
        return BenchmarkClass(rng=1)

    benchmark = _fresh_benchmark()

    agent = TabularNASAgent(
        api_key=api_key,
        model=llm_model,
        base_url=base_url,
    )
    cs = benchmark.get_configuration_space(seed=1)

    try:
        fs = benchmark.get_fidelity_space(seed=1)
        max_fidelity = {}
        for f in fs.values():
            if hasattr(f, 'upper') and f.upper is not None:
                max_fidelity[f.name] = f.upper
            elif hasattr(f, 'sequence'):
                max_fidelity[f.name] = f.sequence[-1]
            else:
                max_fidelity[f.name] = getattr(f, 'default_value', None)
    except Exception:
        max_fidelity = {}

    config_params = _extract_config_info(cs)
    config_space_str = _format_config_for_prompt(config_params)

    metric_name = "validation_error"
    metric_direction = "lower_is_better"
    task_description, targets_and_parameters, dataset_profile = _get_prompt_sections(benchmark_name)

    # ── Header + prompt-visible context ───────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Benchmark : {benchmark_name}")
    print(f"  LLM       : {llm_model}")
    if task_id:
        print(f"  Task ID   : {task_id}")
    print(f"  Metric    : {metric_name}  [{metric_direction}]")
    print(f"{'='*65}")

    _print_text_block("DATASET PROFILE", dataset_profile)
    _print_text_block("TASK DESCRIPTION", task_description)
    _print_text_block("TARGETS & PARAMETERS", targets_and_parameters)
    _print_text_block("HYPERPARAMETER SPACE (sent to LLM)", config_space_str)
    if max_fidelity:
        print(f"  Fidelity   : {max_fidelity}")
    print(f"  ConfigSpace: {len(config_params)} tunable parameters")

    # ── Evaluate baseline ─────────────────────────────────────────────────
    print(f"\n>> Evaluating default configuration (baseline)...")
    default_config = dict(cs.get_default_configuration())
    default_config = {k: (v.item() if hasattr(v, 'item') else v) for k, v in default_config.items()}

    try:
        default_result = _fresh_benchmark().objective_function(configuration=default_config, fidelity=max_fidelity, rng=1)
        default_loss = default_result.get('function_value', None)
        default_cost = default_result.get('cost', 0.0)
    except Exception as e:
        print(f"   [WARN] Default evaluation failed: {e}")
        default_loss, default_cost = None, None

    # ── Evaluate 3 recommendations ────────────────────────────────────────
    evaluation_results = []
    per_run_tokens = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print(f"\n>> Querying LLM with 3 separate n=1 calls for 3 warm-start candidates...")
    recs, batch_tokens, prompt = agent.recommend_batch(
        benchmark_name=benchmark_name,
        config_space_info=config_space_str,
        n=3,
        previous_results=[],
        task_description=task_description,
        targets_and_parameters=targets_and_parameters,
        dataset_profile=dataset_profile,
    )
    per_run_tokens.append(batch_tokens)
    for k in total_tokens:
        total_tokens[k] += batch_tokens[k]

    print(
        f"  Total Tokens: {batch_tokens['prompt_tokens']} in / "
        f"{batch_tokens['completion_tokens']} out / "
        f"{batch_tokens['total_tokens']} total"
    )
    batch_cost_usd = calculate_cost(model_key, batch_tokens["prompt_tokens"], batch_tokens["completion_tokens"]) if model_key else None
    if batch_cost_usd is not None:
        print(f"  API Cost: ${batch_cost_usd:.6f}")

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
            result["raw_response"] = rec.raw_response  # May be None or contain thinking tags
            evaluation_results.append(result)
            continue

        # Successful LLM response
        trace = rec.reasoning_trace
        result["reasoning_trace"] = trace
        result["raw_response"] = rec.raw_response  # May be None or contain thinking tags
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
        config_dict = {k: (v.item() if hasattr(v, 'item') else v) for k, v in config_dict.items()}

        try:
            ws_result = _fresh_benchmark().objective_function(configuration=config_dict, fidelity=max_fidelity, rng=1)
            ws_loss = ws_result.get('function_value', None)
            ws_cost = ws_result.get('cost', 0.0)

            if ws_loss is not None:
                ws_acc = _loss_to_accuracy(benchmark_name, ws_loss)
                if ws_acc is not None:
                    print(f"  Result: Accuracy = {ws_acc:.2f}% | Error = {ws_loss:.5f} | Time = {ws_cost:.4f}s")
                else:
                    print(f"  Result: Error = {ws_loss:.5f} | Time = {ws_cost:.4f}s")
                result["loss"] = ws_loss
                result["accuracy"] = ws_acc
                result["cost"] = ws_cost
            else:
                print("  VALIDATION RETURNED NONE.")
                result["skipped"] = True
                result["error_reason"] = "Config validation returned None"
        except Exception as e:
            print(f"  VALIDATION CRASHED: {e}")
            result["skipped"] = True
            result["error_reason"] = f"HPOBench evaluation crashed: {str(e)}"
        
        evaluation_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────
    # Check if this benchmark has accuracy metrics
    has_accuracy = _loss_to_accuracy(benchmark_name, 0.5) is not None
    
    W = 65
    print(f"\n{'━'*W}")
    print(f"  {'FINAL PERFORMANCE SUMMARY':^{W-4}}")
    print(f"{'━'*W}")
    if has_accuracy:
        print(f"  {'Configuration':<20} | {'Bal. Acc':>9} | {'Error':>7} | {'Time (s)':>8}")
        print(f"  {'─'*20}-+-{'─'*9}-+-{'─'*7}-+-{'─'*8}")
    else:
        print(f"  {'Configuration':<20} | {'Error':>9} | {'Time (s)':>8}")
        print(f"  {'─'*20}-+-{'─'*9}-+-{'─'*8}")

    default_acc = _loss_to_accuracy(benchmark_name, default_loss)
    if default_loss is not None:
        if has_accuracy:
            default_acc_text = f"{default_acc:>8.2f}%" if default_acc is not None else f"{'N/A':>9}"
            print(f"  {'Default Config':<20} | {default_acc_text} | {default_loss:.5f} | {default_cost:>8.4f}")
        else:
            print(f"  {'Default Config':<20} | {default_loss:>9.5f} | {default_cost:>8.4f}")

    best_loss = default_loss if default_loss is not None else float('inf')
    best_config_idx = "Default"

    for res in evaluation_results:
        if res.get("skipped"):
            reason_short = res["error_reason"][:40]
            print(f"  Config {res['config_num']:<13} | {'—':>9} | {'—':>7} | {reason_short}")
        else:
            if has_accuracy:
                acc = res.get("accuracy")
                acc_text = f"{acc:>8.2f}%" if acc is not None else f"{'N/A':>9}"
                print(f"  Config {res['config_num']:<13} | {acc_text} | {res['loss']:.5f} | {res['cost']:>8.4f}")
            else:
                print(f"  Config {res['config_num']:<13} | {res['loss']:>9.5f} | {res['cost']:>8.4f}")
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
        print(f"  BEST: {best_config_idx}   |   TOKENS: {total_tokens['total_tokens']}")
    print(f"{'━'*W}")

    # ── Build full artifact ───────────────────────────────────────────────
    default_baseline = {
        "config": default_config,
        "loss": default_loss,
        "accuracy": default_acc,
        "cost": default_cost,
    }
    artifact = {
        "schema_version": 2,
        "artifact_type": "nas_hpo",
        "benchmark": benchmark_name,
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
    if task_id is not None:
        artifact["task_id"] = task_id

    # ── Save JSON ─────────────────────────────────────────────────────────
    safe_benchmark_name = str(benchmark_name).replace("/", "_").replace(":", "_")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_llm_model = str(llm_model).replace("/", "_").replace(":", "_")
    out_dir = os.path.join(RESULTS_DIR, safe_benchmark_name, safe_llm_model)
    os.makedirs(out_dir, exist_ok=True)
    artifact_benchmark_name = str(artifact["benchmark"]).replace("/", "_").replace(":", "_")
    if artifact_benchmark_name != safe_benchmark_name:
        raise RuntimeError(
            f"Refusing to save artifact for benchmark={artifact['benchmark']} "
            f"under path benchmark={safe_benchmark_name}"
        )
    log_filename = os.path.join(
        out_dir,
        f"nas_{ts_file}.json"
    )
    with open(log_filename, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"  Saved: {log_filename}\n")

    return artifact


def print_benchmark_status():
    """Print the status of all NAS benchmarks."""
    print("=" * 70)
    print("NAS Benchmark Availability Status (Local Mode)")
    print("=" * 70)
    
    print("\n✅ NASBench-201 (Data files only - WORKING):")
    if NASBENCH_201_AVAILABLE:
        print("  - Cifar10ValidNasBench201: Available")
        print("  - Cifar100NasBench201: Available")
        print("  - ImageNetNasBench201: Available")
        # Check data files
        data_dir = os.path.expanduser("~/.local/share/hpobench/nasbench_201")
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                print(f"  Data files found: {len(json_files)}")
                for f in json_files:
                    size_gb = os.path.getsize(os.path.join(data_dir, f)) / (1024**3)
                    print(f"    - {f} ({size_gb:.2f} GB)")
            else:
                print("  ⚠️  Warning: No .json data files found!")
                print(f"     Expected in: {data_dir}")
        else:
            print(f"  ⚠️  Warning: Data directory not found: {data_dir}")
    else:
        print(f"  ❌ Not available: {NASBENCH_201_ERROR}")
    
    print("\n❌ NASBench-101 (Needs tensorflow==1.15.0, nasbench from GitHub):")
    if NASBENCH_101_AVAILABLE:
        print("  - NASCifar10A: Available")
    else:
        print(f"  - NASCifar10A: Not available")
        print(f"    Error: {NASBENCH_101_ERROR}")
    
    print("\n❌ Tabular Benchmarks (Needs tensorflow==1.15.0, tabular_benchmarks from GitHub):")
    if TABULAR_AVAILABLE:
        print("  - SliceLocalization: Available")
        print("  - ProteinStructure: Available")
        print("  - NavalPropulsion: Available")
        print("  - ParkinsonsTelemonitoring: Available")
    else:
        print("  - SliceLocalization: Not available")
        print("  - ProteinStructure: Not available")
        print("  - NavalPropulsion: Not available")
        print("  - ParkinsonsTelemonitoring: Not available")
        print(f"    Error: {TABULAR_ERROR}")
    
    print("\n" + "=" * 70)
    print(f"Available benchmarks: {list(NAS_BENCHMARK_MAP.keys())}")
    print("=" * 70)


if __name__ == "__main__":
    print_benchmark_status()
