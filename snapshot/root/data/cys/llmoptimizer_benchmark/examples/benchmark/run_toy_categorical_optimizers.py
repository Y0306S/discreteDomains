import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from ConfigSpace.hyperparameters import CategoricalHyperparameter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_utils import build_dataset_profile, get_dataset_info
from experiment_utils import (
    apply_hybrid_settings,
    apply_prompt_settings,
    list_v2_dataset_jobs,
    load_datasets_v2,
    make_llm_variant_name,
    optimizer_base_name,
    parse_variant_tags,
)
from hf_qwen_client import get_hf_qwen_client, ollama_chat_healthy
from mindware import CASH
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import AVAILABLE_LLMs, CONFIGS, LLMCONFIGS
from mindware.utils.data_manager import DataManager


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALGORITHM = "random_forest"

CLASSICAL_OPTIMIZERS = ["smac", "tpe", "cmaes", "hebo"]
# Named classical replicates: disk dir / summary label differs, CASH dispatch is the base.
CLASSICAL_ALIASES = {
    "smac_2": "smac",
}
LLM_VARIANTS = {
    "ptllm": ("ptllm", "gpt4o-mini"),
    "ptllm_tpe": ("ptllm_tpe", "gpt4o-mini"),
    "ptllm_smac": ("ptllm_smac", "gpt4o-mini"),
    "ptllm_cmaes": ("ptllm_cmaes", "gpt4o-mini"),
    "ptllm_qwen": ("ptllm", "qwen-sft6"),
    "ptllm_tpe_qwen": ("ptllm_tpe", "qwen-sft6"),
    "ptllm_smac_qwen": ("ptllm_smac", "qwen-sft6"),
    "ptllm_hebo": ("ptllm_hebo", "gpt4o-mini"),
    "ptllm_hebo_qwen": ("ptllm_hebo", "qwen-sft6"),
}
# PT4Opt paper baselines. Selectable via --optimizer but deliberately kept out of
# CLASSICAL_OPTIMIZERS so that "--optimizer all" keeps its existing meaning.
BASELINE_OPTIMIZERS = ["random_search", "lolbo"]
ALL_OPTIMIZER_VARIANTS = (
    CLASSICAL_OPTIMIZERS
    + BASELINE_OPTIMIZERS
    + list(CLASSICAL_ALIASES.keys())
    + list(LLM_VARIANTS.keys())
)

# Trajectory CSV suffix uses dispatch optimizer_label, not variant dir name.
TRAJ_LABELS = {
    "smac": "smac",
    "tpe": "tpe",
    "cmaes": "cmaes",
    "hebo": "hebo",
    "random_search": "random_search",
    "lolbo": "lolbo",
    "ptllm": "ptllm",
    "ptllm_tpe": "ptllm_tpe",
    "ptllm_smac": "ptllm_smac",
    "ptllm_cmaes": "ptllm_cmaes",
    "ptllm_qwen": "ptllm",
    "ptllm_tpe_qwen": "ptllm_tpe",
    "ptllm_smac_qwen": "ptllm_smac",
    "ptllm_hebo": "ptllm_hebo",
    "ptllm_hebo_qwen": "ptllm_hebo",
}

DEFAULT_QWEN_MODEL_PATH = "/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_full_sft_6"

LLM_BACKEND_DEFAULTS = {
    "gpt4o-mini": {
        "chat_engine": "gpt-4o-mini",
    },
}


def resolve_qwen_transport(args):
    """Pick Ollama API or in-process HuggingFace for Qwen SFT6."""
    mode = (args.qwen_backend or "auto").lower()
    if mode == "hf":
        return "hf"
    if mode == "ollama":
        return "ollama"
    # auto: prefer HF (Ollama GGUF conversion fails on Qwen3.5 multimodal weights)
    if ollama_chat_healthy(
        model=os.getenv("QWEN_OLLAMA_MODEL", "qwen35-sft6:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
    ):
        return "ollama"
    print("[Qwen] Ollama unavailable or unhealthy — using HuggingFace Transformers")
    return "hf"


def apply_qwen_backend_params(params, args):
    adapter = getattr(args, "qwen_adapter_path", None) or None
    # LoRA adapters require in-process HF load (not Ollama GGUF).
    transport = "hf" if adapter else resolve_qwen_transport(args)
    if transport == "ollama":
        params.update(
            {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
                "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
                "llm_model": os.getenv("QWEN_OLLAMA_MODEL", "qwen35-sft6:latest"),
            }
        )
        params["llm_transport"] = "ollama"
        return params
    params["llm_client"] = get_hf_qwen_client(
        model_path=args.qwen_model_path,
        adapter_path=adapter,
        temperature=args.llm_temperature,
        max_new_tokens=args.llm_max_tokens,
        enable_thinking=bool(getattr(args, "qwen_enable_thinking", False)),
    )
    params["llm_transport"] = "hf"
    if adapter:
        params["qwen_adapter_path"] = adapter
    return params


def list_cls_datasets():
    info_path = REPO_ROOT / "datasets_bundle" / "dataset_info" / "classification.csv"
    df = pd.read_csv(info_path)
    return df["name"].astype(str).tolist()


def list_rgs_datasets():
    info_path = REPO_ROOT / "datasets_bundle" / "dataset_info" / "regression.csv"
    df = pd.read_csv(info_path)
    return df["name"].astype(str).tolist()


def list_v2_datasets(task_type: str):
    manifest = load_datasets_v2()
    if task_type == "cls":
        return list(manifest["classification"]["datasets"])
    if task_type == "rgs":
        return list(manifest["regression"]["datasets"])
    raise ValueError(f"Unknown task_type {task_type!r}")


def default_metric_for_task(task_type: str) -> str:
    manifest = load_datasets_v2()
    if task_type == "cls":
        return manifest["classification"]["metric"]
    return manifest["regression"]["metric"]


def resolve_variant(variant_name):
    base = optimizer_base_name(variant_name)
    if base in LLM_VARIANTS:
        dispatch, backend = LLM_VARIANTS[base]
        return dispatch, variant_name, backend
    if variant_name in CLASSICAL_ALIASES:
        return CLASSICAL_ALIASES[variant_name], variant_name, None
    return variant_name, variant_name, None


def traj_label_for_variant(variant_name: str) -> str:
    base = optimizer_base_name(variant_name)
    if variant_name in CLASSICAL_ALIASES:
        base = CLASSICAL_ALIASES[variant_name]
    return TRAJ_LABELS.get(base, base)


def parse_optimizers(values, llm_backends):
    names = []
    for value in values:
        if value == "all":
            names.extend(CLASSICAL_OPTIMIZERS)
            if "gpt4o-mini" in llm_backends:
                names.extend(["ptllm", "ptllm_tpe"])
            if "qwen-sft6" in llm_backends:
                names.extend(["ptllm_qwen", "ptllm_tpe_qwen"])
        else:
            names.append(value)

    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def parse_llm_backends(value):
    if not value:
        return ["gpt4o-mini", "qwen-sft6"]
    backends = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"gpt4o-mini", "qwen-sft6"}
    unknown = set(backends) - allowed
    if unknown:
        raise ValueError(f"Unknown llm backends: {sorted(unknown)}. Allowed: {sorted(allowed)}")
    return backends


def configure_llm_runtime(args, optimizers):
    llm_variants = [name for name in optimizers if optimizer_base_name(name) in LLM_VARIANTS]
    if not llm_variants:
        return
    if args.llm_api_key_env:
        api_key = os.environ.get(args.llm_api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {args.llm_api_key_env!r} is not set.")
        LLMCONFIGS.OPENAI_KEY = api_key
        for cfg in AVAILABLE_LLMs.values():
            cfg["api_key"] = api_key
    if args.llm_base_url:
        LLMCONFIGS.OPENAI_BASE_URL = args.llm_base_url
        for cfg in AVAILABLE_LLMs.values():
            cfg["base_url"] = args.llm_base_url


def load_real_data(args):
    task_type = args.task_type
    task_const = CLASSIFICATION if task_type == "cls" else REGRESSION
    dataset_path, label_column, header, sep = get_dataset_info(task_type, args.dataset)
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")

    print(f"[Dataset] loading {dataset_path} ({task_type})")
    dm = DataManager()
    df = dm.load_csv(str(dataset_path), header=header, sep=sep)
    print(f"[Dataset] shape={df.shape}")

    train_df, test_df = dm.split_data(
        df,
        label_col=label_column,
        test_size=args.test_size,
        random_state=args.seed,
        task_type=task_const,
    )

    train_node = dm.from_train_df(train_df, label_col=label_column)
    train_node = dm.preprocess_fit(train_node, task_const)

    test_node = dm.from_test_df(test_df, has_label=True)
    test_node = dm.preprocess_transform(test_node)

    profile = build_dataset_profile(df, label_column, task_type, args.metric)
    profile["fixed_algorithm"] = args.algorithm[0]
    profile["split"] = {
        "test_size": args.test_size,
        "validation_size": args.valid_size,
        "evaluation": args.evaluation,
    }
    return dm, train_node, test_node, dataset_path, profile


def build_optimizer_params(args, variant_name, algorithms, dataset_profile):
    dispatch, _, backend = resolve_variant(variant_name)
    if dispatch == "tpe":
        return {"n_startup_trials": args.tpe_startup_trials}
    if dispatch == "hebo":
        return {
            "n_startup_trials": args.tpe_startup_trials,
            "hebo_model_name": "gp",
            "hebo_duplicate_retries": 50,
        }
    if dispatch not in ["ptllm", "ptllm_tpe", "ptllm_smac", "ptllm_cmaes", "ptllm_hebo"]:
        return None
    if len(algorithms) != 1:
        raise ValueError(f"{variant_name} only supports single-algorithm HPO. Pass exactly one --algorithm value.")

    _, prompt_id, hybrid_id = parse_variant_tags(variant_name)
    if prompt_id:
        apply_prompt_settings(args, prompt_id)
    if dispatch in ("ptllm_tpe", "ptllm_smac", "ptllm_cmaes", "ptllm_hebo") and hybrid_id:
        apply_hybrid_settings(args, hybrid_id)

    params = {
        "dataset_profile": dataset_profile,
        "n_initial_samples": args.llm_n_initial_samples,
        "history_window": args.llm_history_window,
        "top_k": args.llm_top_k,
        "recent_k": args.llm_recent_k,
        "worst_k": args.llm_worst_k,
        "run_summary": args.llm_run_summary,
        "tcs_analysis": getattr(args, "llm_tcs_analysis", False),
        "tcs_single_agent": getattr(args, "llm_tcs_single_agent", False),
        "prompt_blocks": dict(getattr(args, "llm_prompt_blocks", None) or {}),
        "experience_self_mode": getattr(args, "llm_experience_self_mode", "cohort"),
        "experience_self_start_trial": int(getattr(args, "llm_experience_self_start_trial", 1) or 1),
        "experience_explore_mode": getattr(args, "llm_experience_explore_mode", "off"),
        "experience_explore_prob": float(getattr(args, "llm_experience_explore_prob", 0.5) or 0.5),
        "experience_blame": dict(getattr(args, "llm_experience_blame", None) or {}),
        "dataset_id": getattr(args, "dataset", None),
        "task_type": "classification" if getattr(args, "task_type", "cls") == "cls" else "regression",
        "experience_retrieve_k": int(getattr(args, "llm_experience_retrieve_k", 3) or 3),
        "experience_retrieve_start_trial": int(getattr(args, "llm_experience_retrieve_start_trial", 1) or 1),
        "experience_retrieve_rerank_trial": int(getattr(args, "llm_experience_retrieve_rerank_trial", 8) or 8),
        "experience_retrieve_ranker": str(getattr(args, "llm_experience_retrieve_ranker", "cascade") or "cascade"),
        "experience_retrieve_fade": bool(getattr(args, "llm_experience_retrieve_fade", False)),
        "experience_retrieve_stop_trial": getattr(args, "llm_experience_retrieve_stop_trial", None),
        "experience_retrieve_spearman_floor": getattr(args, "llm_experience_retrieve_spearman_floor", None),
        "experience_retrieve_include_top_bottom": bool(
            getattr(args, "llm_experience_retrieve_include_top_bottom", False)
        ),
        "experience_retrieve_include_prefer": bool(
            getattr(args, "llm_experience_retrieve_include_prefer", True)
        ),
        "experience_retrieve_region_style": str(
            getattr(args, "llm_experience_retrieve_region_style", "prefer") or "prefer"
        ),
        "experience_retrieve_tb_layout": str(
            getattr(args, "llm_experience_retrieve_tb_layout", "flat") or "flat"
        ),
        "experience_retrieve_top_k": (
            1
            if getattr(args, "llm_experience_retrieve_top_k", None) in (None, "")
            else max(0, int(args.llm_experience_retrieve_top_k))
        ),
        "experience_retrieve_bottom_k": (
            1
            if getattr(args, "llm_experience_retrieve_bottom_k", None) in (None, "")
            else max(0, int(args.llm_experience_retrieve_bottom_k))
        ),
        "experience_self_gate": str(getattr(args, "llm_experience_self_gate", "clock") or "clock"),
        "experience_explore_tpe_min_trial": int(getattr(args, "llm_experience_explore_tpe_min_trial", 0) or 0),
        "experience_bank_path": None,
        "temperature": args.llm_temperature,
        "max_tokens": args.llm_max_tokens,
        "llm_timeout": args.llm_timeout,
        "fallback_on_transport_error": args.llm_fallback_on_transport_error,
        "prompt_variant": getattr(args, "prompt_variant", None),
        "hybrid_variant": getattr(args, "hybrid_variant", None),
        "response_format": getattr(args, "llm_response_format", None) or "rationale_json",
    }
    bank = getattr(args, "experience_bank", None) or getattr(args, "llm_experience_bank_path", None)
    if bank:
        bp = Path(bank)
        if not bp.is_absolute():
            bp = (REPO_ROOT / bp).resolve()
        params["experience_bank_path"] = str(bp)
    preset_path = getattr(args, "preset_warm_start_json", None)
    if preset_path:
        payload = json.loads(Path(preset_path).read_text(encoding="utf-8"))
        algo = algorithms[0]
        dataset = args.dataset
        key = f"{dataset}:{algo}"
        entry = None
        if isinstance(payload, dict):
            entry = payload.get(key) or payload.get(dataset)
        configs = []
        if isinstance(entry, dict):
            if isinstance(entry.get("config"), dict):
                configs = [dict(entry["config"])]
            elif isinstance(entry.get("best_config"), dict):
                configs = [dict(entry["best_config"])]
            elif isinstance(entry.get("preset_warm_start_configs"), list):
                configs = [dict(c) for c in entry["preset_warm_start_configs"] if isinstance(c, dict)]
            elif any(":" in str(k) for k in entry.keys()):
                # Flat HP dict keyed like "lightgbm:n_estimators".
                configs = [dict(entry)]
        elif isinstance(entry, list):
            configs = [dict(c) for c in entry if isinstance(c, dict)]
        if not configs:
            raise ValueError(
                f"--preset-warm-start-json has no config for key {key!r} "
                f"(file={preset_path})"
            )
        for cfg in configs:
            cfg.setdefault("algorithm", algo)
        params["preset_warm_start_configs"] = configs
        params["_preset_warm_start_json"] = str(preset_path)
        params["_preset_warm_start_key"] = key
    if dispatch == "ptllm_tpe":
        params.update(
            {
                "llm_prob": args.llm_tpe_llm_prob,
                "llm_phase_trials": args.llm_tpe_phase_trials,
                "bo_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "tpe_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "tpe_params": {"n_startup_trials": args.llm_tpe_tpe_startup_trials},
                "n_startup_trials": args.llm_tpe_tpe_startup_trials,
                "hybrid_selector": getattr(args, "hybrid_selector", "bernoulli"),
                "ucb_c": getattr(args, "ucb_c", 1.4),
                "pucb_c2": getattr(args, "pucb_c2", 1.4),
                "pucb_p1": getattr(args, "pucb_p1", 0.9),
                "pucb_p_end": getattr(args, "pucb_p_end", None),
                "pucb_floor": getattr(args, "pucb_floor", None),
                "pucb_floor_after_frac": getattr(args, "pucb_floor_after_frac", None),
                "pucb_switch_after_frac": getattr(args, "pucb_switch_after_frac", None),
                "pucb_q": getattr(args, "pucb_q", "window_minmax"),
                "pucb_window": getattr(args, "pucb_window", 10),
                "dts_gamma": getattr(args, "dts_gamma", 0.95),
                "ap_beta": getattr(args, "ap_beta", 0.3),
                "ap_pmin": getattr(args, "ap_pmin", 0.2),
            }
        )
    if dispatch == "ptllm_smac":
        params.update(
            {
                "llm_prob": args.llm_tpe_llm_prob,
                "llm_phase_trials": args.llm_tpe_phase_trials,
                "bo_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "tpe_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "n_startup_trials": args.llm_tpe_tpe_startup_trials,
                "smac_params": {
                    "rand_prob": 0.1,
                    "acq_optimizer_rand_prob": 0,
                    "initial_runs": args.llm_tpe_tpe_startup_trials,
                },
            }
        )
    if dispatch == "ptllm_cmaes":
        params.update(
            {
                "llm_prob": args.llm_tpe_llm_prob,
                "llm_phase_trials": args.llm_tpe_phase_trials,
                "bo_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "tpe_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "cmaes_params": {"n_startup_trials": args.llm_tpe_tpe_startup_trials},
                "n_startup_trials": args.llm_tpe_tpe_startup_trials,
            }
        )
    if dispatch == "ptllm_hebo":
        params.update(
            {
                "llm_prob": args.llm_tpe_llm_prob,
                "llm_phase_trials": args.llm_tpe_phase_trials,
                "bo_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "hebo_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "tpe_duplicate_retries": args.llm_tpe_tpe_duplicate_retries,
                "n_startup_trials": args.llm_tpe_tpe_startup_trials,
                "bo_arm": "hebo",
                "hybrid_selector": getattr(args, "hybrid_selector", "bernoulli"),
                "ucb_c": getattr(args, "ucb_c", 1.4),
                "pucb_c2": getattr(args, "pucb_c2", 1.4),
                "pucb_p1": getattr(args, "pucb_p1", 0.9),
                "pucb_p_end": getattr(args, "pucb_p_end", None),
                "pucb_floor": getattr(args, "pucb_floor", None),
                "pucb_floor_after_frac": getattr(args, "pucb_floor_after_frac", None),
                "pucb_switch_after_frac": getattr(args, "pucb_switch_after_frac", None),
                "pucb_q": getattr(args, "pucb_q", "window_minmax"),
                "pucb_window": getattr(args, "pucb_window", 10),
                "dts_gamma": getattr(args, "dts_gamma", 0.95),
                "ap_beta": getattr(args, "ap_beta", 0.3),
                "ap_pmin": getattr(args, "ap_pmin", 0.2),
            }
        )
    if backend == "gpt4o-mini":
        params.update(dict(LLM_BACKEND_DEFAULTS[backend]))
    elif backend == "qwen-sft6":
        # Skip HF load for --llm-mock dry-runs (adapter path would otherwise force load).
        if not args.llm_mock:
            apply_qwen_backend_params(params, args)
        elif getattr(args, "qwen_adapter_path", None):
            params["qwen_adapter_path"] = args.qwen_adapter_path
            params["llm_transport"] = "hf"
    if args.llm_mock:
        params["mock_llm"] = True
    if args.llm_response:
        params["mock_llm_responses"] = args.llm_response
    if args.llm_chat_engine:
        params["chat_engine"] = args.llm_chat_engine
    if args.llm_model:
        params["llm_model"] = args.llm_model
    return params


def validate_optimizer_request(optimizers, algorithms):
    llm_variants = [name for name in optimizers if optimizer_base_name(name) in LLM_VARIANTS]
    if llm_variants and len(algorithms) != 1:
        raise ValueError(
            f"{', '.join(llm_variants)} only supports single-algorithm HPO. Pass exactly one --algorithm value."
        )


def categorical_hyperparameters(config_space):
    return [
        hp.name
        for hp in config_space.get_hyperparameters()
        if isinstance(hp, CategoricalHyperparameter)
    ]


def find_latest_run_dir(dataset_output_dir: Path, variant_name: str, algorithm: str | None = None):
    _, output_name, _ = resolve_variant(variant_name)
    opt_dir = dataset_output_dir / output_name
    if not opt_dir.is_dir():
        return None
    candidates = sorted(p for p in opt_dir.glob("CASH-*") if p.is_dir())
    if not algorithm:
        return candidates[-1] if candidates else None
    matching = [
        run_dir
        for run_dir in candidates
        if count_trajectory_trials(run_dir, variant_name, algorithm=algorithm) > 0
    ]
    return matching[-1] if matching else None


def _traj_majority_algorithm(traj_path: Path) -> str | None:
    with traj_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
    if not algos:
        return None
    return Counter(algos).most_common(1)[0][0]


def count_trajectory_trials(run_dir: Path, variant_name: str, algorithm: str | None = None) -> int:
    if run_dir is None:
        return 0
    label = traj_label_for_variant(variant_name)
    traj_files = sorted(run_dir.glob(f"*_opt_{label}_traj.csv"))
    if not traj_files:
        traj_files = sorted(run_dir.glob("*_traj.csv"))
    if not traj_files:
        return 0
    traj = traj_files[-1]
    with traj.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if algorithm:
        mode = _traj_majority_algorithm(traj)
        if mode != algorithm:
            return 0
    return len(rows)


def count_variant_trials(
    dataset_output_dir: Path, variant_name: str, algorithm: str | None = None
) -> int:
    """Max trial rows across CASH runs for this variant.

    When ``algorithm`` is set (algo40 single-algorithm jobs), only runs whose
    trajectory majority algorithm matches are counted. Shared dataset/variant
    dirs otherwise made resume treat a sibling algorithm's 80-row traj as done.
    """
    opt_dir = dataset_output_dir / variant_name
    if not opt_dir.is_dir():
        return 0
    counts = [
        count_trajectory_trials(run_dir, variant_name, algorithm=algorithm)
        for run_dir in sorted(p for p in opt_dir.glob("CASH-*") if p.is_dir())
    ]
    return max(counts) if counts else 0


def variant_is_complete(
    dataset_output_dir: Path, variant_name: str, n_trials: int, algorithm: str | None = None
) -> bool:
    return count_variant_trials(dataset_output_dir, variant_name, algorithm=algorithm) >= n_trials


def load_existing_summary(dataset_output_dir: Path):
    summary_path = dataset_output_dir / "summary.json"
    if not summary_path.is_file():
        return {}, None
    try:
        with summary_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        corrupt = summary_path.with_suffix(
            f".corrupt.{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            summary_path.rename(corrupt)
        except OSError:
            corrupt = None
        print(
            f"[Resume] ignoring corrupt summary.json in {dataset_output_dir}: {exc}"
            + (f" (moved to {corrupt.name})" if corrupt else "")
        )
        return {}, None
    by_optimizer = {row["optimizer"]: row for row in payload.get("results", [])}
    return by_optimizer, payload


def save_dataset_summary(
    dataset_output_dir: Path,
    dataset_name: str,
    dataset_path,
    algorithms,
    args,
    results,
    completed_variants=None,
):
    summary_path = dataset_output_dir / "summary.json"
    payload = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "algorithms": algorithms,
        "metric": args.metric,
        "task_type": args.task_type,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "results": results,
    }
    if completed_variants is not None:
        payload["completed_variants"] = completed_variants
    tmp_path = summary_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, summary_path)
    return summary_path


def dataset_is_fully_complete(
    dataset_output_dir: Path, optimizers, n_trials: int, algorithm: str | None = None
) -> bool:
    return all(
        variant_is_complete(dataset_output_dir, name, n_trials, algorithm=algorithm)
        for name in optimizers
    )


RETRYABLE_DATASET_ERRORS = (
    "Resource temporarily unavailable",
    "can't start new thread",
    "Too many open files",
    "Cannot allocate memory",
)


def is_retryable_dataset_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in RETRYABLE_DATASET_ERRORS)


def history_scope_dir(
    history_dir: Path, task_type: str, metric: str, dataset_name: str, variant_name: str
) -> Path:
    """Isolated BO history dir per optimizer variant (avoids cross-run races)."""
    return history_dir / task_type / metric / dataset_name / variant_name


def save_histories(
    history_dir: Path,
    metric: str,
    dataset_name: str,
    algorithms,
    task_type: str,
    variant_name: str,
):
    kd = KnowledgeDatabase.instance()
    saved = {}
    target_dir = history_scope_dir(history_dir, task_type, metric, dataset_name, variant_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    for algorithm in algorithms:
        history = kd.bo_history_mab.get(algorithm)
        if history is None:
            continue
        history_path = target_dir / f"{algorithm}.json"
        history.save_json(str(history_path))
        saved[algorithm] = str(history_path)
    return saved


def run_one_optimizer(args, variant_name, train_node, test_node, algorithms, dataset_profile):
    dispatch, output_name, _ = resolve_variant(variant_name)
    metric = args.metric
    task_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    output_dir = Path(args.output_dir).resolve() / output_name
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)
    optimizer_params = build_optimizer_params(args, variant_name, algorithms, dataset_profile)

    KnowledgeDatabase._instance = None

    task_id = f"{args.dataset}_{output_name}"
    print(f"\n[CASH] variant={variant_name}, dispatch={dispatch}, task_id={task_id}")

    opt = CASH(
        include_algorithms=algorithms,
        task_type=task_const,
        metric=metric,
        data_node=train_node,
        evaluation=args.evaluation,
        resampling_params={"test_size": args.valid_size, "shuffle": False},
        optimizer=dispatch,
        time_limit=args.time_limit,
        amount_of_resource=args.n_trials,
        per_run_time_limit=args.per_run_time_limit,
        inner_iter_num_per_iter=1,
        output_dir=str(output_dir),
        seed=args.seed,
        n_jobs=1,
        task_id=task_id,
        topk=args.n_trials,
        optimizer_params=optimizer_params,
    )

    conf = opt.get_conf(save=True)
    # Strip live HF client before any downstream JSON (best_model_info / summary).
    if isinstance(conf.get("optimizer_params"), dict) and "llm_client" in conf["optimizer_params"]:
        conf = dict(conf)
        opt_params = dict(conf["optimizer_params"])
        client = opt_params.pop("llm_client", None)
        if client is not None:
            if getattr(client, "adapter_path", None):
                opt_params.setdefault("qwen_adapter_path", client.adapter_path)
            if getattr(client, "model_path", None):
                opt_params.setdefault("qwen_model_path", client.model_path)
        conf["optimizer_params"] = opt_params
    cat_hps = categorical_hyperparameters(opt.cs)
    print(f"[Search Space] {len(opt.cs.get_hyperparameters())} hyperparameters")
    print(f"[Search Space] categorical hps: {cat_hps}")

    best_perf, stats_path = opt.run(refit=args.refit)
    print(f"[CASH] best validation score: {best_perf:.6f}")
    print(f"[CASH] top-k stats: {stats_path}")

    test_perf = None
    if not args.skip_test_eval:
        scorer = get_metric(metric)
        pred = opt.predict(test_node, refit=args.refit, ens=False)
        test_perf = scorer._score_func(test_node.data[1], pred) * scorer._sign
        print(f"[CASH] test score: {test_perf:.6f}")

    histories = save_histories(
        history_dir, metric, args.dataset, algorithms, args.task_type, output_name
    )
    extra_info = {
        "stats_path": stats_path,
        "history_paths": histories,
        "categorical_hyperparameters": cat_hps,
        "config": conf,
        "variant": variant_name,
        "dispatch": dispatch,
        "prompt_variant": getattr(args, "prompt_variant", None),
        "hybrid_variant": getattr(args, "hybrid_variant", None),
    }
    if test_perf is not None:
        extra_info["test_perf"] = float(test_perf)
    model_info = opt.get_model_info(save=True, extra_info=extra_info)

    return {
        "optimizer": variant_name,
        "dispatch": dispatch,
        "task_id": task_id,
        "best_perf": float(best_perf),
        "test_perf": float(test_perf) if test_perf is not None else None,
        "stats_path": stats_path,
        "histories": histories,
        "model_info": model_info,
        "output_dir": opt.output_dir,
    }


def run_one_dataset(args):
    optimizers = expand_optimizer_variants(parse_optimizers(args.optimizer, args.llm_backends), args)
    algorithms = list(args.algorithm)
    validate_optimizer_request(optimizers, algorithms)
    configure_llm_runtime(args, optimizers)

    dataset_output_dir = Path(args.output_dir).resolve()
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(dataset_output_dir)

    existing_by_optimizer, _ = ({}, None)
    if args.resume:
        existing_by_optimizer, _ = load_existing_summary(dataset_output_dir)

    algo_filter = algorithms[0] if len(algorithms) == 1 else None

    print("[Setup]")
    print(f"  task_type={args.task_type}")
    print(f"  dataset={args.dataset}")
    print(f"  optimizers={optimizers}")
    print(f"  algorithms={algorithms}")
    print(f"  n_trials={args.n_trials}, seed={args.seed}")
    if args.resume:
        print("  resume=True (skip completed optimizer variants)")
        if algo_filter:
            print(f"  resume_algorithm={algo_filter} (only matching trajs count as complete)")

    pending = []
    for variant_name in optimizers:
        if args.resume and variant_is_complete(
            dataset_output_dir, variant_name, args.n_trials, algorithm=algo_filter
        ):
            trials = count_trajectory_trials(
                find_latest_run_dir(dataset_output_dir, variant_name, algorithm=algo_filter),
                variant_name,
                algorithm=algo_filter,
            )
            print(
                f"[Resume] skip {variant_name} ({trials}/{args.n_trials} trials complete"
                + (f", algorithm={algo_filter}" if algo_filter else "")
                + ")"
            )
            pending.append(variant_name)
            continue
        pending.append(variant_name)

    need_data = [
        v
        for v in optimizers
        if not (
            args.resume
            and variant_is_complete(
                dataset_output_dir, v, args.n_trials, algorithm=algo_filter
            )
        )
    ]
    if not need_data:
        print("[Resume] all requested optimizers already complete for this dataset")
        summary_path = dataset_output_dir / "summary.json"
        if summary_path.is_file():
            return summary_path, list(existing_by_optimizer.values())
        return summary_path, []

    _, train_node, test_node, dataset_path, dataset_profile = load_real_data(args)

    results = []
    for variant_name in optimizers:
        if args.resume and variant_is_complete(
            dataset_output_dir, variant_name, args.n_trials, algorithm=algo_filter
        ):
            if variant_name in existing_by_optimizer:
                results.append(existing_by_optimizer[variant_name])
            continue

        try:
            rounds = 0
            max_rounds = 12
            while not variant_is_complete(
                dataset_output_dir, variant_name, args.n_trials, algorithm=algo_filter
            ):
                rounds += 1
                if rounds > max_rounds:
                    trials = count_variant_trials(
                        dataset_output_dir, variant_name, algorithm=algo_filter
                    )
                    print(
                        f"[Warn] {variant_name} still incomplete after {max_rounds} rounds "
                        f"({trials}/{args.n_trials} trials"
                        + (f", algorithm={algo_filter}" if algo_filter else "")
                        + ")"
                    )
                    break
                result = run_one_optimizer(
                    args, variant_name, train_node, test_node, algorithms, dataset_profile
                )
                existing_by_optimizer[variant_name] = result
                if variant_is_complete(
                    dataset_output_dir, variant_name, args.n_trials, algorithm=algo_filter
                ):
                    break
            if variant_name in existing_by_optimizer:
                results.append(existing_by_optimizer[variant_name])
            completed = [
                name
                for name in optimizers
                if variant_is_complete(
                    dataset_output_dir, name, args.n_trials, algorithm=algo_filter
                )
            ]
            save_dataset_summary(
                dataset_output_dir,
                args.dataset,
                dataset_path,
                algorithms,
                args,
                list(existing_by_optimizer.values()),
                completed_variants=completed,
            )
            print(f"[Checkpoint] saved progress after {variant_name}")
        except Exception as exc:
            if existing_by_optimizer:
                save_dataset_summary(
                    dataset_output_dir,
                    args.dataset,
                    dataset_path,
                    algorithms,
                    args,
                    list(existing_by_optimizer.values()),
                )
            raise exc

    print("\n[Summary]")
    for result in results:
        test_str = f"{result['test_perf']:.6f}" if result["test_perf"] is not None else "n/a"
        print(
            f"{result['optimizer']}: "
            f"valid_best={result['best_perf']:.6f}, "
            f"test={test_str}, "
            f"output={result['output_dir']}"
        )

    all_results = [existing_by_optimizer[name] for name in optimizers if name in existing_by_optimizer]
    summary_path = save_dataset_summary(
        dataset_output_dir,
        args.dataset,
        dataset_path,
        algorithms,
        args,
        all_results,
        completed_variants=optimizers,
    )
    print(f"[Summary] saved {summary_path}")
    return summary_path, all_results


def update_summary_all(output_root, dataset_name, summary_path):
    summary_all_path = Path(output_root) / "summary_all.json"
    if summary_all_path.is_file():
        with summary_all_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"datasets": {}}
    payload["datasets"][dataset_name] = str(summary_path)
    with summary_all_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Mindware CASH HPO on real classification datasets from datasets_bundle/, "
            "comparing smac, tpe, cmaes, ptllm, ptllm_tpe (gpt-4o-mini), and Qwen SFT6 variants."
        )
    )
    parser.add_argument(
        "--optimizer",
        nargs="+",
        choices=["all"] + ALL_OPTIMIZER_VARIANTS,
        default=["all"],
    )
    parser.add_argument("--dataset", default="ada")
    parser.add_argument("--task-type", choices=["cls", "rgs"], default="cls")
    parser.add_argument("--v2-datasets", action="store_true", help="Use datasets_v2.json list for --all-datasets.")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="With --v2-datasets: run only the 5 pilot datasets (3 cls + 2 rgs).",
    )
    parser.add_argument(
        "--remaining-only",
        action="store_true",
        help="With --v2-datasets: run only non-pilot datasets (rollout after pilot tuning).",
    )
    parser.add_argument(
        "--include-datasets",
        default="",
        help="Comma-separated dataset names to keep (e.g. yeast,adult,boston,elevators).",
    )
    parser.add_argument("--algorithm", nargs="+", default=[DEFAULT_ALGORITHM])
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metric", default="bal_acc")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--valid-size", type=float, default=0.35)
    parser.add_argument("--evaluation", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument(
        "--per-run-time-limit",
        type=int,
        default=300,
        help="Per-trial wall-clock limit (s). 300 gives lightgbm/xgboost headroom; trees usually finish far sooner.",
    )
    parser.add_argument(
        "--dataset-retries",
        type=int,
        default=3,
        help="Retries per dataset on thread/resource exhaustion (fresh attempt after backoff).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmark_data" / "real_hpo_v2"),
    )
    parser.add_argument("--history-dir", default=str(REPO_ROOT / "history_data"))
    parser.add_argument("--refit", choices=["partial", "full", "cv"], default="partial")
    parser.add_argument("--skip-test-eval", action="store_true", help="Skip held-out test evaluation (HTML plan uses valid only).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume interrupted runs: skip optimizer variants that already have a full "
            "trajectory (n_trials rows), and continue remaining variants for each dataset."
        ),
    )
    parser.add_argument("--tpe-startup-trials", type=int, default=3)
    parser.add_argument("--llm-backends", default="gpt4o-mini", help="Comma-separated LLM backends for --optimizer all.")
    parser.add_argument(
        "--qwen-backend",
        choices=["auto", "hf", "ollama"],
        default="auto",
        help="Qwen SFT6 transport: HuggingFace in-process (hf), Ollama API, or auto.",
    )
    parser.add_argument(
        "--qwen-model-path",
        default=DEFAULT_QWEN_MODEL_PATH,
        help="Local HF checkpoint for Qwen when --qwen-backend=hf/auto. "
        "With --qwen-adapter-path this should be the base model (Qwen3.5-9B).",
    )
    parser.add_argument(
        "--qwen-adapter-path",
        default=None,
        help="Optional Peft LoRA adapter dir (e.g. lora_p5_s0_s1a). Loaded on top of --qwen-model-path.",
    )
    parser.add_argument(
        "--qwen-enable-thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking mode in chat template (default off; SFT used nothink).",
    )
    parser.add_argument("--llm-mock", action="store_true")
    parser.add_argument("--llm-response", action="append", default=None)
    parser.add_argument("--llm-chat-engine", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key-env", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--llm-response-format",
        default="rationale_json",
        choices=["rationale_json", "json_only"],
        help="Match SFT target: json_only for S0/S1a; rationale_json for S1b/gpt (default).",
    )
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-n-initial-samples", type=int, default=4)
    parser.add_argument("--llm-history-window", type=int, default=20)
    parser.add_argument("--llm-top-k", type=int, default=6)
    parser.add_argument("--llm-recent-k", type=int, default=10)
    parser.add_argument("--llm-worst-k", type=int, default=4)
    parser.add_argument("--llm-run-summary", action="store_true")
    parser.add_argument("--llm-tcs-analysis", action="store_true", help="Paper-faithful TCS + Analysis Agent before optimizer.")
    parser.add_argument("--prompt-variant", default=None, help="Prompt recipe id (P0-P25); see prompt_variants.json.")
    parser.add_argument(
        "--llm-experience-self-start-trial",
        type=int,
        default=None,
        help="1-indexed trial when P5+ experience_self block activates (default from prompt variant, else 1).",
    )
    parser.add_argument(
        "--experience-bank",
        default=None,
        help="JSONL experience bank for experience_retrieve (overrides prompt variant path).",
    )
    parser.add_argument("--hybrid-variant", default=None, help="Hybrid schedule id (H0-H7); see hybrid_variants.json.")
    parser.add_argument(
        "--variant-suffix",
        default=None,
        help="Optional tag appended to variant dir (e.g. json_only -> ptllm_tpe__P5__H4__json_only).",
    )
    parser.add_argument(
        "--preset-warm-start-json",
        default=None,
        help=(
            "JSON map of dataset:algorithm -> best warm-start HP dict (or {config: ...}). "
            "Used as trial-1 llm_warm_start when --llm-n-initial-samples >= 1; "
            "does not call the HPO LLM for those slots."
        ),
    )
    parser.add_argument("--llm-fallback-on-transport-error", action="store_true")
    parser.add_argument("--llm-tpe-llm-prob", type=float, default=0.5)
    parser.add_argument("--llm-tpe-phase-trials", type=int, default=0)
    parser.add_argument("--llm-tpe-tpe-startup-trials", type=int, default=3)
    parser.add_argument("--llm-tpe-tpe-duplicate-retries", type=int, default=5)
    return parser.parse_args()


HYBRID_OPTIMIZERS = ("ptllm_tpe", "ptllm_smac", "ptllm_cmaes", "ptllm_hebo")


def expand_optimizer_variants(names, args):
    expanded = []
    for name in names:
        base = optimizer_base_name(name)
        if base not in LLM_VARIANTS:
            expanded.append(name)
            continue
        dispatch, _backend = LLM_VARIANTS[base]
        _, prompt_id, hybrid_id = parse_variant_tags(name)
        if prompt_id is None and args.prompt_variant:
            prompt_id = args.prompt_variant
        # Use dispatch name (ptllm_tpe) so ptllm_tpe_qwen also gets hybrid tags.
        if hybrid_id is None and dispatch in HYBRID_OPTIMIZERS and args.hybrid_variant:
            hybrid_id = args.hybrid_variant
        if prompt_id or hybrid_id:
            expanded.append(
                make_llm_variant_name(
                    base,
                    prompt_id,
                    hybrid_id if dispatch in HYBRID_OPTIMIZERS else None,
                    getattr(args, "variant_suffix", None),
                )
            )
        else:
            expanded.append(name)
    return expanded


def _filter_dataset_jobs(jobs, include_csv: str):
    if not include_csv.strip():
        return jobs
    allow = {name.strip() for name in include_csv.split(",") if name.strip()}
    return [(task_type, dataset_name) for task_type, dataset_name in jobs if dataset_name in allow]


def resolve_dataset_jobs(args):
    if args.pilot_only and args.remaining_only:
        raise ValueError("Use only one of --pilot-only or --remaining-only.")
    if args.v2_datasets and (args.all_datasets or args.pilot_only or args.remaining_only):
        if args.pilot_only:
            jobs = list_v2_dataset_jobs("pilot")
        elif args.remaining_only:
            jobs = list_v2_dataset_jobs("remaining")
        else:
            jobs = list_v2_dataset_jobs("all")
        return _filter_dataset_jobs(jobs, args.include_datasets)
    if args.all_datasets:
        jobs = [("cls", name) for name in list_cls_datasets()] + [
            ("rgs", name) for name in list_rgs_datasets()
        ]
        return _filter_dataset_jobs(jobs, args.include_datasets)
    return [(args.task_type, args.dataset)]


def main():
    args = parse_args()
    args.llm_backends = parse_llm_backends(args.llm_backends)
    _cli_exp_start = args.llm_experience_self_start_trial
    if args.prompt_variant:
        apply_prompt_settings(args, args.prompt_variant)
    if args.hybrid_variant:
        apply_hybrid_settings(args, args.hybrid_variant)
    if _cli_exp_start is not None:
        args.llm_experience_self_start_trial = int(_cli_exp_start)
    elif not getattr(args, "llm_experience_self_start_trial", None):
        args.llm_experience_self_start_trial = 1
    if getattr(args, "experience_bank", None):
        args.llm_experience_bank_path = args.experience_bank

    os.chdir(REPO_ROOT)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    optimizers = expand_optimizer_variants(parse_optimizers(args.optimizer, args.llm_backends), args)
    for task_type, dataset_name in resolve_dataset_jobs(args):
        args.task_type = task_type
        args.metric = default_metric_for_task(task_type) if args.v2_datasets else (
            default_metric_for_task(task_type) if task_type == "rgs" and args.metric == "bal_acc" else args.metric
        )
        dataset_dir = output_root / task_type / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(dataset_dir)
        args.dataset = dataset_name
        summary_path = dataset_dir / "summary.json"

        algo_filter = args.algorithm[0] if len(args.algorithm) == 1 else None
        if args.resume:
            if dataset_is_fully_complete(
                dataset_dir, optimizers, args.n_trials, algorithm=algo_filter
            ):
                if summary_path.is_file():
                    print(
                        f"[Resume] already complete: {task_type}/{dataset_name}"
                        + (f" algorithm={algo_filter}" if algo_filter else "")
                    )
                    update_summary_all(output_root, f"{task_type}/{dataset_name}", summary_path)
                    continue
                print(
                    f"[Resume] trajectories complete but summary missing: {task_type}/{dataset_name}"
                    + (f" algorithm={algo_filter}" if algo_filter else "")
                    + " — rebuilding"
                )
        elif summary_path.is_file():
            if args.all_datasets or args.v2_datasets:
                print(f"[Skip] already complete: {task_type}/{dataset_name}")
                update_summary_all(output_root, f"{task_type}/{dataset_name}", summary_path)
                continue
            print(f"[Skip] summary exists: {summary_path}")
            continue

        last_exc = None
        for attempt in range(1, max(1, int(args.dataset_retries)) + 1):
            try:
                summary_path, _ = run_one_dataset(args)
                update_summary_all(output_root, f"{task_type}/{dataset_name}", summary_path)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if is_retryable_dataset_error(exc) and attempt < int(args.dataset_retries):
                    wait_s = 30 * attempt
                    print(
                        f"[Retry] {task_type}/{dataset_name} attempt {attempt}/"
                        f"{args.dataset_retries} after resource error: {exc} "
                        f"(sleep {wait_s}s)"
                    )
                    time.sleep(wait_s)
                    continue
                failed_path = output_root / "failed_runs.jsonl"
                with failed_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "task_type": task_type,
                                "dataset": dataset_name,
                                "error": str(exc),
                                "resume": bool(args.resume),
                                "attempt": attempt,
                            }
                        )
                        + "\n"
                    )
                print(f"[Error] {task_type}/{dataset_name}: {exc}")
                if not (args.all_datasets or args.v2_datasets):
                    raise
                break
        if last_exc is not None and not (args.all_datasets or args.v2_datasets):
            raise last_exc


if __name__ == "__main__":
    main()
