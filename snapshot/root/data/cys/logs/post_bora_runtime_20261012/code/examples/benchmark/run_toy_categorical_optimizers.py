import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
from ConfigSpace.hyperparameters import CategoricalHyperparameter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_utils import build_dataset_profile, get_dataset_info
from ml_runtime_policy import apply_policy, optional_seconds
import ml_run_archive as run_archive
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
    "ptllm_rgpe_qwen": ("ptllm_rgpe", "qwen-sft6"),
    "ptllm_smac_qwen": ("ptllm_smac", "qwen-sft6"),
    "ptllm_hebo": ("ptllm_hebo", "gpt4o-mini"),
    "ptllm_hebo_qwen": ("ptllm_hebo", "qwen-sft6"),
}
# PT4Opt paper baselines. Selectable via --optimizer but deliberately kept out of
# CLASSICAL_OPTIMIZERS so that "--optimizer all" keeps its existing meaning.
BASELINE_OPTIMIZERS = ["random_search", "lolbo", "fsbo", "rgpe", "optformer", "bora", "tosfit", "bolt_ml", "semanticopt"]
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
    "fsbo": "fsbo",
    "rgpe": "rgpe",
    "optformer": "optformer",
    "ptllm": "ptllm",
    "ptllm_tpe": "ptllm_tpe",
    "ptllm_smac": "ptllm_smac",
    "ptllm_cmaes": "ptllm_cmaes",
    "ptllm_qwen": "ptllm",
    "ptllm_tpe_qwen": "ptllm_tpe",
    "ptllm_rgpe_qwen": "ptllm_rgpe",
    "ptllm_smac_qwen": "ptllm_smac",
    "ptllm_hebo": "ptllm_hebo",
    "ptllm_hebo_qwen": "ptllm_hebo",
}

DEFAULT_QWEN_MODEL_PATH = "/root/blockdata/model-weights/llamafactory_qwen35_9b_full_sft_6"

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
    if dispatch in ("bora", "tosfit", "bolt_ml", "semanticopt"):
        from mindware.components.optimizers.online_llm_optimizer import QWEN_BASE
        if len(algorithms) != 1:
            raise ValueError("These ML baselines require one fixed ML algorithm")
        if args.qwen_adapter_path or args.qwen_model_path not in (DEFAULT_QWEN_MODEL_PATH, QWEN_BASE):
            raise ValueError("These baselines select their approved model; omit model/adapter overrides")
        if args.llm_mock or args.llm_response:
            raise ValueError("Use the baseline tests for fixtures; this CLI runs live models")
        limits = (args.extractor_max_tokens, args.extractor_timeout, args.extractor_max_calls)
        if any(value is None or value <= 0 for value in limits):
            raise ValueError("Specify positive --extractor-max-tokens, --extractor-timeout and --extractor-max-calls")
        if dispatch == "tosfit" and (not args.tosfit_device or not args.tosfit_reference_device):
            raise ValueError("Specify --tosfit-device and --tosfit-reference-device")
        if dispatch == "tosfit" and not args.skip_test_eval:
            raise ValueError("ToSFiT's evaluation protocol requires --skip-test-eval")
        params = {"dataset_profile": dataset_profile,
                "extractor_max_tokens": args.extractor_max_tokens,
                "extractor_timeout": args.extractor_timeout,
                "extractor_max_calls": args.extractor_max_calls,
                "generator_max_tokens": args.llm_max_tokens,
                "device": args.tosfit_device, "reference_device": args.tosfit_reference_device}
        if args.recommendation_fallback == "default":
            params["recommendation_default_fallback"] = True
        if dispatch == "bora" and args.bora_gpu_memory_gib is not None:
            params["gpu_memory_limit_gib"] = args.bora_gpu_memory_gib
        if dispatch == "bora":
            params["generator_max_tokens"] = (args.llm_max_tokens
                if args.llm_max_tokens_explicit else 4096)
        if dispatch == "bolt_ml":
            if not args.skip_test_eval:
                raise ValueError("BOLT-ML's 100-evaluation protocol requires --skip-test-eval")
            params.update(task_type=args.task_type, recommendation_default_fallback=True)
        if dispatch == "semanticopt":
            if not args.skip_test_eval:
                raise ValueError("SemanticOpt's evaluation protocol requires --skip-test-eval")
            params.update(task_type=args.task_type, recommendation_default_fallback=True)
        return params
    if dispatch == "tpe":
        return {"n_startup_trials": args.tpe_startup_trials}
    if dispatch == "hebo":
        return {
            "n_startup_trials": args.tpe_startup_trials,
            "hebo_model_name": "gp",
            "hebo_duplicate_retries": 50,
        }
    if dispatch in ("fsbo", "rgpe", "optformer"):
        dump_dir = getattr(args, "transfer_dump_dir", None) or str(
            REPO_ROOT / "benchmark_data" / "ml_train_80_transfer_v2"
        )
        ckpt_dir = getattr(args, "fsbo_ckpt_dir", None) or str(
            REPO_ROOT / "benchmark_data" / "ml_train_80_fsbo_ckpts"
        )
        return {
            "n_startup_trials": args.tpe_startup_trials,
            "task_type": args.task_type,
            "dump_dir": dump_dir,
            "ckpt_dir": ckpt_dir,
            "rgpe_cache_dir": args.rgpe_cache_dir,
            "optformer_model_dir": args.optformer_model_dir,
            "allow_cuda": args.transfer_allow_cuda,
            "worker_timeout": args.transfer_worker_timeout,
            "n_pending": int(getattr(args, "transfer_n_pending", 256) or 256),
            "n_trials": int(args.n_trials),
        }
    if dispatch not in ["ptllm", "ptllm_tpe", "ptllm_rgpe", "ptllm_smac", "ptllm_cmaes", "ptllm_hebo"]:
        return None
    if len(algorithms) != 1:
        raise ValueError(f"{variant_name} only supports single-algorithm HPO. Pass exactly one --algorithm value.")

    _, prompt_id, hybrid_id = parse_variant_tags(variant_name)
    if prompt_id:
        apply_prompt_settings(args, prompt_id)
    if dispatch in ("ptllm_tpe", "ptllm_rgpe", "ptllm_smac", "ptllm_cmaes", "ptllm_hebo") and hybrid_id:
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
        "source_gp_cache": getattr(args, "source_gp_cache", None),
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
        params["preset_warm_start_sha256"] = hashlib.sha256(Path(preset_path).read_bytes()).hexdigest()
    if dispatch in ("ptllm_tpe", "ptllm_rgpe"):
        params.update(
            {
                "llm_duplicate_policy": getattr(args, "llm_duplicate_policy", "legacy"),
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
    if dispatch == "ptllm_rgpe":
        params.update(bo_arm="rgpe", rgpe_params={
            "dump_dir": args.transfer_dump_dir,
            "rgpe_cache_dir": args.rgpe_cache_dir,
            "worker_timeout": args.transfer_worker_timeout,
            "n_pending": int(args.transfer_n_pending),
            "allow_cuda": False,
        })
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
    transfer = [name for name in optimizers if optimizer_base_name(name) in ("fsbo", "rgpe", "optformer", "bora", "tosfit", "bolt_ml", "semanticopt")]
    if transfer and len(algorithms) != 1:
        raise ValueError(
            f"{', '.join(transfer)} is grouped by (task_type, algorithm). Pass exactly one --algorithm value."
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
    if not candidates:
        return None

    def _run_score(run_dir: Path) -> tuple:
        n = count_trajectory_trials(run_dir, variant_name, algorithm=algorithm)
        return (n, run_dir.name)

    if not algorithm:
        return max(candidates, key=_run_score)
    matching = [
        run_dir
        for run_dir in candidates
        if count_trajectory_trials(run_dir, variant_name, algorithm=algorithm) > 0
    ]
    return max(matching, key=_run_score) if matching else None


def _traj_majority_algorithm(traj_path: Path) -> str | None:
    with traj_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
    if not algos:
        return None
    return Counter(algos).most_common(1)[0][0]


def _best_traj_in_run(run_dir: Path, variant_name: str, algorithm: str | None = None):
    """(traj_path, n_rows) for the longest usable traj in ``run_dir``.

    Picks by row count, not by filename order. The old ``traj_files[-1]`` took
    the lexicographically last file, so when the labelled glob missed and the
    ``*_traj.csv`` fallback matched several files, resume could count -- and
    then restore from -- a shorter traj and silently redo trials. Counting and
    restoring must agree, so find_run_traj_file() shares this helper.
    """
    if run_dir is None:
        return None, 0
    label = traj_label_for_variant(variant_name)
    traj_files = sorted(run_dir.glob(f"*_opt_{label}_traj.csv"))
    if not traj_files:
        traj_files = sorted(run_dir.glob("*_traj.csv"))
    best, best_n = None, 0
    for traj in traj_files:
        try:
            with traj.open(newline="", encoding="utf-8") as f:
                n = sum(1 for _ in csv.DictReader(f))
        except OSError:
            continue
        if algorithm and _traj_majority_algorithm(traj) != algorithm:
            continue
        if best is None or n > best_n:
            best, best_n = traj, n
    return best, best_n


def count_trajectory_trials(run_dir: Path, variant_name: str, algorithm: str | None = None) -> int:
    return _best_traj_in_run(run_dir, variant_name, algorithm=algorithm)[1]


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
    dataset_output_dir: Path, variant_name: str, n_trials: int, algorithm: str | None = None,
    *, run_context=None, durable=False, require_success=False,
) -> bool:
    if run_context is not None:
        if (run_context.get("variant") != variant_name or not run_context.get("identity")
                or run_context.get("output_dir") != str(Path(dataset_output_dir).resolve())
                or run_context.get("algorithm") != algorithm):
            return False
        path = Path(run_context["session_dir"]) / "progress.json"
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        matched = (state.get("identity") == run_context["identity"]
                   and state.get("attempts") == n_trials and state.get("pending") is False)
        return matched and (not require_success or (state.get("clean_complete") is True
                                                    and state.get("successful") == n_trials))
    # A count alone cannot establish which model, seed, or protocol produced it.
    if durable or os.environ.get("ML_SESSION_DIR") or optimizer_base_name(variant_name) in (
            "fsbo", "rgpe", "optformer", "bora", "tosfit", "bolt_ml", "semanticopt"):
        return False
    return count_variant_trials(dataset_output_dir, variant_name, algorithm=algorithm) >= n_trials


def durable_session_directory(args, variant_name):
    root = getattr(args, "session_dir", None) or getattr(args, "_session_env_root", None)
    transfer = optimizer_base_name(variant_name) in ("fsbo", "rgpe", "optformer", "bora", "tosfit", "bolt_ml", "semanticopt")
    if not root and not getattr(args, "runtime_policy_details", None) and not transfer:
        return None
    _, output_name, _ = resolve_variant(variant_name)
    if root:
        root = Path(root).resolve()
        if getattr(args, "_session_scoped", False):
            algo = args.algorithm[0] if len(args.algorithm) == 1 else "multi"
            return root / args.task_type / args.dataset / algo / output_name / f"seed-{args.seed}"
        return root
    return Path(args.output_dir).resolve() / output_name / "session"


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
    if getattr(args, "runtime_policy_details", None):
        payload["runtime_policy"] = args.runtime_policy_details
    if completed_variants is not None:
        payload["completed_variants"] = completed_variants
    progress_by_optimizer = {row["optimizer"]: row["progress"] for row in results if "progress" in row}
    if progress_by_optimizer:
        payload["progress_by_optimizer"] = progress_by_optimizer
        payload["completed_variants"] = [name for name in (completed_variants or [])
                                         if name not in progress_by_optimizer
                                         or progress_by_optimizer[name].get("clean_complete")]
        if len(results) == 1:
            payload["progress"] = results[0]["progress"]
            payload["run_id"] = results[0]["run_id"]
    with (dataset_output_dir / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.NamedTemporaryFile(mode="w", dir=dataset_output_dir, delete=False) as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
            temporary = f.name
        os.replace(temporary, summary_path)
    return summary_path


def dataset_is_fully_complete(
    dataset_output_dir: Path, optimizers, n_trials: int, algorithm: str | None = None, *, durable=False
) -> bool:
    return all(
        variant_is_complete(dataset_output_dir, name, n_trials, algorithm=algorithm, durable=durable)
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


def find_run_traj_file(run_dir: Path, variant_name: str, algorithm: str | None = None):
    return _best_traj_in_run(run_dir, variant_name, algorithm=algorithm)[0]


def try_resume_inner_optimizer(opt, args, variant_name, algorithms, prev_run):
    """Replay the latest incomplete traj into the live optimizer. Returns restored trial count."""
    inner = getattr(opt, "optimizer", None)
    if prev_run is None or inner is None or not hasattr(inner, "restore_from_traj"):
        return 0
    algo = algorithms[0] if len(algorithms) == 1 else None
    # Same algorithm filter the completion count uses, so the traj we restore
    # from is the traj that was counted as done.
    traj = find_run_traj_file(prev_run, variant_name, algorithm=algo)
    if traj is None:
        return 0
    if optimizer_base_name(variant_name) in ("fsbo", "rgpe", "optformer"):
        raise RuntimeError("Transfer methods require a durable ML session checkpoint; CSV-only resume is unsupported")
    if os.environ.get("ML_ISOLATED_EVALUATION") == "1":
        from mindware.components.optimizers.resume_traj import study_pickle_path, try_load_study_pickle

        study = try_load_study_pickle(study_pickle_path(str(traj)))
        if study is None:
            raise RuntimeError(f"Cannot resume without a readable sampler checkpoint: {traj}")
        trials = {trial.number: trial for trial in study.trials}
        with traj.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            trial = trials.get(int(row["iter_idx"]))
            perf = float(row["perf"])
            if trial is None or not trial.state.is_finished():
                raise RuntimeError(f"Sampler checkpoint does not match trajectory: {traj}")
            if math.isfinite(perf) and (
                trial.value is None or not math.isclose(trial.value, -perf, rel_tol=1e-12)
            ):
                raise RuntimeError(f"Sampler checkpoint has a different objective: {traj}")
            if not math.isfinite(perf) and trial.value is not None:
                raise RuntimeError(f"Sampler checkpoint has a different failure state: {traj}")
    history_json = None
    if algo:
        history_json = str(
            history_scope_dir(
                Path(args.history_dir).resolve(),
                args.task_type,
                args.metric,
                args.dataset,
                variant_name,
            )
            / f"{algo}.json"
        )
    print(f"[Resume] restoring {variant_name} from {prev_run.name} traj={traj.name}")
    return int(inner.restore_from_traj(str(traj), history_json=history_json) or 0)


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
    if (optimizer_params or {}).get("experience_retrieve_ranker") == "source_gp":
        from mindware.components.experience.retrieve import ExperienceBank
        from mindware.components.experience.source_gp import attach_source_gp
        bank_path = optimizer_params["experience_bank_path"]
        checked = attach_source_gp(ExperienceBank.load(Path(bank_path)), bank_path,
                                   optimizer_params.get("source_gp_cache"))
        checked.source_gp.preflight(algorithms[0], optimizer_params["task_type"])
        optimizer_params["source_gp_manifest_sha256"] = checked.source_gp.manifest_sha256
    client = (optimizer_params or {}).get("llm_client")
    if client is not None and getattr(client, "adapter_path", None):
        # Fail the task before any trials if the requested adapter cannot be loaded.
        client._ensure_loaded()
        optimizer_params["adapter_verification"] = client.adapter_verification

    KnowledgeDatabase._instance = None

    algo_tag = algorithms[0] if len(algorithms) == 1 else "multi"
    task_id = f"{args.dataset}_{algo_tag}_{output_name}"
    session_dir = durable_session_directory(args, variant_name)
    if session_dir is not None:
        os.environ["ML_SESSION_DIR"] = str(session_dir)
        policy = getattr(args, "runtime_policy_details", None)
        identity = {"task": task_id, "seed": args.seed, "metric": args.metric,
                    "policy": policy["sha256"] if policy else {
                        "attempts": args.n_trials, "overall_seconds": args.time_limit,
                        "per_evaluation_seconds": args.per_run_time_limit},
                    "campaign": str(Path(args.output_dir).resolve())}
        if (optimizer_params or {}).get("source_gp_manifest_sha256"):
            identity["source_gp_manifest_sha256"] = optimizer_params["source_gp_manifest_sha256"]
        if (optimizer_params or {}).get("preset_warm_start_sha256"):
            identity["preset_warm_start_sha256"] = optimizer_params["preset_warm_start_sha256"]
        if (optimizer_params or {}).get("adapter_verification"):
            identity["adapter_sha256"] = optimizer_params["adapter_verification"]["sha256"]
        os.environ["ML_RUN_ID"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        previous_policy = policy.get("resume_from_policy_sha256") if policy else None
        if previous_policy:
            previous = dict(identity, policy=previous_policy)
            os.environ["ML_PREVIOUS_RUN_ID"] = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
        else:
            os.environ.pop("ML_PREVIOUS_RUN_ID", None)
    else:
        os.environ.pop("ML_SESSION_DIR", None)
        os.environ.pop("ML_RUN_ID", None)
        os.environ.pop("ML_PREVIOUS_RUN_ID", None)
    print(f"\n[CASH] variant={variant_name}, dispatch={dispatch}, task_id={task_id}")

    prev_run = None
    if args.resume:
        prev_run = find_latest_run_dir(
            Path(args.output_dir), variant_name, algorithm=algorithms[0] if len(algorithms) == 1 else None
        )

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

    durable = session_dir is not None
    if durable:
        from mindware.components.optimizers.ml_checkpoint import attach
        if dispatch in ("fsbo", "rgpe", "ptllm_rgpe", "optformer", "bora", "tosfit", "bolt_ml", "semanticopt"):
            identity = {"run": os.environ["ML_RUN_ID"], "assets": opt.optimizer.asset_identity,
                        "schema": opt.optimizer.schema_fingerprint,
                        "settings": {k: v for k, v in optimizer_params.items() if k != "llm_client"}}
            os.environ["ML_RUN_ID"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        checkpoint = Path(os.environ["ML_SESSION_DIR"]) / "checkpoint.pkl"
        if checkpoint.exists() and not args.resume:
            raise RuntimeError("An existing durable run requires --resume")
        restored = attach(opt.optimizer)
        contexts = getattr(args, "_run_contexts", {})
        contexts[variant_name] = {"variant": variant_name, "session_dir": str(session_dir),
                                  "identity": os.environ["ML_RUN_ID"],
                                  "output_dir": str(Path(args.output_dir).resolve()),
                                  "algorithm": algorithms[0] if len(algorithms) == 1 else None}
        args._run_contexts = contexts
        opt.output_dir = opt.optimizer.output_dir
        opt.datetime = opt.optimizer.topk_saver.identifier
        opt.amount_of_resource = max(0, int(args.n_trials) - restored)
        print(f"[Durable resume] {restored}/{args.n_trials} committed evaluations")

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
    if getattr(args, "runtime_policy_details", None):
        conf["runtime_policy"] = args.runtime_policy_details
        with (Path(opt.output_dir) / "config.json").open("w") as stream:
            json.dump(conf, stream, indent=2)
    cat_hps = categorical_hyperparameters(opt.cs)
    print(f"[Search Space] {len(opt.cs.get_hyperparameters())} hyperparameters")
    print(f"[Search Space] categorical hps: {cat_hps}")

    if args.resume and not durable:
        restored = try_resume_inner_optimizer(opt, args, variant_name, algorithms, prev_run)
        if restored > 0:
            remaining = max(0, int(args.n_trials) - restored)
            opt.amount_of_resource = remaining
            print(
                f"[Resume] {variant_name}: restored {restored}/{args.n_trials} trials; "
                f"running remaining {remaining}"
            )
            if remaining == 0:
                print(f"[Resume] {variant_name}: nothing remaining, skipping CASH.run")
                # Fall through to save_histories with the restored incumbent.

    if opt.amount_of_resource > 0:
        if dispatch in ("bora", "tosfit", "bolt_ml", "semanticopt") and args.skip_test_eval:
            # Validation-only runs must not add an unjournaled fit after their quota.
            best_perf, stats_path = opt.run(refit=args.refit, refit_final=False)
        else:
            best_perf, stats_path = opt.run(refit=args.refit)
    else:
        best_perf = float(getattr(opt.optimizer, "incumbent_perf", float("nan")))
        opt.incumbent_perf = best_perf
        opt.incumbent = opt.optimizer.incumbent_config
        saved_stats = Path(opt.output_dir) / f"{opt.datetime}_topk_config.pkl"
        stats_path = str(saved_stats) if saved_stats.exists() else None
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
    if dispatch in ("bora", "tosfit", "bolt_ml", "semanticopt") and args.skip_test_eval:
        extra_info["fitted_model_saved"] = False
        model_info = opt.get_model_info(save=True, extra_info=extra_info, include_model_path=False)
    else:
        model_info = opt.get_model_info(save=True, extra_info=extra_info)

    result = {
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
    if durable:
        result.update(session_dir=str(session_dir), run_id=os.environ["ML_RUN_ID"],
                      progress=json.loads((session_dir / "progress.json").read_text()))
    return result


def run_one_dataset(args):
    optimizers = expand_optimizer_variants(parse_optimizers(args.optimizer, args.llm_backends), args)
    algorithms = list(args.algorithm)
    validate_optimizer_request(optimizers, algorithms)
    configure_llm_runtime(args, optimizers)
    args._run_contexts = {}
    args._session_scoped = getattr(args, "_session_scoped", False) or len(optimizers) > 1

    dataset_output_dir = Path(args.output_dir).resolve()
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(dataset_output_dir)

    existing_by_optimizer, _ = ({}, None)
    if args.resume:
        existing_by_optimizer, _ = load_existing_summary(dataset_output_dir)

    algo_filter = algorithms[0] if len(algorithms) == 1 else None

    def complete(name, require_success=False):
        return variant_is_complete(
            dataset_output_dir, name, args.n_trials, algorithm=algo_filter,
            run_context=args._run_contexts.get(name),
            durable=durable_session_directory(args, name) is not None,
            require_success=require_success)

    print("[Setup]")
    print(f"  task_type={args.task_type}")
    print(f"  dataset={args.dataset}")
    print(f"  optimizers={optimizers}")
    print(f"  algorithms={algorithms}")
    print(f"  n_trials={args.n_trials}, seed={args.seed}")
    if args.resume:
        print("  resume=True (skip completed; continue incomplete from last traj trial)")
        if algo_filter:
            print(f"  resume_algorithm={algo_filter} (only matching trajs count as complete)")

    # Reporting only -- `need_data` below and the run loop each re-check
    # completion themselves. `pending` used to be built here and never read,
    # with both branches appending, which made the "skip" line look like it
    # controlled scheduling when it did not.
    for variant_name in optimizers:
        if args.resume and complete(variant_name):
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

    need_data = [
        v
        for v in optimizers
        if not (
            args.resume
            and complete(v)
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
        if args.resume and complete(variant_name):
            if variant_name in existing_by_optimizer:
                results.append(existing_by_optimizer[variant_name])
            continue

        try:
            rounds = 0
            max_rounds = args.max_rounds
            while not complete(variant_name):
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
                if complete(variant_name):
                    break
            if variant_name in existing_by_optimizer:
                results.append(existing_by_optimizer[variant_name])
            completed = [
                name
                for name in optimizers
                if complete(name, require_success=True)
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
            if not complete(variant_name):
                raise RuntimeError(f"{variant_name} stopped before reaching {args.n_trials} trials")
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
        completed_variants=[
            name for name in optimizers
            if complete(name, require_success=True)
        ],
    )
    print(f"[Summary] saved {summary_path}")
    return summary_path, all_results


def update_summary_all(output_root, dataset_name, summary_path):
    summary_all_path = Path(output_root) / "summary_all.json"
    with (Path(output_root) / "summary_all.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if summary_all_path.is_file():
            with summary_all_path.open(encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = {"datasets": {}}
        payload["datasets"][dataset_name] = str(summary_path)
        with tempfile.NamedTemporaryFile(mode="w", dir=output_root, delete=False) as f:
            json.dump(payload, f, indent=2)
            temporary = f.name
        os.replace(temporary, summary_all_path)


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
    parser.add_argument("--max-rounds", type=int, default=12,
                        help="Maximum execution rounds per optimizer; use 1 for managed queues.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metric", default="bal_acc")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--valid-size", type=float, default=0.35)
    parser.add_argument("--evaluation", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--time-limit", type=optional_seconds, default=600,
                        help="Overall optimization seconds, or 'none' to stop by trial count.")
    parser.add_argument("--runtime-policy", type=Path, default=None,
                        help="ML policy JSON; overrides trial and time budgets and enforces evaluator isolation.")
    parser.add_argument("--session-dir", type=Path, default=None,
                        help="Durable directory; multi-task/method invocations create separate task/method/seed subdirectories.")
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
    parser.add_argument(
        "--transfer-dump-dir",
        default=str(REPO_ROOT / "benchmark_data" / "ml_train_80_transfer_v2"),
        help="Per-group JSON dumps of train-80 classical BO (FSBO/RGPE sources).",
    )
    parser.add_argument(
        "--fsbo-ckpt-dir",
        default=str(REPO_ROOT / "benchmark_data" / "ml_train_80_fsbo_ckpts"),
        help="Root of per-group FSBO meta-train checkpoints (group/weights).",
    )
    parser.add_argument(
        "--transfer-n-pending",
        type=int,
        default=256,
        help="FSBO/RGPE discrete candidate pool size per suggest.",
    )
    parser.add_argument("--rgpe-cache-dir", default=str(REPO_ROOT / "benchmark_data/ml_train_80_rgpe_sources"))
    parser.add_argument("--optformer-model-dir")
    parser.add_argument("--transfer-allow-cuda", action="store_true")
    parser.add_argument("--transfer-worker-timeout", type=float, default=300)
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
    class ExplicitTokenLimit(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            namespace.llm_max_tokens_explicit = True

    parser.set_defaults(llm_max_tokens_explicit=False)
    parser.add_argument("--llm-max-tokens", type=int, default=2048, action=ExplicitTokenLimit,
                        help="Output tokens (default: 4096 for BORA, 2048 for other methods)")
    parser.add_argument(
        "--llm-response-format",
        default="rationale_json",
        choices=["rationale_json", "json_only"],
        help="Match SFT target: json_only for S0/S1a; rationale_json for S1b/gpt (default).",
    )
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--extractor-max-tokens", type=int, help="GPT-4o-mini tokens per extraction call")
    parser.add_argument("--extractor-timeout", type=float, help="Seconds per extraction call; no automatic retries")
    parser.add_argument("--extractor-max-calls", type=int, help="Total GPT extraction calls per target run")
    parser.add_argument("--recommendation-fallback", choices=("reject", "default"), default="reject",
                        help="BORA/ToSFiT: evaluate the ConfigSpace default after unrecoverable extraction")
    parser.add_argument("--bora-gpu-memory-gib", type=float, help="BORA PyTorch allocation cap on its single visible GPU")
    parser.add_argument("--tosfit-device", help="Explicit policy device, e.g. cuda:0")
    parser.add_argument("--tosfit-reference-device", help="Explicit frozen reference device, e.g. cuda:1")
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
    parser.add_argument("--source-gp-cache", default=None,
                        help="Verified training-source GP artifact directory for source_gp retrieval.")
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


HYBRID_OPTIMIZERS = ("ptllm_tpe", "ptllm_rgpe", "ptllm_smac", "ptllm_cmaes", "ptllm_hebo")


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

    apply_policy(args)
    args._session_env_root = os.environ.get("ML_SESSION_DIR")

    session = args.session_dir or args._session_env_root
    if session:
        packed, _ = run_archive.locate(Path(session) / "progress.json")
        if packed is not None and packed.exists():
            saved = run_archive.manifest(packed)
            if not args.resume:
                raise RuntimeError("This session is archived and immutable; use --resume to skip it or a new session for a new run")
            config = run_archive.load_json(Path(saved['progress']['trajectory']).parent / 'config.json')
            requested = [resolve_variant(name)[0] for name in parse_optimizers(args.optimizer, args.llm_backends)]
            if requested != [config.get('optimizer')]:
                raise RuntimeError("The archived run does not match the requested optimizer")
            if (saved['seed'] != args.seed or saved['progress']['attempts'] != args.n_trials
                    or saved['task']['dataset'] != args.dataset
                    or saved['task']['task_type'] != args.task_type
                    or [saved['task']['algorithm']] != list(args.algorithm)):
                raise RuntimeError("The archived run does not match the requested task, seed or budget")
            print(f"[Resume] archived run {saved['run_id']} is complete; no evaluations started: {packed}")
            return

    os.chdir(REPO_ROOT)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    optimizers = expand_optimizer_variants(parse_optimizers(args.optimizer, args.llm_backends), args)
    jobs = resolve_dataset_jobs(args)
    args._session_scoped = len(jobs) > 1 or len(optimizers) > 1
    for task_type, dataset_name in jobs:
        args.task_type = task_type
        args.metric = default_metric_for_task(task_type) if args.v2_datasets else (
            default_metric_for_task(task_type) if task_type == "rgs" and args.metric == "bal_acc" else args.metric
        )
        algo_filter = args.algorithm[0] if len(args.algorithm) == 1 else None
        # Isolate single-algorithm jobs so a finished (dataset, algo) slot cannot
        # skip sibling algorithms on the same dataset via shared summary.json.
        if algo_filter:
            dataset_dir = output_root / task_type / dataset_name / algo_filter
        else:
            dataset_dir = output_root / task_type / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(dataset_dir)
        args.dataset = dataset_name
        summary_path = dataset_dir / "summary.json"
        slot_key = f"{task_type}/{dataset_name}" + (f"/{algo_filter}" if algo_filter else "")

        durable = any(durable_session_directory(args, name) is not None for name in optimizers)
        slot_complete = dataset_is_fully_complete(
            dataset_dir, optimizers, args.n_trials, algorithm=algo_filter, durable=durable
        )
        if slot_complete and summary_path.is_file():
            print(f"[Resume] already complete: {slot_key}")
            update_summary_all(output_root, slot_key, summary_path)
            continue
        if slot_complete:
            print(
                f"[Resume] trajectories complete but summary missing: {slot_key} — rebuilding"
            )
        elif summary_path.is_file() and not algo_filter and not durable:
            print(f"[Skip] already complete: {slot_key}")
            update_summary_all(output_root, slot_key, summary_path)
            continue

        last_exc = None
        for attempt in range(1, max(1, int(args.dataset_retries)) + 1):
            try:
                summary_path, _ = run_one_dataset(args)
                update_summary_all(output_root, slot_key, summary_path)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if is_retryable_dataset_error(exc) and attempt < int(args.dataset_retries):
                    wait_s = 30 * attempt
                    print(
                        f"[Retry] {slot_key} attempt {attempt}/"
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
                                "algorithm": algo_filter,
                                "error": str(exc),
                                "resume": bool(args.resume),
                                "attempt": attempt,
                            }
                        )
                        + "\n"
                    )
                print(f"[Error] {slot_key}: {exc}")
                traceback.print_exc()
                if not (args.all_datasets or args.v2_datasets):
                    raise
                break
        if last_exc is not None and not (args.all_datasets or args.v2_datasets):
            raise last_exc


if __name__ == "__main__":
    main()
