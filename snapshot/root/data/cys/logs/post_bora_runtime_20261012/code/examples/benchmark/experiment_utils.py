"""Load prompt/hybrid variant definitions and build optimizer output names."""

from __future__ import annotations

import csv
import json
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent


def _load_manifest(name: str) -> dict:
    path = BENCHMARK_DIR / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_prompt_variants() -> dict[str, dict]:
    payload = _load_manifest("prompt_variants.json")
    return {row["id"]: row for row in payload["variants"]}


def load_hybrid_variants() -> dict[str, dict]:
    payload = _load_manifest("hybrid_variants.json")
    return {row["id"]: row for row in payload["variants"]}


def load_datasets_v2() -> dict:
    return _load_manifest("datasets_v2.json")


EVAL8_DATASETS = (
    "yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol"
)


def score_higher_is_better(_task_type: str | None = None) -> bool:
    """Whether larger stored scores are better (cls accuracy; rgs negated holdout MSE)."""
    return True


def traj_label_for_variant(variant_dir_name: str, traj_labels: dict[str, str] | None = None) -> str:
    base = variant_dir_name.split("__", 1)[0]
    if traj_labels:
        return traj_labels.get(base, base)
    # Default dispatch labels for Qwen LoRA variant dirs (traj CSV uses optimizer dispatch name).
    if base.startswith("ptllm_tpe_qwen"):
        return "ptllm_tpe"
    if base.startswith("ptllm_rgpe_qwen"):
        return "ptllm_rgpe"
    if base.startswith("ptllm_hebo_qwen"):
        return "ptllm_hebo"
    if base.startswith("ptllm_qwen"):
        return "ptllm"
    return base


def best_run_dir(
    variant_dir: Path,
    *,
    traj_label: str | None = None,
    traj_labels: dict[str, str] | None = None,
) -> Path | None:
    """Pick the CASH run with the best validation score (tie-break: more trials)."""
    if not variant_dir.is_dir():
        return None
    label = traj_label or traj_label_for_variant(variant_dir.name, traj_labels)
    best_dir = None
    best_score = float("-inf")
    best_n = -1
    for run_dir in sorted(p for p in variant_dir.glob("CASH-*") if p.is_dir()):
        traj_files = sorted(run_dir.glob(f"*_opt_{label}_traj.csv"))
        if not traj_files:
            continue
        with traj_files[-1].open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        scores = []
        for row in rows:
            raw = row.get("best_so_far")
            if raw in (None, ""):
                continue
            try:
                scores.append(float(raw))
            except ValueError:
                continue
        if not scores:
            continue
        score = max(scores)
        n = len(rows)
        if score > best_score or (score == best_score and n > best_n):
            best_dir, best_score, best_n = run_dir, score, n
    return best_dir


def jobs_from_dataset_names(names_csv: str) -> list[tuple[str, str]]:
    """Resolve (task_type, dataset) from comma-separated names (cls or rgs bundle)."""
    import pandas as pd

    repo = BENCHMARK_DIR.parents[1]
    cls_names = set(
        pd.read_csv(repo / "datasets_bundle" / "dataset_info" / "classification.csv")["name"]
        .astype(str)
    )
    rgs_names = set(
        pd.read_csv(repo / "datasets_bundle" / "dataset_info" / "regression.csv")["name"]
        .astype(str)
    )
    jobs = []
    for raw in names_csv.split(","):
        name = raw.strip()
        if not name:
            continue
        if name in cls_names:
            jobs.append(("cls", name))
        elif name in rgs_names:
            jobs.append(("rgs", name))
        else:
            raise ValueError(f"Unknown dataset name: {name!r}")
    return jobs


def list_v2_dataset_jobs(task_filter: str = "all") -> list[tuple[str, str]]:
    """Return (task_type, dataset) jobs from datasets_v2.json.

    task_filter: all | pilot | remaining
    """
    manifest = load_datasets_v2()
    cls_all = list(manifest["classification"]["datasets"])
    rgs_all = list(manifest["regression"]["datasets"])
    pilot = manifest.get("pilot", {})
    cls_pilot = set(pilot.get("classification", []))
    rgs_pilot = set(pilot.get("regression", []))

    jobs = []
    for task_type, all_names, pilot_names in [
        ("cls", cls_all, cls_pilot),
        ("rgs", rgs_all, rgs_pilot),
    ]:
        if task_filter == "pilot":
            names = [n for n in all_names if n in pilot_names]
        elif task_filter == "remaining":
            names = [n for n in all_names if n not in pilot_names]
        else:
            names = all_names
        jobs.extend((task_type, name) for name in names)
    return jobs


def optimizer_base_name(variant_name: str) -> str:
    return variant_name.split("__", 1)[0]


def parse_variant_tags(variant_name: str) -> tuple[str, str | None, str | None]:
    parts = variant_name.split("__")
    base = parts[0]
    prompt_id = None
    hybrid_id = None
    try:
        known_prompts = set(load_prompt_variants())
    except Exception:
        known_prompts = set()
    try:
        known_hybrids = set(load_hybrid_variants())
    except Exception:
        known_hybrids = set()
    for part in parts[1:]:
        if part in known_prompts or (part.startswith("P") and part[1:].isdigit()):
            prompt_id = part
        elif part in known_hybrids or (part.startswith("H") and part[1:].isdigit()):
            hybrid_id = part
    return base, prompt_id, hybrid_id


def make_llm_variant_name(
    base: str,
    prompt_id: str | None,
    hybrid_id: str | None = None,
    suffix: str | None = None,
) -> str:
    parts = [base]
    if prompt_id:
        parts.append(prompt_id)
    if hybrid_id:
        parts.append(hybrid_id)
    name = "__".join(parts)
    if suffix:
        name += suffix if suffix.startswith("__") else f"__{suffix}"
    return name


def _resolve_prompt_variant(row: dict, variants: dict[str, dict]) -> dict:
    """Merge extends chain (e.g. P17 extends P5; P26 extends P22) into a flat settings dict."""
    merged = dict(row)
    extends = merged.pop("extends", None)
    accumulated_blocks = dict(merged.get("prompt_blocks") or {})
    if extends:
        if extends not in variants:
            raise ValueError(f"Unknown extends base {extends!r} for {row.get('id')}")
        base_row = _resolve_prompt_variant(variants[extends], variants)
        accumulated_blocks = {**dict(base_row.get("prompt_blocks") or {}), **accumulated_blocks}
        base = {
            k: v
            for k, v in base_row.items()
            if k not in ("id", "label", "notes", "extends", "optimizer", "prompt_blocks")
        }
        merged = {**base, **merged}
    if accumulated_blocks:
        merged["prompt_blocks"] = accumulated_blocks
    return merged


def apply_prompt_settings(args, prompt_id: str | None) -> None:
    variants = load_prompt_variants()
    if prompt_id is None:
        prompt_id = getattr(args, "prompt_variant", None) or "P0"
    if prompt_id not in variants:
        raise ValueError(f"Unknown prompt variant {prompt_id!r}")
    row = _resolve_prompt_variant(variants[prompt_id], variants)
    args.prompt_variant = prompt_id
    args.llm_top_k = int(row["top_k"])
    args.llm_worst_k = int(row["worst_k"])
    args.llm_recent_k = int(row["recent_k"])
    args.llm_history_window = int(row.get("history_window", args.llm_history_window))
    args.llm_run_summary = bool(row.get("run_summary", False))
    args.llm_tcs_analysis = bool(row.get("tcs_analysis", False))
    args.llm_tcs_single_agent = bool(row.get("tcs_single_agent", False))
    blocks = dict(row.get("prompt_blocks") or {})
    args.llm_prompt_blocks = blocks
    args.prompt_optimizer = row.get("optimizer")
    args.llm_experience_self_mode = str(row.get("experience_self_mode") or "cohort")
    args.llm_experience_self_start_trial = int(row.get("experience_self_start_trial") or 1)
    args.llm_experience_explore_mode = str(row.get("experience_explore_mode") or "off")
    args.llm_experience_explore_prob = float(row.get("experience_explore_prob") or 0.5)
    args.llm_experience_blame = dict(row.get("experience_blame") or {})
    args.llm_experience_retrieve_k = int(row.get("experience_retrieve_k") or 3)
    args.llm_experience_retrieve_start_trial = int(row.get("experience_retrieve_start_trial") or 1)
    args.llm_experience_retrieve_rerank_trial = int(row.get("experience_retrieve_rerank_trial") or 8)
    args.llm_experience_bank_path = str(row.get("experience_bank_path") or "") or None
    args.llm_experience_retrieve_ranker = str(row.get("experience_retrieve_ranker") or "surrogate")
    args.llm_experience_retrieve_fade = bool(row.get("experience_retrieve_fade", False))
    stop = row.get("experience_retrieve_stop_trial")
    args.llm_experience_retrieve_stop_trial = None if stop in (None, "", 0, "0") else int(stop)
    floor = row.get("experience_retrieve_spearman_floor")
    args.llm_experience_retrieve_spearman_floor = None if floor in (None, "") else float(floor)
    args.llm_experience_retrieve_include_top_bottom = bool(
        row.get("experience_retrieve_include_top_bottom", False)
    )
    args.llm_experience_retrieve_include_prefer = bool(
        row.get("experience_retrieve_include_prefer", True)
    )
    args.llm_experience_retrieve_region_style = str(
        row.get("experience_retrieve_region_style") or "prefer"
    )
    args.llm_experience_retrieve_tb_layout = str(
        row.get("experience_retrieve_tb_layout") or "flat"
    )
    _tk = row.get("experience_retrieve_top_k")
    args.llm_experience_retrieve_top_k = 1 if _tk in (None, "") else max(0, int(_tk))
    _bk = row.get("experience_retrieve_bottom_k")
    args.llm_experience_retrieve_bottom_k = 1 if _bk in (None, "") else max(0, int(_bk))
    if bool(row.get("experience_retrieve_until_self", False)) and args.llm_experience_retrieve_stop_trial is None:
        args.llm_experience_retrieve_stop_trial = int(row.get("experience_self_start_trial") or 50)
    args.llm_experience_self_gate = str(row.get("experience_self_gate") or "clock")
    tpe_min = row.get("experience_explore_tpe_min_trial")
    args.llm_experience_explore_tpe_min_trial = 0 if tpe_min in (None, "", 0, "0") else int(tpe_min)


def apply_hybrid_settings(args, hybrid_id: str | None) -> None:
    variants = load_hybrid_variants()
    if hybrid_id is None:
        hybrid_id = getattr(args, "hybrid_variant", None) or "H0"
    if hybrid_id not in variants:
        raise ValueError(f"Unknown hybrid variant {hybrid_id!r}")
    row = variants[hybrid_id]
    args.hybrid_variant = hybrid_id
    args.llm_tpe_llm_prob = float(row["llm_prob"])
    args.llm_tpe_phase_trials = int(row.get("llm_phase_trials", 0))
    args.hybrid_selector = str(row.get("hybrid_selector") or "bernoulli")
    args.ucb_c = float(row.get("ucb_c", 1.4))
    args.pucb_c2 = float(row.get("pucb_c2", 1.4))
    args.pucb_p1 = float(row.get("pucb_p1", 0.9))
    args.pucb_p_end = None if row.get("pucb_p_end") is None else float(row.get("pucb_p_end"))
    args.pucb_floor = None if row.get("pucb_floor") is None else float(row.get("pucb_floor"))
    args.pucb_floor_after_frac = (
        None if row.get("pucb_floor_after_frac") is None else float(row.get("pucb_floor_after_frac"))
    )
    args.pucb_switch_after_frac = (
        None if row.get("pucb_switch_after_frac") is None else float(row.get("pucb_switch_after_frac"))
    )
    args.pucb_q = str(row.get("pucb_q") or "window_minmax")
    args.pucb_window = int(row.get("pucb_window", 10))
    args.dts_gamma = float(row.get("dts_gamma", 0.95))
    args.ap_beta = float(row.get("ap_beta", 0.3))
    args.ap_pmin = float(row.get("ap_pmin", 0.2))
    args.llm_duplicate_policy = str(row.get("llm_duplicate_policy", "legacy"))
    for key, attr in (("n_initial_samples", "llm_n_initial_samples"),
                      ("history_window", "llm_history_window"), ("recent_k", "llm_recent_k"),
                      ("top_k", "llm_top_k"), ("worst_k", "llm_worst_k"),
                      ("response_format", "llm_response_format")):
        if key in row:
            setattr(args, attr, row[key])


def list_stage1_core_variants() -> list[str]:
    payload = _load_manifest("prompt_variants.json")
    return list(payload["recommended_batches"]["stage1_core"])


def list_stage2_hybrid_variants() -> list[str]:
    payload = _load_manifest("hybrid_variants.json")
    return list(payload["recommended_batches"]["stage2_all"])


# Measured Qwen3.5-9B completion lengths (assistant turn only):
#   json_only (s0 mltest20 eval): p99=228, max=228; old cap 512 caused leak truncation
#   rationale_json (s1b SFT targets): p99=310, max=352
LLM_MAX_TOKENS_JSON_ONLY = 768
LLM_MAX_TOKENS_RATIONALE_JSON = 768

LLM_MAX_TOKENS_BY_RESPONSE_FORMAT: dict[str, int] = {
    "json_only": LLM_MAX_TOKENS_JSON_ONLY,
    "rationale_json": LLM_MAX_TOKENS_RATIONALE_JSON,
}


def default_llm_max_tokens(response_format: str | None) -> int:
    """Return a safe max_new_tokens for P5 LoRA eval / RL rollouts."""
    fmt = (response_format or "rationale_json").strip().lower()
    if fmt in ("json", "json_only", "config_only"):
        return LLM_MAX_TOKENS_JSON_ONLY
    return LLM_MAX_TOKENS_RATIONALE_JSON
