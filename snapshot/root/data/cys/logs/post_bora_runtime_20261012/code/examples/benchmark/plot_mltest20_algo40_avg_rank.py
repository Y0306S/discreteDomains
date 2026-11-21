#!/usr/bin/env python3
"""Build avg_rank_table.csv for mltest20 algo40.

Default ranking set is tasks_30.json: 30 unique datasets (15 cls + 15 rgs),
one locked algorithm each. tasks_40.json (40 dataset×algorithm slots) is kept
for reference and repair jobs; pass --tasks-json to rebuild that table.

Includes classical, gpt-4o-mini baselines, SFT LoRA (s1a/s1b), and phase-2 LoRA (s2_A–H).
Task keys: {task_type}/{dataset}/{algorithm}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ALGO40 = REPO / "benchmark_data" / "real_hpo_v2_mltest20_algo40"

# Standard pooled leaderboard: drop worst 20% of trial scores per task before min-max.
PRIMARY_DROP_FRAC = 0.20

# s2_* eval arms (trained on p5_s2 DPO data) and their SFT (s1) checkpoint lineage.
S2_S1_BASE: dict[str, str] = {
    "A": "s1b_ep7p5",
    "B": "s1b_ep7p5",
    "C": "s1b_ep7p5",
    "D": "s1b_ep2p5",
    "E": "s1b_ep7p5",
    "F": "s1b_ep7p5",
    "G": "s1b_ep7p5",
    "H": "s1b_ep7p5",
}
# s1a lineage s2 (separate eval output dirs): C/G + conservative + micro (O3/D1m)
S1A_S2_LETTERS = ("C", "G", "D1", "D1m", "D2", "D3", "O1", "O3", "O4")
S1A_REPLICATION_MODELS = {
    "s1a_ep7p5_r2": ("s1a_ep7p5", "qwen LoRA s1a_ep7p5_r2 (P5/H0 tpe)"),
    "s1a_ep7p5_r3": ("s1a_ep7p5", "qwen LoRA s1a_ep7p5_r3 (P5/H0 tpe)"),
    # seed=2 replications of leaderboard arms (same adapters, fresh eval dirs)
    "dpo_G_r2": ("dpo_G", "qwen LoRA s1b_ep7p5_s2_G_r2 (P5/H0 tpe)"),
    "s1a_s2_D1_r2": ("s1a_s2_D1", "qwen LoRA s1a_ep7p5_s2_D1_r2 (P5/H0 tpe)"),
    "dpo_rationale_G_r2": ("dpo_rationale_G", "qwen LoRA s1b_ep7p5_dpo_rationale_G_r2 (P5/H0 tpe)"),
}
# REINFORCE light arms (continue s1a_ep7p5; separate eval dirs / adapters)
S1A_R0_MODELS = {
    "s1a_r0a": "qwen LoRA s1a_ep7p5_r0a T0.7 light16 (P5/H0 tpe)",
    "s1a_r0b": "qwen LoRA s1a_ep7p5_r0b T1.0 light16 (P5/H0 tpe)",
    "s1a_r0c": "qwen LoRA s1a_ep7p5_r0c T0.7 medium32 (P5/H0 tpe)",
    "s1a_r0d": "qwen LoRA s1a_ep7p5_r0d T1.0 medium32 (P5/H0 tpe)",
}
# BO-seeded / prefix RL ablations (continue s1a_ep7p5)
S1A_SETUP_AB_MODELS = {
    "s1a_a_boseed_global": "qwen LoRA s1a_ep7p5_a_boseed_global T1.0 light16 (P5/H0 tpe)",
    "s1a_a_boseed_dsseq": "qwen LoRA s1a_ep7p5_a_boseed_dsseq T1.0 light16 (P5/H0 tpe)",
    "s1a_b_prefix_grpo": "qwen LoRA s1a_ep7p5_b_prefix_grpo T1.0 light16 (P5/H0 tpe)",
    "s1a_b_prefix_dsseq": "qwen LoRA s1a_ep7p5_b_prefix_dsseq T1.0 light16 (P5/H0 tpe)",
    "s1a_b_prefix_dsseq_hg8": (
        "qwen LoRA s1a_ep7p5_b_prefix_dsseq_hg8 T1.0 light8 high-growth (P5/H0 tpe)"
    ),
    "s1a_b_prefix_dsseq_m32": (
        "qwen LoRA s1a_ep7p5_b_prefix_dsseq_m32 T1.0 medium32 (P5/H0 tpe)"
    ),
}


def discover_reward_sweep_models() -> dict[str, str]:
    """Auto-pick up s1a_b_rw_{arm}_ckpt{N} eval dirs under algo40 parent."""
    parent = ALGO40.parent
    out: dict[str, str] = {}
    for root in sorted(parent.glob("real_hpo_v2_mltest20_algo40_p5_s1a_b_rw_*")):
        model_id = root.name.replace("real_hpo_v2_mltest20_algo40_p5_", "", 1)
        # s1a_b_rw_hybrid_ckpt12 -> arm=hybrid, ckpt=12
        label = f"qwen LoRA {model_id} (P5/H0 tpe)"
        out[model_id] = label
    return out


def discover_s1b_rg_dbest_hp_models() -> dict[str, str]:
    """Auto-pick up s1b_rg_dbest_lr*_kl*_ckpt16 eval dirs (RL on dpo_rationale_G)."""
    parent = ALGO40.parent
    out: dict[str, str] = {}
    for root in sorted(parent.glob("real_hpo_v2_mltest20_algo40_p5_s1b_rg_dbest_*")):
        model_id = root.name.replace("real_hpo_v2_mltest20_algo40_p5_", "", 1)
        label = f"qwen LoRA {model_id} (P5/H0 tpe)"
        out[model_id] = label
    return out


def discover_s1b_rg_orpo_hp_models() -> dict[str, str]:
    """Auto-pick up s1b_rg_orpo_lr*_b* eval dirs (ORPO HP sweep on rationale data)."""
    parent = ALGO40.parent
    out: dict[str, str] = {}
    for root in sorted(parent.glob("real_hpo_v2_mltest20_algo40_p5_s1b_rg_orpo_*")):
        model_id = root.name.replace("real_hpo_v2_mltest20_algo40_p5_", "", 1)
        label = f"qwen LoRA {model_id} (P5/H0 tpe)"
        out[model_id] = label
    return out


def discover_s1b_s2g_orpo_hp_models() -> dict[str, str]:
    """Auto-pick up s1b_s2g_orpo_lr*_b* eval dirs (ORPO HP sweep on p5_s2_dpo / s2_G)."""
    parent = ALGO40.parent
    out: dict[str, str] = {}
    for root in sorted(parent.glob("real_hpo_v2_mltest20_algo40_p5_s1b_s2g_orpo_*")):
        model_id = root.name.replace("real_hpo_v2_mltest20_algo40_p5_", "", 1)
        label = f"qwen LoRA {model_id} (P5/H0 tpe)"
        out[model_id] = label
    return out


_RS_GPSIM_MESH_RE = re.compile(r"RS\d+f\d+")


def is_legacy_rs_retrieve_variant(variant: str) -> bool:
    """Drop pre-v3 RS/RC retrieve arms; keep RS*k*f* gpsim mesh (e.g. RS10f05)."""
    if _RS_GPSIM_MESH_RE.search(variant):
        return False
    legacy_markers = (
        "P5plusA5T50X50RS10gp",
        "P5plusA5T50X50RS10",
        "P5plusA5T50X50RSWarm",
        "P5plusA5T50X50RS__",
        "P5plusA5T50X50RC",
        "P5plusA5TRdyFanovaX50RSWarm",
        "P5plusA5TRdyFanovaX50RS",
    )
    return any(m in variant for m in legacy_markers)


def discover_p5plus_qwen_variants(root: Path) -> dict[str, tuple[Path, str]]:
    """Register Qwen attr_t2 P5+ arms from on-disk variant folder names."""
    out: dict[str, tuple[Path, str]] = {}
    if not root.is_dir():
        return out
    seen: set[str] = set()
    for tt in ("cls", "rgs"):
        tt_dir = root / tt
        if not tt_dir.is_dir():
            continue
        for ds_dir in sorted(tt_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for var_dir in sorted(ds_dir.iterdir()):
                disk = var_dir.name
                if not disk.startswith("ptllm_tpe_qwen__") or disk in seen:
                    continue
                seen.add(disk)
                label = disk.replace("ptllm_tpe_qwen__", "ptllm_tpe_qwen_s1b_attr_t2__", 1)
                out[label] = (root, disk)
            if seen:
                break
        if seen:
            break
    return out


def discover_s1b_attr_orpo_models() -> dict[str, str]:
    """Auto-pick up s1b_attr_t2/t3 attribution ORPO eval dirs (from s1b_ep7p5)."""
    parent = ALGO40.parent
    out: dict[str, str] = {}
    labels = {
        "s1b_attr_t2": "qwen LoRA s1b_attr_t2 partial-edit (P5/H0 tpe)",
        "s1b_attr_t3": "qwen LoRA s1b_attr_t3 per-HP soft (P5/H0 tpe)",
    }
    for root in sorted(parent.glob("real_hpo_v2_mltest20_algo40_p5_s1b_attr_t*")):
        model_id = root.name.replace("real_hpo_v2_mltest20_algo40_p5_", "", 1)
        if model_id.startswith("s1b_attr_t2_orpo_lr"):
            out[model_id] = f"qwen LoRA {model_id} T2+s2gHP (P5/H0 tpe)"
        else:
            out[model_id] = labels.get(model_id, f"qwen LoRA {model_id} (P5/H0 tpe)")
    return out


def discover_grpo_rsmix_p5plus_models() -> dict[str, tuple[str, str, str]]:
    """RS-mix GRPO adapters evaluated under version-matched P5+ RS3/RS5/RS10 prompts."""
    parent = ALGO40.parent
    out: dict[str, tuple[str, str, str]] = {}
    root_globs = (
        "real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_p5_",
        "real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_top3_strict4_p5_",
        "real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_top3_online_p5_",
        "real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v13_p5_",
        "real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12gap_p5_",
        "real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12gap_top3_strict4_p5_",
    )
    prompts_by_ver = {
        "v3": ("P5plusA5T50X50RS3f05", "P5plusA5T50X50RS5f05", "P5plusA5T50X50RS10f03"),
        "v4": ("P5plusA5T50X50RS3f05v4", "P5plusA5T50X50RS5f05v4", "P5plusA5T50X50RS10f05v4"),
        "v5": ("P5plusA5T50X50RS3f05v5", "P5plusA5T50X50RS5f05v5", "P5plusA5T50X50RS10f05v5"),
    }
    for prefix in root_globs:
        for root in sorted(parent.glob(f"{prefix}*")):
            if not root.is_dir():
                continue
            model_id = root.name.replace(prefix, "", 1)
            ver = "v3"
            for candidate in ("v5", "v4", "v3"):
                if f"_{candidate}_" in model_id:
                    ver = candidate
                    break
            for prompt in prompts_by_ver[ver]:
                disk = f"ptllm_tpe_qwen__{prompt}__HPUCB_arm"
                label = f"ptllm_tpe_qwen_{model_id}__{prompt}__HPUCB_arm"
                display = f"qwen LoRA {model_id} ({prompt} / HPUCB_arm tpe)"
                out[label] = (display, disk, str(root))
    return out


def s2_variant_id(letter: str, planner: str) -> str:
    """planner: smac | tpe"""
    s1 = S2_S1_BASE[letter]
    return f"ptllm_{planner}_qwen_{s1}_s2_{letter}__P5__H0"


def is_s2_variant(variant: str) -> bool:
    return "_s2_" in variant and "ptllm_" in variant and "_qwen_" in variant


def fnum(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_tasks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_key(t: dict) -> str:
    return f"{t['task_type']}/{t['dataset']}/{t['algorithm']}"


def find_traj(
    root: Path,
    tt: str,
    ds: str,
    variant: str,
    n_trials: int,
    algorithm: str | None = None,
) -> Path | None:
    dirs = []
    if algorithm:
        isolated = root / tt / ds / algorithm / variant
        if isolated.is_dir():
            dirs.append(isolated)
    legacy = root / tt / ds / variant
    if legacy.is_dir():
        dirs.append(legacy)
    if not dirs:
        return None
    complete: list[tuple[float, int, Path]] = []
    for d in dirs:
        for run_dir in sorted(p for p in d.glob("CASH-*") if p.is_dir()):
            trajs = sorted(run_dir.glob("*_traj.csv"))
            if not trajs:
                continue
            traj = trajs[-1]
            rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
            if len(rows) < n_trials:
                continue
            if algorithm:
                algos = [str(r.get("algorithm") or "").strip() for r in rows if str(r.get("algorithm") or "").strip()]
                if not algos:
                    continue
                from collections import Counter

                mode = Counter(algos).most_common(1)[0][0]
                if mode != algorithm:
                    continue
            bests = [fnum(r.get("best_so_far")) for r in rows]
            bests = [b for b in bests if not math.isnan(b)]
            if not bests:
                continue
            complete.append((max(bests), len(rows), traj))
    if not complete:
        return None
    complete.sort(key=lambda x: (x[0], x[1], str(x[2])))
    return complete[-1][2]


def final_best(traj: Path) -> float | None:
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    if not rows:
        return None
    bests = [fnum(r.get("best_so_far")) for r in rows]
    bests = [b for b in bests if not math.isnan(b) and b != float("-inf")]
    if not bests:
        return None
    return max(bests)


def average_rank(ordered: list[tuple[str, float]]) -> dict[str, float]:
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and abs(ordered[j + 1][1] - ordered[i][1]) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


def task_ranks(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return average_rank(ordered)


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if len(scores) < 2:
        return {k: 1.0 for k in scores}
    vals = list(scores.values())
    best, worst = max(vals), min(vals)
    return {k: 1.0 if best == worst else (v - worst) / (best - worst) for k, v in scores.items()}


def percentile(data: list[float], q: float) -> float:
    """Linear interpolation percentile, q in [0, 100]."""
    if not data:
        return float("nan")
    xs = sorted(data)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def trial_perfs(traj: Path, n_trials: int) -> list[float]:
    """Per-trial validation scores from traj (one value per HPO trial)."""
    rows = list(csv.DictReader(traj.open(newline="", encoding="utf-8")))
    out: list[float] = []
    for r in rows[:n_trials]:
        p = fnum(r.get("perf"))
        if math.isnan(p) or p == float("-inf"):
            continue
        out.append(p)
    return out


def normalize_from_pooled_drop_worst(
    scores: dict[str, float],
    pooled_trial_perfs: list[float],
    *,
    drop_worst_frac: float = PRIMARY_DROP_FRAC,
) -> dict[str, float]:
    """Higher-is-better norm from a **global** trial pool (cls or rgs).

    1. Pool all methods' trial ``perf`` values on this task.
    2. Drop the worst ``drop_worst_frac`` (lowest scores).
    3. Scale method scores with min(kept)→0, max(kept)→1 (p100 of kept).

    Method ``scores`` are typically each method's final best; only the
    reference scale uses the pooled drop.
    """
    if len(scores) < 2 or len(pooled_trial_perfs) < 2:
        return normalize(scores)
    xs = sorted(pooled_trial_perfs)
    n_drop = int(math.floor(drop_worst_frac * len(xs)))
    kept = xs[n_drop:] if n_drop > 0 else xs
    if not kept:
        return normalize(scores)
    p_lo, p_hi = kept[0], kept[-1]
    if p_hi <= p_lo:
        return normalize(scores)
    out: dict[str, float] = {}
    for k, v in scores.items():
        if v <= p_lo:
            out[k] = 0.0
        elif v >= p_hi:
            out[k] = 1.0
        else:
            out[k] = (v - p_lo) / (p_hi - p_lo)
    return out


# Back-compat alias
normalize_rgs_from_pooled_drop_worst = normalize_from_pooled_drop_worst


def display_name(variant: str) -> str:
    mapping = {
        "smac": "smac",
        "tpe": "tpe",
        "cmaes": "cmaes",
        "hebo": "hebo",
        "ptllm_smac__P5__H0": "gpt-4o-mini (P5/H0 smac)",
        "ptllm_tpe__P5__H0": "gpt-4o-mini (P5/H0 tpe)",
        "ptllm_tpe__P16__H2": "gpt-4o-mini (P16/H2 tpe)",
        "ptllm_tpe__P5plusA5T50X50__H2": "gpt-4o-mini (P5+ A5 T50 X50 / H2)",
        "ptllm_tpe__P5plusA5T50X50RC__H2": "gpt-4o-mini (P5+ A5 T50 X50 RC / H2)",
        "ptllm_tpe__P5plusA5T50X50RS__H2": "gpt-4o-mini (P5+ A5 T50 X50 RS / H2)",
        "ptllm_tpe__P5plusA5T50X50__H60": "gpt-4o-mini (P5+ A5 T50 X50 / H60)",
        "ptllm_tpe__P5plusA5T50X50__H0": "gpt-4o-mini (P5+ A5 T50 X50 / H0)",
        "ptllm_tpe__P5plusA5T50X50__H40": "gpt-4o-mini (P5+ A5 T50 X50 / H40)",
        "ptllm_tpe__P5plusA5T50X50__HUCB": "gpt-4o-mini (P5+ A5 T50 X50 / HUCB)",
        "ptllm_tpe__P5plusA5T50X50__HPUCB": "gpt-4o-mini (P5+ A5 T50 X50 / HPUCB)",
        "ptllm_tpe__P5plusA5T50X50__HPUCB_breakthrough": "gpt-4o-mini (P5+ A5 T50 X50 / HPUCB_bt)",
        "ptllm_tpe__P5plusA5T50X50__HPUCB_arm": "gpt-4o-mini (P5+ A5 T50 X50 / HPUCB_arm)",
        "ptllm_tpe__P5plusA5T50X50__HDTS": "gpt-4o-mini (P5+ A5 T50 X50 / HDTS)",
        "ptllm_tpe__P5plusA5T50X50__HAP": "gpt-4o-mini (P5+ A5 T50 X50 / HAP)",
        "ptllm_tpe__P5plusA5T50X50__HDUCB": "gpt-4o-mini (P5+ A5 T50 X50 / HDUCB)",
        "ptllm_tpe__P5plusA5T50X50__HDMAB": "gpt-4o-mini (P5+ A5 T50 X50 / HDMAB)",
        "ptllm_tpe__P5plusA5T50X50__HTRAN": "gpt-4o-mini (P5+ A5 T50 X50 / HTRAN)",
        "ptllm_tpe__P5plusA5TRdyX50__H2": "gpt-4o-mini (P5+ A5 TRdy X50 / H2)",
        "ptllm_tpe__P5plusA5TRaceX50__H2": "gpt-4o-mini (P5+ A5 TRace X50 / H2)",
        "ptllm_tpe__P5plusA5TRdyFanovaX50__H2": "gpt-4o-mini (P5+ A5 TRdyFanova X50 / H2)",
        "ptllm_tpe__P5plusA5TRdyTurboX50__H2": "gpt-4o-mini (P5+ A5 TRdyTurbo X50 / H2)",
        "ptllm_tpe__P5plusA5TRdyStrengthX50__H2": "gpt-4o-mini (P5+ A5 TRdyStrength X50 / H2)",
        "ptllm_tpe__P5plusA5TContrastX50__H2": "gpt-4o-mini (P5+ A5 TContrast X50 / H2)",
        "ptllm_tpe__P5plusA5TStagX50__H2": "gpt-4o-mini (P5+ A5 TStag X50 / H2)",
        "ptllm_tpe__P5plusA5T50XStag__H2": "gpt-4o-mini (P5+ A5 T50 XStag / H2)",
        "ptllm_tpe__P5plusA5T50XCov__H2": "gpt-4o-mini (P5+ A5 T50 XCov / H2)",
        "ptllm_tpe__P5plusA5T50XInt__H2": "gpt-4o-mini (P5+ A5 T50 XInt / H2)",
        "ptllm_tpe__P5plusA5T50XRefine__H2": "gpt-4o-mini (P5+ A5 T50 XRefine / H2)",
        "ptllm_tpe__P5plusA5T50XHAP__H2": "gpt-4o-mini (P5+ A5 T50 XHAP / H2)",
        "ptllm_smac_qwen_s1a_ep7p5__P5__H0": "qwen LoRA s1a_ep7p5 (P5/H0 smac)",
        "ptllm_tpe_qwen_s1a_ep7p5__P5__H0": "qwen LoRA s1a_ep7p5 (P5/H0 tpe)",
        "ptllm_smac_qwen_s1b_ep7p5__P5__H0": "qwen LoRA s1b_ep7p5 (P5/H0 smac)",
        "ptllm_tpe_qwen_s1b_ep7p5__P5__H0": "qwen LoRA s1b_ep7p5 (P5/H0 tpe)",
        "ptllm_tpe_qwen_s1b_attr_t2__P5plusA5T50X50__H2": (
            "qwen LoRA s1b_attr_t2 partial-edit (P5+ A5 T50 X50 / H2 tpe)"
        ),
        "ptllm_tpe_qwen_s1b_attr_t2__P5plusA5T50X50__H0": (
            "qwen LoRA s1b_attr_t2 partial-edit (P5+ A5 T50 X50 / H0 tpe)"
        ),
        "ptllm_tpe_qwen_s1b_attr_t2__P5plusA5T50X50__HPUCB_arm": (
            "qwen LoRA s1b_attr_t2 partial-edit (P5+ A5 T50 X50 / HPUCB_arm tpe)"
        ),
        "ptllm_tpe_qwen_s1b_attr_t2__P5plusA5TRdyFanovaX50RS__HPUCB_arm": (
            "qwen LoRA s1b_attr_t2 partial-edit (P5+ A5 Fanova RS / HPUCB_arm tpe)"
        ),
    }
    for letter in "ABCDEFGH":
        s1 = S2_S1_BASE[letter]
        mapping[s2_variant_id(letter, "smac")] = f"qwen LoRA {s1}_s2_{letter} (P5/H0 smac)"
        mapping[s2_variant_id(letter, "tpe")] = f"qwen LoRA {s1}_s2_{letter} (P5/H0 tpe)"
    for letter in S1A_S2_LETTERS:
        mapping[f"ptllm_smac_qwen_s1a_ep7p5_s2_{letter}__P5__H0"] = (
            f"qwen LoRA s1a_ep7p5_s2_{letter} (P5/H0 smac)"
        )
        mapping[f"ptllm_tpe_qwen_s1a_ep7p5_s2_{letter}__P5__H0"] = (
            f"qwen LoRA s1a_ep7p5_s2_{letter} (P5/H0 tpe)"
        )
    for model_id, (_base, label) in S1A_REPLICATION_MODELS.items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
    for model_id, label in S1A_R0_MODELS.items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
    for model_id, label in S1A_SETUP_AB_MODELS.items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
    for model_id, label in discover_reward_sweep_models().items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
    for model_id, label in discover_s1b_rg_dbest_hp_models().items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
    for model_id, label in discover_s1b_rg_orpo_hp_models().items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
    for model_id, label in discover_s1b_s2g_orpo_hp_models().items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
        mapping[f"ptllm_hebo_qwen_{model_id}__P5__H0"] = f"qwen LoRA {model_id} (P5/H0 hebo)"
    for model_id, label in discover_s1b_attr_orpo_models().items():
        mapping[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = label
    for label, (display, _disk, _root) in discover_grpo_rsmix_p5plus_models().items():
        mapping[label] = display
    for letter in ("C", "G"):
        mapping[f"ptllm_tpe_qwen_s1b_ep7p5_dpo_rationale_{letter}__P5__H0"] = (
            f"qwen LoRA s1b_ep7p5_dpo_rationale_{letter} (P5/H0 tpe)"
        )
        mapping[f"ptllm_smac_qwen_s1b_ep7p5_dpo_rationale_{letter}__P5__H0"] = (
            f"qwen LoRA s1b_ep7p5_dpo_rationale_{letter} (P5/H0 smac)"
        )
    name = mapping.get(variant)
    if name:
        return name
    if variant.startswith("ptllm_tpe_qwen_s1b_attr_t2__"):
        rest = variant[len("ptllm_tpe_qwen_s1b_attr_t2__") :].replace("__", " / ")
        return f"qwen LoRA s1b_attr_t2 partial-edit ({rest} tpe)"
    return variant


def method_meaning(variant: str, display: str | None = None) -> str:
    """Human-readable one-line description of how this leaderboard arm was trained/eval'd."""
    name = display or display_name(variant)
    # Shared tag scraps
    planner = "TPE planner" if "tpe" in variant or "(P5/H0 tpe)" in name else (
        "SMAC planner" if "smac" in variant or "smac" in name.lower() else (
            "HEBO planner" if "hebo" in variant or "hebo" in name.lower() else "planner"
        )
    )
    prompt = "P5/H0" if "P5__H0" in variant or "P5/H0" in name else (
        "P16/H2" if "P16" in variant or "P16" in name else "prompt"
    )
    if _RS_GPSIM_MESH_RE.search(variant):
        m = _RS_GPSIM_MESH_RE.search(variant)
        tag = m.group(0) if m else "RS"
        extra = ""
        if "v5" in variant:
            extra = ", v5 cards: exclusive top / mixed overlap / exclusive bottom 20% ranges plus grouped top3/bottom3 configs"
        elif "v4" in variant:
            extra = ", v4 cards: prefer lines plus top-3 and bottom-3 configs"
        prompt = (
            f"P5+ A5 T50 X50 {tag} (v3 bank, landscape GP retrieve from trial 1{extra})"
        )
    elif "P5plusA5T50X50RC" in variant or "P5+ A5 T50 X50 RC" in name:
        prompt = "P5+ A5 T50 X50 RC (cascade similar-task retrieve)"
    elif "P5plusA5T50X50RS" in variant or "P5+ A5 T50 X50 RS" in name:
        prompt = "P5+ A5 T50 X50 RS (GP surrogate similar-task retrieve)"
    elif "P5plusA5T50X50__HPUCB_arm" in variant or "P5+ A5 T50 X50 / HPUCB_arm" in name:
        prompt = "P5+ A5 T50 X50/HPUCB_arm"
    elif "P5plusA5T50X50__H0" in variant or "P5+ A5 T50 X50 / H0" in name:
        prompt = "P5+ A5 T50 X50/H0"
    elif "P5plusA5T50X50__H2" in variant or "P5+ A5 T50 X50 / H2" in name:
        prompt = "P5+ A5 T50 X50/H2"

    # Classical / API LLM
    if variant == "smac":
        return "Classical SMAC HPO (no LLM)."
    if variant == "tpe":
        return "Classical TPE HPO (no LLM)."
    if variant == "cmaes":
        return "Classical CMA-ES HPO (no LLM)."
    if variant == "hebo":
        return "Classical HEBO HPO (Huawei Noah, no LLM)."
    if variant.startswith("ptllm_smac__") or variant.startswith("ptllm_tpe__"):
        return f"gpt-4o-mini LLM proposer + {planner}; {prompt}."

    # SFT bases
    if "s1a_ep7p5__" in variant or name.endswith("s1a_ep7p5 (P5/H0 tpe)") or name.endswith("s1a_ep7p5 (P5/H0 smac)"):
        return f"Qwen3.5-9B LoRA SFT s1a_ep7p5 (json-only lineage); eval {prompt} + {planner}."
    if "s1b_ep7p5__" in variant and "s2_" not in variant and "dpo_rationale" not in variant and "rg_" not in variant:
        return f"Qwen3.5-9B LoRA SFT s1b_ep7p5 (rationale+json lineage); eval {prompt} + {planner}."

    # Letter DPO s2 (s1b unless ep2p5)
    s2_meanings = {
        "A": "DPO-A conservative sigmoid (β=0.1, lr=5e-6, 1ep) on p5_s2_dpo from s1b_ep7p5",
        "B": "DPO-B strong sigmoid (β=0.3, lr=1e-5, 2ep) on p5_s2_dpo from s1b_ep7p5",
        "C": "DPO-C sigmoid+SFT mix (β=0.1, pref_ftx, 2ep) on p5_s2_dpo from s1b_ep7p5",
        "D": "DPO-D explore sigmoid (β=0.2, lr=8e-6, 1.5ep) on p5_s2_dpo from s1b_ep2p5",
        "E": "DPO-E ultra-conservative sigmoid (β=0.05, lr=3e-6, 1ep) on p5_s2_dpo from s1b_ep7p5",
        "F": "DPO-F strong-β sigmoid (β=0.5, lr=5e-6, 1ep) on p5_s2_dpo from s1b_ep7p5",
        "G": "ORPO-G (β=0.1, lr=5e-6, 1ep) on p5_s2_dpo from s1b_ep7p5",
        "H": "IPO-H (β=0.1, lr=8e-6, 1.5ep) on p5_s2_dpo from s1b_ep7p5",
    }
    if "_s2_" in variant and "s1a_ep7p5_s2_" not in variant:
        for letter, desc in s2_meanings.items():
            if f"_s2_{letter}__" in variant or f"_s2_{letter} (" in name:
                return f"{desc}; eval {prompt} + {planner}."

    # s1a preference letters
    s1a_s2 = {
        "C": "s1a DPO-C sigmoid+ftx (β=0.1, 2ep) on p5_s2_dpo from s1a_ep7p5",
        "G": "s1a ORPO-G (β=0.1, lr=5e-6, 1ep) on p5_s2_dpo from s1a_ep7p5",
        "D1": "s1a conservative DPO-D1 sigmoid (β=0.05, ftx=0.25, lr=2e-6, 1ep)",
        "D1m": "s1a D1-micro (same as D1, 0.5 epoch)",
        "D2": "s1a conservative IPO-D2 (β=0.1, ftx=0.15, lr=3e-6, 1ep)",
        "D3": "s1a conservative cDPO-D3 sigmoid (β=0.08, ftx=0.2, lr=2e-6, 1ep)",
        "O1": "s1a soft ORPO-O1 (β=0.03, lr=2e-6, 1ep)",
        "O3": "s1a micro ORPO-O3 (β=0.1, lr=5e-6, 0.5ep)",
        "O4": "s1a gentle SimPO-O4 (β=0.1, lr=2e-6, 1ep)",
    }
    for letter, desc in s1a_s2.items():
        if f"s1a_ep7p5_s2_{letter}__" in variant or f"s1a_ep7p5_s2_{letter} (" in name:
            return f"{desc}; eval {prompt} + {planner}."

    # Rationale preference
    if "dpo_rationale_G" in variant or "dpo_rationale_G" in name:
        return (
            "ORPO on rationale+JSON pairs (p5_s2_dpo_rationale) from s1b_ep7p5 "
            f"(β=0.1, lr=5e-6, 1ep); eval {prompt} + {planner}, rationale_json."
        )
    if "dpo_rationale_C" in variant or "dpo_rationale_C" in name:
        return (
            "DPO-C sigmoid on rationale+JSON pairs (p5_s2_dpo_rationale) from s1b_ep7p5 "
            f"(β=0.1, 2ep); eval {prompt} + {planner}, rationale_json."
        )

    # ORPO HP sweep (rationale pairs)
    if "s1b_rg_orpo_" in variant or "s1b_rg_orpo_" in name:
        import re

        m = re.search(r"lr(\w+)_b(\d+)", name)
        lr_map = {
            "2e6": "2e-6",
            "5e6": "5e-6",
            "7p5e6": "7.5e-6",
            "1e5": "1e-5",
            "1p5e5": "1.5e-5",
        }
        b_map = {
            "05": "0.05",
            "07": "0.07",
            "10": "0.1",
            "15": "0.15",
            "20": "0.2",
            "25": "0.25",
            "30": "0.3",
            "35": "0.35",
            "40": "0.4",
        }
        lr = lr_map.get(m.group(1), m.group(1)) if m else "?"
        beta = b_map.get(m.group(2), m.group(2)) if m else "?"
        return (
            f"ORPO HP sweep from s1b_ep7p5 on p5_s2_dpo_rationale "
            f"(lr={lr}, pref_beta={beta}, 1ep); eval {prompt} + {planner}, rationale_json."
        )

    # ORPO HP sweep (s2_G / p5_s2_dpo)
    if "s1b_s2g_orpo_" in variant or "s1b_s2g_orpo_" in name:
        import re

        m = re.search(r"lr(\w+)_b(\d+)", name)
        lr_map = {
            "2e6": "2e-6",
            "5e6": "5e-6",
            "7p5e6": "7.5e-6",
            "1e5": "1e-5",
            "1p5e5": "1.5e-5",
        }
        b_map = {
            "05": "0.05",
            "07": "0.07",
            "10": "0.1",
            "15": "0.15",
            "20": "0.2",
            "25": "0.25",
            "30": "0.3",
            "35": "0.35",
            "40": "0.4",
        }
        lr = lr_map.get(m.group(1), m.group(1)) if m else "?"
        beta = b_map.get(m.group(2), m.group(2)) if m else "?"
        return (
            f"ORPO HP sweep from s1b_ep7p5 on p5_s2_dpo (s2_G recipe) "
            f"(lr={lr}, pref_beta={beta}, 1ep); eval {prompt} + {planner}, rationale_json."
        )

    # Attribution ORPO (T2 partial-edit / T3 per-HP soft) from s1b_ep7p5
    if "s1b_attr_t2_orpo_lr" in variant or "s1b_attr_t2_orpo_lr" in name:
        import re

        m = re.search(r"lr(\w+)_b(\d+)", name)
        lr_map = {"1e5": "1e-5", "5e6": "5e-6"}
        b_map = {"30": "0.3", "35": "0.35", "10": "0.1"}
        lr = lr_map.get(m.group(1), m.group(1)) if m else "?"
        beta = b_map.get(m.group(2), m.group(2)) if m else "?"
        return (
            f"ORPO T2 partial-edit data from s1b_ep7p5 (lr={lr}, pref_beta={beta}, 1ep); "
            f"eval {prompt} + {planner}, rationale_json."
        )
    if "s1b_attr_t2" in variant or "s1b_attr_t2" in name:
        return (
            f"ORPO attribution T2 (partial-edit, all blamed HPs repaired) from s1b_ep7p5 "
            f"on p5_s2_dpo (lr=5e-6, pref_beta=0.1, 1ep); eval {prompt} + {planner}, rationale_json."
        )
    if "s1b_attr_t3" in variant or "s1b_attr_t3" in name:
        return (
            f"ORPO attribution T3 (per-HP soft, upsampled by blame) from s1b_ep7p5 "
            f"on p5_s2_dpo (lr=5e-6, pref_beta=0.1, 1ep); eval {prompt} + {planner}, rationale_json."
        )

    # RL on rationale_G
    if "s1b_rg_dbest_" in variant or "s1b_rg_dbest_" in name:
        import re

        m = re.search(r"lr(\w+)_kl(\d+)", name)
        lr_map = {"5e7": "5e-7", "1e6": "1e-6", "2e6": "2e-6"}
        kl_map = {"02": "0.02", "05": "0.05", "10": "0.1"}
        lr = lr_map.get(m.group(1), "?") if m else "?"
        kl = kl_map.get(m.group(2), "?") if m else "?"
        return (
            f"REINFORCE on dpo_rationale_G with Setup-B dbest light16 "
            f"(lr={lr}, kl_beta={kl}, ckpt after 16 datasets); eval {prompt} + {planner}."
        )

    # Reward sweep / Setup B RL on s1a
    if "s1a_b_rw_" in variant or "s1a_b_rw_" in name:
        import re

        m_hp = re.search(r"s1a_b_rw_dbest_lr(\w+)_kl(\d+)", name)
        if m_hp:
            lr_map = {
                "5e7": "5e-7",
                "7p5e7": "7.5e-7",
                "1e6": "1e-6",
                "1p5e6": "1.5e-6",
            }
            kl_map = {"03": "0.03", "05": "0.05", "07": "0.07"}
            lr = lr_map.get(m_hp.group(1), m_hp.group(1))
            kl = kl_map.get(m_hp.group(2), m_hp.group(2))
            return (
                f"REINFORCE local refine from s1a_ep7p5 on Setup-B dbest light16 "
                f"(lr={lr}, kl_beta={kl}, ckpt16); eval {prompt} + {planner}."
            )
        m = re.search(r"s1a_b_rw_(\w+)_ckpt(\d+)", name)
        arm = m.group(1) if m else "?"
        ckpt = m.group(2) if m else "?"
        arm_desc = {
            "dbest": "pure Δbest reward",
            "hybrid": "hybrid Δbest+0.1·abs reward",
            "abs": "absolute-norm reward only",
            "mix_strong": "Δbest+0.5·abs reward",
        }.get(arm, arm)
        return (
            f"REINFORCE from s1a_ep7p5 on Setup-B prefix rollouts, {arm_desc}, "
            f"dsseq light16, mid-ckpt after {ckpt} datasets; eval {prompt} + {planner}."
        )

    # Setup A/B ablations
    if "a_boseed_global" in variant or "a_boseed_global" in name:
        return f"REINFORCE Setup-A BO-seeded rollouts, global advantage, T1.0 light16 from s1a_ep7p5; eval {prompt} + {planner}."
    if "a_boseed_dsseq" in variant or "a_boseed_dsseq" in name:
        return f"REINFORCE Setup-A BO-seeded rollouts, per-dataset sequential, T1.0 light16 from s1a_ep7p5; eval {prompt} + {planner}."
    if "b_prefix_grpo" in variant or "b_prefix_grpo" in name:
        init = "s1b_ep7p5_attr_t2" if "s1b_attr_t2" in variant or "s1b_attr_t2" in name else "s1a_ep7p5"
        if "grpo_rsmix" in variant or "grpo_rsmix" in name:
            return (
                f"GRPO-group RS-mix Setup-B prefix rollouts (12-prefix schedule, 4→10 adaptive), "
                f"T1.0 pareto32 from {init}; eval {prompt} + {planner}."
            )
        return f"REINFORCE Setup-B BO-prefix rollouts, GRPO-group advantage, T1.0 light16 from {init}; eval {prompt} + {planner}."
    if "b_prefix_dsseq_hg8" in variant or "b_prefix_dsseq_hg8" in name:
        return f"REINFORCE Setup-B prefix, dsseq, mid-ckpt after 8 high-growth datasets (light8) from s1a_ep7p5; eval {prompt} + {planner}."
    if "b_prefix_dsseq_m32" in variant or "b_prefix_dsseq_m32" in name:
        return f"REINFORCE Setup-B prefix, dsseq on medium32 from s1a_ep7p5; eval {prompt} + {planner}."
    if "b_prefix_dsseq" in variant or "b_prefix_dsseq" in name:
        return f"REINFORCE Setup-B BO-prefix rollouts, per-dataset sequential, T1.0 light16 from s1a_ep7p5; eval {prompt} + {planner}."

    # R0 light/medium
    if "r0a" in variant or "r0a" in name:
        return f"REINFORCE R0 early arm: T0.7 light16 from s1a_ep7p5; eval {prompt} + {planner}."
    if "r0b" in variant or "r0b" in name:
        return f"REINFORCE R0 arm: T1.0 light16 from s1a_ep7p5; eval {prompt} + {planner}."
    if "r0c" in variant or "r0c" in name:
        return f"REINFORCE R0 arm: T0.7 medium32 from s1a_ep7p5; eval {prompt} + {planner}."
    if "r0d" in variant or "r0d" in name:
        return f"REINFORCE R0 arm: T1.0 medium32 from s1a_ep7p5; eval {prompt} + {planner}."

    return f"Qwen LoRA HPO arm; eval {prompt} + {planner}."


def family(variant: str) -> str:
    if variant in ("smac", "tpe", "cmaes", "hebo"):
        return "classical"
    if variant.startswith("ptllm_smac__") or variant.startswith("ptllm_tpe__"):
        return "gpt4o_mini"
    if is_s2_variant(variant) and "ptllm_smac_qwen" in variant:
        return "qwen_lora_s2_smac"
    if is_s2_variant(variant) and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s2"
    if "s1a_ep7p5_s2_" in variant and "ptllm_smac_qwen" in variant:
        return "qwen_lora_s1a_s2_smac"
    if "s1a_ep7p5_s2_" in variant and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s1a_s2"
    if any(x in variant for x in ("s1a_r0a", "s1a_r0b", "s1a_r0c", "s1a_r0d")) and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s1a_r0"
    if any(
        x in variant
        for x in (
            "s1a_a_boseed_global",
            "s1a_a_boseed_dsseq",
            "s1a_b_prefix_grpo",
            "s1a_b_prefix_dsseq",
            "s1a_b_prefix_dsseq_hg8",
            "s1a_b_prefix_dsseq_m32",
            "s1a_b_rw_",
        )
    ) and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s1a_setup_ab"
    if "s1b_rg_dbest_" in variant and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s1b_rg_rl"
    if "s1b_rg_orpo_" in variant and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s1b_rg_orpo"
    if "s1b_s2g_orpo_" in variant and "ptllm_hebo_qwen" in variant:
        return "qwen_lora_s1b_s2g_orpo_hebo"
    if "s1b_s2g_orpo_" in variant and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s1b_s2g_orpo"
    if "s1b_attr_t" in variant and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s1b_attr_orpo"
    if "dpo_rationale" in variant and "ptllm_tpe_qwen" in variant:
        return "qwen_lora_s2_rationale"
    if "dpo_rationale" in variant and "ptllm_smac_qwen" in variant:
        return "qwen_lora_s2_rationale_smac"
    if "ptllm_smac_qwen" in variant:
        return "qwen_lora_p5_smac"
    if "ptllm_tpe_qwen" in variant:
        return "qwen_lora_p5"
    return "other"


def build_sources(*, include_replications: bool = True) -> dict[str, tuple[Path, str]]:
    classical = ALGO40 / "classical"
    gpt = ALGO40
    s1a = ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5"
    s1b = ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1b_ep7p5"
    sources = {
        "smac": (classical, "smac"),
        "tpe": (classical, "tpe"),
        "cmaes": (classical, "cmaes"),
        "hebo": (classical, "hebo"),
        "ptllm_smac__P5__H0": (gpt, "ptllm_smac__P5__H0"),
        "ptllm_tpe__P5__H0": (gpt, "ptllm_tpe__P5__H0"),
        "ptllm_tpe__P16__H2": (gpt, "ptllm_tpe__P16__H2"),
        # Dynamic-gates sweep: same ALGO40 root, new prompt/hybrid variants.
        "ptllm_tpe__P5plusA5T50X50__H2": (gpt, "ptllm_tpe__P5plusA5T50X50__H2"),
        # Similar-task bake-off: cascade (RC) vs per-algo GP surrogate (RS).
        "ptllm_tpe__P5plusA5T50X50RC__H2": (gpt, "ptllm_tpe__P5plusA5T50X50RC__H2"),
        "ptllm_tpe__P5plusA5T50X50RS__H2": (gpt, "ptllm_tpe__P5plusA5T50X50RS__H2"),
        "ptllm_tpe__P5plusA5T50X50__H60": (gpt, "ptllm_tpe__P5plusA5T50X50__H60"),
        "ptllm_tpe__P5plusA5T50X50__H0": (gpt, "ptllm_tpe__P5plusA5T50X50__H0"),
        "ptllm_tpe__P5plusA5T50X50__H40": (gpt, "ptllm_tpe__P5plusA5T50X50__H40"),
        "ptllm_tpe__P5plusA5T50X50__HUCB": (gpt, "ptllm_tpe__P5plusA5T50X50__HUCB"),
        "ptllm_tpe__P5plusA5T50X50__HPUCB": (gpt, "ptllm_tpe__P5plusA5T50X50__HPUCB"),
        "ptllm_tpe__P5plusA5T50X50__HPUCB_breakthrough": (gpt, "ptllm_tpe__P5plusA5T50X50__HPUCB_breakthrough"),
        "ptllm_tpe__P5plusA5T50X50__HPUCB_arm": (gpt, "ptllm_tpe__P5plusA5T50X50__HPUCB_arm"),
        "ptllm_tpe__P5plusA5T50X50__HDTS": (gpt, "ptllm_tpe__P5plusA5T50X50__HDTS"),
        "ptllm_tpe__P5plusA5T50X50__HAP": (gpt, "ptllm_tpe__P5plusA5T50X50__HAP"),
        "ptllm_tpe__P5plusA5T50X50__HDUCB": (gpt, "ptllm_tpe__P5plusA5T50X50__HDUCB"),
        "ptllm_tpe__P5plusA5T50X50__HDMAB": (gpt, "ptllm_tpe__P5plusA5T50X50__HDMAB"),
        "ptllm_tpe__P5plusA5T50X50__HTRAN": (gpt, "ptllm_tpe__P5plusA5T50X50__HTRAN"),
        "ptllm_tpe__P5plusA5TRdyX50__H2": (gpt, "ptllm_tpe__P5plusA5TRdyX50__H2"),
        "ptllm_tpe__P5plusA5TRaceX50__H2": (gpt, "ptllm_tpe__P5plusA5TRaceX50__H2"),
        "ptllm_tpe__P5plusA5TRdyFanovaX50__H2": (gpt, "ptllm_tpe__P5plusA5TRdyFanovaX50__H2"),
        "ptllm_tpe__P5plusA5TRdyTurboX50__H2": (gpt, "ptllm_tpe__P5plusA5TRdyTurboX50__H2"),
        "ptllm_tpe__P5plusA5TRdyStrengthX50__H2": (gpt, "ptllm_tpe__P5plusA5TRdyStrengthX50__H2"),
        "ptllm_tpe__P5plusA5TContrastX50__H2": (gpt, "ptllm_tpe__P5plusA5TContrastX50__H2"),
        "ptllm_tpe__P5plusA5TStagX50__H2": (gpt, "ptllm_tpe__P5plusA5TStagX50__H2"),
        "ptllm_tpe__P5plusA5T50XStag__H2": (gpt, "ptllm_tpe__P5plusA5T50XStag__H2"),
        "ptllm_tpe__P5plusA5T50XCov__H2": (gpt, "ptllm_tpe__P5plusA5T50XCov__H2"),
        "ptllm_tpe__P5plusA5T50XInt__H2": (gpt, "ptllm_tpe__P5plusA5T50XInt__H2"),
        "ptllm_tpe__P5plusA5T50XRefine__H2": (gpt, "ptllm_tpe__P5plusA5T50XRefine__H2"),
        "ptllm_tpe__P5plusA5T50XHAP__H2": (gpt, "ptllm_tpe__P5plusA5T50XHAP__H2"),
        "ptllm_smac_qwen_s1a_ep7p5__P5__H0": (s1a, "ptllm_smac_qwen__P5__H0"),
        "ptllm_tpe_qwen_s1a_ep7p5__P5__H0": (s1a, "ptllm_tpe_qwen__P5__H0"),
        "ptllm_smac_qwen_s1b_ep7p5__P5__H0": (s1b, "ptllm_smac_qwen__P5__H0"),
        "ptllm_tpe_qwen_s1b_ep7p5__P5__H0": (s1b, "ptllm_tpe_qwen__P5__H0"),
    }
    for letter in "ABCDEFGH":
        dpo_root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_dpo_{letter}"
        sources[s2_variant_id(letter, "smac")] = (dpo_root, "ptllm_smac_qwen__P5__H0")
        sources[s2_variant_id(letter, "tpe")] = (dpo_root, "ptllm_tpe_qwen__P5__H0")
    s1a_s2_root = ALGO40.parent
    for letter in S1A_S2_LETTERS:
        root = s1a_s2_root / f"real_hpo_v2_mltest20_algo40_p5_s1a_s2_{letter}"
        sources[f"ptllm_smac_qwen_s1a_ep7p5_s2_{letter}__P5__H0"] = (root, "ptllm_smac_qwen__P5__H0")
        sources[f"ptllm_tpe_qwen_s1a_ep7p5_s2_{letter}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    if include_replications:
        for model_id, _base in S1A_REPLICATION_MODELS.items():
            root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
            sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    for model_id in S1A_R0_MODELS:
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
        sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    for model_id in S1A_SETUP_AB_MODELS:
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
        sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    for model_id in discover_reward_sweep_models():
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
        sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    for model_id in discover_s1b_rg_dbest_hp_models():
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
        sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    for model_id in discover_s1b_rg_orpo_hp_models():
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
        sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    for model_id in discover_s1b_s2g_orpo_hp_models():
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
        sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
        sources[f"ptllm_hebo_qwen_{model_id}__P5__H0"] = (root, "ptllm_hebo_qwen__P5__H0")
    for model_id in discover_s1b_attr_orpo_models():
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_{model_id}"
        sources[f"ptllm_tpe_qwen_{model_id}__P5__H0"] = (root, "ptllm_tpe_qwen__P5__H0")
    for label, (_display, disk, root_s) in discover_grpo_rsmix_p5plus_models().items():
        sources[label] = (Path(root_s), disk)
    # Same attr_t2 adapter under P5+ experience (p5plus sweep output root).
    qwen_p5plus_root = ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus"
    summary_csv = ALGO40 / "analysis/p5plus/gpt_p5_p5plus_summary.csv"
    if summary_csv.is_file():
        for row in csv.DictReader(summary_csv.open(encoding="utf-8")):
            gv = row["variant"].strip()
            if not gv.startswith("ptllm_tpe__"):
                continue
            disk = gv.replace("ptllm_tpe__", "ptllm_tpe_qwen__", 1)
            label = gv.replace("ptllm_tpe__", "ptllm_tpe_qwen_s1b_attr_t2__", 1)
            sources[label] = (qwen_p5plus_root, disk)
    else:
        sources["ptllm_tpe_qwen_s1b_attr_t2__P5plusA5T50X50__HPUCB_arm"] = (
            qwen_p5plus_root,
            "ptllm_tpe_qwen__P5plusA5T50X50__HPUCB_arm",
        )
    sources["ptllm_tpe_qwen_s1b_attr_t2__P5plusA5TRdyFanovaX50RS__HPUCB_arm"] = (
        qwen_p5plus_root,
        "ptllm_tpe_qwen__P5plusA5TRdyFanovaX50RS__HPUCB_arm",
    )
    for label, pair in discover_p5plus_qwen_variants(qwen_p5plus_root).items():
        sources[label] = pair
    for letter in ("C", "G"):
        root = ALGO40.parent / f"real_hpo_v2_mltest20_algo40_p5_dpo_rationale_{letter}"
        sources[f"ptllm_tpe_qwen_s1b_ep7p5_dpo_rationale_{letter}__P5__H0"] = (
            root,
            "ptllm_tpe_qwen__P5__H0",
        )
        sources[f"ptllm_smac_qwen_s1b_ep7p5_dpo_rationale_{letter}__P5__H0"] = (
            root,
            "ptllm_smac_qwen__P5__H0",
        )
    return sources


def is_smac_arm(variant: str) -> bool:
    """Classical SMAC or any ptllm_smac* LLM arm."""
    return variant == "smac" or variant.startswith("ptllm_smac")


def build_metric_rows(
    variants: list[str],
    summary: dict[tuple[str, str], float],
    task_keys: list[str],
    coverage: dict[str, int],
    pooled_trial_norm: bool = False,
    trial_pools_by_task: dict[str, list[float]] | None = None,
    drop_worst_frac: float = PRIMARY_DROP_FRAC,
) -> tuple[list[dict], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    per_ds_ranks: dict[str, dict[str, float]] = {}
    per_ds_norm: dict[str, dict[str, float]] = {}
    for key in task_keys:
        scores = {v: summary[(key, v)] for v in variants if (key, v) in summary}
        if len(scores) < 2:
            continue
        per_ds_ranks[key] = task_ranks(scores)
        if pooled_trial_norm:
            pool = (trial_pools_by_task or {}).get(key, [])
            per_ds_norm[key] = normalize_from_pooled_drop_worst(
                scores, pool, drop_worst_frac=drop_worst_frac
            )
        else:
            per_ds_norm[key] = normalize(scores)

    def is_cls(k: str) -> bool:
        return k.startswith("cls/")

    rows: list[dict] = []
    for v in variants:
        ranks_all = [per_ds_ranks[k][v] for k in per_ds_ranks if v in per_ds_ranks[k]]
        ranks_cls = [per_ds_ranks[k][v] for k in per_ds_ranks if v in per_ds_ranks[k] and is_cls(k)]
        ranks_rgs = [per_ds_ranks[k][v] for k in per_ds_ranks if v in per_ds_ranks[k] and not is_cls(k)]
        cn = [per_ds_norm[k][v] for k in per_ds_norm if v in per_ds_norm[k] and is_cls(k)]
        rn = [per_ds_norm[k][v] for k in per_ds_norm if v in per_ds_norm[k] and not is_cls(k)]
        mean_cn = sum(cn) / len(cn) if cn else float("nan")
        mean_rn = sum(rn) / len(rn) if rn else float("nan")
        bal = (mean_cn + mean_rn) / 2 if not (math.isnan(mean_cn) or math.isnan(mean_rn)) else float("nan")
        rows.append(
            {
                "display_name": display_name(v),
                "variant": v,
                "family": family(v),
                "datasets": coverage[v],  # algo-correct slots in the active tasks json (30 default)
                "avg_rank": sum(ranks_all) / len(ranks_all) if ranks_all else float("nan"),
                "avg_rank_cls": sum(ranks_cls) / len(ranks_cls) if ranks_cls else float("nan"),
                "avg_rank_rgs": sum(ranks_rgs) / len(ranks_rgs) if ranks_rgs else float("nan"),
                "mean_cls_norm": mean_cn,
                "mean_rgs_norm": mean_rn,
                "mean_balanced_norm": bal,
            }
        )
    rows.sort(key=lambda r: (math.isnan(r["avg_rank"]), r["avg_rank"]))
    return rows, per_ds_ranks, per_ds_norm


def write_rank_csv(
    path: Path,
    rows: list[dict],
    fields: list[str],
    sort_by_balanced_norm: bool,
    sort_key: str | None = None,
) -> None:
    ordered = rows
    key = sort_key or ("mean_balanced_norm" if sort_by_balanced_norm else None)
    if key is not None:
        ordered = sorted(
            rows,
            key=lambda r: (
                math.isnan(r[key]),
                -r[key] if not math.isnan(r[key]) else 0.0,
            ),
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(ordered, start=1):
            out: dict[str, str | int] = {
                "rank": i,
                "display_name": row["display_name"],
                "variant": row["variant"],
                "family": row["family"],
                "datasets": row["datasets"],
            }
            if "method_meaning" in fields:
                out["method_meaning"] = method_meaning(row["variant"], row["display_name"])
            for k in ("avg_rank", "avg_rank_cls", "avg_rank_rgs"):
                out[k] = f"{row[k]:.2f}" if not math.isnan(row[k]) else ""
            for k in ("mean_cls_norm", "mean_rgs_norm", "mean_balanced_norm"):
                out[k] = f"{row[k]:.6f}" if not math.isnan(row[k]) else ""
            w.writerow(out)


def write_standard_pooled_leaderboards(
    out_dir: Path,
    rows: list[dict],
    rows_no_smac: list[dict],
    fields: list[str],
    *,
    seed1_suffix: str = "",
) -> Path:
    """Write drop-20% pooled tables as the standard algo40 leaderboard.

    Canonical file: mean_norm_table_pooled_drop20pct{suffix}.csv
    Legacy aliases (same content, drop-20%): mean_norm_table_no_smac_rgs_pooled{suffix}.csv
    """
    suffix = seed1_suffix
    primary = out_dir / f"mean_norm_table_pooled_drop20pct{suffix}.csv"
    write_rank_csv(primary, rows, fields, sort_by_balanced_norm=True)
    write_rank_csv(
        out_dir / f"mean_norm_table_no_smac_rgs_pooled{suffix}.csv",
        rows,
        fields,
        sort_by_balanced_norm=True,
    )
    write_rank_csv(
        out_dir / f"mean_norm_table_no_smac_pooled{suffix}.csv",
        rows_no_smac,
        fields,
        sort_by_balanced_norm=True,
    )
    if not suffix:
        write_rank_csv(out_dir / "mean_norm_table.csv", rows, fields, sort_by_balanced_norm=True)
        write_rank_csv(out_dir / "mean_norm_table_pooled.csv", rows, fields, sort_by_balanced_norm=True)
    pct = int(round(PRIMARY_DROP_FRAC * 100))
    print(f"Wrote {primary} (standard: drop worst {pct}%, cls+rgs+balanced)")
    return primary


def write_drop_norm_tables(
    out_dir: Path,
    variants: list[str],
    summary: dict[tuple[str, str], float],
    task_keys: list[str],
    coverage: dict[str, int],
    trial_pools_by_task: dict[str, list[float]],
    fields: list[str],
    *,
    seed1_suffix: str = "",
    drop_fracs: tuple[float, ...] = (0.30,),
) -> list[Path]:
    """Optional archive tables at non-standard drop fractions (not primary)."""
    written: list[Path] = []
    for frac in drop_fracs:
        if abs(frac - PRIMARY_DROP_FRAC) < 1e-9:
            continue
        pct = int(round(frac * 100))
        rows, _, _ = build_metric_rows(
            variants,
            summary,
            task_keys,
            coverage,
            pooled_trial_norm=True,
            trial_pools_by_task=trial_pools_by_task,
            drop_worst_frac=frac,
        )
        path = out_dir / f"mean_norm_table_pooled_drop{pct}pct{seed1_suffix}.csv"
        write_rank_csv(path, rows, fields, sort_by_balanced_norm=True)
        written.append(path)
        print(f"Wrote {path} (archive: drop worst {pct}%, cls+rgs+balanced)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-json",
        type=Path,
        default=ALGO40 / "tasks_30.json",
        help=(
            "Ranking task list. Default: tasks_30.json (30 unique datasets). "
            "Pass tasks_40.json to rebuild the 40 dataset×algorithm reference table."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ALGO40 / "analysis" / "avg_rank",
    )
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only include variants with every task in --tasks-json",
    )
    parser.add_argument(
        "--rgs-pooled-trial-norm",
        "--pooled-trial-norm",
        dest="pooled_trial_norm",
        action="store_true",
        help=(
            "Cls+rgs: method score = final best; per-task norm pools ALL methods' "
            "trial perfs, drops worst 20%%, then min-max with max(kept)=1.0 "
            "(higher better). Without this flag, both use method-final min-max."
        ),
    )
    parser.add_argument(
        "--seed1-only",
        action="store_true",
        help=(
            "Exclude seed-replication arms (*_r2, *_r3, etc. from "
            "S1A_REPLICATION_MODELS). Writes a separate seed1 table and does "
            "not overwrite the full leaderboard CSVs."
        ),
    )
    parser.add_argument(
        "--ignore-task-algorithm",
        action="store_true",
        help=(
            "When loading trajs, do not require the traj majority algorithm to "
            "match the active tasks json. Default is algo-correct: the "
            "`datasets` column counts matching slots (30 in tasks_30.json, "
            "40 in tasks_40.json), not unique dataset names."
        ),
    )
    args = parser.parse_args()

    tasks_json = args.tasks_json if args.tasks_json.is_absolute() else REPO / args.tasks_json
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(tasks_json)
    task_keys = [task_key(t) for t in tasks]
    n_tasks = len(tasks)
    sources = build_sources(include_replications=not args.seed1_only)
    legacy_rs = [k for k in sources if is_legacy_rs_retrieve_variant(k)]
    for k in legacy_rs:
        del sources[k]

    summary: dict[tuple[str, str], float] = {}
    coverage: dict[str, int] = {label: 0 for label in sources}
    trial_pools_by_task: dict[str, list[float]] = {k: [] for k in task_keys}

    for label, (root, disk_var) in sources.items():
        if not root.is_dir():
            continue
        for t in tasks:
            tt, ds = t["task_type"], t["dataset"]
            key = task_key(t)
            algo = None if args.ignore_task_algorithm else t.get("algorithm")
            traj = find_traj(root, tt, ds, disk_var, args.n_trials, algorithm=algo)
            if traj is None:
                continue
            best = final_best(traj)
            if best is None or math.isnan(best) or best == float("-inf"):
                continue
            if args.pooled_trial_norm:
                # Pool trials for both cls and rgs when using pooled scale.
                trial_pools_by_task[key].extend(trial_perfs(traj, args.n_trials))
            summary[(key, label)] = best
            coverage[label] += 1

    variants = [v for v in sources if coverage[v] >= 1]
    if args.require_complete:
        variants = [v for v in variants if coverage[v] == len(tasks)]

    pools = trial_pools_by_task if args.pooled_trial_norm else None
    drop_frac = PRIMARY_DROP_FRAC if args.pooled_trial_norm else 0.1
    # Full peer set including classical/LLM SMAC.
    rows, per_ds_ranks, per_ds_norm = build_metric_rows(
        variants, summary, task_keys, coverage, args.pooled_trial_norm, pools, drop_frac
    )
    variants_no_smac = [v for v in variants if not is_smac_arm(v)]
    rows_no_smac, _, _ = build_metric_rows(
        variants_no_smac, summary, task_keys, coverage, args.pooled_trial_norm, pools, drop_frac
    )

    fields = [
        "rank",
        "display_name",
        "method_meaning",
        "variant",
        "family",
        "datasets",
        "avg_rank",
        "avg_rank_cls",
        "avg_rank_rgs",
        "mean_cls_norm",
        "mean_rgs_norm",
        "mean_balanced_norm",
    ]

    if args.seed1_only:
        # Separate artifacts so the full (with r2/r3) leaderboard stays intact.
        if args.pooled_trial_norm:
            path_primary = write_standard_pooled_leaderboards(
                out_dir, rows, rows_no_smac, fields, seed1_suffix="_seed1"
            )
            write_drop_norm_tables(
                out_dir,
                variants,
                summary,
                task_keys,
                coverage,
                trial_pools_by_task,
                fields,
                seed1_suffix="_seed1",
            )
        else:
            path_primary = out_dir / "mean_norm_table_no_smac_seed1.csv"
            write_rank_csv(path_primary, rows, fields, sort_by_balanced_norm=True)
        print(f"Wrote {path_primary} (seed1-only, excluded {len(S1A_REPLICATION_MODELS)} replication arms)")
        if legacy_rs:
            print(f"Excluded {len(legacy_rs)} legacy RS/RC retrieve arms")
    else:
        if args.pooled_trial_norm:
            path_primary = write_standard_pooled_leaderboards(
                out_dir, rows, rows_no_smac, fields
            )
            write_drop_norm_tables(
                out_dir,
                variants,
                summary,
                task_keys,
                coverage,
                trial_pools_by_task,
                fields,
            )
        else:
            path_primary = out_dir / "mean_norm_table_no_smac.csv"
            write_rank_csv(path_primary, rows, fields, sort_by_balanced_norm=True)
        print(f"Wrote {path_primary}")
        if legacy_rs:
            print(f"Excluded {len(legacy_rs)} legacy RS/RC retrieve arms")

    complete = sum(1 for v in variants if coverage[v] == n_tasks)
    print(f"Variants: {len(variants)} ({complete} complete at {n_tasks}/{n_tasks})")
    ranked = sorted(
        rows,
        key=lambda r: (
            math.isnan(r["mean_balanced_norm"]),
            -r["mean_balanced_norm"] if not math.isnan(r["mean_balanced_norm"]) else 0.0,
        ),
    )
    for i, row in enumerate(ranked[:15], start=1):
        print(
            f"  {i:2d}. {row['display_name'][:72]:72s} "
            f"bal={row['mean_balanced_norm']:.4f} ds={row['datasets']}/{n_tasks}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())