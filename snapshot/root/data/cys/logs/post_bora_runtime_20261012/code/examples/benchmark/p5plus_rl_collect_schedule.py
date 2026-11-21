#!/usr/bin/env python3
"""GRPO / Setup-B collect schedule: prefix → prompt scenario.

Locked mix (do not generate examples from this module):

  k=0, 5, 10     → rs_warm   (P5plusRS / RSv4 / RSv5 similar-task cards)
  k=20, 30, 40   → normal    (plain P5)
  k=50, 70, 90   → explore   (P5plusA5Explore coverage map)
  k=60, 80, 100  → exploit   (P5plusA5Exploit self-run prefer/avoid)

  python examples/benchmark/p5plus_rl_collect_schedule.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

MIX_PREFIXES: List[int] = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# v13: denser normal band + explore and exploit at each late prefix.
V12_MIX_JOBS: List[Tuple[int, str]] = [
    (0, "rs_warm"),
    (5, "rs_warm"),
    (10, "rs_warm"),
    (20, "normal"),
    (30, "normal"),
    (40, "normal"),
    (50, "explore"),
    (60, "exploit"),
    (70, "explore"),
    (80, "exploit"),
    (90, "explore"),
    (100, "exploit"),
]

V13_MIX_JOBS: List[Tuple[int, str]] = [
    (0, "rs_warm"),
    (5, "rs_warm"),
    (10, "rs_warm"),
    (20, "normal"),
    (25, "normal"),
    (30, "normal"),
    (35, "normal"),
    (40, "normal"),
    (45, "normal"),
    (50, "explore"),
    (50, "exploit"),
    (60, "explore"),
    (60, "exploit"),
    (70, "explore"),
    (70, "exploit"),
    (80, "explore"),
    (80, "exploit"),
    (90, "explore"),
    (90, "exploit"),
    (100, "explore"),
    (100, "exploit"),
]

# v14 (2026-10-05): rerank at k=3,5,7 with the exact benchmark retrieval variant
# (train80 bank, source_gp, global trials 4-8); no empty-history prefix.
V14_RERANK_PROMPT = "P5plusA5T50X50RS5f05v4"
V14_MIX_JOBS: List[Tuple[int, str]] = [
    (3, "rerank"),
    (5, "rerank"),
    (7, "rerank"),
    (10, "normal"),
    (20, "normal"),
    (30, "normal"),
    (40, "normal"),
    (50, "explore"),
    (60, "exploit"),
    (70, "explore"),
    (80, "exploit"),
    (90, "explore"),
    (100, "exploit"),
]

# v15 (2026-10-06): denser k<=40 where v14 groups improved and gained most; late
# checkpoints 90/100 dropped (median gain ~0.002). Still 13 groups per task.
V15_MIX_JOBS: List[Tuple[int, str]] = [
    (3, "rerank"),
    (5, "rerank"),
    (7, "rerank"),
    (10, "normal"),
    (15, "normal"),
    (20, "normal"),
    (25, "normal"),
    (30, "normal"),
    (40, "normal"),
    (50, "explore"),
    (60, "exploit"),
    (70, "explore"),
    (80, "exploit"),
]

PREFIX_SCENARIO: Dict[int, str] = {
    0: "rs_warm",
    5: "rs_warm",
    10: "rs_warm",
    20: "normal",
    30: "normal",
    40: "normal",
    50: "explore",
    70: "explore",
    90: "explore",
    60: "exploit",
    80: "exploit",
    100: "exploit",
}

RS_WARM_PROMPT: Dict[str, str] = {
    "v3": "P5plusRS",
    "v4": "P5plusRSv4",
    "v5": "P5plusRSv5",
}

FIXED_SCENARIO_PROMPT: Dict[str, str] = {
    "normal": "P5",
    "explore": "P5plusA5Explore",
    "exploit": "P5plusA5Exploit",
}

SCENARIO_ALIASES: Dict[str, str] = {
    "similar": "rs_warm",
}

ALL_SCENARIOS = ("rs_warm", "normal", "explore", "exploit")


def canonical_scenario(name: str) -> str:
    key = str(name or "").strip()
    return SCENARIO_ALIASES.get(key, key)


def scenario_for_prefix(prefix_k: int) -> str:
    k = int(prefix_k)
    if k not in PREFIX_SCENARIO:
        raise ValueError(
            f"prefix k={k} is not in the mix schedule {MIX_PREFIXES}; "
            f"map={PREFIX_SCENARIO}"
        )
    return PREFIX_SCENARIO[k]


def prompt_id_for_scenario(scenario: str, rs_version: str = "v3") -> str:
    sc = canonical_scenario(scenario)
    ver = str(rs_version or "v3").strip().lower()
    if ver not in RS_WARM_PROMPT:
        raise ValueError(f"unknown rs_version {rs_version!r}; choose from {sorted(RS_WARM_PROMPT)}")
    if sc == "rs_warm":
        return RS_WARM_PROMPT[ver]
    if sc not in FIXED_SCENARIO_PROMPT:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {ALL_SCENARIOS}")
    return FIXED_SCENARIO_PROMPT[sc]


def prompt_id_for_prefix(prefix_k: int, rs_version: str = "v3") -> str:
    return prompt_id_for_scenario(scenario_for_prefix(prefix_k), rs_version)


def v13_gap_jobs() -> List[Tuple[int, str]]:
    """Jobs in v13 but not in v12 (ablation: collect/train only these additions)."""
    v12 = set(V12_MIX_JOBS)
    return [(int(k), sc) for k, sc in V13_MIX_JOBS if (int(k), sc) not in v12]


V13_GAP_JOBS: List[Tuple[int, str]] = v13_gap_jobs()


def mix_jobs_v13(
    *,
    rs_version: str = "v3",
) -> List[Tuple[int, str, str]]:
    """Return (prefix_k, scenario, prompt_id) rows for the v13 mix schedule."""
    ver = str(rs_version or "v3").strip().lower()
    return [
        (int(k), sc, prompt_id_for_scenario(sc, ver))
        for k, sc in V13_MIX_JOBS
    ]


def mix_jobs_v13_gap(
    *,
    rs_version: str = "v3",
) -> List[Tuple[int, str, str]]:
    """Return only the 9 v13−v12 gap jobs."""
    ver = str(rs_version or "v3").strip().lower()
    return [
        (int(k), sc, prompt_id_for_scenario(sc, ver))
        for k, sc in V13_GAP_JOBS
    ]


def mix_jobs_v14() -> List[Tuple[int, str, str]]:
    """Return (prefix_k, scenario, prompt_id) rows for the v14 schedule."""
    return _strict_jobs(V14_MIX_JOBS)


def mix_jobs_v15() -> List[Tuple[int, str, str]]:
    """Return (prefix_k, scenario, prompt_id) rows for the v15 schedule."""
    return _strict_jobs(V15_MIX_JOBS)


def _strict_jobs(jobs: List[Tuple[int, str]]) -> List[Tuple[int, str, str]]:
    rows = []
    for k, sc in jobs:
        pid = V14_RERANK_PROMPT if sc == "rerank" else FIXED_SCENARIO_PROMPT[sc]
        rows.append((int(k), sc, pid))
    return rows


def mix_jobs(
    prefixes: Optional[Sequence[int]] = None,
    *,
    rs_version: str = "v3",
    scenarios: Optional[Iterable[str]] = None,
) -> List[Tuple[int, str, str]]:
    """Return (prefix_k, scenario, prompt_id) rows for the mix schedule."""
    wanted = None
    if scenarios is not None:
        wanted = {canonical_scenario(s) for s in scenarios}
        unknown = wanted - set(ALL_SCENARIOS)
        if unknown:
            raise ValueError(f"unknown mix scenarios {sorted(unknown)}")
    rows: List[Tuple[int, str, str]] = []
    for k in list(prefixes if prefixes is not None else MIX_PREFIXES):
        sc = scenario_for_prefix(int(k))
        if wanted is not None and sc not in wanted:
            continue
        rows.append((int(k), sc, prompt_id_for_scenario(sc, rs_version)))
    return rows


def default_output_root(rs_version: str = "v3", *, tasks_tag: str = "light16") -> Path:
    ver = str(rs_version or "v3").strip().lower()
    return REPO_ROOT / "benchmark_data" / "rl_rollouts" / f"p5plus_grpo_{tasks_tag}_prompts_rsmix_{ver}"


def apply_prompt_to_optimizer(opt: Any, args: argparse.Namespace) -> None:
    """Copy prompt-variant fields onto a live PT4OptLLMOptimizer (per-prefix switch)."""
    from mindware.components.optimizers.experience_gate import ExperienceGate, ExplorePolicy
    from mindware.components.optimizers.llm_prompt_blocks import validate_prompt_blocks

    opt.prompt_blocks = validate_prompt_blocks(getattr(args, "llm_prompt_blocks", None) or {})
    opt.top_k = int(args.llm_top_k)
    opt.worst_k = int(args.llm_worst_k)
    opt.recent_k = int(args.llm_recent_k)
    opt.history_window = int(args.llm_history_window)
    opt.run_summary = bool(args.llm_run_summary)
    opt.experience_self_mode = str(args.llm_experience_self_mode or "cohort")
    opt.experience_self_start_trial = max(1, int(args.llm_experience_self_start_trial or 1))
    opt.experience_self_gate = str(getattr(args, "llm_experience_self_gate", None) or "clock").strip().lower()
    opt._experience_gate = ExperienceGate(mode=opt.experience_self_gate, cap=opt.experience_self_start_trial)
    opt.experience_explore_mode = str(args.llm_experience_explore_mode or "off").strip().lower()
    opt.experience_explore_prob = float(args.llm_experience_explore_prob or 0.5)
    opt._explore_policy = ExplorePolicy(
        mode=opt.experience_explore_mode,
        seed=int(getattr(opt, "seed", 1) or 1),
        p_default=opt.experience_explore_prob,
    )
    opt.experience_blame = dict(getattr(args, "llm_experience_blame", None) or {})
    opt.experience_retrieve_k = max(1, int(getattr(args, "llm_experience_retrieve_k", 3) or 3))
    opt.experience_retrieve_start_trial = max(1, int(getattr(args, "llm_experience_retrieve_start_trial", 1) or 1))
    opt.experience_retrieve_rerank_trial = max(
        1, int(getattr(args, "llm_experience_retrieve_rerank_trial", 24) or 24)
    )
    opt.experience_retrieve_ranker = str(
        getattr(args, "llm_experience_retrieve_ranker", "surrogate") or "surrogate"
    )
    opt.experience_retrieve_fade = bool(getattr(args, "llm_experience_retrieve_fade", False))
    stop = getattr(args, "llm_experience_retrieve_stop_trial", None)
    opt.experience_retrieve_stop_trial = None if stop in (None, "", 0, "0") else int(stop)
    floor = getattr(args, "llm_experience_retrieve_spearman_floor", None)
    opt.experience_retrieve_spearman_floor = None if floor in (None, "") else float(floor)
    opt.experience_retrieve_include_top_bottom = bool(
        getattr(args, "llm_experience_retrieve_include_top_bottom", False)
    )
    opt.experience_retrieve_include_prefer = bool(
        getattr(args, "llm_experience_retrieve_include_prefer", True)
    )
    opt.experience_retrieve_region_style = str(
        getattr(args, "llm_experience_retrieve_region_style", "prefer") or "prefer"
    )
    opt.experience_retrieve_tb_layout = str(
        getattr(args, "llm_experience_retrieve_tb_layout", "flat") or "flat"
    )
    _tk = getattr(args, "llm_experience_retrieve_top_k", 1)
    opt.experience_retrieve_top_k = 1 if _tk in (None, "") else max(0, int(_tk))
    _bk = getattr(args, "llm_experience_retrieve_bottom_k", 1)
    opt.experience_retrieve_bottom_k = 1 if _bk in (None, "") else max(0, int(_bk))
    fmt = getattr(args, "llm_response_format", None)
    if fmt:
        opt.response_format = str(fmt)
    opt.dataset_id = getattr(args, "dataset", opt.dataset_id)
    opt.task_type = "classification" if getattr(args, "task_type", "cls") == "cls" else "regression"
    opt._experience_bank = None
    opt._experience_bank_warned = False
    bank = getattr(args, "llm_experience_bank_path", None)
    if bank:
        bp = Path(bank)
        if not bp.is_absolute():
            bp = (REPO_ROOT / bp).resolve()
        opt.params["experience_bank_path"] = str(bp)
    else:
        opt.params.pop("experience_bank_path", None)
    opt.params["prompt_blocks"] = opt.prompt_blocks
    opt.params["prompt_variant"] = args.prompt_variant
    opt.params["experience_retrieve_ranker"] = opt.experience_retrieve_ranker
    opt.params["experience_retrieve_rerank_trial"] = opt.experience_retrieve_rerank_trial
    opt.params["experience_retrieve_spearman_floor"] = opt.experience_retrieve_spearman_floor
    opt.params["experience_retrieve_include_top_bottom"] = opt.experience_retrieve_include_top_bottom
    opt.params["experience_retrieve_include_prefer"] = opt.experience_retrieve_include_prefer
    opt.params["experience_retrieve_region_style"] = opt.experience_retrieve_region_style
    opt.params["experience_retrieve_tb_layout"] = opt.experience_retrieve_tb_layout
    opt.params["experience_retrieve_top_k"] = opt.experience_retrieve_top_k
    opt.params["experience_retrieve_bottom_k"] = opt.experience_retrieve_bottom_k
    opt.params["experience_explore_mode"] = opt.experience_explore_mode
    opt.params["experience_explore_prob"] = opt.experience_explore_prob
    opt.params["experience_self_start_trial"] = opt.experience_self_start_trial


def schedule_table(rs_version: str = "v3") -> str:
    lines = [
        f"rs_version={rs_version}",
        "k\tscenario\tprompt_id",
    ]
    for k, sc, pid in mix_jobs(rs_version=rs_version):
        lines.append(f"{k}\t{sc}\t{pid}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Print the GRPO mix collect schedule (no generation).")
    p.add_argument("--rs-version", choices=sorted(RS_WARM_PROMPT), default="v3")
    args = p.parse_args()
    print(schedule_table(args.rs_version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
