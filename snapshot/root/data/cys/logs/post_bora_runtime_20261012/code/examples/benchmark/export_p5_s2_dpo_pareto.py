#!/usr/bin/env python3
"""Rebuild P5 S2 DPO pairs with mixed-similarity + worse-than-x4 Pareto selection.

For each improvement event (prefix x1..x3, chosen x4):
  - Candidate pool = successful BO configs whose identity is NOT in {x1..x4}
  - Keep only configs strictly worse than x4 (higher-is-better score)
  - Score A: mixed HP similarity to x4 in [0, 1] (↑ better)
  - Score B: how much worse than x4, min-max normalized in-pool to [0, 1] (↑ better)
  - Keep Pareto front on (A, B); pick argmax of 0.5*A + 0.5*B
  - Emit LlamaFactory sharegpt ranking JSONL (chosen=x4, rejected=pick)

Does not require rejected to be temporally after x4.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

DEFAULT_EVENTS = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s1b_events.jsonl"
)
DEFAULT_PROMPTS = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s1a_bc.jsonl"
)
DEFAULT_OUT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo.jsonl"
)
DEFAULT_TRAIN = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_train.jsonl"
)
DEFAULT_DATASET_INFO = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/llamafactory_data/dataset_info.json"
)


def cfg_only(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "algorithm"}


def cfg_key(d: dict[str, Any]) -> str:
    c = cfg_only(d)
    return json.dumps(c, sort_keys=True, ensure_ascii=True, default=str)


def score_of(obs: dict[str, Any]) -> float:
    # OpenBox objectives are minimize(-metric) for higher-is-better metrics.
    return -float(obs["objectives"][0])


def load_success_obs(history_path: str, cache: dict[str, list[dict]]) -> list[dict]:
    if history_path in cache:
        return cache[history_path]
    hist = json.loads(Path(history_path).read_text(encoding="utf-8"))
    obs = [
        o
        for o in hist.get("observations", [])
        if o.get("trial_state") in (0, "0") and o.get("objectives")
    ]
    cache[history_path] = obs
    return obs


def parse_search_space(user: str) -> dict[str, dict[str, Any]]:
    """Parse HP specs from the P5 user prompt Search space block."""
    out: dict[str, dict[str, Any]] = {}
    if "Search space:" not in user:
        return out
    block = user.split("Search space:", 1)[1]
    # Stop at next blank-line section header-ish content
    lines = []
    for ln in block.splitlines():
        if not ln.strip():
            if lines:
                break
            continue
        if ln.startswith("Previous ") or ln.startswith("Selected historical"):
            break
        lines.append(ln)
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("- "):
            continue
        body = ln[2:]
        m = re.match(
            r"^(.+?):\s*(categorical|float|integer|constant)\b(.*)$",
            body,
            flags=re.I,
        )
        if not m:
            continue
        name, kind, rest = m.group(1).strip(), m.group(2).lower(), m.group(3)
        spec: dict[str, Any] = {"kind": kind}
        if kind == "categorical":
            cm = re.search(r"choices=\[(.*?)\]", rest)
            if cm:
                choices = []
                for part in re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", cm.group(1)):
                    val = part[0] or part[1] or part[2]
                    if val is not None and val != "":
                        choices.append(val)
                spec["choices"] = choices
        elif kind in ("float", "integer"):
            rm = re.search(r"range=\[([^,\]]+),\s*([^\]]+)\]", rest)
            if rm:
                lo, hi = float(rm.group(1)), float(rm.group(2))
                spec["low"] = lo
                spec["high"] = hi
                spec["log"] = "log_scale=True" in rest or "log_scale=true" in rest
        out[name] = spec
    return out


def mixed_distance(
    a: dict[str, Any],
    b: dict[str, Any],
    space: dict[str, dict[str, Any]],
) -> float:
    """Per-HP distance in [0,1], averaged over union of keys (excl. algorithm)."""
    keys = sorted((set(a) | set(b)) - {"algorithm"})
    if not keys:
        return 0.0
    dists = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va == vb:
            dists.append(0.0)
            continue
        spec = space.get(k) or {}
        kind = spec.get("kind")
        if kind == "categorical" or kind == "constant":
            dists.append(1.0)
            continue
        if kind in ("float", "integer"):
            try:
                fa, fb = float(va), float(vb)
            except (TypeError, ValueError):
                dists.append(1.0)
                continue
            lo, hi = spec.get("low"), spec.get("high")
            if lo is None or hi is None or hi <= lo:
                dists.append(1.0)
                continue
            if spec.get("log") and lo > 0 and fa > 0 and fb > 0:
                span = math.log(hi) - math.log(lo)
                d = abs(math.log(fa) - math.log(fb)) / span if span > 0 else 1.0
            else:
                d = abs(fa - fb) / (hi - lo)
            dists.append(float(min(1.0, max(0.0, d))))
            continue
        # Unknown HP type: exact mismatch
        dists.append(1.0)
    return sum(dists) / len(dists)


def similarity(a: dict, b: dict, space: dict) -> float:
    return 1.0 - mixed_distance(a, b, space)


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated set for maximize (sim, worse_norm)."""
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (
                q["sim"] >= p["sim"]
                and q["worse_norm"] >= p["worse_norm"]
                and (q["sim"] > p["sim"] or q["worse_norm"] > p["worse_norm"])
            ):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def assistant_config_json(cfg: dict[str, Any]) -> str:
    return json.dumps({"config": cfg_only(cfg)}, ensure_ascii=False)


def to_sharegpt_dpo(system: str, user: str, chosen: str, rejected: str, metadata: dict) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human", "value": user},
        ],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
        "metadata": metadata,
    }


def to_train_row(row: dict) -> dict:
    return {
        "conversations": row["conversations"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }


def register_dataset(dataset_info: Path, name: str, file_name: str) -> None:
    info: dict[str, Any] = {}
    if dataset_info.is_file():
        info = json.loads(dataset_info.read_text(encoding="utf-8"))
    info[name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    dataset_info.parent.mkdir(parents=True, exist_ok=True)
    dataset_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def build_pair(
    event: dict[str, Any],
    prompt_row: dict[str, Any],
    hist_cache: dict[str, list[dict]],
) -> tuple[dict | None, str]:
    meta = event["metadata"]
    obs = load_success_obs(meta["history_path"], hist_cache)
    t = int(meta["improvement_trial"]) - 1  # 0-based index of x4
    if t < 0 or t >= len(obs):
        return None, "bad_improvement_trial"
    x4_obs = obs[t]
    x4 = cfg_only(x4_obs["config"])
    target = cfg_only(event["target_config"])
    if cfg_key(x4) != cfg_key(target):
        return None, "target_mismatch"
    x4_score = score_of(x4_obs)

    # Group x1..x4 = successful trials at indices 0..t
    group_keys = {cfg_key(o["config"]) for o in obs[: t + 1]}

    user = prompt_row["messages"][1]["content"]
    system = prompt_row["messages"][0]["content"]
    space = parse_search_space(user)

    # Pool: any successful config not in {x1..x4}, strictly worse than x4
    pool = []
    for i, o in enumerate(obs):
        key = cfg_key(o["config"])
        if key in group_keys:
            continue
        sc = score_of(o)
        if not (sc < x4_score - 1e-12):
            continue  # drop if better than or equal to x4
        cfg = cfg_only(o["config"])
        pool.append({"idx": i, "config": cfg, "score": sc})

    if not pool:
        return None, "empty_pool"

    gaps = [x4_score - p["score"] for p in pool]
    gmin, gmax = min(gaps), max(gaps)
    for p, g in zip(pool, gaps):
        p["sim"] = similarity(p["config"], x4, space)
        p["gap"] = g
        p["worse_norm"] = 1.0 if gmax <= gmin + 1e-15 else (g - gmin) / (gmax - gmin)
        p["weighted"] = 0.5 * p["sim"] + 0.5 * p["worse_norm"]

    front = pareto_front(pool)
    pick = max(front, key=lambda p: (p["weighted"], p["sim"], p["worse_norm"]))

    hamming = sum(
        1
        for k in set(x4) | set(pick["config"])
        if x4.get(k) != pick["config"].get(k)
    )
    row = to_sharegpt_dpo(
        system,
        user,
        assistant_config_json(x4),
        assistant_config_json(pick["config"]),
        {
            "stage": "S2",
            "task_type": meta["task_type"],
            "dataset": meta["dataset"],
            "algorithm": meta["algorithm"],
            "metric": meta["metric"],
            "improvement_trial": meta["improvement_trial"],
            "delta_rel_score": meta.get("delta_rel_score"),
            "sample_weight": meta.get("sample_weight"),
            "chosen_config": x4,
            "rejected_config": pick["config"],
            "chosen_score": x4_score,
            "rejected_score": pick["score"],
            "hamming": hamming,
            "sim": pick["sim"],
            "worse_norm": pick["worse_norm"],
            "weighted": pick["weighted"],
            "pareto_size": len(front),
            "pool_size": len(pool),
            "rejected_trial_index_0based": pick["idx"],
            "selection": "pareto_weighted_0.5sim_0.5worse",
            "history_path": meta["history_path"],
        },
    )
    return row, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--register-name", type=str, default="p5_s2_dpo")
    args = parser.parse_args()

    prompts: dict[tuple, dict] = {}
    with args.prompts.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = row["metadata"]
            key = (
                m["task_type"],
                m["dataset"],
                m["algorithm"],
                int(m["improvement_trial"]),
            )
            prompts[key] = row

    rows: list[dict] = []
    skips: dict[str, int] = {}
    hist_cache: dict[str, list[dict]] = {}

    with args.events.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            m = event["metadata"]
            key = (
                m["task_type"],
                m["dataset"],
                m["algorithm"],
                int(m["improvement_trial"]),
            )
            prompt_row = prompts.get(key)
            if prompt_row is None:
                skips["missing_prompt"] = skips.get("missing_prompt", 0) + 1
                continue
            pair, reason = build_pair(event, prompt_row, hist_cache)
            if pair is None:
                skips[reason] = skips.get(reason, 0) + 1
                continue
            rows.append(pair)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.train_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_train_row(row), ensure_ascii=False) + "\n")

    sims = [r["metadata"]["sim"] for r in rows]
    hamm = [r["metadata"]["hamming"] for r in rows]
    pareto_sizes = [r["metadata"]["pareto_size"] for r in rows]
    manifest = {
        "pairs": len(rows),
        "skips": skips,
        "sim_mean": (sum(sims) / len(sims)) if sims else None,
        "hamming_mean": (sum(hamm) / len(hamm)) if hamm else None,
        "hamming_min": min(hamm) if hamm else None,
        "hamming_max": max(hamm) if hamm else None,
        "pareto_size_mean": (sum(pareto_sizes) / len(pareto_sizes)) if pareto_sizes else None,
        "out": str(args.out),
        "train_out": str(args.train_out),
        "format": "sharegpt_ranking_dpo",
        "selection": "pool=not_in_{x1..x4}; worse_than_x4; mixed_sim; "
        "pareto(sim,worse_norm); pick=0.5*sim+0.5*worse_norm",
        "note": "rejected may be before or after x4; must not be in {x1..x4} and must score < x4",
    }
    man_path = args.out.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    register_dataset(args.dataset_info, args.register_name, f"../data/{args.train_out.name}")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.train_out}")
    print(f"Wrote {man_path}")
    print(f"Registered {args.register_name} -> {args.train_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
