#!/usr/bin/env python3
"""Export P16 Optimizer DPO pairs (CPU-only; no GPU).

For each improvement event with an Analysis Agent report:
  1. Anchor on the BO improving (best) config + its evaluated score.
  2. Find an evaluated neighbor that differs on HPs NOT named in the analysis
     PRIMARY ACTION / SPECIFIC RECOMMENDATION (JSON dumps in sec.4 are stripped
     so a full-config paste does not mark every HP as "recommended").
  3. Prefer Hamming distance 1 (one-HP tweak). Optionally allow up to --max-hamming.
  4. Compare scores from BO history. If the neighbor is better, swap so
     chosen is always the higher-scoring config.

Writes LlamaFactory sharegpt ranking format (conversations + chosen + rejected).
Prompt context is reused from p16_optimizer_sft.jsonl (TCS + analysis).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DATA = Path("/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")
DEFAULT_EVENTS = DEFAULT_DATA / "p16_improvement_events_with_analysis.jsonl"
DEFAULT_SFT = DEFAULT_DATA / "p16_optimizer_sft.jsonl"
DEFAULT_OUT = DEFAULT_DATA / "p16_s2_dpo.jsonl"
DEFAULT_DATASET_INFO = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/llamafactory_data/dataset_info.json"
)


def cfg_for(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "algorithm"}


def hamming(a: dict, b: dict) -> int:
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


def diff_keys(a: dict, b: dict) -> list[str]:
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))


def final_analysis_body(analysis: str) -> str:
    matches = list(re.finditer(r"1\.\s*PROBLEM DIAGNOSIS\s*:", analysis or "", flags=re.I))
    return analysis[matches[-1].start() :] if matches else (analysis or "")


def recommended_hps(analysis: str, search_hps: list[str]) -> set[str]:
    """HPs stated in PRIMARY ACTION + actionable language in SPECIFIC RECOMMENDATION.

    Full JSON config blobs in section 4 are stripped so they do not mark every HP.
    """
    body = final_analysis_body(analysis)
    blob = ""
    sec3 = re.search(
        r"3\.\s*PRIMARY ACTION\s*:\s*(.+?)(?:\n\s*4\.|\Z)",
        body,
        flags=re.I | re.S,
    )
    if sec3:
        blob += sec3.group(1)
    sec4 = re.search(
        r"4\.\s*SPECIFIC RECOMMENDATION\s*:\s*(.+?)(?:\n\s*5\.|\Z)",
        body,
        flags=re.I | re.S,
    )
    out: set[str] = set()
    if sec4:
        chunk = re.sub(r"\{[^{}]*\}", " ", sec4.group(1))
        blob += "\n" + chunk
        for m in re.finditer(
            r"(?:Set|Change|Increase|Decrease|Reduce|Raise|Lower)\s+"
            r"([A-Za-z0-9_]+:[A-Za-z0-9_]+)",
            chunk,
            flags=re.I,
        ):
            out.add(m.group(1))
    for hp in search_hps:
        if hp in blob:
            out.add(hp)
            continue
        short = hp.split(":", 1)[-1]
        if len(short) >= 3 and re.search(
            rf"(?<![\w:]){re.escape(short)}(?![\w:])", blob, flags=re.I
        ):
            out.add(hp)
    return out


def search_hps_from_user(user: str) -> list[str]:
    hps: list[str] = []
    block = user.split("Search space:", 1)
    if len(block) < 2:
        return hps
    rest = block[1].split("\n\n", 1)[0]
    for ln in rest.splitlines():
        ln = ln.strip()
        if not ln.startswith("- "):
            continue
        m = re.match(
            r"^(.+?):\s*(categorical|float|integer|constant)\b",
            ln[2:],
            flags=re.I,
        )
        if m:
            hps.append(m.group(1).strip())
    return hps


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


def score_of(o: dict) -> float:
    # OpenBox stores objectives as minimize(-metric) for higher-is-better metrics.
    return -float(o["objectives"][0])


def assistant_config_json(cfg: dict) -> str:
    return json.dumps({"config": cfg}, ensure_ascii=False)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-hamming", type=int, default=1, help="Prefer 1 (single-HP tweak).")
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--register-name", type=str, default="p16_s2_dpo")
    args = parser.parse_args()

    sft_by_key: dict[tuple, dict] = {}
    with args.sft.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = row["metadata"]
            key = (
                m["task_type"],
                m["dataset"],
                m["algorithm"],
                int(m["improvement_trial"]),
            )
            sft_by_key[key] = row

    rows: list[dict] = []
    skip = Counter()
    hamm = Counter()
    swaps = 0
    hist_cache: dict[str, list[dict]] = {}

    with args.events.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            meta = event["metadata"]
            key = (
                meta["task_type"],
                meta["dataset"],
                meta["algorithm"],
                int(meta["improvement_trial"]),
            )
            sft = sft_by_key.get(key)
            if not sft:
                skip["no_sft"] += 1
                continue

            system = sft["messages"][0]["content"]
            user = sft["messages"][1]["content"]
            search_hps = search_hps_from_user(user) or list(
                cfg_for(event["target_config"]).keys()
            )
            rec = recommended_hps(event.get("analysis_report") or "", search_hps)

            target = cfg_for(event["target_config"])
            target_score = float(meta["improving_perf"])
            obs = load_success_obs(meta["history_path"], hist_cache)

            cands: list[tuple[float, dict, int, list[str]]] = []
            for o in obs:
                cfg = cfg_for(o["config"])
                if cfg == target:
                    continue
                d = hamming(target, cfg)
                if d < 1 or d > args.max_hamming:
                    continue
                dks = diff_keys(target, cfg)
                # Differ only on HPs that analysis did NOT recommend changing.
                if any(k in rec for k in dks):
                    continue
                cands.append((score_of(o), cfg, d, dks))

            if not cands:
                skip["no_cand"] += 1
                continue

            worse = [c for c in cands if c[0] < target_score - 1e-12]
            if worse:
                worse.sort(key=lambda x: (x[2], -x[0]))  # closer Hamming, then closer score
                sc, cfg, d, dks = worse[0]
                chosen_cfg, rejected_cfg = target, cfg
                chosen_score, rejected_score = target_score, sc
                swapped = False
            else:
                # Neighbor scored better — swap places.
                cands.sort(key=lambda x: (x[2], -x[0]))
                sc, cfg, d, dks = cands[0]
                chosen_cfg, rejected_cfg = cfg, target
                chosen_score, rejected_score = sc, target_score
                swapped = True
                swaps += 1

            assert chosen_score >= rejected_score - 1e-12
            hamm[d] += 1
            rows.append(
                to_sharegpt_dpo(
                    system,
                    user,
                    assistant_config_json(chosen_cfg),
                    assistant_config_json(rejected_cfg),
                    metadata={
                        "stage": "P16_S2_DPO",
                        "task_type": meta["task_type"],
                        "dataset": meta["dataset"],
                        "algorithm": meta["algorithm"],
                        "metric": meta.get("metric"),
                        "improvement_trial": meta["improvement_trial"],
                        "delta_rel_score": meta.get("delta_rel_score"),
                        "chosen_config": chosen_cfg,
                        "rejected_config": rejected_cfg,
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "hamming": d,
                        "diff_keys": dks,
                        "analysis_recommended_hps": sorted(rec),
                        "swapped": swapped,
                        "selection": (
                            "hamming_near_miss_non_recommended_hp;"
                            "scores_from_BO_history;swap_if_neighbor_better"
                        ),
                    },
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Slim train file (no metadata) for LlamaFactory.
    train_out = args.out.with_name(args.out.stem + "_train.jsonl")
    with train_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "conversations": row["conversations"],
                        "chosen": row["chosen"],
                        "rejected": row["rejected"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    register_dataset(
        args.dataset_info,
        args.register_name,
        f"../data/{train_out.name}",
    )

    hammings = [r["metadata"]["hamming"] for r in rows]
    manifest = {
        "pairs": len(rows),
        "swaps": swaps,
        "skip": dict(skip),
        "hamming_counts": {str(k): v for k, v in sorted(hamm.items())},
        "hamming_mean": (sum(hammings) / len(hammings)) if hammings else None,
        "max_hamming": args.max_hamming,
        "out": str(args.out),
        "train_out": str(train_out),
        "dataset_name": args.register_name,
        "note": (
            "chosen/rejected ordered by BO score; rejected differs on HPs not "
            "named in analysis PRIMARY ACTION / actionable SPECIFIC RECOMMENDATION"
        ),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))

    # Spot-check a few
    for i, row in enumerate(rows[:3]):
        m = row["metadata"]
        print(
            f"\nspot[{i}] {m['dataset']}/{m['algorithm']} trial={m['improvement_trial']} "
            f"hamming={m['hamming']} swapped={m['swapped']} diffs={m['diff_keys']}"
        )
        print(f"  rec_hps={m['analysis_recommended_hps']}")
        print(f"  scores chosen={m['chosen_score']:.6f} rejected={m['rejected_score']:.6f}")


if __name__ == "__main__":
    main()
