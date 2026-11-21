#!/usr/bin/env python3
"""Investigate s1a_ep7p5 DPO (s2_C / s2_G) eval regression vs s1a base and vs s1b DPO.

Training did not crash; this script focuses on *why eval hurt* despite JSON-only DPO data.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALGO40 = REPO / "benchmark_data" / "real_hpo_v2_mltest20_algo40"
TASKS = json.loads((ALGO40 / "tasks_40.json").read_text(encoding="utf-8"))

ROOTS = {
    "s1a_base": ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5",
    "s1a_s2_C": ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_s2_C",
    "s1a_s2_G": ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_s1a_s2_G",
    "s1b_dpo_C": ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_dpo_C",
    "s1b_dpo_G": ALGO40.parent / "real_hpo_v2_mltest20_algo40_p5_dpo_G",
}
DISK_VAR = "ptllm_tpe_qwen__P5__H0"
N_TRIALS = 80


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def find_best(root: Path, tt: str, ds: str) -> float | None:
    d = root / tt / ds / DISK_VAR
    if not d.is_dir():
        return None
    best_complete = None
    for run_dir in sorted(p for p in d.glob("CASH-*") if p.is_dir()):
        trajs = sorted(run_dir.glob("*_traj.csv"))
        if not trajs:
            continue
        rows = list(csv.DictReader(trajs[-1].open(newline="", encoding="utf-8")))
        if len(rows) < N_TRIALS:
            continue
        bests = [fnum(r.get("best_so_far")) for r in rows]
        bests = [b for b in bests if not math.isnan(b) and b != float("-inf")]
        if not bests:
            continue
        val = max(bests)
        if best_complete is None or val > best_complete:
            best_complete = val
    return best_complete


def load_dpo_sample(path: Path, n: int = 200) -> dict:
    chosen_lens = []
    rejected_lens = []
    hamming = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            chosen_lens.append(len(row["chosen"]["value"]))
            rejected_lens.append(len(row["rejected"]["value"]))
            hamming.append(row.get("metadata", {}).get("hamming"))
    n_read = len(chosen_lens)
    return {
        "n": n_read,
        "chosen_len_mean": sum(chosen_lens) / len(chosen_lens) if chosen_lens else 0,
        "rejected_len_mean": sum(rejected_lens) / len(rejected_lens) if rejected_lens else 0,
        "hamming_mean": sum(h for h in hamming if h is not None) / max(1, len([h for h in hamming if h is not None])),
    }


def trainer_final_loss(adapter_dir: Path) -> float | None:
    log = adapter_dir / "trainer_log.jsonl"
    if not log.is_file():
        return None
    last = None
    for line in log.open(encoding="utf-8"):
        row = json.loads(line)
        if "loss" in row:
            last = row["loss"]
    return last


def main() -> int:
    keys = [f"{t['task_type']}/{t['dataset']}/{t['algorithm']}" for t in TASKS]
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for name, root in ROOTS.items():
        for t in TASKS:
            k = f"{t['task_type']}/{t['dataset']}/{t['algorithm']}"
            b = find_best(root, t["task_type"], t["dataset"])
            if b is not None:
                scores[k][name] = b

    out_dir = ALGO40 / "analysis" / "s1a_dpo_investigation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-task deltas vs s1a base
    delta_rows = []
    wins_c = losses_c = ties_c = 0
    wins_g = losses_g = ties_g = 0
    missing_c = missing_g = 0
    for k in keys:
        base = scores[k].get("s1a_base")
        c = scores[k].get("s1a_s2_C")
        g = scores[k].get("s1a_s2_G")
        bC = scores[k].get("s1b_dpo_C")
        bG = scores[k].get("s1b_dpo_G")
        if base is None:
            continue
        row = {"task": k, "s1a_base": base}
        if c is not None:
            row["s1a_s2_C"] = c
            row["delta_C"] = c - base
            if c > base + 1e-9:
                wins_c += 1
            elif c < base - 1e-9:
                losses_c += 1
            else:
                ties_c += 1
        else:
            missing_c += 1
        if g is not None:
            row["s1a_s2_G"] = g
            row["delta_G"] = g - base
            if g > base + 1e-9:
                wins_g += 1
            elif g < base - 1e-9:
                losses_g += 1
            else:
                ties_g += 1
        else:
            missing_g += 1
        if bC is not None:
            row["s1b_dpo_C"] = bC
        if bG is not None:
            row["s1b_dpo_G"] = bG
        delta_rows.append(row)

    delta_path = out_dir / "per_task_vs_s1a_base.csv"
    with delta_path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in delta_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(delta_rows)

    dpo_path = Path("/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo.jsonl")
    dpo_stats = load_dpo_sample(dpo_path) if dpo_path.is_file() else {}

    weights = {
        "s1a_dpo_C": Path("/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_C"),
        "s1a_dpo_G": Path("/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G"),
        "s1b_dpo_C": Path("/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_C"),
        "s1b_dpo_G": Path("/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_G"),
    }
    train_loss = {k: trainer_final_loss(p) for k, p in weights.items()}

    report = {
        "summary": {
            "s1a_s2_C_vs_base": {"wins": wins_c, "losses": losses_c, "ties": ties_c, "missing_tasks": missing_c},
            "s1a_s2_G_vs_base": {"wins": wins_g, "losses": losses_g, "ties": ties_g, "missing_tasks": missing_g},
            "coverage": {name: sum(1 for k in keys if name in scores[k]) for name in ROOTS},
        },
        "dpo_json_only_data": dpo_stats,
        "final_dpo_train_loss": train_loss,
        "hypotheses": [
            "DPO training completed (loss~0, pref accuracy~1); failure is eval/regression not OOM/crash.",
            "DPO JSONL uses s1a_bc system/user prompts for all arms (not s1b_user_prompt); only assistant targets differ JSON vs rationale+JSON export.",
            "s1a DPO may overfit preference on short JSON strings while SFT never saw rationale tokens — distribution shift is mild for JSON body but DPO still shifts logits.",
            "s1b_ep7p5 + same DPO often ranks better on algo40 — prior capacity / SFT quality may dominate DPO benefit.",
            "s1a_s2_C missing 2/40 tasks (colleges/extra_trees, jasmine/gradient_boosting) — hurts aggregate rank.",
            "Replication s1a_ep7p5_r2 (seed 2) beats s1a seed-1 — TPE stochasticity may exceed DPO effect for s1a json-only stack.",
            "pref_ftx=0.1 (C) mixes SFT; ORPO-G on s1a uses 1 epoch — check if both still collapse preference loss without improving generalization.",
        ],
        "artifacts": {
            "per_task_csv": str(delta_path),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md = out_dir / "README.md"
    md.write_text(
        "# s1a DPO investigation\n\n"
        f"- Per-task deltas: `{delta_path.name}`\n"
        f"- Machine-readable: `{report_path.name}`\n\n"
        "## Headline counts (vs s1a_ep7p5 base)\n\n"
        f"- **s2_C**: {wins_c} wins / {losses_c} losses / {ties_c} ties "
        f"({missing_c} missing tasks)\n"
        f"- **s2_G**: {wins_g} wins / {losses_g} losses / {ties_g} ties "
        f"({missing_g} missing tasks)\n\n"
        "## Training loss at end\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in train_loss.items())
        + "\n\nSee `report.json` hypotheses.\n",
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {report_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
