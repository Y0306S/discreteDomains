#!/usr/bin/env python3
"""Export peptide SFT ablation arm B: round-10 batch that first hit global best MIC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.benchmark.peptide_sft_common import DEFAULT_BO_ROOT, export_arm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Export peptide SFT winbatch10 train/eval JSONL.")
    p.add_argument("--bo-root", type=Path, default=DEFAULT_BO_ROOT)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "benchmark_data" / "rl_task_selection" / "peptide_sft",
    )
    p.add_argument(
        "--split-mode",
        choices=("stratified10", "consistent10_plus90", "last100"),
        default="stratified10",
        help="stratified10: eval pep_000,010,...,990 (900 train / 100 eval). "
        "consistent10_plus90: CONSISTENT10 + 90 random held-out. last100: pep_900-999 eval.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    manifest = export_arm(
        arm="winbatch10",
        bo_root=args.bo_root,
        out_dir=args.out_dir,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
