#!/usr/bin/env python3
"""Export ml_train_80 classical BO dumps for FSBO / RGPE. Does not train anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.components.optimizers.transfer_hpo.history_export import export_all


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_data" / "ml_train_80_transfer_v2",
    )
    p.add_argument(
        "--hist-root",
        type=Path,
        default=Path("/root/blockdata/llmoptimizer_benchmark/history_data/history_data"),
    )
    p.add_argument("--min-obs", type=int, default=10)
    args = p.parse_args()
    summary = export_all(args.out, hist_root=args.hist_root, min_obs=args.min_obs)
    print(json.dumps({k: summary[k] for k in summary if k != "groups"}, indent=2))
    print(f"{'group':24} {'train':>5} {'src':>5} {'n80':>5} {'miss':>5} {'obs':>7} {'dim':>4} ready variants")
    for row in summary["groups"]:
        print(
            f"{row['group']:24} {row['n_train_slots']:5d} {row['n_source_tasks']:5d} "
            f"{row.get('n_full_80', 0):5d} {row['n_missing_history']:5d} {row['n_obs_total']:7d} {row['dim']:4d} "
            f"{str(row['ready_for_meta_train']):5} {row.get('source_variants') or {}}"
        )
    print("wrote", args.out / "coverage.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
