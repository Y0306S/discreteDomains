#!/usr/bin/env python3
"""Train ESM-2 + MLP fitness surrogate on ProteinGym scored splits (GPU 0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.protein.surrogate import train_surrogate_head


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landscape", choices=["gfp", "aav"], required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "history_data/domain_protein/proteingym",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model-id", default="facebook/esm2_t12_35M_UR50D")
    args = p.parse_args()

    landscape_dir = args.data_root / args.landscape
    if not (landscape_dir / "sequences_scored.csv").is_file():
        raise SystemExit(f"missing scored CSV in {landscape_dir}")
    report = train_surrogate_head(
        landscape_dir,
        device=args.device,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps({"done": True, **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
