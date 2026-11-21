#!/usr/bin/env python3
"""Convert zaxie tune_report.json TPE logs → OpenBox History JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo.zaxie import convert_tree
from mindware.domains.paths import ZAXIE_ROOT, domain_history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=ZAXIE_ROOT)
    p.add_argument("--out", type=Path, default=domain_history("dl"))
    p.add_argument("--min-scored", type=int, default=10)
    p.add_argument("--max-tasks", type=int, default=0)
    args = p.parse_args(argv)
    written = convert_tree(args.src, args.out, min_scored=args.min_scored, max_tasks=args.max_tasks)
    print(f"Wrote {len(written)} DL histories under {args.out}")
    for path in written[:12]:
        print(" ", path)
    if len(written) > 12:
        print(f"  ... {len(written) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
