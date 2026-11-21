#!/usr/bin/env python3
"""Build fresh, manifest-checked training-source GP models for RS."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mindware.components.experience.source_gp import build_artifact, ML_TRAIN_80_FIXED_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "train-manifest", "test-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = build_artifact(args.bank, args.train_manifest, args.test_manifest, args.output,
                            expected_train_sha256=ML_TRAIN_80_FIXED_SHA256, workers=args.workers)
    print(f"Built {len(result['sources'])} source entries in {args.output}")


if __name__ == "__main__":
    main()
