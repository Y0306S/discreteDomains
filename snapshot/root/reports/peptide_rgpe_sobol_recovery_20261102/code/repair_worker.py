"""Validated rank-sampling repair around the pinned merged RGPE worker."""
import argparse
import json
from pathlib import Path
import sys
import time
from sobol_repair import TOOLS, digest, install, verify


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sobol-repair', required=True)
    args, remaining = parser.parse_known_args()
    manifest = verify(args.sobol_repair)
    sys.path.insert(0, str(TOOLS))
    install()
    import peptide_rgpe_merged_worker as worker
    telemetry = Path(remaining[remaining.index('--telemetry')+1])
    worker.atomic_json(telemetry.with_name('sobol_repair.json'),
        dict(at=time.time(), manifest_sha256=digest(args.sobol_repair),
             implementation=manifest['format'], original_recipe_unchanged=True))
    sys.argv = [sys.argv[0]]+remaining
    worker.main()


if __name__ == '__main__':
    main()
