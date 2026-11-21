#!/usr/bin/env python3
"""Versioned RGPE model entry: preserve fitting/ranking, replace acquisition only."""
import argparse
import json
from pathlib import Path
import sys
import time

import peptide_transfer_models as original
from peptide_rgpe_merged import VERSION, sample_ensemble
from peptide_rgpe_merged_policy import digest, verify


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class MergedRGPEBackend(original.RGPEBackend):
    def sample(self, candidates, count, seed):
        if self.ensemble is None:
            raise ValueError('RGPE target is not fitted')
        return sample_ensemble(self.ensemble, candidates, count, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amendment', required=True, type=Path)
    parser.add_argument('--telemetry', required=True, type=Path)
    args, remaining = parser.parse_known_args()
    amendment = verify(args.amendment)
    recipe = json.loads((Path(amendment['campaign'])/'recipe.json').read_text())
    if remaining != ['--recipe', recipe['parent_recipe']]:
        raise ValueError('Worker recipe differs from the amended campaign parent')
    # Also verify the inherited code. Source payloads retain per-load checks.
    parent = json.loads(Path(recipe['parent_recipe']).read_text())
    for name, expected in parent['code'].items():
        if digest(name) != expected:
            raise ValueError('Inherited RGPE code changed: '+name)
    original.RGPEBackend = MergedRGPEBackend
    original_send = original.send
    def send(stream, value):
        result = value.get('result')
        if isinstance(result, dict) and 'draws' in result:
            atomic_json(args.telemetry, dict(at=time.time(), info=result['info']))
        return original_send(stream, value)
    original.send = send
    atomic_json(args.telemetry.with_name('rgpe_worker.json'),
                dict(at=time.time(), implementation=VERSION,
                     amendment_sha256=digest(args.amendment), code_sha256=amendment['code']))
    sys.argv = [sys.argv[0]]+remaining
    original.main()


if __name__ == '__main__':
    main()
