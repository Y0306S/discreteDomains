#!/usr/bin/env python3
"""Run the unchanged frozen local wrapper with a validated RGPE acquisition amendment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from peptide_rgpe_merged_policy import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amendment', required=True, type=Path)
    args, remaining = parser.parse_known_args()
    amendment = verify(args.amendment)
    import peptide_local_bo as runner
    original_client = runner.transfer.ModelClient
    class MergedClient(original_client):
        def __init__(self, recipe_path, recipe):
            if recipe['method'] != 'rgpe':
                raise ValueError('Acquisition amendment is RGPE only')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            self.child = subprocess.Popen([recipe['model_python'],
                str(Path(__file__).with_name('peptide_rgpe_merged_worker.py')),
                '--amendment', str(args.amendment),
                '--telemetry', str(Path(amendment['campaign'])/'runs'/task/'rgpe_acquisition.json'),
                '--recipe', str(recipe_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            self.last_sample = None
    if (not remaining or remaining[0] != 'run-task'
            or '--campaign' not in remaining or '--task' not in remaining
            or Path(remaining[remaining.index('--campaign')+1]).resolve() != Path(amendment['campaign']).resolve()):
        raise ValueError('Launcher requires run-task in the amended campaign')
    task = remaining[remaining.index('--task')+1]
    runner.transfer.ModelClient = MergedClient
    sys.argv = [runner.__file__]+remaining
    runner.main()


if __name__ == '__main__':
    main()
