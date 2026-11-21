#!/usr/bin/env python3
"""Archive a completed backfill while retaining the controller's inherited lock."""

import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--campaign', type=Path, required=True)
parser.add_argument('--job', required=True)
parser.add_argument('--lock-fd', type=int, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.campaign / 'code'))
sys.path.insert(0, str(args.campaign / 'code/examples/benchmark/domains'))
from mindware.domains.taskset_archive import pack_job
from run_taskset_cpu_campaign import inspect_history

jobs = json.loads((args.campaign / 'distributed/plan.json').read_text())['jobs']
job = next(j for j in jobs if j['id'] == args.job)
with os.fdopen(args.lock_fd, 'a') as lock:
    if inspect_history(job, validate=True)['n_observations'] != 100:
        raise ValueError('Backfill not complete')
    print(json.dumps(pack_job(args.campaign, job, cleanup=True, lock=lock)))
