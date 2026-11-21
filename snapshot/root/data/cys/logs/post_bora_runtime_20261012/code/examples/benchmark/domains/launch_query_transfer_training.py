#!/usr/bin/env python3
"""Prepare full CEB GP training; --execute is required to launch verified jobs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from query_transfer_common import atomic_json, digest, load_corpus, run_lock

ENV_ROOT = Path('/root/data/cys/baseline/envs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.corpus, args.run_dir = args.corpus.resolve(), args.run_dir.resolve()
    manifest, _ = load_corpus(args.corpus)
    trainer = Path(__file__).with_name('train_query_transfer.py').resolve()
    contract = Path(__file__).with_name('query_transfer_common.py').resolve()
    pending = []
    for method in ('fsbo', 'rgpe'):
        status_path = args.checks / method / 'status.json'
        if not status_path.is_file():
            pending.append(method + ': fit/reload component check not run')
            continue
        status = json.loads(status_path.read_text())
        if (status['status'] != 'complete' or status['identity']['purpose'] != 'component_check'
                or status['identity']['corpus_sha256'] != digest(args.corpus / 'manifest.json')
                or status['identity']['trainer_sha256'] != digest(trainer)
                or status['identity']['contract_sha256'] != digest(contract)):
            pending.append(method + ': component check does not match current corpus/trainer')
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
    if limit - current < 32 * 1024 ** 3:
        pending.append('Less than 32 GiB RAM headroom')
    record = dict(domain='ceb_query_runtime', corpus=str(args.corpus),
                  corpus_sha256=digest(args.corpus / 'manifest.json'),
                  source_count=len(manifest['source_tasks']), observations=manifest['observations'],
                  vae_checkpoint=manifest['vae_path'], vae_sha256=manifest['vae_sha256'],
                  ram_headroom_gib=(limit - current) / 1024 ** 3,
                  execute=args.execute, pending=pending, jobs={})
    for method, env_name, cpus in [('fsbo', 'bl_fsbo_exact', '42-43'), ('rgpe', 'bl_rgpe', '40-41')]:
        output = args.run_dir / method
        command = ['taskset', '-c', cpus, 'nice', '-n', '15',
                   str(ENV_ROOT / env_name / 'bin/python'), str(trainer), method,
                   '--corpus', str(args.corpus), '--output', str(output)]
        record['jobs'][method] = dict(cpus=cpus, threads=2, nice=15, command=command,
                                      log=str(args.run_dir / (method + '.log')), output=str(output))
    print(json.dumps(record, indent=2), flush=True)
    if not args.execute:
        return
    if pending:
        raise RuntimeError('Training launch blocked: ' + '; '.join(pending))
    with run_lock(args.run_dir):
        record_path = args.run_dir / 'launch.json'
        if record_path.exists():
            raise ValueError('Launch record already exists; inspect processes before resuming')
        for method, job in record['jobs'].items():
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                               PYTHONUNBUFFERED='1', PYTHONHASHSEED='0')
            with Path(job['log']).open('xb') as log:
                process = subprocess.Popen(job['command'], env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job['pid'] = process.pid
            atomic_json(record_path, record)
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
