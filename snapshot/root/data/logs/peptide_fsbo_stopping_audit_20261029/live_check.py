"""Verify resumed worker identities, CPU allocation and new fit provenance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
import queue_peptide_transfer as queue


def main(side):
    case = queue.CAMPAIGN / 'fsbo'
    expected = hashlib.sha256((case / 'target_fit_amendment.json').read_bytes()).hexdigest()
    status = queue.read(queue.CAMPAIGN / 'workers' / ('queue_' + side + '.json'))
    issues, new_fits = [], []
    cpus = []
    for row in status['active']:
        current = queue.process(row['pid'])
        if not current or current['start'] != row['process_start']:
            issues.append('Stopped or replaced worker: ' + row['task'])
            continue
        if set(os.sched_getaffinity(row['pid'])) != {row['cpu']}:
            issues.append('Incorrect CPU affinity: ' + row['task'])
        cpus.append(row['cpu'])
        history = queue.read(case / 'runs' / row['task'] / 'history.json', {})
        fits = [r['fit'] for r in history.get('observations', []) if r.get('fit')]
        if fits and fits[-1].get('amendment_sha256'):
            fit = fits[-1]
            if (fit['amendment_sha256'] != expected or fit['maximum_updates'] != 1000
                    or fit['optimizer_updates'] > 1000
                    or abs(fit['best_loss'] - fit['restored_loss']) > 0.00001):
                issues.append('Incorrect amended fit: ' + row['task'])
            new_fits.append(dict(task=row['task'], proposals=history['proposals'], fit=fit))
    if len(cpus) != len(set(cpus)):
        issues.append('Independent workers share a CPU core')
    expected_count = 22 if side == 'local' else 26
    if len(status['active']) != expected_count:
        issues.append('Not all existing worker slots resumed')
    if status['halted_cpus']:
        issues.append('Queue has halted CPU slots')
    value = dict(side=side, passed=not issues, active=len(status['active']),
        expected_workers=expected_count, issues=issues, new_fits=new_fits,
        supervisor_pid=status['pid'], checked_at=time.time())
    queue.write(ROOT / (side + '_live.json'), value)
    print(json.dumps(dict(value, new_fits=len(new_fits))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('side', choices=('local', 'remote'))
    main(parser.parse_args().side)
