#!/usr/bin/env python3
"""Gracefully pause only this campaign's TPE processes on the current host."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time

CAMPAIGN = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
WORKER = '/root/data/cys/llmoptimizer_benchmark/tools/peptide_test200.py'
OPTIMIZED_WORKER = '/root/data/cys/llmoptimizer_benchmark/tools/peptide_tpe_runtime.py'


def processes():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            words = [x.decode() for x in args if x]
            worker = next((p for p in (WORKER, OPTIMIZED_WORKER) if p in words), None)
            if worker is None or '--campaign' not in words:
                continue
            if words[words.index('--campaign') + 1] != str(CAMPAIGN / 'tpe'):
                continue
            action = words[words.index(worker) + 1]
            if action not in ('worker', 'run-task'):
                continue
            found.append(dict(pid=int(entry.name), action=action, command=words))
        except (OSError, UnicodeDecodeError, IndexError):
            continue
    return found


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=('local', 'remote'), required=True)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    marker = CAMPAIGN / 'tpe' / 'paused.json'
    if not marker.exists():
        raise RuntimeError('Create the persistent pause marker before stopping workers')
    original = processes()
    for row in sorted(original, key=lambda value: value['action'] != 'worker'):
        if any(p['pid'] == row['pid'] and p['command'] == row['command'] for p in processes()):
            try:
                os.kill(row['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + args.timeout
    remaining = processes()
    while remaining and time.monotonic() < deadline:
        time.sleep(1)
        remaining = processes()
    report = dict(status='paused' if not remaining else 'draining', host=socket.gethostname(),
                  checked_at=time.time(), signaled=original, remaining=remaining,
                  forced_kills=0, histories_preserved=True)
    write(CAMPAIGN / 'workers' / ('tpe_pause_' + args.side + '.json'), report)
    print(json.dumps(dict(report, signaled=len(original), remaining=len(remaining))), flush=True)
    if remaining:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
