#!/usr/bin/env python3
"""Fill released export/training CPUs and move finished TPE workers to LOLBO."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import launch_peptide_test200 as launch

TRAINING = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def alive(row):
    return launch.worker_alive(row)


def occupied(cpu):
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            command = (item / 'cmdline').read_bytes()
            if b'peptide_transfer.py' in command and cpu in os.sched_getaffinity(int(item.name)):
                return True
        except OSError:
            pass
    return False


def start(cpu, method, side, index):
    if launch.method_paused(method):
        return None
    directory = launch.CAMPAIGN / method
    command = [launch.PYTHON, str(launch.worker_script(method)), 'worker', '--campaign', str(directory),
        '--cpu', str(cpu), '--index', str(index), '--workers', '70',
        '--server', 'http://127.0.0.1:%d' % (18871,18872,18873)[index % 3]]
    log = launch.CAMPAIGN / 'workers' / ('handoff_%s_%s_cpu%d.log' % (side, method, cpu))
    with log.open('ab') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, env=launch.ENV, start_new_session=True)
    return dict(cpu=cpu, method=method, pid=child.pid, command=command, started=time.time(), log=str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side', choices=('local', 'remote'))
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        log = launch.CAMPAIGN / 'workers' / ('handoff_%s_supervisor.log' % args.side)
        with log.open('ab') as stream:
            child = subprocess.Popen([launch.PYTHON, str(Path(__file__).resolve()), args.side],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                env=launch.ENV, start_new_session=True)
        launch.write(log.with_suffix('.json'), dict(pid=child.pid, side=args.side, started=time.time()))
        print(json.dumps(dict(pid=child.pid, side=args.side)), flush=True)
        return
    os.sched_setaffinity(0, {39} if args.side == 'local' else {47})
    record = launch.CAMPAIGN / 'workers' / ('handoff_%s.json' % args.side)
    rows = read(record) or []
    while True:
        complete = {m: len(list((launch.CAMPAIGN / m / 'runs').glob('*.tar.gz'))) == 200 for m in ('tpe','lolbo')}
        if all(complete[m] or launch.method_paused(m) for m in complete):
            return
        candidates = {}
        if args.side == 'local':
            if read(TRAINING / 'corpus/status.json').get('status') == 'complete':
                candidates[18] = 11
        else:
            if read(TRAINING / 'fsbo/status.json').get('status') == 'complete':
                candidates.update({8:38, 9:39})
            if read(TRAINING / 'rgpe/status.json').get('status') == 'complete':
                candidates.update({cpu:30+cpu for cpu in range(8)})
        if complete['tpe']:
            original = read(launch.CAMPAIGN / 'workers' / (args.side + '.json')) or []
            for row in original:
                if row['method'] == 'tpe' and not alive(row):
                    candidates[row['cpu']] = 40 + row['index']
        for cpu, index in candidates.items():
            previous = next((r for r in rows if r['cpu'] == cpu), None)
            if previous and alive(previous):
                continue
            if occupied(cpu):
                continue
            method = 'lolbo' if complete['tpe'] or cpu != 18 else 'tpe'
            if complete[method] or launch.method_paused(method):
                method = 'tpe' if method == 'lolbo' else 'lolbo'
            if complete[method] or launch.method_paused(method):
                continue
            if previous:
                rows.remove(previous)
            replacement = start(cpu, method, args.side, index)
            if replacement is not None:
                rows.append(replacement)
            launch.write(record, rows)
        time.sleep(30)


if __name__ == '__main__':
    main()
