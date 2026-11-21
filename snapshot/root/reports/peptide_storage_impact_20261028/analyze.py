#!/usr/bin/env python3
"""Read-only timing snapshots and storage-removal upper bounds."""

import argparse
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time

BASE = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
TRAIN = Path('/root/data/cys/baseline/artifacts/bolt_taskset_top10_full_sft/run_20261027T084502Z')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
IO = ('checkpoint_get', 'checkpoint_put', 'checkpoint_add_trial', 'checkpoint_trials',
      'checkpoint_commit', 'profile_persistence')
TOP = IO + ('sampler_ask', 'sampler_tell', 'rpc_decode', 'rpc_oracle', 'rpc_weights',
            'rpc_encode', 'rpc_serialization')


def local():
    tasks = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            words = [w.decode() for w in (item / 'cmdline').read_bytes().split(b'\0') if w]
            if ('/root/data/cys/llmoptimizer_benchmark/tools/peptide_tpe_runtime.py' not in words
                    or 'run-task' not in words or '--task' not in words):
                continue
            tasks.append(dict(pid=int(item.name), task=words[words.index('--task') + 1]))
        except (OSError, ValueError, IndexError):
            pass
    return tasks


def snapshot():
    locations = {'local': local()}
    remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'LLMOptimizer4', PYTHON, '-', '--local'], input=Path(__file__).read_text(),
        text=True, capture_output=True, check=True, timeout=30).stdout
    locations['remote'] = json.loads(remote)
    hosts = {(r['task'], r['pid']): host for host, rows in locations.items() for r in rows}
    with sqlite3.connect('file:' + str(BASE / 'workers/tpe_profile.sqlite') + '?mode=ro',
                         uri=True, timeout=10) as db:
        values = [json.loads(row[0]) for row in db.execute('SELECT value FROM profiles')]
    profiles = []
    for row in values:
        key = (row['task'], row['pid'])
        if key in hosts and row['tell_calls'] > 0:
            profiles.append(dict(row, host=hosts[key]))
    return dict(time=time.time(), active={k: len(v) for k, v in locations.items()}, profiles=profiles)


def summarize(rows):
    stages = {}
    wall = 0.0
    for row in rows:
        wall += max(0.0, row['updated'] - row['started'])
        for name, value in row['stages'].items():
            total = stages.setdefault(name, dict(calls=0, seconds=0.0))
            total['calls'] += value['calls']
            total['seconds'] += value['seconds']
    total = sum(stages.get(name, {}).get('seconds', 0.0) for name in TOP)
    io = sum(stages.get(name, {}).get('seconds', 0.0) for name in IO)
    commit = stages.get('checkpoint_commit', {}).get('seconds', 0.0)
    asks = stages.get('sampler_ask', {}).get('calls', 0)
    return dict(profile_tasks=len(rows), asks=asks, observed_wall_worker_hours=wall / 3600,
        measured_stage_worker_hours=total / 3600,
        io_worker_hours=io / 3600, io_fraction_of_measured=io / total if total else None,
        commit_fraction_of_measured=commit / total if total else None,
        perfect_io_speedup_upper_bound=total / (total - io) if total > io else None,
        measured_seconds_per_proposal=total / asks if asks else None,
        perfect_io_seconds_per_proposal=(total - io) / asks if asks else None,
        stages={name: dict(value, mean_seconds=value['seconds'] / value['calls'] if value['calls'] else None)
                for name, value in stages.items()})


def compare(first, last):
    earlier = {(r['task'], r['pid'], r['started']): r for r in first['profiles']}
    rows = []
    for row in last['profiles']:
        old = earlier.get((row['task'], row['pid'], row['started']))
        if not old or row['updated'] <= old['updated']:
            continue
        delta = dict(row, started=old['updated'], stages={})
        for name, value in row['stages'].items():
            previous = old['stages'].get(name, dict(calls=0, seconds=0.0))
            delta['stages'][name] = {k: value[k] - previous[k] for k in ('calls', 'seconds')}
        rows.append(delta)
    return {host: summarize([r for r in rows if host == 'all' or r['host'] == host])
            for host in ('all', 'local', 'remote')}


def elapsed(value):
    parts = [int(v) for v in value.split(':')]
    return sum(v * 60**i for i, v in enumerate(reversed(parts)))


def bolt():
    text = (TRAIN / 'training.log').read_text(errors='replace')
    entries = {}
    for step, clock in re.findall(r'(\d+)/2620 \[([\d:]+)<', text):
        entries[int(step)] = elapsed(clock)
    pauses = []
    for step in (500, 1000, 1500):
        if step in entries and step + 1 in entries and step + 2 in entries:
            ordinary = entries[step + 2] - entries[step + 1]
            pauses.append(dict(step=step,
                checkpoint_pause_seconds=entries[step + 1] - entries[step] - ordinary,
                adjacent_training_step_seconds=ordinary))
    save_events = re.findall(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) >> Saving model checkpoint to [^\n]+checkpoint-(\d+)', text)
    health = json.loads((TRAIN / 'rank0_health.json').read_text())
    return dict(health=health, checkpoint_pauses=pauses, save_events=save_events,
                note='Checkpoint pauses include gathering, serialization, writes and barriers; not pure storage time.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.local:
        print(json.dumps(local()))
        return
    data = snapshot()
    result = dict(snapshot=data,
        cumulative={host: summarize([r for r in data['profiles'] if host == 'all' or r['host'] == host])
                    for host in ('all', 'local', 'remote')}, bolt=bolt())
    if args.compare:
        result['interval'] = compare(json.loads(args.compare.read_text())['snapshot'], data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    concise = dict(time=data['time'], active=data['active'],
        cumulative={host: {k: v for k, v in row.items() if k != 'stages'} for host, row in result['cumulative'].items()},
        bolt=result['bolt'])
    if 'interval' in result:
        concise['interval'] = {host: {k: v for k, v in row.items() if k != 'stages'}
                               for host, row in result['interval'].items()}
    print(json.dumps(concise, indent=2))


if __name__ == '__main__':
    main()
