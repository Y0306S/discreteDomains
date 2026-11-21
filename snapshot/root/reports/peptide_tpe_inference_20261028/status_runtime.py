#!/usr/bin/env python3
"""Read-only progress and resource snapshot for resumed peptide TPE workers."""

import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
TRANSFER = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')


def local_status(side):
    path = CAMPAIGN / 'workers' / (side + '.json')
    records = json.loads(path.read_text()) if path.exists() else []
    workers, children = [], []
    for row in records:
        if row['method'] != 'tpe':
            continue
        try:
            words = Path('/proc/%d/cmdline' % row['pid']).read_bytes().split(b'\0')
            if row['command'][1].encode() not in words:
                continue
            workers.append(dict(pid=row['pid'], cpu=row['cpu'], port=row['server_port']))
        except OSError:
            pass
    parents = {row['pid'] for row in workers}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = dict(line.split(':', 1) for line in (entry / 'status').read_text().splitlines()
                          if ':' in line)
            if int(status['PPid']) not in parents:
                continue
            words = (entry / 'cmdline').read_bytes().split(b'\0')
            if b'--task' in words:
                children.append(dict(pid=int(entry.name), task=words[words.index(b'--task') + 1].decode(),
                    cpus=sorted(os.sched_getaffinity(int(entry.name))),
                    rss_gib=int(status['VmRSS'].split()[0]) / 2**20))
        except (OSError, KeyError, ValueError):
            pass
    cgroup = Path('/sys/fs/cgroup')
    return dict(host=socket.gethostname(), side=side, worker_count=len(workers),
                workers=workers, active_tasks=children,
                memory_current_gib=int((cgroup / 'memory.current').read_text()) / 2**30,
                memory_max_gib=int((cgroup / 'memory.max').read_text()) / 2**30,
                memory_events=(cgroup / 'memory.events').read_text().strip())


def main():
    if len(sys.argv) > 1:
        print(json.dumps(local_status(sys.argv[1])))
        return
    local = local_status('local')
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'LLMOptimizer4',
        '/usr/bin/python3', '-S', '-', 'remote'], input=Path(__file__).read_text(),
        text=True, capture_output=True, check=True, timeout=20)
    remote = json.loads(response.stdout)
    progress = []
    for path in (CAMPAIGN / 'tpe' / 'runs').glob('*/progress.json'):
        value = json.loads(path.read_text())
        progress.append(dict(task=path.parent.name, proposals=value['proposals'],
                             fresh_evaluations=value['fresh_evaluations'], cache_hits=value['cache_hits']))
    profile_path = CAMPAIGN / 'workers' / 'tpe_profile.sqlite'
    with sqlite3.connect('file:' + str(profile_path) + '?mode=ro', uri=True, timeout=10) as db:
        profiles = [json.loads(row[0]) for row in db.execute('SELECT value FROM profiles')]
    active_pids = {row['pid'] for machine in (local, remote) for row in machine['active_tasks']}
    profiles = [value for value in profiles if value['pid'] in active_pids]
    stages = {}
    for profile in profiles:
        for name, value in profile['stages'].items():
            total = stages.setdefault(name, dict(calls=0, seconds=0.0))
            total['calls'] += value['calls']
            total['seconds'] += value['seconds']
    for value in stages.values():
        value['mean_seconds'] = value['seconds'] / max(1, value['calls'])
    services = {}
    for port in (18871,18872,18873,18874,18875,18876,18877,18878):
        with urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=5) as response:
            services[str(port)] = json.load(response)
    output = dict(checked_at=time.time(), local=local, remote=remote,
        completed_tpe=len(list((CAMPAIGN / 'tpe' / 'runs').glob('*.tar.gz'))),
        completed_lolbo=len(list((CAMPAIGN / 'lolbo' / 'runs').glob('*.tar.gz'))),
        unfinished_progress=progress, unfinished_proposals=sum(v['proposals'] for v in progress),
        profile_tasks=len(profiles), profile_stages=stages, services=services,
        export=json.loads((TRANSFER / 'corpus/status.json').read_text()),
        training_queue=json.loads((TRANSFER / 'queue.json').read_text()))
    target = ROOT / 'live_status.json'
    temp = target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(output, indent=2) + '\n')
    os.replace(temp, target)
    print(json.dumps(dict(workers_local=local['worker_count'], workers_remote=remote['worker_count'],
        active_tasks_local=len(local['active_tasks']), active_tasks_remote=len(remote['active_tasks']),
        completed_tpe=output['completed_tpe'], completed_lolbo=output['completed_lolbo'],
        unfinished_proposals=output['unfinished_proposals'], profile_tasks=len(profiles),
        mean_stage_seconds={name:stages[name]['mean_seconds'] for name in
            ('sampler_ask','rpc_decode','rpc_oracle','checkpoint_commit','server_queue_decode') if name in stages},
        export_completed=output['export']['completed'],
        training_status=output['training_queue']['status'],
        service_errors={p:v['errors'] for p,v in services.items()}, report=str(target))))


if __name__ == '__main__':
    main()
