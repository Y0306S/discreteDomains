#!/usr/bin/env python3
"""Restore the saved database and validate the paused worker before restart."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


ROOT = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
OUT = ROOT / 'maintenance/LLMOptimizer3_gpu_change_20261005'
sys.path.insert(0, str(ROOT / 'code'))
import query_train800_fleet as fleet


def verify_running():
    plan, worker = fleet.allocation(ROOT, 'LLMOptimizer3.w00')
    paused = fleet.read(OUT / 'worker_paused.json')
    directory = ROOT / 'outputs' / paused['task'] / 'lolbo/seed0'
    deadline = time.monotonic() + 300
    while True:
        state = fleet.read(ROOT / 'workers/LLMOptimizer3.w00.json')
        if state['status'] != 'running' or state['task'] != paused['task']:
            raise RuntimeError('Unexpected worker status after restart: ' + state['status'])
        summary = fleet.read(directory / 'status.json')
        if summary['status'] == 'running' and summary['proposals'] > paused['saved_proposals']:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('No newly committed batch observed within five minutes')
        time.sleep(5)
    processes = {}
    database = fleet.read(ROOT / 'databases/LLMOptimizer3.w00.json')
    for label, pid, expected in (
            ('supervisor', state['pid'], [worker['bo_cpu']]),
            ('optimizer', state['child_pid'], [worker['bo_cpu']]),
            ('database', database['pid'], worker['db_cpus'])):
        proc = Path('/proc') / str(pid)
        assert proc.exists()
        assert sorted(os.sched_getaffinity(pid)) == sorted(expected)
        processes[label] = {'pid': pid, 'cpus': sorted(os.sched_getaffinity(pid))}
    rows = [fleet.read(directory / f'proposal_{index:04d}.json')
            for index in range(paused['saved_proposals'], summary['proposals'])]
    assert rows and all(math.isfinite(row['runtime']) and row['runtime'] > 0 for row in rows)
    assert summary['unresolved_database_calls'] == 0 and summary['error'] is None
    for relative, expected in paused['artifacts_sha256'].items():
        if relative.endswith('/checkpoint.pt'):
            continue
        assert fleet.digest(ROOT / relative) == expected, relative
    result = {'verified_at': datetime.now(timezone.utc).isoformat(),
              'status': 'resumed_and_advancing', 'worker': worker['name'], 'task': state['task'],
              'resumed_from_proposals': paused['saved_proposals'],
              'committed_proposals': summary['proposals'], 'new_rows_checked': len(rows),
              'old_proposal_files_unchanged': paused['saved_proposals'],
              'database_calls': summary['database_calls'], 'cache_hits': summary['cache_hits'],
              'processes': processes, 'gpu': worker['gpu'], 'new_gpu_model': 'NVIDIA A800-SXM4-80GB',
              'machine_gpu_count': 4, 'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
              'memory_limit_bytes': int(Path('/sys/fs/cgroup/memory.max').read_text()),
              'cpu_model': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                if line.startswith('model name')),
              'hardware_change_at_proposal': paused['saved_proposals'],
              'timing_note': 'The task spans the pre-maintenance and A800 hosts; preserve this boundary when interpreting runtime.',
              'extra_workers_launched': 0}
    fleet.write(OUT / 'worker_resumed.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-running', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'cs-0b2cd-18deb-server'
    if args.verify_running:
        verify_running()
        return
    plan, worker = fleet.allocation(ROOT, 'LLMOptimizer3.w00')
    state = fleet.read(ROOT / 'workers/LLMOptimizer3.w00.json')
    assert state['status'] == 'paused' and state['task'] == 'CEB_11B16'
    assert state['saved_proposals'] == 420
    with fleet.lock(ROOT / 'workers/LLMOptimizer3.w00.lock'):
        backup = fleet.read(OUT / 'database_archive.json')
        archive = Path(backup['archive'])
        target = Path(worker['pgdata'])
        if target.exists():
            raise RuntimeError('Restore target already exists; inspect before replacing anything')
        free_bytes = shutil.disk_usage('/dev/shm').free
        assert free_bytes > backup['bytes'] + 4 * 1024**3
        assert archive.stat().st_size == backup['bytes']
        print('Verifying the saved database archive checksum.', flush=True)
        with archive.open('rb') as stream:
            archive_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert archive_sha == backup['sha256'], 'Database archive checksum mismatch'
        paused = fleet.read(OUT / 'worker_paused.json')
        for relative, expected in paused['artifacts_sha256'].items():
            assert fleet.digest(ROOT / relative) == expected, relative
        for relative, expected in plan['files_sha256'].items():
            assert fleet.digest(ROOT / relative) == expected, relative
        print('Archive, saved proposals, checkpoint, and campaign files verified; restoring database.', flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '--numeric-owner', '--keep-old-files', '-xpf', str(archive),
                        '-C', str(target.parent)], check=True)
        assert target.stat().st_uid == 10000 and target.stat().st_gid == 10000
        assert not (target / 'postmaster.pid').exists()
        print('Checking restored database contents.', flush=True)
        subprocess.run(['tar', '--compare', '-f', str(archive), '-C', str(target.parent)], check=True)
        control = subprocess.check_output([str(fleet.PG_BIN / 'pg_controldata'), str(target)],
                                         text=True, env={**os.environ, 'LC_ALL': 'C'})
        cluster_state = next(line.split(':', 1)[1].strip() for line in control.splitlines()
                             if line.startswith('Database cluster state:'))
        assert cluster_state == 'shut down'
        result = {'restored_at': datetime.now(timezone.utc).isoformat(),
                  'worker': worker['name'], 'hostname': socket.gethostname(),
                  'database_path': str(target), 'database_archive_sha256': archive_sha,
                  'archive_comparison': 'passed', 'cluster_state_before_start': cluster_state,
                  'campaign_files_verified': len(plan['files_sha256']),
                  'saved_artifacts_verified': len(paused['artifacts_sha256']),
                  'resume_task': state['task'], 'resume_proposals': state['saved_proposals'],
                  'cpu_allocation': [worker['bo_cpu'], *worker['db_cpus']], 'gpu': worker['gpu']}
        fleet.write(OUT / 'database_restored.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
