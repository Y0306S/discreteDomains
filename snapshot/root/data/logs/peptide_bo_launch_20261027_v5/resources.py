"""Read container limits and relevant physical CPU reservations without exposing arguments."""
import json
import os
from pathlib import Path
import socket
import subprocess

def snapshot():
    root = Path('/sys/fs/cgroup')
    maximum = int((root / 'memory.max').read_text())
    current = int((root / 'memory.current').read_text())
    memory = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
    processes = []
    reserved = set()
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            args = (item / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            relevant = any(token in args for token in ('/pg16/bin/postgres', 'postgres:',
                'manage.py worker', 'train_bolt_taskset_sft.py', 'train_semanticopt',
                'peptide_bo_v5.py', 'peptide_vae_server.py', 'peptide_test200.py',
                'peptide_transfer.py', 'peptide_training_queue.py', 'peptide_tpe_runtime.py',
                'peptide_rgpe_fit.py', 'peptide_rgpe_queue.py')) or (
                    'bolt_taskset_top10_full_sft/' in args and '/train.py' in args)
            if not relevant or 'resources.py' in args:
                continue
            affinity = sorted(os.sched_getaffinity(int(item.name)))
            if len(affinity) < os.cpu_count():
                for cpu in affinity:
                    siblings = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip()
                    for part in siblings.split(','):
                        if '-' in part:
                            low, high = map(int, part.split('-'))
                            reserved.update(range(low, high + 1))
                        else:
                            reserved.add(int(part))
            status = dict(line.split(':', 1) for line in (item / 'status').read_text().splitlines() if ':' in line)
            kind = 'postgres' if 'postgres' in args else ('peptide' if 'peptide_' in args else 'training/query manager')
            processes.append(dict(pid=int(item.name), kind=kind, affinity=affinity,
                rss_kib=int(status.get('VmRSS', '0 kB').split()[0])))
        except (OSError, ProcessLookupError, ValueError):
            continue
    free = []
    seen = set()
    pinned_physical = 0
    for cpu in sorted(os.sched_getaffinity(0)):
        siblings = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip()
        if siblings in seen:
            continue
        seen.add(siblings)
        if cpu in reserved:
            pinned_physical += 1
        else:
            free.append(cpu)
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    cpu_max = (root / 'cpu.max').read_text().strip()
    quota, period = cpu_max.split()
    quota_cores = len(seen) if quota == 'max' else min(len(seen), int(quota) / int(period))
    return dict(host=socket.gethostname(), cpu_max=cpu_max, cpu_quota_cores=quota_cores,
        pinned_physical_cores=pinned_physical,
        unreserved_cpu_quota=max(0, quota_cores-pinned_physical),
        available_gib=(maximum-current+int(memory.get('inactive_file', 0)))/2**30,
        memory_current_gib=current/2**30, memory_max_gib=maximum/2**30,
        free_physical_cpus=free, processes=processes, gpu=gpu)

if __name__ == '__main__':
    print(json.dumps(snapshot()))
