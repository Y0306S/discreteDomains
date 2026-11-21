"""Bounded, shared GPU probing even when the NVIDIA driver cannot reap a child."""

import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


_CHILDREN = {}
COMMAND = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
           '--format=csv,noheader,nounits']


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, IndexError):
        return None


def driver_waits():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if ((proc / 'comm').read_text().strip() == 'nvidia-smi'
                    and (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[0] == 'D'):
                return int(proc.name)
        except (OSError, IndexError):
            continue
    return None


def parse_memory(output):
    result = {}
    for line in output.splitlines():
        gpu, used, total = line.split(',')
        gpu, used, total = int(gpu), float(used) / 1024, float(total) / 1024
        if (gpu in result or gpu < 0 or not all(map(math.isfinite, (used, total)))
                or not 0 <= used <= total or total <= 0):
            raise ValueError('Invalid GPU memory report')
        result[gpu] = {'used': used, 'total': total}
    if not result:
        raise ValueError('Empty GPU memory report')
    return result


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record))
    temporary.replace(path)


def gpu_memory(directory=None, *, timeout=10, cache_seconds=5, retry_seconds=30):
    directory = Path(directory or f'/tmp/ml_gpu_health_{socket.gethostname()}')
    directory.mkdir(parents=True, exist_ok=True)
    for pid, child in list(_CHILDREN.items()):
        if child.poll() is not None:
            del _CHILDREN[pid]
    with (directory / 'probe.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('GPU health probe is owned by another controller') from None
        path = directory / 'probe.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        now = time.time()
        original = record.get('identity')
        if original and identity(original['pid']) == original:
            if now - record['started_at'] >= timeout and not record.get('timed_out'):
                # Never wait/communicate after kill: D-state children cannot exit yet.
                try:
                    os.kill(original['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                record['timed_out'] = True
                save(path, record)
            raise RuntimeError(f"GPU probe PID {original['pid']} is "
                               + ('stuck after its deadline' if record.get('timed_out') else 'pending'))
        blocked = driver_waits()
        if blocked:
            raise RuntimeError(f'NVIDIA driver wait in PID {blocked}; new GPU admissions disabled')
        if original and not record.get('collected'):
            record['collected'] = True
            record['finished_at'] = now
            try:
                if record.get('timed_out'):
                    raise ValueError('GPU probe exceeded its deadline')
                record['memory'] = parse_memory((directory / 'output.txt').read_text())
            except (OSError, ValueError) as exc:
                record['error'] = str(exc)
            save(path, record)
        if record.get('memory') and now - record['started_at'] <= cache_seconds:
            return {int(gpu): info for gpu, info in record['memory'].items()}
        if record.get('error') and now - record['finished_at'] < retry_seconds:
            raise RuntimeError('GPU probe failed: ' + record['error'])
        with (directory / 'output.txt').open('w') as stream:
            child = subprocess.Popen(COMMAND, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        _CHILDREN[child.pid] = child
        original = identity(child.pid)
        # A fast successful child can already be a zombie; retain its start identity
        # so the next poll still consumes its output.
        if original is None:
            original = {'pid': child.pid, 'start_ticks': 'exited', 'boot_id': ''}
        save(path, {'identity': original, 'started_at': now})
        raise RuntimeError(f'GPU probe PID {child.pid} started; awaiting its result')
