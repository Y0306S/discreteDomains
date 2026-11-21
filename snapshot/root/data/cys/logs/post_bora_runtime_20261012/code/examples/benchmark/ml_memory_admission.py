"""Serialize admissions across ML campaigns and reserve memory for pending fits."""

import contextlib
import fcntl
import json
import os
from pathlib import Path


def process_info(pid):
    try:
        fields = Path('/proc/%s/stat' % pid).read_text().rsplit(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'birth': fields[19], 'parent': int(fields[1]),
                'rss_gib': int(fields[21]) * os.sysconf('SC_PAGE_SIZE') / 2**30}
    except (OSError, IndexError, ValueError):
        return None


def forecast(working_gib, reservations, new_gib, limit_gib, headroom_gib):
    pending = sum(max(0, r['reserved_gib'] - r['resident_gib']) for r in reservations)
    return working_gib + pending + new_gib + headroom_gib <= limit_gib


class Admission:
    def __init__(self, path, spec):
        self.path, self.spec = Path(path), spec
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.processes = {int(p.name): info for p in Path('/proc').iterdir()
                          if p.name.isdigit() and (info := process_info(p.name))}
        self.rows = {key: row for key, row in self.rows.items()
                     if row.get('boot_id') == self.boot_id
                     and self.processes.get(row['pid'], {}).get('birth') == row['birth']}

    def estimate(self, method, task):
        key = '/'.join(task[k] for k in ('task_type', 'dataset', 'algorithm'))
        estimates = self.spec.get('task_memory_gib', {})
        estimate = estimates.get(key, estimates.get(task['algorithm'], 16))
        if method == 'tosfit' and self.spec.get('local_tosfit'):
            estimate = max(estimate, 40)
        return float(estimate)

    def allowed(self, estimate):
        cg = Path('/sys/fs/cgroup')
        stats = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        working = (int((cg/'memory.current').read_text()) - int(stats.get('inactive_file', 0))
                   - int(stats.get('slab_reclaimable', 0))) / 2**30
        reservations = []
        for row in self.rows.values():
            ids = {row['pid']}
            while True:
                more = {pid for pid, info in self.processes.items() if info['parent'] in ids}
                if more <= ids:
                    break
                ids |= more
            # The largest process underestimates resident task RAM conservatively.
            resident = max((self.processes.get(pid, {}).get('rss_gib', 0) for pid in ids), default=0)
            reservations.append(dict(row, resident_gib=resident))
        return forecast(working, reservations, estimate, self.spec['memory_gib'],
                        self.spec.get('memory_headroom_gib', 32))

    def reserve(self, state, estimate):
        self.rows[str(state['pid'])] = {'pid': state['pid'], 'birth': state['start_ticks'],
            'boot_id': self.boot_id, 'reserved_gib': estimate, 'session': state['session']}
        from ml_fixed_campaign import save
        save(self.path, self.rows)


@contextlib.contextmanager
def admission(spec):
    name = spec.get('memory_admission_registry')
    if not name:
        yield None
        return
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield Admission(path, spec)
