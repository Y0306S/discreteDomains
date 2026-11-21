"""Serialize resource admission across independent, ordered ML campaigns."""

from collections import Counter
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import socket
import time

from ml_gpu_health import gpu_memory


def configuration(root, host):
    from ml_fixed_campaign import digest, load

    protocol = load(root / 'protocol.json')
    if not protocol.get('resource_pool'):
        return None
    path = Path(protocol['resource_pool'])
    if digest(path) != protocol['resource_pool_sha256']:
        raise RuntimeError('Shared resource pool changed: ' + str(path))
    config = load(path)
    roots = [str(Path(p).resolve()) for p in config['campaigns']]
    if (config['host'] != host or config['hostname'] != socket.gethostname()
            or str(root.resolve()) not in roots or len(roots) != len(set(roots))):
        raise RuntimeError('Shared resource pool identity mismatch')
    spec = protocol['hosts'][host]
    if (not set(spec['cpus']) <= set(config['cpus'])
            or set(config['cpus']) & set(config['scheduler_cpus'])):
        raise RuntimeError('Shared resource pool CPU allocation mismatch')
    return path, config


def occupied_cpus(pool):
    occupied = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            state = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
            affinity = os.sched_getaffinity(int(proc.name))
            if state not in ('Z', 'X') and len(affinity) <= len(pool):
                occupied.update(affinity & pool)
        except (OSError, ValueError, IndexError):
            continue
    return occupied


class Allocation:
    def __init__(self, config, states, pending, memory, occupied, blocked_reason=None):
        self.config = config
        self.states = states
        self.pending = pending
        self.memory = {gpu: dict(info) for gpu, info in memory.items()}
        self.used_cpus = set(occupied)
        self.used_gpu = Counter()
        self.workers = len(states)
        self.blocked_reason = blocked_reason
        for state in states:
            self.used_cpus.update(state['cpus'])
            self.used_gpu.update(self.request(state['method'], state['gpu']))

    def request(self, method, slot):
        if slot is None:
            return {}
        devices = [int(gpu) for gpu in str(slot).split(',')]
        sizes = self.config['gpu_reservations_gib'][method]
        if len(devices) != len(sizes) or len(devices) != len(set(devices)):
            raise RuntimeError('Invalid ordered GPU reservation for ' + method)
        return dict(zip(devices, sizes))

    def waiting_for(self, root):
        for earlier in self.config['campaigns']:
            if Path(earlier).resolve() == root.resolve():
                return None
            if self.pending.get(earlier):
                return earlier
        raise RuntimeError('Campaign is not registered in its resource pool')

    def available(self, root):
        return (not self.blocked_reason and self.workers < self.config['max_workers']
                and self.waiting_for(root) is None)

    def choose_gpu(self, method, slots):
        if self.blocked_reason:
            return None
        feasible = []
        for slot in dict.fromkeys(slots):
            request = self.request(method, slot)
            if all(gpu in self.memory
                   and self.used_gpu[gpu] + size <= self.config['gpu_budget_gib'][str(gpu)]
                   and self.memory[gpu]['used'] + size
                   <= self.memory[gpu]['total'] - self.config['gpu_headroom_gib']
                   for gpu, size in request.items()):
                # Fill the least reserved devices first; account for every side of a pair.
                feasible.append((max(self.used_gpu[gpu] for gpu in request), str(slot), slot))
        return min(feasible)[2] if feasible else None

    def reserve(self, method, cpus, gpu):
        if self.used_cpus.intersection(cpus):
            raise RuntimeError('Overlapping shared-pool CPU reservation')
        self.used_cpus.update(cpus)
        self.workers += 1
        request = self.request(method, gpu)
        self.used_gpu.update(request)
        for device, size in request.items():
            self.memory[device]['used'] += size


def snapshot(config, memory, blocked_reason=None):
    from ml_fixed_campaign import load, process_alive, progress, task_key

    states, pending = [], {}
    for name in config['campaigns']:
        root = Path(name)
        protocol = load(root / 'protocol.json')
        if protocol['hosts'][config['host']]['hostname'] != config['hostname']:
            raise RuntimeError('Peer campaign belongs to another machine')
        setting = configuration(root, config['host'])
        if setting is None or setting[1] != config:
            raise RuntimeError('Peer campaign is not using the same shared resource pool')
        for path in (root / 'state').rglob('*.json'):
            state = load(path, {})
            if state.get('host') == config['host'] and process_alive(state):
                states.append(state)
        waiting = 0
        for method in protocol['methods']:
            for task in load(root / 'queues' / f'{method}.json'):
                state = load(root / 'state' / method / f'{task_key(task)}.json', {})
                if state.get('status') not in (None, 'pending'):
                    continue
                if not progress(root, method, task)['budget_exhausted']:
                    waiting += 1
        pending[name] = waiting
    return Allocation(config, states, pending, memory, occupied_cpus(set(config['cpus'])), blocked_reason)


@contextmanager
def resources(root, host):
    setting = configuration(root, host)
    if setting is None:
        yield None
        return
    path, config = setting
    memory, blocked_reason = {}, config.get('maintenance_reason')
    if not blocked_reason:
        try:
            memory = gpu_memory(config.get('gpu_probe_directory'), cache_seconds=15)
        except (OSError, ValueError, RuntimeError) as exc:
            blocked_reason = str(exc)
    # Hold the shared lock until launched workers have durable task-state records.
    with path.with_suffix('.lock').open('a') as lock:
        deadline = time.monotonic() + config.get('lock_timeout_seconds', 1)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    yield Allocation(config, [], {}, {}, set(config['cpus']),
                                     'Shared allocation lock is busy; admissions disabled')
                    return
                time.sleep(.05)
        yield snapshot(config, memory, blocked_reason)


def readiness(root, host):
    from ml_fixed_campaign import load

    with resources(root, host) as pool:
        if pool is None:
            raise RuntimeError('Campaign has no shared resource pool')
        if pool.blocked_reason:
            return [pool.blocked_reason]
        earlier = pool.waiting_for(root)
        if earlier:
            return ['Earlier campaign still has unassigned tasks: ' + Path(earlier).name]
        if not pool.available(root):
            return ['Shared worker slots are occupied']
        spec = load(root / 'protocol.json')['hosts'][host]
        free = set(spec['cpus']) - pool.used_cpus
        for method in spec['limits']:
            if not free:
                continue
            if pool.choose_gpu(method, spec['gpu_slots']) is not None:
                return []
        return ['Waiting for free shared CPU/GPU capacity']
