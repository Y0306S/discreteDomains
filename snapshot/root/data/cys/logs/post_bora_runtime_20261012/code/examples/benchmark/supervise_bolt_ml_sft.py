#!/usr/bin/env python3
"""Own BOLT's process group, CPU reservations, and optional memory guards."""

import argparse
import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import psutil
import yaml
from packaging.version import Version

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mindware.components.optimizers.transfer_hpo.protocol import atomic_json

ARTIFACTS = Path('/root/data/cys/baseline/artifacts/bolt_ml_top10_full_sft')


def parse_cpu_cores(value):
    cores = tuple(int(item) for item in value.split(','))
    if len(cores) != 8 or len(set(cores)) != 8 or min(cores) < 0:
        raise ValueError('BOLT requires eight distinct CPU cores, four per rank')
    return cores


def ram_resources(cgroup=Path('/sys/fs/cgroup'), host_available=None):
    host_available = psutil.virtual_memory().available if host_available is None else host_available
    result = {'host_available_ram_gib': host_available / 2**30,
              'available_ram_gib': host_available / 2**30}
    limit_path, current_path = cgroup / 'memory.max', cgroup / 'memory.current'
    if not limit_path.exists():
        return result
    limit = limit_path.read_text().strip()
    if limit == 'max':
        return result
    limit, current = int(limit), int(current_path.read_text())
    stats = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
    # Inactive file cache can be reclaimed; shared-memory pages cannot.
    file_cache = max(0, int(stats.get('file', 0)) - int(stats.get('shmem', 0)))
    reclaimable = min(file_cache, int(stats.get('inactive_file', 0)))
    available = max(0, limit - current + reclaimable)
    result.update(container_ram_limit_gib=limit / 2**30,
                  container_ram_used_gib=current / 2**30,
                  reclaimable_file_cache_gib=reclaimable / 2**30,
                  available_ram_gib=min(host_available, available) / 2**30)
    return result


def sample_resources():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'
    ], text=True, timeout=10)
    rows = {int(parts[0]): (int(parts[1]) / 1024, int(parts[2]) / 1024)
            for parts in (row.split(',') for row in output.splitlines())}
    return {'gpu_free_gib': {str(gpu): rows[gpu][0] for gpu in (0, 1)},
            'gpu_total_gib': {str(gpu): rows[gpu][1] for gpu in (0, 1)}, **ram_resources()}


def gpu_launch_requirements(resources):
    # Cover the allocator cap and CUDA/NCCL allocations outside that allocator.
    return {gpu: total * 0.30 + 2 for gpu, total in resources['gpu_total_gib'].items()}


def ram_reserve_gib(resources):
    return min(100, resources.get('container_ram_limit_gib', 1000) * 0.10)


def stop_reason(resources, disable_memory_limits=False):
    if disable_memory_limits:
        return None
    reserve = ram_reserve_gib(resources)
    if resources['available_ram_gib'] < reserve:
        return f'Available CPU RAM below {reserve:g} GiB'
    return None


def stop_child(child):
    # child starts its own session, so this process group contains only our training workers.
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=20)


def checkpoint_step(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    step = state['global_step']
    tag = (checkpoint / 'latest').read_text().strip()
    if tag != f'global_step{step}':
        raise ValueError(f'Checkpoint step/tag mismatch: {checkpoint}')
    single = checkpoint / 'model.safetensors'
    if single.is_file():
        required = [single]
    else:
        index = checkpoint / 'model.safetensors.index.json'
        if not index.is_file():
            raise ValueError(f'Incomplete model weights: {checkpoint}')
        weight_map = json.loads(index.read_text()).get('weight_map', {})
        shards = set(weight_map.values())
        if not shards or any(Path(shard).name != shard for shard in shards):
            raise ValueError(f'Invalid model shard index: {checkpoint}')
        required = [index, *(checkpoint / shard for shard in shards)]
    for rank in (0, 1):
        required.extend([checkpoint / f'rng_state_{rank}.pth',
                         checkpoint / tag / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt',
                         checkpoint / tag / f'zero_pp_rank_{rank}_mp_rank_00_model_states.pt'])
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError(f'Incomplete two-rank checkpoint: {checkpoint}')
    return step


def supervise(run_dir, resume_from_checkpoint=None, disable_memory_limits=False,
              verify_first_checkpoint=False, cpu_cores=tuple(range(24, 32))):
    with (ARTIFACTS / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not set(cpu_cores).issubset(os.sched_getaffinity(0)):
            raise ValueError('Requested CPU cores are outside the supervisor allocation')
        os.sched_setaffinity(0, set(cpu_cores))
        core_locks = []
        taskset_locks = Path('/root/data/logs/taskset_major_families_cpu_20260929/locks')
        for cpu in cpu_cores:
            core_lock = (taskset_locks / f'core.LLMOptimizer2.{cpu}.lock').open('a')
            fcntl.flock(core_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            core_locks.append(core_lock)
        resources = sample_resources()
        launch_requirements = gpu_launch_requirements(resources)
        if any(resources['gpu_free_gib'][gpu] < required for gpu, required in launch_requirements.items()):
            raise RuntimeError(f'Insufficient GPU memory at launch: {resources}')
        reason = stop_reason(resources, disable_memory_limits)
        if reason:
            raise RuntimeError(f'{reason}: {resources}')
        config = yaml.safe_load((REPO / 'configs/bolt_ml/full_sft.yaml').read_text())
        resumed_step = None
        if resume_from_checkpoint:
            if Version(importlib.metadata.version('torch')) < Version('2.6'):
                raise RuntimeError('Checkpoint resume requires PyTorch >= 2.6 in the BOLT environment')
            resumed_step = checkpoint_step(resume_from_checkpoint)
            config['resume_from_checkpoint'] = str(Path(resume_from_checkpoint).resolve())
        ds_config = json.loads(Path(config['deepspeed']).read_text())
        atomic_json(run_dir / 'deepspeed_config.json', ds_config)
        config['deepspeed'] = str(run_dir / 'deepspeed_config.json')
        atomic_json(run_dir / 'training_config.json', config)
        env = dict(os.environ, BOLT_RUN_DIR=str(run_dir))
        env['BOLT_CPU_CORES'] = ','.join(str(cpu) for cpu in cpu_cores)
        env['PATH'] = os.pathsep.join([str(Path(sys.executable).parent),
                                      str(Path(sys.executable).resolve().parent), env.get('PATH', '')])
        if not shutil.which('ninja', path=env['PATH']):
            raise RuntimeError('The BOLT environment requires ninja for the CPU Adam extension')
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['BOLT_EXPECTED_RESUME_STEP'] = str(resumed_step or 0)
        env['BOLT_DISABLE_MEMORY_LIMITS'] = '1' if disable_memory_limits else '0'
        env['BOLT_VERIFY_FIRST_CHECKPOINT'] = '1' if verify_first_checkpoint else '0'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc_per_node=2', str(REPO / 'examples/benchmark/bolt_ml_train_rank.py'),
                   str(run_dir / 'training_config.json')]
        with (run_dir / 'training.log').open('w') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=tuple(item.fileno() for item in core_locks))
        status = {'supervisor_pid': os.getpid(), 'torchrun_pid': child.pid, 'command': command,
                  'run_dir': str(run_dir), 'status': 'starting', 'started_utc': datetime.datetime.now(
                      datetime.timezone.utc).isoformat(), 'resources': resources,
                  'gpu_allocator_fraction_per_rank': None if disable_memory_limits else 0.30,
                  'cpu_affinity': sorted(os.sched_getaffinity(0)),
                  'gpu_launch_required_gib': launch_requirements,
                  'stop_below_gpu_free_gib': None,
                  'stop_below_available_ram_gib': None if disable_memory_limits else ram_reserve_gib(resources),
                  'disable_memory_limits': disable_memory_limits,
                  'checkpoint_max_shard_size': '1GB', 'verify_first_checkpoint': verify_first_checkpoint,
                  'resume_from_checkpoint': config.get('resume_from_checkpoint'),
                  'resumed_step': resumed_step, 'save_steps': config['save_steps'],
                  'save_total_limit': config['save_total_limit'],
                  'framework_versions': {name: importlib.metadata.version(name)
                                         for name in ('torch', 'transformers', 'deepspeed', 'llamafactory')}}

        def interrupted(signum, frame):
            raise InterruptedError(f'Supervisor received signal {signum}')

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            while child.poll() is None:
                resources = sample_resources()
                reason = stop_reason(resources, disable_memory_limits)
                status.update(resources=resources, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                if reason:
                    status.update(status='stopped_for_headroom', reason=reason)
                    stop_child(child)
                    break
                status['status'] = 'running'
                atomic_json(run_dir / 'status.json', status)
                atomic_json(ARTIFACTS / 'current_run.json', status)
                time.sleep(5)
            if status['status'] != 'stopped_for_headroom':
                status['status'] = 'completed' if child.returncode == 0 else 'failed'
            status['exit_code'] = child.returncode
        except BaseException as exc:
            stop_child(child)
            status.update(status='stopped', reason=str(exc), exit_code=child.returncode)
            raise
        finally:
            atomic_json(run_dir / 'status.json', status)
            atomic_json(ARTIFACTS / 'current_run.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    parser.add_argument('--cpus', type=parse_cpu_cores, default=tuple(range(24, 32)),
                        help='Eight comma-separated CPU cores; first four for rank 0, last four for rank 1.')
    parser.add_argument('--disable-memory-limits', action='store_true',
                        help='Disable the runtime RAM reserve and the per-rank CUDA allocator cap.')
    parser.add_argument('--verify-first-checkpoint', action='store_true',
                        help='Save and validate a checkpoint after the first resumed update.')
    args = parser.parse_args()
    if args.detach:
        run_dir = ARTIFACTS / datetime.datetime.now(datetime.timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
        run_dir.mkdir()
        with (run_dir / 'supervisor.log').open('w') as log:
            command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run_dir)]
            command.extend(['--cpus', ','.join(str(cpu) for cpu in args.cpus)])
            if args.resume_from_checkpoint:
                checkpoint_step(args.resume_from_checkpoint)
                command.extend(['--resume-from-checkpoint', str(args.resume_from_checkpoint.resolve())])
            if args.disable_memory_limits:
                command.append('--disable-memory-limits')
            if args.verify_first_checkpoint:
                command.append('--verify-first-checkpoint')
            proc = subprocess.Popen(command,
                                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'supervisor_pid': proc.pid, 'run_dir': str(run_dir)}))
    else:
        if args.run_dir is None:
            parser.error('--run-dir is required when not detaching')
        supervise(args.run_dir, args.resume_from_checkpoint, args.disable_memory_limits,
                  args.verify_first_checkpoint, args.cpus)


if __name__ == '__main__':
    main()
