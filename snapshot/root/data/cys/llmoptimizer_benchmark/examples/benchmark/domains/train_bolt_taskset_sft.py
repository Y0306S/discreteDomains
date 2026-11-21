#!/usr/bin/env python3
"""Launch complete TaskSet BOLT SFT with isolated processes and durable progress."""

import argparse
from datetime import datetime, timezone
import fcntl
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile

ARTIFACTS = Path('/root/data/cys/baseline/artifacts/bolt_taskset_top10_full_sft')
DATA = Path('/root/data/cys/llmoptimizer_benchmark/benchmark_data/bolt_taskset_top10_train838_v4')
CORES = tuple(range(31, 39))
HOST = 'cs-0b2cd-18deb-server'
HOST_NAME = 'LLMOptimizer3'
GPUS = (0, 1)
RESUME = None
RESUME_SOURCE = None
GPU_RESIDENT_PARAMS = False
GPU_MEMORY_FRACTION = 0.75


def configure_allocation(args):
    global CORES, HOST, HOST_NAME, GPUS, RESUME, RESUME_SOURCE
    global GPU_RESIDENT_PARAMS, GPU_MEMORY_FRACTION
    hosts = {'LLMOptimizer2': 'cs-94bac-0693e-server',
             'LLMOptimizer3': 'cs-0b2cd-18deb-server'}
    HOST_NAME = args.host
    HOST = hosts[HOST_NAME]
    GPUS = tuple(int(value) for value in args.gpus.split(','))
    CORES = tuple(int(value) for value in args.cpus.split(','))
    RESUME = getattr(args, 'resume_from_checkpoint', None)
    RESUME_SOURCE = getattr(args, 'resume_source_run', None)
    GPU_RESIDENT_PARAMS = getattr(args, 'gpu_resident_params', False)
    GPU_MEMORY_FRACTION = getattr(args, 'gpu_memory_fraction', 0.75)
    if not 0.5 <= GPU_MEMORY_FRACTION <= 0.95:
        raise ValueError('GPU memory fraction must be between 0.5 and 0.95')
    if bool(RESUME) != bool(RESUME_SOURCE):
        raise ValueError('Resume requires both the checkpoint and its source run')
    if (not GPUS or len(set(GPUS)) != len(GPUS) or min(GPUS) < 0
            or not CORES or len(set(CORES)) != len(CORES)
            or len(CORES) % len(GPUS) or not set(CORES) <= os.sched_getaffinity(0)):
        raise ValueError('Require unique GPUs and equally partitioned available CPU cores')
    if len(GPUS) not in (1, 2):
        raise ValueError('The verified batch recipe supports one or two GPUs')


def allocation_args():
    result = ['--host', HOST_NAME, '--gpus', ','.join(map(str, GPUS)),
              '--cpus', ','.join(map(str, CORES)),
              '--gpu-memory-fraction', str(GPU_MEMORY_FRACTION)]
    if RESUME:
        result += ['--resume-from-checkpoint', str(RESUME), '--resume-source-run', str(RESUME_SOURCE)]
    if GPU_RESIDENT_PARAMS:
        result.append('--gpu-resident-params')
    return result


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def resources():
    import psutil

    cg = Path('/sys/fs/cgroup')
    stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    current = int((cg / 'memory.current').read_text())
    limit = int((cg / 'memory.max').read_text())
    reclaimable = min(int(stats.get('inactive_file', 0)),
                      max(0, int(stats.get('file', 0)) - int(stats.get('shmem', 0))))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [list(map(int, line.split(','))) for line in gpu.splitlines()]
    return dict(time=time.time(), gpu_free_gib={str(r[0]): r[1] / 1024 for r in rows},
                available_ram_gib=min(psutil.virtual_memory().available, limit - current + reclaimable) / 2**30,
                container_ram_gib=current / 2**30, disk_free_gib=shutil.disk_usage(ARTIFACTS).free / 2**30,
                memory_events=(cg / 'memory.events').read_text())


def validated_config():
    import yaml

    preflight = json.loads((DATA / 'preflight.json').read_text())
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if not preflight['passed'] or preflight['truncated_examples'] or preflight['examples'] != 8380:
        raise ValueError('A complete, untruncated 8380-example preflight is required')
    if digest(DATA / 'manifest.json') != preflight['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed after preflight')
    if digest(DATA / 'full_sft.yaml') != preflight['review_config_sha256']:
        raise ValueError('Training configuration changed after preflight')
    for name, expected in manifest['files'].items():
        if digest(DATA / name) != expected:
            raise ValueError('Dataset changed after preflight: ' + name)
    config = yaml.safe_load((DATA / 'full_sft.yaml').read_text())
    expected = dict(num_train_epochs=5, max_steps=-1, learning_rate=1e-5, finetuning_type='full',
                    per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    train_on_prompt=False, packing=False, val_size=0.0, resume_from_checkpoint=None)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Full training recipe changed')
    if config.get('max_samples') is not None or Path(config['dataset_dir']) != DATA:
        raise ValueError('Unexpected dataset subset')
    if Path(config['output_dir']) != ARTIFACTS / 'model':
        raise ValueError('Unexpected model output directory')
    if RESUME:
        if RESUME.resolve().parent != (ARTIFACTS / 'model').resolve():
            raise ValueError('Resume checkpoint is outside this campaign')
        state = json.loads((RESUME / 'trainer_state.json').read_text())
        if not 0 < state['global_step'] < 2620 or state['max_steps'] != 2620:
            raise ValueError('Invalid checkpoint training extent')
        verify_checkpoint(RESUME, state['global_step'])
        previous = json.loads((RESUME_SOURCE / 'training_config.json').read_text())
        previous_status = json.loads((RESUME_SOURCE / 'status.json').read_text())
        if (previous_status['dataset_manifest_sha256'] != preflight['dataset_manifest_sha256']
                or previous_status['gpus'] != list(GPUS)
                or previous_status['effective_batch_size'] != 16):
            raise ValueError('Resume data, GPU world size, or effective batch differs')
        effective = dict(config, gradient_accumulation_steps=16 // len(GPUS), preprocessing_num_workers=1)
        ignored = {'deepspeed', 'resume_from_checkpoint'}
        if {k: v for k, v in effective.items() if k not in ignored} != {
                k: v for k, v in previous.items() if k not in ignored}:
            raise ValueError('Resume would change the numerical training recipe')
        config['resume_from_checkpoint'] = str(RESUME)
    elif (ARTIFACTS / 'model').exists() and any((ARTIFACTS / 'model').iterdir()):
        raise FileExistsError('Refusing to overwrite model output; resume must be explicit')
    return config


def admission():
    if socket.gethostname() != HOST:
        raise RuntimeError(f'This allocation is for {HOST_NAME} only')
    sample = resources()
    reasons = []
    for gpu in GPUS:
        if str(gpu) not in sample['gpu_free_gib']:
            reasons.append(f'GPU {gpu} does not exist')
        elif sample['gpu_free_gib'][str(gpu)] < 65:
            reasons.append(f'GPU {gpu} has less than the required 65 GiB free')
    if sample['available_ram_gib'] < 200:
        reasons.append('Less than the required 200 GiB available RAM')
    if sample['disk_free_gib'] < 400:
        reasons.append('Less than the required 400 GiB free output storage')
    return dict(passed=not reasons, reasons=reasons, resources=sample)


def verify_checkpoint(path, step):
    state = json.loads((path / 'trainer_state.json').read_text())
    tag = (path / 'latest').read_text().strip()
    if state['global_step'] != step or tag != f'global_step{step}':
        raise ValueError('Checkpoint step mismatch')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        names = set(json.loads(index.read_text())['weight_map'].values())
        if not names or any(Path(n).name != n for n in names):
            raise ValueError('Invalid checkpoint shard index')
        weights = [path / name for name in names]
    else:
        weights = [path / 'model.safetensors']
    rng_names = ['rng_state.pth'] if len(GPUS) == 1 else [f'rng_state_{rank}.pth' for rank in range(len(GPUS))]
    required = weights + [path / name for name in rng_names]
    for rank in range(len(GPUS)):
        for name in (f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt',
                     f'zero_pp_rank_{rank}_mp_rank_00_model_states.pt'):
            state_path = path / tag / name
            with zipfile.ZipFile(state_path) as archive:
                if not archive.namelist():
                    raise ValueError('Empty DeepSpeed state')
            required.append(state_path)
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        raise ValueError('Incomplete checkpoint')


def worker(run_dir):
    rank = int(os.environ['LOCAL_RANK'])
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, GPUS))
            or rank not in range(len(GPUS))):
        raise ValueError('Worker GPU allocation does not match its assigned ranks')
    threads = len(CORES) // len(GPUS)
    os.sched_setaffinity(0, set(CORES[rank * threads:(rank + 1) * threads]))
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, rank)
    config = json.loads((run_dir / 'training_config.json').read_text())
    initial_step = json.loads((RESUME / 'trainer_state.json').read_text())['global_step'] if RESUME else 0

    class HealthCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step != initial_step or state.max_steps != 2620:
                raise ValueError(f'Unexpected training extent: {state.global_step}/{state.max_steps}')
            model = kwargs['model']
            model.save_pretrained = partial(model.save_pretrained, max_shard_size='1GB')
            write_json(run_dir / f'rank{rank}_started.json', dict(
                pid=os.getpid(), time=time.time(), max_steps=state.max_steps,
                initial_step=initial_step, resume_from_checkpoint=str(RESUME) if RESUME else None,
                learning_rates=kwargs['lr_scheduler'].get_last_lr(),
                trainable_parameters=sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                                         for p in model.parameters() if p.requires_grad),
                cpu_affinity=sorted(os.sched_getaffinity(0))))

        def on_step_end(self, args, state, control, **kwargs):
            write_json(run_dir / f'rank{rank}_health.json', dict(
                pid=os.getpid(), time=time.time(), step=state.global_step, max_steps=state.max_steps,
                epoch=state.epoch, gpu_allocated_gib=torch.cuda.memory_allocated(rank) / 2**30,
                gpu_reserved_gib=torch.cuda.memory_reserved(rank) / 2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(rank) / 2**30))
            if state.global_step == initial_step + 1:
                control.should_save = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                if not math.isfinite(float(logs['loss'])):
                    raise FloatingPointError('Non-finite SFT loss')
                write_json(run_dir / f'rank{rank}_loss.json', dict(step=state.global_step, time=time.time(), **logs))

        def on_save(self, args, state, control, **kwargs):
            torch.distributed.barrier(device_ids=[rank])
            path = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            verify_checkpoint(path, state.global_step)
            write_json(run_dir / f'rank{rank}_checkpoint.json', dict(
                step=state.global_step, path=str(path), time=time.time(), complete=True))

    run_exp(args=config, callbacks=[HealthCallback()])
    write_json(run_dir / f'rank{rank}_finished.json', dict(time=time.time(), complete=True))


def supervise(run_dir):
    if socket.gethostname() != HOST:
        raise RuntimeError(f'This allocation is for {HOST_NAME} only')
    with (ARTIFACTS / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.sched_setaffinity(0, set(CORES))
        os.nice(10)
        locks = []
        lock_root = Path('/root/data/logs/taskset_major_families_cpu_20260929/locks')
        for core in CORES:
            handle = (lock_root / f'core.{HOST_NAME}.{core}.lock').open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        config = validated_config()
        # Preserve the original two-GPU effective batch of sixteen conversations.
        config['gradient_accumulation_steps'] = 16 // len(GPUS)
        config['preprocessing_num_workers'] = 1
        check = admission()
        write_json(run_dir / 'admission.json', check)
        if not check['passed']:
            raise RuntimeError('; '.join(check['reasons']))
        deepspeed_config = json.loads(Path(config['deepspeed']).read_text())
        if GPU_RESIDENT_PARAMS:
            deepspeed_config['zero_optimization'].pop('offload_param', None)
        write_json(run_dir / 'deepspeed_config.json', deepspeed_config)
        config['deepspeed'] = str(run_dir / 'deepspeed_config.json')
        write_json(run_dir / 'training_config.json', config)
        shutil.copy2(DATA / 'preflight.json', run_dir / 'preflight.json')
        threads = str(len(CORES) // len(GPUS))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, GPUS)),
                   OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS=threads, MAX_JOBS=threads,
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   HF_DATASETS_CACHE=str(DATA / 'token_cache'),
                   TORCH_EXTENSIONS_DIR=str(ARTIFACTS / 'torch_extensions'),
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   f'--nproc_per_node={len(GPUS)}', str(Path(__file__).resolve()),
                   '--worker', '--run-dir', str(run_dir), *allocation_args()]
        with (run_dir / 'training.log').open('w') as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=tuple(handle.fileno() for handle in locks))
        status = dict(supervisor_pid=os.getpid(), torchrun_pid=child.pid, host=HOST_NAME,
                      run_dir=str(run_dir), status='starting', started_at=time.time(),
                      cpus=list(CORES), gpus=list(GPUS), examples=8380, epochs=5, max_steps=2620,
                      effective_batch_size=16, dataset_manifest_sha256=digest(DATA / 'manifest.json'),
                      script_sha256=digest(__file__), gpu_allocator_fraction=GPU_MEMORY_FRACTION,
                      gpu_resident_params=GPU_RESIDENT_PARAMS,
                      resume_from_checkpoint=str(RESUME) if RESUME else None,
                      resume_source_run=str(RESUME_SOURCE) if RESUME_SOURCE else None)

        def interrupted(signum, frame):
            raise InterruptedError(f'Received signal {signum}')

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            while child.poll() is None:
                sample = resources()
                if sample['available_ram_gib'] < 40 or sample['disk_free_gib'] < 80:
                    raise RuntimeError('RAM or storage reserve reached')
                status.update(status='running', checked_at=time.time(), resources=sample)
                write_json(run_dir / 'status.json', status)
                write_json(ARTIFACTS / 'current_run.json', status)
                time.sleep(10)
            status.update(status='completed' if child.returncode == 0 else 'failed', exit_code=child.returncode)
        except BaseException as exc:
            status.update(status='stopped', reason=str(exc))
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
            raise
        finally:
            status['checked_at'] = time.time()
            write_json(run_dir / 'status.json', status)
            write_json(ARTIFACTS / 'current_run.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--detach', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--supervise', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--host', choices=['LLMOptimizer2', 'LLMOptimizer3'], default='LLMOptimizer3')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--cpus', default='31,32,33,34,35,36,37,38')
    parser.add_argument('--resume-from-checkpoint', type=Path)
    parser.add_argument('--resume-source-run', type=Path)
    parser.add_argument('--gpu-resident-params', action='store_true')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.75)
    args = parser.parse_args()
    configure_allocation(args)
    if args.detach or args.check:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        validated_config()
        check = admission()
        check.update(data_preflight_passed=True, examples=8380, epochs=5, max_steps=2620,
                     training_started=False, launch_authorized=True,
                     host=HOST_NAME, gpus=list(GPUS), cpus=list(CORES), effective_batch_size=16,
                     dataset_manifest_sha256=digest(DATA / 'manifest.json'))
        write_json(ARTIFACTS / 'readiness.json', check)
        if args.check or not check['passed']:
            print(json.dumps(check, indent=2))
            return
        run_dir = ARTIFACTS / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
        run_dir.mkdir()
        script = run_dir / Path(__file__).name
        shutil.copy2(__file__, script)
        with (run_dir / 'supervisor.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(script), '--supervise', '--run-dir', str(run_dir),
                                        *allocation_args()],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(json.dumps(dict(pid=process.pid, run_dir=str(run_dir))))
    elif args.run_dir is None:
        parser.error('--run-dir is required')
    elif args.worker:
        worker(args.run_dir)
    else:
        supervise(args.run_dir)


if __name__ == '__main__':
    main()
