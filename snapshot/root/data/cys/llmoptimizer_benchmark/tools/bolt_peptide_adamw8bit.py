#!/usr/bin/env python3
"""Checkpoint-bound migration of peptide full SFT to one-GPU 8-bit AdamW."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import bolt_peptide_sft as original

ROOT = original.ARTIFACTS
DEPENDENCIES = ROOT / 'optimizer_dependencies/bitsandbytes_0_46_1'
GPU = 2
CPU = 18
CONTROL_CPU = 44
TARGET_STEP = 100
STATUS = ROOT / 'adamw8bit_migration.json'


def config_for_migration(old, resume, output):
    config = dict(old)
    config.update(model_name_or_path=str(resume), resume_from_checkpoint=str(resume),
                  output_dir=str(output), deepspeed=None, optim='adamw_bnb_8bit',
                  bf16=True, pure_bf16=True, per_device_train_batch_size=4,
                  gradient_accumulation_steps=4, save_steps=100, save_total_limit=2)
    return config


def cuda_environment():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PYTHONPATH'] = str(DEPENDENCIES) + os.pathsep + env.get('PYTHONPATH', '')
    for name in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    return env


def kernel_check(run):
    os.sched_setaffinity(0, {CONTROL_CPU})
    import bitsandbytes as bnb
    import torch
    from transformers import Trainer, TrainingArguments
    from llamafactory.hparams import get_train_args

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if bnb.__version__ != '0.46.1' or not torch.cuda.is_bf16_supported():
        raise ValueError('Require the pinned package and BF16 GPU support')
    candidate = config_for_migration(json.loads((original.DATA/'training_config.json').read_text()),
                                     original.BASE, run/'parser_scratch')
    candidate['resume_from_checkpoint'] = None
    _, _, training, finetuning, _ = get_train_args(candidate)
    if (finetuning.finetuning_type != 'full' or not finetuning.pure_bf16
            or training.deepspeed is not None
            or training.per_device_train_batch_size * training.gradient_accumulation_steps != 16):
        raise ValueError('LLaMA-Factory did not accept the one-GPU full SFT recipe')
    args = TrainingArguments(output_dir=str(run / 'kernel_scratch'), optim='adamw_bnb_8bit',
                             learning_rate=1e-5, weight_decay=0.01, report_to='none')
    cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    parameter = torch.nn.Parameter(torch.randn(256, 256, device='cuda', dtype=torch.bfloat16) * 0.01)
    before = parameter.detach().clone()
    optimizer = cls([parameter], **kwargs)
    for _ in range(3):
        parameter.grad = torch.full_like(parameter, 0.25)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    state = optimizer.state[parameter]
    if (state['state1'].dtype != torch.uint8 or state['state2'].dtype != torch.uint8
            or not torch.isfinite(parameter).all() or torch.equal(before, parameter)):
        raise ValueError('8-bit BF16 optimizer did not produce valid GPU updates')
    report = dict(passed=True, bitsandbytes=bnb.__version__, torch=torch.__version__,
                  physical_gpu=GPU, optimizer=type(optimizer).__name__, optim_bits=8,
                  parameter_dtype=str(parameter.dtype), gradient_dtype='torch.bfloat16',
                  moment_dtype=str(state['state1'].dtype), updates=3,
                  changed_values=int(torch.count_nonzero(before != parameter).item()),
                  gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  llamafactory_recipe_accepted=True, effective_batch_size=16, time=time.time())
    original.atomic_json(run / 'kernel_preflight.json', report)
    print(json.dumps(report), flush=True)


def restore_schedule(trainer, step):
    # DeepSpeed's optimizer checkpoint cannot be loaded by bitsandbytes.
    # Advance only the existing cosine scheduler, preserving its original horizon.
    scheduler = trainer.lr_scheduler
    if hasattr(scheduler, 'scheduler'):
        scheduler = scheduler.scheduler
    scheduler.last_epoch = step
    scheduler._step_count = step + 1
    rates = [base * fn(step) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    scheduler._last_lr = rates
    for group, rate in zip(trainer.optimizer.param_groups, rates):
        group['lr'] = rate
    return rates


def prepare_resume(source, destination):
    original.verify_checkpoint(source, TARGET_STEP)
    destination.mkdir()
    for path in source.iterdir():
        if path.name == 'trainer_state.json':
            state = json.loads(path.read_text())
            state['train_batch_size'] = 4
            original.atomic_json(destination / path.name, state)
        elif path.is_file() and (path.suffix in ('.json', '.safetensors', '.jinja')
                               or path.name == 'training_args.bin'):
            (destination / path.name).symlink_to(path)
    import torch
    rng = torch.load(source / 'rng_state_0.pth', map_location='cpu', weights_only=False)
    if isinstance(rng.get('cuda'), list):
        rng['cuda'] = rng['cuda'][0]
    torch.save(rng, destination / 'rng_state.pth')
    if not (destination / 'trainer_state.json').is_file():
        raise ValueError('Missing training position')


def worker(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(GPU):
        raise ValueError('Require one visible GPU')
    os.sched_setaffinity(0, {CPU})
    import bitsandbytes as bnb
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from llamafactory.train.tuner import run_exp

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.92)
    config = json.loads((run / 'training_config.json').read_text())
    resume = Path(config['resume_from_checkpoint']).resolve()
    expected_step = json.loads((resume / 'trainer_state.json').read_text())['global_step']
    load_original = CustomSeq2SeqTrainer._load_optimizer_and_scheduler

    def load_migrated(trainer, checkpoint):
        if checkpoint is not None and Path(checkpoint).resolve() == resume:
            if trainer.optimizer.state:
                raise ValueError('Expected fresh 8-bit optimizer moments')
            rates = restore_schedule(trainer, expected_step)
            original.atomic_json(run / 'scheduler_restored.json', dict(step=expected_step,
                rates=rates, optimizer_moments='reset_at_precision_change', time=time.time()))
        else:
            return load_original(trainer, checkpoint)

    CustomSeq2SeqTrainer._load_optimizer_and_scheduler = load_migrated
    step_started = time.monotonic()

    class Health(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            nonlocal step_started
            optimizer = kwargs['optimizer']
            while hasattr(optimizer, 'optimizer'):
                optimizer = optimizer.optimizer
            model = kwargs['model']
            parameters = [p for p in model.parameters() if p.requires_grad]
            if (state.global_step != expected_step or state.max_steps != 2500
                    or sum(p.numel() for p in parameters) != 8953803264
                    or any(p.dtype != torch.bfloat16 or p.device.type != 'cuda' for p in parameters)
                    or not isinstance(optimizer, bnb.optim.AdamW)
                    or optimizer.args.optim_bits != 8 or optimizer.is_paged):
                raise ValueError('Wrong training position, precision, full SFT extent or optimizer')
            model.save_pretrained = partial(model.save_pretrained, max_shard_size='1GB')
            original.atomic_json(run / 'started.json', dict(pid=os.getpid(), time=time.time(),
                step=state.global_step, max_steps=state.max_steps, physical_gpu=GPU,
                cpu_affinity=sorted(os.sched_getaffinity(0)), trainable_parameters=8953803264,
                parameter_dtype='bfloat16', optimizer=type(optimizer).__name__, optim_bits=8,
                optimizer_on_gpu=True, cpu_optimizer_offload=False))
            step_started = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            nonlocal step_started
            optimizer = kwargs['optimizer']
            while hasattr(optimizer, 'optimizer'):
                optimizer = optimizer.optimizer
            moments = Counter()
            for values in optimizer.state.values():
                for name in ('state1', 'state2'):
                    value = values.get(name)
                    if value is not None:
                        if value.device.type != 'cuda':
                            raise ValueError('Optimizer moment left GPU')
                        moments[str(value.dtype)] += value.numel()
            if not moments['torch.uint8']:
                raise ValueError('No 8-bit optimizer moments')
            elapsed = time.monotonic() - step_started
            step_started = time.monotonic()
            original.atomic_json(run / 'health.json', dict(pid=os.getpid(), time=time.time(),
                step=state.global_step, max_steps=state.max_steps, epoch=state.epoch,
                step_seconds=elapsed, gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                gpu_reserved_gib=torch.cuda.memory_reserved()/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                moment_elements_by_dtype=dict(moments), cpu_affinity=sorted(os.sched_getaffinity(0))))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                if not math.isfinite(float(logs['loss'])):
                    raise FloatingPointError('Nonfinite SFT loss')
                original.atomic_json(run / 'loss.json', dict(step=state.global_step,
                                     time=time.time(), **logs))

        def on_save(self, args, state, control, **kwargs):
            path = Path(args.output_dir) / ('checkpoint-%d' % state.global_step)
            for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                if not (path / name).is_file() or not (path / name).stat().st_size:
                    raise ValueError('Incomplete new checkpoint: ' + name)
            original.atomic_json(run / 'checkpoint.json', dict(step=state.global_step,
                                 path=str(path), complete=True, time=time.time()))

        def on_train_end(self, args, state, control, **kwargs):
            if state.global_step != 2500:
                raise ValueError('Training stopped before five epochs')
            original.atomic_json(run / 'finished.json', dict(step=state.global_step,
                                 epoch=state.epoch, time=time.time()))

    run_exp(args=config, callbacks=[Health()])


def check_cpu(cpu, allowed=()):
    import psutil
    siblings = {cpu, cpu + 128}
    for process in psutil.process_iter(['pid']):
        if process.pid in {*allowed, os.getpid()}:
            continue
        try:
            affinity = set(process.cpu_affinity())
            if len(affinity) <= 8 and affinity & siblings:
                raise ValueError('CPU %d already reserved by PID %d' % (cpu, process.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def controller(run):
    import psutil
    os.sched_setaffinity(0, {CONTROL_CPU})
    plan = json.loads((run / 'plan.json').read_text())
    source_run = Path(plan['source_run'])
    state = dict(plan, controller_pid=os.getpid(), supervisor_pid=os.getpid(),
                 status='waiting_for_checkpoint', checked_at=time.time())
    child = None

    def interrupted(signum, frame):
        raise InterruptedError('Migration controller received signal %d' % signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (ROOT / 'adamw8bit_migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                records = [source_run / ('rank%d_checkpoint.json' % rank) for rank in range(2)]
                if all(p.is_file() and json.loads(p.read_text()).get('step') == TARGET_STEP
                       and json.loads(p.read_text()).get('complete') for p in records):
                    break
                proc = psutil.Process(plan['source_supervisor_pid'])
                if str(source_run) not in proc.cmdline():
                    raise ValueError('Original supervisor identity changed')
                health = source_run / 'rank0_health.json'
                state.update(source_step=json.loads(health.read_text())['step'] if health.exists() else 0,
                             checked_at=time.time())
                original.atomic_json(STATUS, state)
                time.sleep(5)
            checkpoint = ROOT / 'model' / ('checkpoint-%d' % TARGET_STEP)
            original.verify_checkpoint(checkpoint, TARGET_STEP)
            source_script = source_run / 'bolt_peptide_sft.py'
            if original.digest(source_script) != plan['source_script_sha256']:
                raise ValueError('Original launch source changed')
            original.validate_data()
            for pid in plan['source_worker_pids']:
                if str(source_run) not in psutil.Process(pid).cmdline():
                    raise ValueError('Original worker identity changed')
            state.update(status='checkpoint_verified', source_step=TARGET_STEP, checked_at=time.time())
            original.atomic_json(STATUS, state)
            os.kill(plan['source_supervisor_pid'], signal.SIGTERM)
            deadline = time.monotonic() + 120
            while any(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
                      for pid in [plan['source_supervisor_pid'], *plan['source_worker_pids']]):
                if time.monotonic() > deadline:
                    raise TimeoutError('Old training did not release its resources')
                time.sleep(2)
            check_cpu(CPU)
            resources = original.resources()
            if resources['gpu_free_gib'][str(GPU)] < 68 or resources['disk_free_gib'] < 150:
                raise RuntimeError('Insufficient one-GPU or checkpoint headroom')
            resume = run / 'resume_checkpoint_100'
            prepare_resume(checkpoint, resume)
            config = config_for_migration(json.loads((source_run / 'training_config.json').read_text()),
                                          resume, ROOT / 'model_adamw8bit')
            original.atomic_json(run / 'training_config.json', config)
            original.atomic_json(run / 'provenance.json', dict(plan,
                source_checkpoint=str(checkpoint), source_state_sha256=original.digest(checkpoint/'trainer_state.json'),
                source_model_index_sha256=original.digest(checkpoint/'model.safetensors.index.json'),
                training_config_sha256=original.digest(run/'training_config.json'),
                moments_reset=True, scheduler_position_retained=True, global_step_retained=True,
                cpu_offload=False, model_and_gradient_dtype='bfloat16',
                data_order_bitwise_equivalent=False, effective_batch_size=16,
                precision_change='FP32-master AdamW to BF16 full weights and 8-bit AdamW moments'))
            with (run / 'training.log').open('w') as stream:
                child = subprocess.Popen([sys.executable, str(run/'bolt_peptide_adamw8bit.py'),
                    'worker', '--run-dir', str(run)], env=cuda_environment(), stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='starting', worker_pid=child.pid, gpus=[GPU], rank_cpus=[CPU],
                         control_cpu=CONTROL_CPU, run_dir=str(run), optimizer='adamw_bnb_8bit',
                         output_dir=config['output_dir'], epochs=5, examples=8000, tasks=800,
                         max_steps=2500, effective_batch_size=16)
            while child.poll() is None:
                sample = original.resources()
                if sample['available_ram_gib'] < 30 or sample['disk_free_gib'] < 80:
                    raise RuntimeError('RAM or disk reserve reached')
                state.update(checked_at=time.time(), resources=sample)
                if (run / 'health.json').exists():
                    health = json.loads((run / 'health.json').read_text())
                    state.update(status='running', step=health['step'], health=health)
                    if health['step'] >= TARGET_STEP + 2:
                        state['real_updates_validated'] = True
                original.atomic_json(STATUS, state)
                original.atomic_json(ROOT / 'current_run.json', state)
                time.sleep(10)
            state.update(status='completed' if child.returncode == 0 else 'failed',
                         exit_code=child.returncode, checked_at=time.time())
        except BaseException as exc:
            state.update(status='migration_failed', reason=repr(exc), checked_at=time.time())
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
            raise
        finally:
            original.atomic_json(STATUS, state)
            if state.get('source_step') == TARGET_STEP:
                original.atomic_json(ROOT / 'current_run.json', state)


def schedule():
    if not DEPENDENCIES.is_dir():
        raise ValueError('Install the isolated pinned optimizer first')
    check_cpu(CONTROL_CPU)
    current = json.loads((ROOT / 'current_run.json').read_text())
    source = Path(current['run_dir'])
    if current['status'] != 'running' or current['gpus'] != [2, 3]:
        raise ValueError('Require the expected running two-GPU source')
    source_step = json.loads((source / 'rank0_health.json').read_text())['step']
    if source_step >= TARGET_STEP:
        raise ValueError('Target checkpoint already passed; select a new safe boundary')
    stamp = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ_adamw8bit')
    run = ROOT / stamp
    run.mkdir()
    shutil.copy2(__file__, run / 'bolt_peptide_adamw8bit.py')
    shutil.copy2(source / 'bolt_peptide_sft.py', run / 'bolt_peptide_sft.py')
    plan = dict(source_run=str(source), source_supervisor_pid=current['supervisor_pid'],
        source_worker_pids=[json.loads((source/('rank%d_started.json'%r)).read_text())['pid'] for r in range(2)],
        source_script_sha256=original.digest(source/'bolt_peptide_sft.py'),
        source_manifest_sha256=original.digest(original.DATA/'manifest.json'),
        migration_script_sha256=original.digest(run/'bolt_peptide_adamw8bit.py'),
        target_checkpoint_step=TARGET_STEP, scheduled_at=time.time(), launch_authorized=True,
        source_step_at_schedule=source_step, optimizer_moments='reset_at_migration',
        single_gpu=GPU, training_cpu=CPU, control_cpu=CONTROL_CPU, bitsandbytes_version='0.46.1',
        optimizer_dependencies=str(DEPENDENCIES))
    original.atomic_json(run / 'plan.json', plan)
    subprocess.run([sys.executable, str(run/'bolt_peptide_adamw8bit.py'), 'kernel-check',
                    '--run-dir', str(run)], env=cuda_environment(), check=True)
    with (run / 'controller.log').open('w') as stream:
        child = subprocess.Popen([sys.executable, str(run/'bolt_peptide_adamw8bit.py'),
            'controller', '--run-dir', str(run)], stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(controller_pid=child.pid, run_dir=str(run),
                          status='waiting_for_checkpoint', checkpoint_step=TARGET_STEP)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('schedule', 'controller', 'worker', 'kernel-check'))
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.action != 'schedule' and args.run_dir is None:
        parser.error('--run-dir is required')
    if args.action == 'schedule':
        schedule()
    elif args.action == 'kernel-check':
        kernel_check(args.run_dir)
    elif args.action == 'worker':
        worker(args.run_dir)
    else:
        controller(args.run_dir)


if __name__ == '__main__':
    main()
