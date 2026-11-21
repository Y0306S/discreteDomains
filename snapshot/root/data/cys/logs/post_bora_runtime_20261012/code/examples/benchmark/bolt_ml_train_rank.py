#!/usr/bin/env python3
"""Apply BOLT's resource policy before LLaMA-Factory loads weights."""

from functools import partial
import math
import json
import os
import time
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def restore_training_policy(args, state):
    # TrainerState loading restores the old intervals after parsing the new config.
    state.compute_steps(args, state.max_steps)


def load_training_config(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())


def configure_checkpoint_shards(model):
    model.save_pretrained = partial(model.save_pretrained, max_shard_size='1GB')


def main():
    rank = int(os.environ['LOCAL_RANK'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or rank not in (0, 1):
        raise ValueError('This launch is restricted to two workers on GPUs 0 and 1')
    from examples.benchmark.supervise_bolt_ml_sft import parse_cpu_cores
    cores = parse_cpu_cores(os.environ.get('BOLT_CPU_CORES', '24,25,26,27,28,29,30,31'))
    os.sched_setaffinity(0, set(cores[4 * rank:4 * (rank + 1)]))

    import torch
    from transformers import TrainerCallback
    from mindware.components.optimizers.transfer_hpo.protocol import atomic_json

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)
    unlimited = os.environ.get('BOLT_DISABLE_MEMORY_LIMITS') == '1'
    allocator_fraction = None if unlimited else 0.30
    if allocator_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, rank)
    status_path = Path(os.environ['BOLT_RUN_DIR']) / f'rank{rank}_health.json'

    class HealthCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            expected_step = int(os.environ.get('BOLT_EXPECTED_RESUME_STEP', '0'))
            if state.global_step != expected_step:
                raise RuntimeError(f'Expected checkpoint step {expected_step}, got {state.global_step}')
            restore_training_policy(args, state)
            configure_checkpoint_shards(kwargs['model'])
            atomic_json(status_path.with_name(f'rank{rank}_restored.json'), {
                'pid': os.getpid(), 'rank': rank, 'updated_at': time.time(),
                'restored_step': state.global_step, 'max_steps': state.max_steps,
                'save_steps': state.save_steps, 'save_total_limit': args.save_total_limit,
                'checkpoint_max_shard_size': '1GB',
            })

        def on_step_end(self, args, state, control, **kwargs):
            # Both ranks release unused allocations before the next training step.
            torch.distributed.barrier(device_ids=[rank])
            torch.cuda.empty_cache()
            memory = torch.cuda.memory_stats(rank)
            atomic_json(status_path, {
                'pid': os.getpid(), 'rank': rank, 'step': state.global_step, 'epoch': state.epoch,
                'updated_at': time.time(),
                'max_steps': state.max_steps, 'gpu_allocated_gib': torch.cuda.memory_allocated(rank) / 2**30,
                'gpu_reserved_gib': torch.cuda.memory_reserved(rank) / 2**30,
                'peak_allocated_gib': torch.cuda.max_memory_allocated(rank) / 2**30,
                'gpu_allocator_retries': memory.get('num_alloc_retries', 0),
                'gpu_allocator_ooms': memory.get('num_ooms', 0),
                'cpu_affinity': sorted(os.sched_getaffinity(0)), 'allocator_fraction': allocator_fraction,
            })
            expected_step = int(os.environ.get('BOLT_EXPECTED_RESUME_STEP', '0'))
            if os.environ.get('BOLT_VERIFY_FIRST_CHECKPOINT') == '1' and state.global_step == expected_step + 1:
                control.should_save = True

        def on_save(self, args, state, control, **kwargs):
            torch.distributed.barrier(device_ids=[rank])
            from examples.benchmark.supervise_bolt_ml_sft import checkpoint_step
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            if checkpoint_step(checkpoint) != state.global_step:
                raise RuntimeError(f'Invalid saved checkpoint: {checkpoint}')
            atomic_json(status_path.with_name(f'rank{rank}_checkpoint.json'), {
                'rank': rank, 'step': state.global_step, 'path': str(checkpoint),
                'updated_at': time.time(), 'complete': True,
            })

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs and not math.isfinite(float(logs['loss'])):
                raise FloatingPointError(f'Non-finite training loss at step {state.global_step}')

    print(f'BOLT rank={rank} pid={os.getpid()} GPU allocator fraction={allocator_fraction} CPU affinity='
          f'{sorted(os.sched_getaffinity(0))}', flush=True)
    from llamafactory.train.tuner import run_exp
    config = load_training_config(sys.argv[1])
    run_exp(args=config, callbacks=[HealthCallback()])


if __name__ == '__main__':
    main()
