#!/usr/bin/env python3
"""CPU-only provenance, actual LLaMA-Factory preprocessing, and optimizer checks."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from examples.benchmark.export_bolt_ml_top10 import TRAIN80, canonical, digest, write_json
from examples.benchmark.supervise_bolt_ml_sft import checkpoint_step, gpu_launch_requirements, sample_resources
from mindware.components.optimizers.transfer_hpo.protocol import allowed_task_ids, checked_source_path, sha256_file
from mindware.components.optimizers.transfer_hpo.schema import config_from_dict, cs_from_spec

CONFIG = REPO / 'configs/bolt_ml/full_sft.yaml'
REPORT = REPO / 'benchmark_data/bolt_ml_top10_train80_v1/preflight.json'


def resource_check(config):
    output = Path(config['output_dir'])
    if config.get('resume_from_checkpoint'):
        checkpoint_step(config['resume_from_checkpoint'])
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f'Output already contains files; this launch starts from the clean base: {output}')
    storage = output
    while not storage.exists():
        storage = storage.parent
    free_disk = shutil.disk_usage(storage).free / 2**30
    memory = sample_resources()
    ram = memory['available_ram_gib']
    gpu_free = memory['gpu_free_gib']
    required = gpu_launch_requirements(memory)
    if free_disk < 300 or ram < 200 or any(gpu_free[gpu] < need for gpu, need in required.items()):
        raise RuntimeError(f'Resource preflight failed: disk={free_disk:.1f} GiB, RAM={ram:.1f} GiB, GPUs={gpu_free}')
    return {'free_disk_gib': free_disk, **memory,
            'gpu_launch_required_gib': required, 'proposed_gpus': [0, 1], 'gpu_training_tested': False}


def validate_export(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['train_manifest'] != str(TRAIN80) or manifest['train_manifest_sha256'] != sha256_file(TRAIN80):
        raise ValueError('The fixed train80 manifest changed')
    for name, checksum in manifest['files'].items():
        if sha256_file(directory / name) != checksum:
            raise ValueError(f'Export file changed: {name}')
    examples = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
    records = [json.loads(line) for line in (directory / 'provenance.jsonl').read_text().splitlines()]
    tasks = json.loads((directory / 'tasks.json').read_text())
    schemas = {key: cs_from_spec(value) for key, value in json.loads((directory / 'schemas.json').read_text()).items()}
    allowed = allowed_task_ids(TRAIN80)
    if {task['task_id'] for task in tasks} != allowed:
        raise ValueError('Task coverage does not equal the fixed train80 allowlist')
    source_data = {}
    checked_contexts = set()
    for task in tasks:
        path = checked_source_path(task['source_path'])
        if sha256_file(path) != task['source_sha256']:
            raise ValueError(f'History changed: {path}')
        source_data[str(path)] = json.loads(path.read_text())['observations']
        context = task['context_source']
        if context['path'] not in checked_contexts:
            if sha256_file(context['path']) != context['sha256']:
                raise ValueError(f'Dataset context changed: {context["path"]}')
            checked_contexts.add(context['path'])
    if len(records) != len(examples) or len(examples) != manifest['examples']:
        raise ValueError('Misaligned examples/provenance')
    seen = set()
    for index, (example, record) in enumerate(zip(examples, records)):
        if record['example_index'] != index or record['example_sha256'] != digest(example):
            raise ValueError(f'Example/provenance mismatch: {index}')
        task_id = record['task_id']
        if task_id not in allowed or not 1 <= record['rank'] <= 10:
            raise ValueError(f'Invalid source task/rank: {record}')
        algorithm = task_id.split('/')[-1]
        config = json.loads(example['messages'][-1]['content'])
        cs = schemas[record['schema_fingerprint']]
        if config_from_dict(cs, config, algorithm) is None or config != record['config']:
            raise ValueError(f'Invalid configuration: {index}')
        obs = source_data[record['source_path']][record['observation_index']]
        raw_config = dict(obs['config'])
        raw_config.setdefault('algorithm', algorithm)
        if canonical(raw_config) != canonical(config) or obs['trial_state'] != 0:
            raise ValueError(f'Historical configuration was altered or failed: {index}')
        if obs['objectives'] != [record['objective_minimize']]:
            raise ValueError(f'Historical score mismatch: {index}')
        key = (task_id, canonical(config))
        if key in seen:
            raise ValueError(f'Duplicate task/configuration: {index}')
        seen.add(key)
    return manifest, examples


def preflight(config_path, report_path, compile_cpu_adam):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run this preflight with CUDA_VISIBLE_DEVICES= to guarantee CPU-only checks')
    import torch
    import yaml
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    config = yaml.safe_load(config_path.read_text())
    manifest, examples = validate_export(Path(config['dataset_dir']))
    # Parse with the real LF parser. Only device/distributed switches change for this CPU check.
    cpu_config = dict(config, deepspeed=None, bf16=False, use_cpu=True,
                      report_to='none', dataloader_pin_memory=False)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(cpu_config)
    if finetuning_args.finetuning_type != 'full' or data_args.train_on_prompt or data_args.packing:
        raise ValueError('Expected full SFT with assistant-only loss and no packing')
    if data_args.val_size or data_args.eval_dataset or data_args.max_samples is not None:
        raise ValueError('Expected the complete authorized training dataset')

    ds = HfTrainerDeepSpeedConfig(config['deepspeed'])
    ds_args = type('DeepSpeedArguments', (), dict(vars(training_args), world_size=2, bf16=True))()
    ds.trainer_config_process(ds_args)
    if ds.mismatches:
        raise ValueError(ds.mismatches)
    if ds.config['train_batch_size'] != 16:
        raise ValueError('The effective batch must match the published 16')
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, 'sft', **tokenizer_module)
    dataset = dataset_module['train_dataset']
    if len(dataset) != len(examples):
        raise ValueError('LLaMA-Factory dropped training examples')
    expected_hashes, actual_hashes = Counter(), Counter()
    lengths, answer_lengths = [], []
    for example in examples:
        system, user, assistant = example['messages']
        pairs = template.encode_multiturn(tokenizer, [user, assistant], system['content'], None)
        if len(pairs) != 1 or template.efficient_eos:
            raise ValueError('Unexpected single-turn template layout')
        source, target = pairs[0]
        if len(source) + len(target) > data_args.cutoff_len:
            raise ValueError('Training cutoff would truncate an example')
        decoded = tokenizer.decode(target, skip_special_tokens=False)
        if decoded != assistant['content'] + '<|im_end|>\n':
            raise ValueError('The supervised answer is not exactly configuration JSON plus the end token')
        expected_hashes[digest([source + target, [-100] * len(source) + target])] += 1
        lengths.append(len(source) + len(target))
        answer_lengths.append(len(target))
    for row in dataset:
        if row['attention_mask'] != [1] * len(row['input_ids']):
            raise ValueError('Unexpected attention mask')
        actual_hashes[digest([row['input_ids'], row['labels']])] += 1
    if actual_hashes != expected_hashes:
        raise ValueError('LLaMA-Factory tokenization/masking differs from the complete expected examples')

    cpu_adam_result = 'not_run'
    if compile_cpu_adam:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        reference = torch.nn.Parameter(param.detach().clone())
        kwargs = dict(lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        optimizer = DeepSpeedCPUAdam([param], adamw_mode=True, **kwargs)
        reference_optimizer = torch.optim.AdamW([reference], **kwargs)
        for _ in range(3):
            param.grad = torch.tensor([0.2, -0.3])
            reference.grad = param.grad.clone()
            optimizer.step()
            reference_optimizer.step()
        torch.testing.assert_close(param, reference, rtol=1e-6, atol=1e-7)
        cpu_adam_result = 'passed_against_torch_adamw_on_two_cpu_parameters'
        del optimizer
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight unexpectedly initialized CUDA')
    report = {
        'passed': True, 'examples': len(examples), 'tasks': manifest['represented_tasks'],
        'min_tokens': min(lengths), 'max_tokens': max(lengths),
        'min_answer_tokens': min(answer_lengths), 'max_answer_tokens': max(answer_lengths),
        'cutoff_len': data_args.cutoff_len, 'truncated_examples': 0,
        'assistant_only_loss_verified': True, 'source_provenance_verified': True,
        'effective_batch_size': ds.config['train_batch_size'],
        'planned_optimizer_steps': math.ceil(len(examples) / 16) * config['num_train_epochs'],
        'cpu_adam': cpu_adam_result, 'gpu_training_tested': False, 'training_started': False,
        'config_sha256': sha256_file(config_path), 'deepspeed_sha256': sha256_file(config['deepspeed']),
        'dataset_manifest_sha256': sha256_file(Path(config['dataset_dir']) / 'manifest.json'),
        'python': sys.executable, 'versions': {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'llamafactory', 'deepspeed', 'datasets', 'accelerate', 'ConfigSpace')},
        'resolved_deepspeed': ds.config,
    }
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--compile-cpu-adam', action='store_true')
    parser.add_argument('--resource-check-only', action='store_true')
    args = parser.parse_args()
    if args.resource_check_only:
        import yaml
        result = resource_check(yaml.safe_load(args.config.read_text()))
    else:
        result = preflight(args.config, args.report, args.compile_cpu_adam)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
