#!/usr/bin/env python3
"""Prepare/check SemanticOpt ML SFT; GPU training requires --execute --gpu N.

Input is annotated *historical* trajectories, not newly evaluated ML tasks.
Prediction-label generation is a separate stage. Missing labels are never guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mindware.components.optimizers.transfer_hpo.protocol import (
    HISTORY_ROOT, allowed_task_ids, checked_source_path, sha256_file,
)

TRAIN80 = Path('/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl')
CONFIG = REPO / 'configs/semanticopt_ml/sft.json'
SYSTEM = (
    'You are a black-box optimization assistant for tabular machine learning. '
    'Return exactly one JSON object with next_configuration, mean, and std. '
    'After at least one observed evaluation failure, also include failure_probability. '
    'Predict the objective distribution for the proposed configuration: mean is its mean, '
    'std is its nonnegative standard deviation, and failure_probability is between 0 and 1. '
    'Include every active hyperparameter and constant with its exact ConfigSpace name and '
    'type, omit inactive parameters, and obey all bounds, choices, conditions and forbiddens. '
    'Seek the lowest objective within the remaining evaluation budget. Return JSON only.'
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def history_prefix_digest(observations):
    # The archive uses Infinity/NaN for failed trials; these tokens are hashed,
    # never emitted as training targets or numeric evaluation feedback.
    encoded = json.dumps(observations, sort_keys=True, separators=(',', ':'), allow_nan=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def number(value, name, lower=None, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f'{name} is outside its allowed range')
    return value


def sig4(value):
    return float(format(value, '.4g'))


def load_config(path):
    config = json.loads(Path(path).read_text())
    published = {'lora_rank': 16, 'lora_alpha': 16, 'learning_rate': 1e-5,
                 'lr_scheduler_type': 'cosine', 'num_train_epochs': 1,
                 'effective_batch_size': 64, 'bf16': True}
    for key, value in published.items():
        if config.get(key) != value:
            raise ValueError(f'The prepared paper recipe requires {key}={value!r}')
    for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'save_steps',
                'max_seq_length'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    batch = config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
    if batch != config['effective_batch_size']:
        raise ValueError('Single-GPU microbatch times accumulation must equal 64 trajectories')
    if any(key in config for key in ('max_steps', 'max_samples', 'eval_dataset')):
        raise ValueError('This recipe uses one complete epoch, with no sample/step cap or test data')
    if not Path(config['model_name_or_path']).is_dir():
        raise ValueError('The local base model directory does not exist')
    if (Path(config['model_name_or_path']) / 'adapter_config.json').exists():
        raise ValueError('Start from the clean Qwen base, not an existing adapter')
    model_config = json.loads((Path(config['model_name_or_path']) / 'config.json').read_text())
    text_config = model_config.get('text_config', model_config)
    context_limit = text_config.get('max_position_embeddings')
    if type(context_limit) is not int or context_limit < 1:
        raise ValueError('The local model must declare its native context limit')
    if config['max_seq_length'] > context_limit:
        raise ValueError(f'max_seq_length exceeds the local model context limit of {context_limit}')
    return config


def original_history_length(observations):
    """Only the original BO prefix is permitted, including in prompt context."""
    end = len(observations)
    for index, obs in enumerate(observations):
        source = (obs.get('extra_info') or {}).get('source')
        if isinstance(source, str) and source.startswith('topup_'):
            end = min(end, index)
        elif source not in (None, 'original_bo'):
            raise ValueError(f'Unrecognized observation source at index {index}: {source!r}')
        elif index >= end:
            raise ValueError('Top-up observations must form a trailing segment')
    return end


def validate_observations(observations, cs, algorithm, task_type):
    """Recover exact recorded configurations and outcomes without repairing them."""
    from ConfigSpace import Configuration

    recorded = []
    for obs in observations:
        values = dict(obs['config'])
        values.setdefault('algorithm', algorithm)
        if values['algorithm'] != algorithm:
            raise ValueError('Recorded configuration has the wrong algorithm')
        config = Configuration(cs, values=values)
        cs.check_configuration(config)
        if canonical(dict(config)) != canonical(values):
            raise ValueError('Do not alter/default scored historical configurations')
        state = obs.get('trial_state')
        if type(state) is not int or state not in (0, 1, 2, 3):
            raise ValueError('Unknown historical trial state')
        entry = {'configuration': values, 'status': 'success' if state == 0 else 'failure'}
        if state == 0:
            scores = obs.get('objectives')
            if not isinstance(scores, list) or len(scores) != 1:
                raise ValueError('Expected a single recorded objective')
            score = number(scores[0], 'recorded objective', -1 if task_type == 'cls' else 0,
                           0 if task_type == 'cls' else None)
            if abs(score) >= 1e18:
                raise ValueError('Sentinel objective cannot label a successful evaluation')
            entry['objective'] = sig4(score)
        recorded.append(entry)
    return recorded


def validate_steps(row, observations, cs, algorithm, task_type):
    """Check prefix attestations and recover actions/outcomes from the original file."""
    warmup = row.get('initial_observations')
    steps = row.get('steps')
    if type(warmup) is not int or not 1 <= warmup < len(observations):
        raise ValueError('initial_observations must leave at least one supervised decision')
    if not isinstance(steps, list) or not steps or warmup + len(steps) > len(observations):
        raise ValueError('Missing or out-of-range trajectory steps')
    if warmup + len(steps) > original_history_length(observations):
        raise ValueError('Top-up observations cannot enter prediction targets or prefix context')
    teacher = row.get('teacher') or {}
    if not teacher.get('name') or teacher.get('causal_prefix_only') is not True:
        raise ValueError('Prediction-label producer must declare a causal teacher')
    if not isinstance(teacher.get('settings'), dict) or not teacher.get('code_sha256'):
        raise ValueError('Prediction-label producer settings and code checksum are required')
    if row.get('objective_scale') != 'archive_minimize':
        raise ValueError('Predictions must use the original negative-balanced-accuracy or MSE scale')

    recorded = validate_observations(observations[:warmup + len(steps)], cs, algorithm, task_type)

    answers = []
    for index, step in enumerate(steps, warmup):
        if type(step.get('observation_index')) is not int or step['observation_index'] != index:
            raise ValueError('Steps must be contiguous in original archive order')
        if step.get('prefix_sha256') != history_prefix_digest(observations[:index]):
            raise ValueError('Prediction label does not attest to the exact preceding history')
        answer = {'next_configuration': recorded[index]['configuration'],
                  'mean': sig4(number(step.get('mean'), 'mean')),
                  'std': sig4(number(step.get('std'), 'std', 0))}
        had_failure = any(obs['trial_state'] != 0 for obs in observations[:index])
        if had_failure:
            answer['failure_probability'] = sig4(number(
                step.get('failure_probability'), 'failure_probability', 0, 1))
        elif 'failure_probability' in step:
            raise ValueError('failure_probability must be omitted before an observed failure')
        answers.append(answer)
    return recorded, answers


def trajectory_messages(row, observations, cs, profile, algorithm, task_type):
    from mindware.components.optimizers.transfer_hpo.schema import configspace_spec, schema_spec

    recorded, answers = validate_steps(row, observations, cs, algorithm, task_type)
    warmup = row['initial_observations']
    space = configspace_spec(cs)
    metric = 'negative validation balanced accuracy' if task_type == 'cls' else 'validation mean squared error'
    context = {
        'dataset_profile': profile, 'fixed_algorithm': algorithm,
        'objective': {'name': metric, 'direction': 'minimize'},
        'total_evaluations_including_initial_observations': len(recorded),
        'ConfigSpace': {'hyperparameters': schema_spec(cs),
                        'conditions': space['conditions'], 'forbiddens': space['forbiddens']},
        'initial_evaluations': recorded[:warmup],
    }
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': canonical(context) + '\nOutput the next configuration.'}]
    for index, answer in enumerate(answers, warmup):
        messages.append({'role': 'assistant', 'content': canonical(answer)})
        feedback = {'evaluation': index + 1,
                    **{key: value for key, value in recorded[index].items() if key != 'configuration'}}
        suffix = '\nOutput the next configuration.' if index + 1 < len(recorded) else '\nEvaluation budget exhausted.'
        messages.append({'role': 'user', 'content': canonical(feedback) + suffix})
    return messages


def tokenize_trajectory(tokenizer, messages, max_length):
    """Mask by structured message boundaries, checking against the native Qwen template."""
    if messages[-1]['role'] != 'user':
        raise ValueError('A trajectory must end with the last recorded evaluation feedback')
    pieces, spans, offset = [], [], 0
    for message in messages:
        content = message['content']
        if content != content.strip() or any(token in content for token in ('<|im_start|>', '<|im_end|>', '<think>', '</think>')):
            raise ValueError('Unexpected chat control token or whitespace in trajectory content')
        header = f'<|im_start|>{message["role"]}\n'
        piece = header + content + '<|im_end|>\n'
        if message['role'] == 'assistant':
            spans.append((offset + len(header), offset + len(piece)))
        pieces.append(piece)
        offset += len(piece)
    text = ''.join(pieces)
    native = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False)
    if native != text:
        raise ValueError('Native tokenizer template changed; review assistant masking before training')
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    if len(encoded['input_ids']) > max_length:
        raise ValueError(f'Trajectory has {len(encoded["input_ids"])} tokens, above {max_length}; no silent truncation')
    labels, counts = [], [0] * len(spans)
    for token_id, (start, stop) in zip(encoded['input_ids'], encoded['offset_mapping']):
        active = [i for i, (left, right) in enumerate(spans) if start < right and stop > left]
        if active:
            left, right = spans[active[0]]
            if len(active) != 1 or not left <= start < stop <= right:
                raise ValueError('A token crosses the assistant-only supervision boundary')
            labels.append(token_id)
            counts[active[0]] += 1
        else:
            labels.append(-100)
    if not counts or not all(counts):
        raise ValueError('Every assistant turn must contribute supervised tokens')
    return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'], 'labels': labels}


def validate_coverage(manifest, allowed, represented, trajectories, dataset_hash, train_hash):
    if manifest.get('format_version') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Annotation production must be marked complete before SFT')
    if manifest.get('dataset_sha256') != dataset_hash or manifest.get('train_manifest_sha256') != train_hash:
        raise ValueError('Annotation manifest checksums do not match the dataset and fixed split')
    included = manifest.get('represented_tasks', [])
    exclusions = manifest.get('excluded_tasks', {})
    if not isinstance(included, list) or len(included) != len(set(included)):
        raise ValueError('Invalid represented-task list in annotation manifest')
    if not isinstance(exclusions, dict) or any(not isinstance(reason, str) or not reason.strip()
                                             for reason in exclusions.values()):
        raise ValueError('Every excluded train80 task needs an explicit reason')
    if set(included) != represented or represented & set(exclusions) or represented | set(exclusions) != allowed:
        raise ValueError('Full train80 coverage requires every task to be represented or explicitly excluded')
    if manifest.get('trajectories') != trajectories or not manifest.get('horizon_policy'):
        raise ValueError('Annotation manifest needs the complete trajectory count and horizon policy')


def load_training_data(config, tokenizer):
    from examples.benchmark.export_bolt_ml_top10 import dataset_context
    from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs

    if os.environ.get('ML_TRANSFER_HISTORY_MIRROR'):
        raise ValueError('SemanticOpt requires the original authorized history archive')
    path = Path(config['dataset'])
    if not path.is_file():
        raise ValueError(f'Prediction-annotated trajectories are not ready: {path}')
    allowed = allowed_task_ids(TRAIN80)
    manifest_hash = sha256_file(TRAIN80)
    dataset_hash = sha256_file(path)
    contexts, sources, seen, data = {}, {}, set(), []
    represented = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row['task_id']
        if task_id not in allowed or row.get('train_manifest_sha256') != manifest_hash:
            raise ValueError(f'Line {line_number}: task/manifest is not the fixed train80 source')
        tt, dataset, algorithm = task_id.split('/')
        metric = 'bal_acc' if tt == 'cls' else 'mse'
        expected = checked_source_path(HISTORY_ROOT / tt / metric / dataset / f'{algorithm}.json')
        if Path(row['source_path']).resolve() != expected:
            raise ValueError('Trajectory source is not this task\'s original archive file')
        if task_id not in sources:
            sources[task_id] = {'sha256': sha256_file(expected),
                                'observations': json.loads(expected.read_text())['observations']}
        source = sources[task_id]
        if source['sha256'] != row.get('source_sha256'):
            raise ValueError('Source history checksum changed')
        key = (task_id, row.get('initial_observations'), len(row.get('steps', [])))
        if key in seen:
            raise ValueError('Duplicate trajectory horizon would silently overweight a task')
        seen.add(key)
        if (tt, dataset) not in contexts:
            contexts[tt, dataset] = dataset_context(tt, dataset)
        profile, schema_kwargs, _ = contexts[tt, dataset]
        cs = build_hpo_cs(tt, algorithm, **schema_kwargs)
        messages = trajectory_messages(row, source['observations'], cs, profile, algorithm, tt)
        data.append(tokenize_trajectory(tokenizer, messages, config['max_seq_length']))
        represented.add(task_id)
    if not data:
        raise ValueError('No annotated trajectories')
    if dataset_hash != sha256_file(path):
        raise ValueError('Annotated trajectories changed during preprocessing')
    manifest_path = path.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        raise ValueError(f'Completed annotation manifest is not ready: {manifest_path}')
    annotation_manifest = json.loads(manifest_path.read_text())
    validate_coverage(annotation_manifest, allowed, represented, len(data), dataset_hash, manifest_hash)
    lengths = [len(row['input_ids']) for row in data]
    return data, {'trajectories': len(data), 'represented_train80_tasks': len(represented),
                  'allowed_train80_tasks': len(allowed), 'unrepresented_tasks': sorted(allowed - represented),
                  'dataset_sha256': dataset_hash, 'train_manifest_sha256': manifest_hash,
                  'annotation_manifest_sha256': sha256_file(manifest_path),
                  'excluded_tasks': annotation_manifest['excluded_tasks'],
                  'source_sha256': {key: value['sha256'] for key, value in sources.items()},
                  'min_tokens': min(lengths), 'max_tokens': max(lengths),
                  'supervised_tokens': sum(sum(label != -100 for label in row['labels']) for row in data),
                  'assistant_only_loss_verified': True, 'truncated_trajectories': 0,
                  'predictions_recomputed_by_this_check': False}


def check(config, prepare=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU checks require CUDA_VISIBLE_DEVICES=')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], local_files_only=True)
    data, report = load_training_data(config, tokenizer)
    report = {**report, 'passed': True, 'training_started': False,
              'max_seq_length': config['max_seq_length'],
              'optimizer_steps': math.ceil(len(data) / config['effective_batch_size']),
              'config_sha256': digest(config), 'script_sha256': sha256_file(__file__),
              'tokenizer_vocab_sha256': digest(tokenizer.get_vocab()),
              'gpu_training_verified': False}
    if prepare:
        directory = Path(config['prepared_dir'])
        directory.mkdir(parents=True, exist_ok=True)
        payload = ''.join(canonical(row) + '\n' for row in data)
        cache = directory / 'tokens.jsonl'
        report['tokens_sha256'] = hashlib.sha256(payload.encode()).hexdigest()
        # A completed manifest is published last; an interrupted preparation cannot train.
        from mindware.components.optimizers.transfer_hpo.protocol import atomic_json
        temporary = cache.with_name(f'tokens.{os.getpid()}.tmp')
        try:
            temporary.write_text(payload)
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(directory / 'preflight.json', report)
    return report


def load_prepared(config):
    directory = Path(config['prepared_dir'])
    report = json.loads((directory / 'preflight.json').read_text())
    expected = {'config_sha256': digest(config), 'script_sha256': sha256_file(__file__),
                'dataset_sha256': sha256_file(config['dataset']),
                'annotation_manifest_sha256': sha256_file(Path(config['dataset']).with_suffix('.manifest.json')),
                'train_manifest_sha256': sha256_file(TRAIN80),
                'tokens_sha256': sha256_file(directory / 'tokens.jsonl')}
    if report.get('passed') is not True or any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Prepared dataset/preflight is missing, stale, or changed')
    data = [json.loads(line) for line in (directory / 'tokens.jsonl').read_text().splitlines()]
    if len(data) != report['trajectories']:
        raise ValueError('Prepared trajectory count changed')
    return data, report


def execute(config, gpu, resume, config_path=CONFIG):
    if gpu is None or gpu < 0:
        raise ValueError('Training requires an explicit --gpu N assignment')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This prepared recipe is single-GPU; distributed batch accounting needs a separate setup')
    output = Path(config['output_dir']).resolve()
    dataset_hash = sha256_file(config['dataset'])
    run_identity = {'config_sha256': digest(config), 'dataset_sha256': dataset_hash,
                    'annotation_manifest_sha256': sha256_file(Path(config['dataset']).with_suffix('.manifest.json')),
                    'train_manifest_sha256': sha256_file(TRAIN80)}
    if resume:
        resume = Path(resume).resolve()
        if resume.parent != output or not (resume / 'trainer_state.json').is_file():
            raise ValueError('Resume must name a trainer checkpoint inside this output directory')
        if not (resume / 'optimizer.pt').is_file() or not (resume / 'scheduler.pt').is_file():
            raise ValueError('Resume checkpoint is missing optimizer/scheduler state')
        previous = json.loads((output / 'semanticopt_run.json').read_text())
        if any(previous.get(key) != value for key, value in run_identity.items()):
            raise ValueError('Resume recipe or training sources changed')
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output: {output}')
    # Keep legacy ConfigSpace/estimator imports outside the isolated Unsloth environment.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run([config['preprocessing_python'], str(Path(__file__).resolve()),
                    '--config', str(Path(config_path).resolve()), '--prepare'], env=env, check=True)
    data, report = load_prepared(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    # Unsloth must patch the training libraries before their first import.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], local_files_only=True)
    if digest(tokenizer.get_vocab()) != report['tokenizer_vocab_sha256']:
        raise ValueError('Training and preprocessing tokenizer vocabularies differ')
    model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name_or_path'], max_seq_length=config['max_seq_length'],
        dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False,
    )
    if hasattr(loaded_tokenizer, 'tokenizer'):
        loaded_tokenizer = loaded_tokenizer.tokenizer
    if loaded_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError('Training loader changed the checked tokenizer vocabulary')
    model = FastLanguageModel.get_peft_model(
        model, r=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'], bias='none', target_modules=config['target_modules'],
        use_gradient_checkpointing='unsloth', random_state=config['seed'],
    )
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name or 'visual' in name for name, _ in trainable):
        raise ValueError('Expected language-model LoRA parameters only')
    config_keys = ('learning_rate', 'lr_scheduler_type', 'num_train_epochs',
                   'per_device_train_batch_size', 'gradient_accumulation_steps', 'bf16',
                   'optim', 'weight_decay', 'warmup_steps', 'max_grad_norm', 'seed',
                   'save_steps', 'save_total_limit')
    args = SFTConfig(
        output_dir=str(output), **{key: config[key] for key in config_keys},
        max_length=None, packing=False, dataset_kwargs={'skip_prepare_dataset': True},
        completion_only_loss=False, assistant_only_loss=False,
        dataloader_num_workers=0, dataloader_drop_last=False, report_to=[],
        logging_steps=1, save_strategy='steps', eval_strategy='no',
        gradient_checkpointing=True, remove_unused_columns=False,
    )
    # The precomputed -100 labels supervise every assistant turn, even on VLM backbones.
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=Dataset.from_list(data),
                         data_collator=collator, args=args)
    first = next(iter(trainer.get_train_dataloader()))
    if not (first['labels'] == -100).any() or not (first['labels'] != -100).any():
        raise ValueError('Trainer lost the assistant-only label mask')
    output.mkdir(parents=True, exist_ok=True)
    run_record = {**run_identity, **report, 'gpu': gpu,
                  'base_model': config['model_name_or_path'],
                  'trainable_parameters': sum(p.numel() for _, p in trainable),
                  'versions': {name: importlib.metadata.version(name) for name in
                               ('torch', 'transformers', 'trl', 'unsloth', 'peft')}}
    (output / 'semanticopt_run.json').write_text(json.dumps(run_record, indent=2) + '\n')
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(output / 'final_adapter'))
    tokenizer.save_pretrained(str(output / 'final_adapter'))
    trainer.save_state()
    (output / 'training_complete.json').write_text(json.dumps({
        **run_identity, 'metrics': result.metrics, 'global_step': trainer.state.global_step,
        'epoch': trainer.state.epoch, 'checkpoint_reload_verified': False,
    }, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='CPU-only source/token/mask validation')
    mode.add_argument('--prepare', action='store_true', help='CPU validation and tokenized training export')
    mode.add_argument('--execute', action='store_true', help='Explicitly launch the complete SFT epoch')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check or args.prepare:
        result = check(config, prepare=args.prepare)
    elif args.execute:
        execute(config, args.gpu, args.resume, args.config)
        return
    else:
        result = {'mode': 'plan_only', 'training_started': False, 'config': config,
                  'annotated_trajectories_present': Path(config['dataset']).is_file(),
                  'training_labels': ['next_configuration', 'mean', 'std',
                                      'failure_probability only after a prior failure'],
                  'launch_requires': ['annotated train80 trajectories', 'CPU preflight',
                                      'approved GPU allocation and --execute --gpu N'],
                  'automatic_launch_after_evaluations': False}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
