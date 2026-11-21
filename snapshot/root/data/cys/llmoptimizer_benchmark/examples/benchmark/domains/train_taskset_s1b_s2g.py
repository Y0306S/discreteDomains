#!/usr/bin/env python3
"""Prepare, supervise, and verify full TaskSet S1B -> S2-G and ORPO-only training."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from taskset_rationale_validation import (GENERATION_LIMITS, VALIDATION_VERSION,
                                        generation_details, generation_error,
                                        rationale_text, record_error)
from taskset_rationale_pool import (checked_shard, pool_progress, shard_path,
                                   validate_pool, worker_status_path)

MODEL = '/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B'
CONFIGS = Path('/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def validate_data(data):
    manifest = json.loads((data / 'manifest.json').read_text())
    if (manifest['source_tasks'] != 838 or manifest['prompt_mode'] != 'p5_h0'
            or manifest['min_successful_prefix'] != 2 or manifest['nesterov_enabled']):
        raise ValueError('Unexpected TaskSet source/prompt contract')
    for name, expected in manifest['files'].items():
        if digest(data / name) != expected:
            raise ValueError('Changed training data: ' + name)
    for name, expected in {**manifest['input_sha256'], **manifest['code_sha256']}.items():
        if digest(name) != expected:
            raise ValueError('Changed frozen input or data-generation code: ' + name)
    return manifest


def recipe(stage, data, output):
    import yaml

    path = (CONFIGS / 'llamafactory_qwen35_9b_lora_p5_s1b_ep7p5.yaml' if stage == 'sft' else
            CONFIGS / 's1b_s2g_orpo_sweep/s1b_s2g_orpo_lr1e5_b30.yaml')
    config = yaml.safe_load(path.read_text())
    config.update(model_name_or_path=MODEL, dataset_dir=str(data), dataset='taskset_s1b' if stage == 'sft' else 'taskset_s2g',
                  output_dir=str(output), overwrite_output_dir=False, report_to='none', plot_loss=False,
                  save_steps=500, save_total_limit=2, logging_steps=1, save_only_model=False,
                  per_device_train_batch_size=1, gradient_accumulation_steps=16 if stage == 'sft' else 8,
                  preprocessing_num_workers=1, dataloader_num_workers=0, overwrite_cache=False,
                  train_on_prompt=False, val_size=0.0, packing=False, max_steps=-1,
                  flash_attn='sdpa', resume_from_checkpoint=None)
    config.pop('adapter_name_or_path', None)
    config.pop('run_name', None)
    return config, path


def prepare(stage, data, campaign):
    manifest = validate_data(data)
    import torch
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    config, original = recipe(stage, data, campaign / 'models' / stage)
    import yaml
    original_config = yaml.safe_load(original.read_text())
    if stage == 'sft':
        validate_rationale_gate(data, campaign)
        sft_manifest = json.loads((data / 'sft_manifest.json').read_text())
        if (sft_manifest['examples'] != manifest['sft_examples']
                or digest(data / 'sft_train.jsonl') != sft_manifest['training_sha256']
                or digest(data / 'rationales.jsonl') != sft_manifest['rationales_sha256']):
            raise ValueError('SFT assembly changed')
    cpu_config = dict(config, bf16=False, use_cpu=True, dataloader_pin_memory=False, cutoff_len=131072)
    model_args, data_args, training_args, _, _ = get_train_args(cpu_config)
    module = load_tokenizer(model_args)
    tokenizer = module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    rows = jsonl(data / ('sft_train.jsonl' if stage == 'sft' else 'orpo_train.jsonl'))
    expected, lengths = Counter(), []
    for row in rows:
        if stage == 'sft':
            system, user, answer = row['messages']
            source, target = template.encode_multiturn(tokenizer, [user, answer], system['content'], None)[0]
            expected[digest_arrays([source + target, [-100] * len(source) + target])] += 1
            lengths.append(len(source) + len(target))
            if tokenizer.decode(target) != answer['content'] + '<|im_end|>\n':
                raise ValueError('SFT answer tokenization changed')
        else:
            system, user = row['conversations']
            arms = []
            for side in ('chosen', 'rejected'):
                answer = dict(role='assistant', content=row[side]['value'])
                source, target = template.encode_multiturn(tokenizer, [dict(role='user', content=user['value']), answer], system['value'], None)[0]
                if tokenizer.decode(target) != answer['content'] + '<|im_end|>\n':
                    raise ValueError('ORPO answer tokenization changed')
                arms.extend([source + target, [-100] * len(source) + target])
                lengths.append(len(source) + len(target))
            expected[digest_arrays(arms)] += 1
    cutoff = max(6016, math.ceil((max(lengths) + (256 if stage == 'orpo' else 0)) / 1024) * 1024)
    if stage == 'sft':
        cutoff = max(cutoff, json.loads((campaign / 'orpo_preflight.json').read_text())['cutoff_len'])
    data_args.cutoff_len = config['cutoff_len'] = cutoff
    dataset = get_dataset(template, model_args, data_args, training_args,
                          'sft' if stage == 'sft' else 'rm', **module)['train_dataset']
    actual = Counter()
    for row in dataset:
        if stage == 'sft':
            actual[digest_arrays([row['input_ids'], row['labels']])] += 1
        else:
            actual[digest_arrays([row['chosen_input_ids'], row['chosen_labels'], row['rejected_input_ids'], row['rejected_labels']])] += 1
    if actual != expected or len(dataset) != len(rows):
        raise ValueError('LLaMA-Factory changed, dropped, or truncated supervised tokens')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight initialized CUDA')
    atomic_json(campaign / f'{stage}_config.json', config)
    report = dict(passed=True, examples=len(rows), min_tokens=min(lengths), max_tokens=max(lengths),
                  cutoff_len=cutoff, truncated_examples=0, assistant_only_labels_verified=True,
                  data_manifest_sha256=digest(data / 'manifest.json'), data_file_sha256=digest(data / ('sft_train.jsonl' if stage == 'sft' else 'orpo_train.jsonl')),
                  ml_recipe=str(original), ml_recipe_sha256=digest(original), effective_batch=16 if stage == 'sft' else 8,
                  epochs=config['num_train_epochs'], recipe_changes={k: dict(ml=original_config.get(k), taskset=v)
                  for k, v in config.items() if original_config.get(k) != v})
    atomic_json(campaign / f'{stage}_preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def digest_arrays(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def rationale_cache(data):
    cache = data / 'rationales.jsonl'
    return {r['event_index']: r for r in jsonl(cache)} if cache.exists() else {}


def audit_rationales(data, campaign, tokenizer, events, require_complete=False):
    records = rationale_cache(data)
    rejected, verified, inferred = [], 0, 0
    for i, row in records.items():
        if i < 0 or i >= len(events) or row['event_sha256'] != digest_arrays(events[i]):
            raise ValueError('Rationale resume input changed')
        error = record_error(row, tokenizer)
        if error:
            rejected.append(dict(event_index=i, reason=error))
        elif row.get('validation_version') == VALIDATION_VERSION:
            verified += 1
        else:
            inferred += 1
    report = dict(passed=not rejected and len(records) == len(events),
                  validation_version=VALIDATION_VERSION, examples=len(records), expected_examples=len(events),
                  eos_verified_records=verified, legacy_completion_inferred_records=inferred,
                  generation_limits=list(GENERATION_LIMITS), script_sha256=digest(__file__),
                  validation_helper_sha256=digest(Path(__file__).with_name('taskset_rationale_validation.py')),
                  rejected=rejected, events_sha256=digest(data / 'sft_events.jsonl'),
                  rationales_sha256=digest(data / 'rationales.jsonl') if records else None,
                  updated_at=time.time())
    atomic_json(campaign / 'rationale_audit.json', report)
    if require_complete and not report['passed']:
        raise ValueError('Rationale completion audit failed: ' + json.dumps(report))
    for rejection in rejected:
        del records[rejection['event_index']]
    return records, report


def validate_rationale_gate(data, campaign):
    audit_path = campaign / 'rationale_audit.json'
    audit = json.loads(audit_path.read_text())
    sft_manifest = json.loads((data / 'sft_manifest.json').read_text())
    if (not audit['passed'] or audit['validation_version'] != VALIDATION_VERSION
            or audit['examples'] != audit['expected_examples']
            or audit['rationales_sha256'] != digest(data / 'rationales.jsonl')
            or audit['events_sha256'] != digest(data / 'sft_events.jsonl')
            or sft_manifest.get('rationale_audit_sha256') != digest(audit_path)
            or sft_manifest['examples'] != audit['examples']
            or sft_manifest['training_sha256'] != digest(data / 'sft_train.jsonl')):
        raise ValueError('A complete, unchanged rationale audit is required before SFT')


def generate_rationales(data, campaign, worker_id=None):
    validate_data(data)
    events = jsonl(data / 'sft_events.jsonl')
    pool_path = campaign / 'rationale_pool.json'
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else None
    pool_sha256 = digest(pool_path) if pool else None
    if pool:
        validate_pool(pool, len(events))
        if pool['events_sha256'] != digest(data / 'sft_events.jsonl'):
            raise ValueError('Rationale pool source changed')
        worker_id = worker_id or 'local'
        allocation = pool['workers'][worker_id]
        cache = shard_path(campaign, worker_id)
        cache.parent.mkdir(parents=True, exist_ok=True)
        existing = {r['event_index']: r for r in jsonl(cache)} if cache.exists() else {}
        selected_indices = allocation['indices']
        status_path = worker_status_path(campaign, worker_id)
    else:
        if worker_id:
            raise ValueError('A worker requires a rationale pool plan')
        cache = data / 'rationales.jsonl'
        existing = rationale_cache(data)
        selected_indices = list(range(len(events)))
        status_path = campaign / 'rationale_progress.json'

    def progress(status):
        atomic_json(status_path, dict(status=status, completed=len(existing), total=len(selected_indices),
                    pool_sha256=pool_sha256, pid=os.getpid(), updated_at=time.time()))
        if pool and worker_id == 'local':
            combined = pool_progress(campaign, pool, pool_sha256)
            atomic_json(campaign / 'rationale_progress.json', dict(combined, status=status, updated_at=time.time()))

    progress('loading_model')
    for i, row in existing.items():
        if row['event_sha256'] != digest_arrays(events[i]):
            raise ValueError('Rationale resume input changed')
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(allocation['memory_fraction'] if pool else 0.85)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if pool:
        existing = {i: r for i, r in existing.items() if not record_error(r, tokenizer)}
        if any(i not in selected_indices for i in existing):
            raise ValueError('Rationale worker cache contains unassigned events')
    else:
        existing, audit = audit_rationales(data, campaign, tokenizer, events)
        if audit['rejected']:
            atomic_json(campaign / f'rationale_repairs_{int(time.time())}.json', audit)
            print('Regenerating rejected cached rationales: ' + json.dumps(audit['rejected']), flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                  attn_implementation='sdpa', trust_remote_code=True).to('cuda').eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [tokenizer.eos_token_id])
    if not eos_ids or None in eos_ids:
        raise ValueError('Rationale generation requires known EOS token IDs')

    def render_prompt(index, retry=False):
        messages = [dict(message) for message in events[index]['rationale_messages']]
        messages[-1]['content'] += ('\nReturn one complete plain-English sentence of at most 60 words. '
                                  "Start with 'Rationale:' and end with sentence punctuation. "
                                  'Do not output JSON, XML tags, markdown, or a list.')
        if retry:
            messages[-1]['content'] += '\nThe previous attempt was incomplete or malformed. Be concise and finish the sentence.'
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate(prompts, limit):
        inputs = tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                       eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
        details = []
        for prompt, ids, mask in zip(prompts, generated[:, inputs['input_ids'].shape[1]:], inputs['attention_mask']):
            item = generation_details(tokenizer, ids.tolist(), limit, eos_ids)
            item.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        prompt_input_tokens=int(mask.sum().item()))
            details.append(item)
        return details

    batch_size = allocation['batch_size'] if pool else 4
    for start in range(0, len(selected_indices), batch_size):
        indices = [i for i in selected_indices[start:start + batch_size] if i not in existing]
        if not indices:
            continue
        responses = generate([render_prompt(i) for i in indices], GENERATION_LIMITS[0])
        with cache.open('a') as stream:
            for i, response in zip(indices, responses):
                generations = [response]
                for limit in GENERATION_LIMITS[1:]:
                    if not generation_error(generations[-1]):
                        break
                    generations.append(generate([render_prompt(i, retry=True)], limit)[0])
                if generation_error(generations[-1]):
                    atomic_json(campaign / f'rationale_failure_{i}.json', dict(event_index=i, generation_attempts=generations))
                    raise ValueError(f'Incomplete or malformed rationale at event {i}')
                record = dict(event_index=i, event_sha256=digest_arrays(events[i]),
                              rationale=rationale_text(generations[-1]['text']),
                              raw_response=generations[0]['text'], attempts=[g['text'] for g in generations],
                              generation_attempts=generations, validation_version=VALIDATION_VERSION,
                              model=MODEL, temperature=0, max_new_tokens=generations[-1]['max_new_tokens'])
                if pool:
                    record.update(worker_id=worker_id, pool_sha256=pool_sha256)
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                existing[i] = record
            stream.flush()
            os.fsync(stream.fileno())
        progress('running')
        print(f'Rationales {worker_id or "local"}: {len(existing)}/{len(selected_indices)}', flush=True)
    if pool:
        progress('completed')
        del model
        torch.cuda.empty_cache()
        if worker_id != 'local':
            return
        while True:
            combined = pool_progress(campaign, pool, pool_sha256)
            failed = {name: s for name, s in combined['workers'].items() if s.get('status') == 'failed'}
            if failed:
                raise RuntimeError('Rationale pool worker failed: ' + json.dumps(failed))
            atomic_json(campaign / 'rationale_progress.json', dict(combined, status='waiting_for_workers', updated_at=time.time()))
            if all(s.get('status') == 'completed' for s in combined['workers'].values()):
                break
            time.sleep(10)
        original_cache = data / 'rationales.jsonl'
        retained = rationale_cache(data)
        with original_cache.open('a') as stream:
            for name, worker in pool['workers'].items():
                records = checked_shard(jsonl(shard_path(campaign, name)), worker['indices'], events, tokenizer, digest_arrays)
                for i, record in sorted(records.items()):
                    if record.get('pool_sha256') != pool_sha256:
                        raise ValueError('Rationale shard belongs to another pool')
                    if i not in retained or record_error(retained[i], tokenizer):
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
                        retained[i] = record
            stream.flush()
            os.fsync(stream.fileno())
        cache = original_cache
    existing, audit = audit_rationales(data, campaign, tokenizer, events, require_complete=True)
    output = data / 'sft_train.jsonl'
    temporary = output.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for i, event in enumerate(events):
            answer = 'Rationale: ' + existing[i]['rationale'] + '\n\nJSON:\n' + json.dumps({'config': event['config']}, sort_keys=True, allow_nan=False)
            row = dict(messages=event['messages'] + [dict(role='assistant', content=answer)])
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(output)
    atomic_json(data / 'sft_manifest.json', dict(examples=len(events), events_sha256=digest(data / 'sft_events.jsonl'),
                rationales_sha256=digest(cache), training_sha256=digest(output), model=MODEL, format='rationale_json',
                rationale_audit_sha256=digest(campaign / 'rationale_audit.json'), rationale_validation_version=VALIDATION_VERSION))
    atomic_json(campaign / 'rationale_progress.json', dict(status='completed', completed=len(events), total=len(events), updated_at=time.time()))


def prepare_rationale_pool(data, campaign):
    validate_data(data)
    if (campaign / 'rationale_pool.json').exists():
        raise FileExistsError('A rationale pool is already configured')
    from transformers import AutoTokenizer

    events = jsonl(data / 'sft_events.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    retained, _ = audit_rationales(data, campaign, tokenizer, events)
    pending = [i for i in range(len(events)) if i not in retained]
    workers = {'local': dict(host='LLMOptimizer2', gpu=0, cpus=[38, 39], memory_fraction=0.85, batch_size=4)}
    for gpu in range(4):
        workers[f'remote{gpu}'] = dict(host='LLMOptimizer3', gpu=gpu, cpus=[2 * gpu, 2 * gpu + 1],
                                      memory_fraction=0.8, batch_size=4)
    for position, worker in enumerate(workers.values()):
        worker['indices'] = pending[position::len(workers)]
    plan = dict(data=str(data), events_sha256=digest(data / 'sft_events.jsonl'), total_examples=len(events),
                retained_examples=len(retained), pending_indices=pending, workers=workers,
                model=MODEL, created_at=time.time(), validation_version=VALIDATION_VERSION)
    validate_pool(plan, len(events))
    atomic_json(campaign / 'rationale_pool.json', plan)
    print(json.dumps(dict(retained_examples=len(retained), pending_examples=len(pending),
                         workers={name: dict(worker, indices=len(worker['indices'])) for name, worker in workers.items()}), indent=2))


def rationale_pool_worker(args):
    pool_path = args.campaign / 'rationale_pool.json'
    pool = json.loads(pool_path.read_text())
    allocation = pool['workers'][args.worker_id]
    status_path = worker_status_path(args.campaign, args.worker_id)
    locks = []
    try:
        os.sched_setaffinity(0, set(allocation['cpus']))
        os.nice(5)
        lock_root = Path('/root/data/logs/taskset_major_families_cpu_20260929/locks')
        for path in [args.campaign / f'rationale_{args.worker_id}.lock',
                     *(lock_root / f"core.{allocation['host']}.{cpu}.lock" for cpu in allocation['cpus'])]:
            handle = path.open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        generate_rationales(args.data, args.campaign, args.worker_id)
    except BaseException as error:
        atomic_json(status_path, dict(status='failed', reason=str(error), pid=os.getpid(),
                    pool_sha256=digest(pool_path), updated_at=time.time()))
        raise


def launch_rationale_pool(data, campaign, host):
    validate_data(data)
    pool_path = campaign / 'rationale_pool.json'
    pool = json.loads(pool_path.read_text())
    validate_pool(pool, len(jsonl(data / 'sft_events.jsonl')))
    workers = {name: worker for name, worker in pool['workers'].items() if worker['host'] == host and name != 'local'}
    if not workers:
        raise ValueError('No remote workers allocated to this host')
    launch_path = campaign / f'rationale_pool_launch_{host}.json'
    with (campaign / f'rationale_pool_launch_{host}.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if launch_path.exists():
            raise FileExistsError('Remote rationale workers have already been launched')
        gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        gpu_free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in gpu_rows.splitlines()}
        for worker in workers.values():
            if gpu_free.get(worker['gpu'], 0) < 40000:
                raise RuntimeError('A remote rationale GPU has less than 40,000 MiB free')
        result = dict(host=host, pool_sha256=digest(pool_path), script_sha256=digest(__file__), workers={})
        for name, worker in workers.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker['gpu']), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            command = [sys.executable, str(Path(__file__).resolve()), 'rationale-worker', '--worker-id', name,
                       '--data', str(data), '--campaign', str(campaign)]
            with (campaign / f'rationale_{name}.log').open('a') as stream:
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            result['workers'][name] = dict(pid=child.pid, gpu=worker['gpu'], cpus=worker['cpus'], command=command)
        atomic_json(launch_path, result)
        print(json.dumps(result, indent=2))


def tensor_digest(state):
    result = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        result.update(key.encode())
        result.update(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return result.hexdigest()


def verify_adapter(model, path, adapter_name='default'):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    actual = get_peft_model_state_dict(model, adapter_name=adapter_name)
    expected = load_file(str(Path(path) / 'adapter_model.safetensors'), device='cpu')
    if set(actual) != set(expected):
        raise ValueError('Adapter key coverage mismatch: ' + str(set(actual) ^ set(expected)))
    for key, value in actual.items():
        if not torch.equal(value.detach().cpu(), expected[key]):
            raise ValueError('Adapter tensor mismatch: ' + key)
    return dict(tensors=len(expected), matched=len(expected), all_tensors_exact=True,
                adapter_sha256=digest(Path(path) / 'adapter_model.safetensors'))


def worker(config_path, status_path):
    import torch
    from peft import get_peft_model_state_dict
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.85)
    config = json.loads(config_path.read_text())
    if config['stage'] == 'sft':
        validate_rationale_gate(Path(config['dataset_dir']), status_path.parent)
    atomic_json(status_path, dict(status='loading_model', model=MODEL, pid=os.getpid(),
                epochs=config['num_train_epochs'], learning_rate=config['learning_rate'], updated_at=time.time()))
    from taskset_answer_logits import install_orpo_answer_logits, install_sft_answer_logits
    if config['stage'] == 'dpo':
        install_orpo_answer_logits()
    output = Path(config['output_dir'])
    initial_step = 0
    resume = None
    if output.exists() and any(output.iterdir()):
        checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        complete = [p for p in checkpoints if all((p / name).is_file() for name in
                    ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth'))]
        if not complete:
            raise FileExistsError('Existing output has no verified resumable checkpoint')
        resume = complete[-1]
        initial_step = json.loads((resume / 'trainer_state.json').read_text())['global_step']
        config['resume_from_checkpoint'] = str(resume)
    loaded_model = []
    initial = []

    class Verification(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            model = kwargs['model']
            if config['stage'] == 'sft':
                install_sft_answer_logits(model)
            loaded_model.append(model)
            initial.append(tensor_digest(get_peft_model_state_dict(model)))
            parent = config.get('adapter_name_or_path')
            coverage = verify_adapter(model, resume or parent) if (resume or parent) else None
            atomic_json(status_path, dict(status='training', step=state.global_step, max_steps=state.max_steps,
                       epochs=config['num_train_epochs'], parent_adapter=parent, parent_verification=coverage,
                       resumed_from=str(resume) if resume else None,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       initial_adapter_digest=initial[0], answer_only_vocab_projection=True,
                       initial_learning_rates=[g['lr'] for g in kwargs['optimizer'].param_groups],
                       script_sha256=digest(__file__),
                       updated_at=time.time()))

        def on_step_end(self, args, state, control, **kwargs):
            previous = json.loads(status_path.read_text())
            previous.update(step=state.global_step, max_steps=state.max_steps, epoch=state.epoch,
                       updated_at=time.time(), peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
            if state.global_step in (initial_step + 1, initial_step + 2):
                previous['first_update_verified'] = tensor_digest(get_peft_model_state_dict(kwargs['model'])) != initial[0]
                if not previous['first_update_verified'] and state.global_step == initial_step + 2:
                    raise RuntimeError('The optimizer did not update adapter tensors')
                if previous['first_update_verified']:
                    control.should_save = True
            atomic_json(status_path, previous)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                if not math.isfinite(float(logs['loss'])):
                    raise FloatingPointError('Non-finite training loss')
                previous = json.loads(status_path.read_text())
                previous['latest_metrics'] = logs
                atomic_json(status_path, previous)

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            verify_adapter(kwargs['model'], checkpoint)
            if not (checkpoint / 'optimizer.pt').exists() or not (checkpoint / 'trainer_state.json').exists():
                raise ValueError('Checkpoint is not resumable')

    run_exp(args=config, callbacks=[Verification()])
    model = loaded_model[0]
    final = verify_adapter(model, config['output_dir'])
    if tensor_digest(get_peft_model_state_dict(model)) == initial[0]:
        raise RuntimeError('Final adapter equals the initial adapter')
    model.load_adapter(config['output_dir'], adapter_name='reload_check', is_trainable=False)
    reload_check = verify_adapter(model, config['output_dir'], adapter_name='reload_check')
    previous = json.loads(status_path.read_text())
    previous.update(status='completed', final_verification=final, reload_verification=reload_check, updated_at=time.time())
    atomic_json(status_path, previous)


def branch(args):
    cpus = {int(x) for x in args.cpus.split(',')}
    os.sched_setaffinity(0, cpus)
    os.nice(5)
    args.campaign.mkdir(parents=True, exist_ok=True)
    validate_data(args.data)
    preflight = json.loads((args.campaign / 'orpo_preflight.json').read_text())
    if not preflight['passed'] or preflight['data_manifest_sha256'] != digest(args.data / 'manifest.json'):
        raise ValueError('Full ORPO token preflight is required')
    if shutil.disk_usage(args.campaign).free < 30 * 2**30:
        raise RuntimeError('Less than 30 GiB free campaign storage')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    gpu_free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in gpu_rows.splitlines()}
    if gpu_free.get(args.gpu, 0) < 65 * 1024:
        raise RuntimeError('The selected GPU has less than 65 GiB free')
    locks = []
    lock_root = Path('/root/data/logs/taskset_major_families_cpu_20260929/locks')
    lock_root.mkdir(parents=True, exist_ok=True)
    for path in [args.campaign / f'{args.branch}.lock', *(lock_root / f'core.LLMOptimizer2.{cpu}.lock' for cpu in cpus)]:
        handle = path.open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS=str(len(cpus)),
               MKL_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(args.campaign / 'hf_home'),
               HF_DATASETS_CACHE=str(args.campaign / 'hf_cache'), TORCH_EXTENSIONS_DIR=str(args.campaign / 'torch_extensions'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', DISABLE_VERSION_CHECK='1')
    child = None
    status = args.campaign / f'{args.branch}_queue.json'

    def run(command, log, gpu=True):
        nonlocal child
        child_env = dict(env)
        if not gpu:
            child_env['CUDA_VISIBLE_DEVICES'] = ''
        atomic_json(status, dict(status='running', phase=log, pid=os.getpid(), gpu=args.gpu, cpus=sorted(cpus), updated_at=time.time()))
        with (args.campaign / f'{args.branch}_{log}.log').open('a') as stream:
            child = subprocess.Popen(command, env=child_env, stdin=subprocess.DEVNULL, stdout=stream,
                      stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(h.fileno() for h in locks))
        code = child.wait()
        if code:
            raise RuntimeError(f'{log} exited with code {code}')

    def stop(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, str(Path(__file__).resolve())]
    try:
        if args.branch == 'chain':
            run(command + ['rationales', '--data', str(args.data), '--campaign', str(args.campaign)], 'rationales')
            run(command + ['prepare', '--stage', 'sft', '--data', str(args.data), '--campaign', str(args.campaign)], 'sft_preflight', False)
            config = json.loads((args.campaign / 'sft_config.json').read_text())
            config['output_dir'] = str(args.campaign / 'models/sft')
            path = args.campaign / 'chain_sft_config.json'
            atomic_json(path, config)
            run(command + ['worker', '--config', str(path), '--status', str(args.campaign / 'sft_status.json')], 'sft')
        config = json.loads((args.campaign / 'orpo_config.json').read_text())
        config['output_dir'] = str(args.campaign / f'models/{args.branch}_orpo')
        if args.branch == 'chain':
            config['adapter_name_or_path'] = str(args.campaign / 'models/sft')
        path = args.campaign / f'{args.branch}_orpo_config.json'
        atomic_json(path, config)
        run(command + ['worker', '--config', str(path), '--status', str(args.campaign / f'{args.branch}_orpo_status.json')], 'orpo')
        atomic_json(status, dict(status='completed', gpu=args.gpu, cpus=sorted(cpus), updated_at=time.time()))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
        atomic_json(status, dict(status='failed', reason=str(error), gpu=args.gpu, cpus=sorted(cpus), updated_at=time.time()))
        raise


def launch(data, campaign, selected_branch=None):
    validate_data(data)
    preflight = json.loads((campaign / 'orpo_preflight.json').read_text())
    if not preflight['passed'] or preflight['data_manifest_sha256'] != digest(data / 'manifest.json'):
        raise ValueError('The complete ORPO token preflight must pass before launch')
    with (campaign / 'launch.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = campaign / 'launch.json'
        allocations = {'orpo_only': (1, '40,41'), 'chain': (0, '38,39')}
        old = json.loads(record.read_text()) if record.exists() else None
        if old and selected_branch is None:
            raise FileExistsError('This campaign has already been launched; specify the stopped branch explicitly')
        if selected_branch:
            allocations = {selected_branch: allocations[selected_branch]}
        if old:
            for name in allocations:
                pid = old['branches'][name]['pid']
                stat = Path(f'/proc/{pid}/stat')
                if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                    raise RuntimeError('The selected branch is still active')
            atomic_json(campaign / f'launch_previous_{int(time.time())}.json', old)
        result = dict(data=str(data), data_manifest_sha256=digest(data / 'manifest.json'),
                      script_sha256=digest(__file__), model=MODEL, launched_at=time.time(), branches={})
        if old:
            result['branches'] = old['branches']
            for branch_record in result['branches'].values():
                branch_record.setdefault('script_sha256', old['script_sha256'])
        result['answer_logits_helper_sha256'] = digest(Path(__file__).with_name('taskset_answer_logits.py'))
        result['rationale_validation_helper_sha256'] = digest(Path(__file__).with_name('taskset_rationale_validation.py'))
        for name, (gpu, cpus) in allocations.items():
            command = [sys.executable, str(Path(__file__).resolve()), 'branch', '--branch', name,
                       '--gpu', str(gpu), '--cpus', cpus, '--data', str(data), '--campaign', str(campaign)]
            with (campaign / f'{name}_supervisor.log').open('a') as stream:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            result['branches'][name] = dict(pid=child.pid, gpu=gpu, cpus=cpus, command=command,
                                           script_sha256=digest(__file__))
        atomic_json(record, result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'rationales', 'worker', 'branch', 'launch',
                                        'rationale-pool-prepare', 'rationale-pool-launch', 'rationale-worker'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--stage', choices=['sft', 'orpo'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--status', type=Path)
    parser.add_argument('--branch', choices=['chain', 'orpo_only'])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cpus')
    parser.add_argument('--worker-id')
    parser.add_argument('--host')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.stage, args.data, args.campaign)
    elif args.mode == 'rationales':
        generate_rationales(args.data, args.campaign)
    elif args.mode == 'worker':
        worker(args.config, args.status)
    elif args.mode == 'branch':
        branch(args)
    elif args.mode == 'rationale-pool-prepare':
        prepare_rationale_pool(args.data, args.campaign)
    elif args.mode == 'rationale-pool-launch':
        launch_rationale_pool(args.data, args.campaign, args.host)
    elif args.mode == 'rationale-worker':
        rationale_pool_worker(args)
    else:
        launch(args.data, args.campaign, args.branch)
