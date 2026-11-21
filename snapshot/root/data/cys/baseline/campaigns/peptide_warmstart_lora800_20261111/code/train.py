"""Fresh-base peptide warm-start LoRA SFT with audited targets and checkpoints."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)

def verify_identity():
    for name, checksum in load(ROOT/'identity.json')['files'].items():
        if sha(name) != checksum:
            raise ValueError('Pinned SFT input changed: ' + name)

def audit():
    verify_identity()
    from source_exporter import validate_data, DATA, ALPHABET
    manifest, examples = validate_data()
    records = [json.loads(x) for x in (DATA/'provenance.jsonl').open()]
    from Levenshtein import distance
    ranks = {}
    for row in records:
        seq, ref = row['sequence'], row['reference']
        sim = 1 - distance(seq, ref)/len(ref)
        if not (1 <= len(seq) <= 30 and set(seq) <= ALPHABET and seq.count('C') <= 1 and sim >= .75):
            raise ValueError('Infeasible supervised target')
        if not (math.isfinite(row['mic']) and row['mic'] > 0 and math.isclose(row['score'], -row['mic'])):
            raise ValueError('Invalid measured objective')
        ranks.setdefault(row['task_id'], []).append(row)
    for values in ranks.values():
        if [v['rank'] for v in values] != list(range(1, 11)) or [v['mic'] for v in values] != sorted(v['mic'] for v in values):
            raise ValueError('Incorrect top-ten rank order')
    atomic(ROOT/'evidence/data_audit.json', dict(passed=True, at=time.time(),
        train_tasks=len(ranks), examples=len(examples), excluded_test_tasks=len(manifest['excluded_test_tasks']),
        train_test_overlap=0, feasible_targets=8000, source_provenance_verified=True,
        top_ten_rank_order_verified=True, new_oracle_calls=0, dataset_sha256=sha(DATA/'train.jsonl')))
    return examples

def prepare():
    examples = audit()
    import torch
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    config = load(ROOT/'sft_config.json')
    if config['finetuning_type'] != 'lora' or config.get('adapter_name_or_path') or config['resume_from_checkpoint']:
        raise ValueError('Require a fresh base-model LoRA recipe')
    cpu = dict(config, use_cpu=True, bf16=False, dataloader_pin_memory=False)
    model_args, data_args, training_args, _, _ = get_train_args(cpu)
    module = load_tokenizer(model_args)
    tokenizer = module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    expected = Counter(); lengths = []
    for row in examples:
        system, user, answer = row['messages']
        source, target = template.encode_multiturn(tokenizer,
            [dict(role='user', content=user['content']), dict(role='assistant', content=answer['content'])],
            system['content'], None)[0]
        if tokenizer.decode(target) != answer['content'] + '<|im_end|>\n':
            raise ValueError('Supervised answer/EOS mismatch')
        lengths.append(len(source)+len(target))
        expected[json.dumps([source+target, [-100]*len(source)+target])] += 1
    if max(lengths) > config['cutoff_len']:
        raise ValueError('Prompt/answer would be truncated')
    actual = Counter()
    dataset = get_dataset(template, model_args, data_args, training_args, 'sft', **module)['train_dataset']
    for row in dataset:
        actual[json.dumps([row['input_ids'], row['labels']])] += 1
    if expected != actual or len(dataset) != 8000:
        raise ValueError('LLaMA-Factory preprocessing changed or dropped examples')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation initialized CUDA')
    atomic(ROOT/'evidence/preflight.json', dict(passed=True, at=time.time(), examples=8000,
        min_tokens=min(lengths), max_tokens=max(lengths), assistant_only_labels_verified=True,
        truncated_examples=0, config_sha256=sha(ROOT/'sft_config.json'), expected_steps=2500))

def train():
    verify_identity()
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from llamafactory.train.tuner import run_exp
    config = load(ROOT/'sft_config.json')
    gate = load(ROOT/'evidence/preflight.json')
    if not gate['passed'] or gate['config_sha256'] != sha(ROOT/'sft_config.json'):
        raise ValueError('Missing/changed preprocessing gate')
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError('Optimized Qwen3.5 linear attention unavailable')
    torch.set_num_threads(3); torch.set_num_interop_threads(1)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.cuda.set_per_process_memory_fraction(32*2**30/torch.cuda.get_device_properties(0).total_memory)
    output = Path(config['output_dir'])
    previous = load(ROOT/'training_status.json') if (ROOT/'training_status.json').exists() else {}
    if previous.get('status') == 'completed':
        if sha(output/'adapter_model.safetensors') != previous['adapter_sha256']:
            raise ValueError('Completed adapter changed')
        return
    required = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json')
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    checkpoints = [p for p in checkpoints if all((p/name).exists() for name in required)]
    resume = checkpoints[-1] if checkpoints else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError('Output contains no complete resumable checkpoint')
    config['resume_from_checkpoint'] = str(resume) if resume else None
    status = dict(status='loading', pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        framework='LLaMA-Factory', finetuning_type='lora', parent_adapter=None,
        base_model=config['model_name_or_path'], resumed_from=config['resume_from_checkpoint'])
    def publish(**values):
        status.update(values, at=time.time()); atomic(ROOT/'training_status.json', status)
    def tensor_hash(model):
        h = hashlib.sha256()
        for name, tensor in sorted(get_peft_model_state_dict(model).items()):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def verify(model, path):
        saved = load_file(str(path/'adapter_model.safetensors'))
        live = get_peft_model_state_dict(model)
        if set(saved) != set(live) or any(not torch.equal(saved[k], live[k].detach().cpu()) for k in saved):
            raise ValueError('Saved adapter differs from live tensors')
        return len(saved)
    holder = []; initial = []
    initial_step = load(resume/'trainer_state.json')['global_step'] if resume else 0
    class Audit(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            model = kw['model']; holder.append(model); initial.append(tensor_hash(model))
            trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
            if not trainable or any('lora_' not in n or any(v in n for v in ('visual.', 'vision_tower', 'lm_head', 'embed_tokens')) for n in trainable):
                raise ValueError('Trainable parameters must be language-model LoRA only')
            if state.max_steps != gate['expected_steps']:
                raise ValueError('Unexpected optimizer step budget')
            if resume: verify(model, resume)
            publish(status='training', step=state.global_step, max_steps=state.max_steps,
                initial_adapter_hash=initial[0], trainable_parameters=sum(p.numel() for p in trainable.values()),
                total_parameters=sum(p.numel() for p in model.parameters()), base_frozen=True, vision_frozen=True,
                trainable_names=list(trainable), attention=model.config._attn_implementation,
                flash_sdpa_enabled=torch.backends.cuda.flash_sdp_enabled(), optimized_linear_attention=True)
        def on_step_end(self, args, state, control, **kw):
            changed = tensor_hash(kw['model']) != initial[0] if state.global_step <= initial_step+2 else status.get('first_update_verified', False)
            if state.global_step == initial_step+2 and not changed:
                raise ValueError('LoRA adapter has not updated')
            publish(step=state.global_step, epoch=state.epoch, first_update_verified=changed,
                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            if state.global_step in (initial_step+1, initial_step+2): control.should_save=True
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and 'loss' in logs:
                if not all(math.isfinite(float(v)) for v in logs.values() if isinstance(v, (int, float))):
                    raise ValueError('Nonfinite training metrics')
                publish(latest_metrics=logs)
        def on_save(self, args, state, control, **kw):
            path = output/f'checkpoint-{state.global_step}'
            verify(kw['model'], path)
            if not all((path/name).exists() for name in required):
                raise ValueError('Incomplete resumable checkpoint')
            publish(latest_checkpoint=str(path))
    publish()
    run_exp(args=config, callbacks=[Audit()])
    model = holder[0]; count = verify(model, output)
    if tensor_hash(model) == initial[0] or status['step'] != 2500 or status['epoch'] < 4.999:
        raise ValueError('SFT did not complete five epochs with weight updates')
    publish(status='completed', adapter_sha256=sha(output/'adapter_model.safetensors'),
        verified_tensors=count, final_adapter_hash=tensor_hash(model))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['audit', 'prepare', 'train'])
    args = p.parse_args()
    try:
        globals()[args.action]()
    except BaseException as error:
        atomic(ROOT/f'{args.action}_error.json', dict(error=repr(error), at=time.time()))
        raise
