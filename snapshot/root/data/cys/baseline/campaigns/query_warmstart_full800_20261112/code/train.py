"""Fresh-base one-rank full SFT with audited targets and ZeRO3 checkpoints."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STEPS=lambda: load(ROOT/'evidence/preflight.json')['expected_steps']

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
    gate=load(ROOT/'evidence/data_audit.json')
    if not gate['passed'] or gate['sft_sha256']!=sha(ROOT/'data/train.jsonl') or gate['train_test_overlap']!=0:
        raise ValueError('Independent query provenance audit missing or changed')
    examples=[json.loads(x) for x in (ROOT/'data/train.jsonl').open()]
    if len(examples)!=gate['sft_examples']:raise ValueError('Query example count changed')
    return examples

def prepare():
    examples = audit()
    import torch
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    config = load(ROOT/'sft_config.json')
    if config['finetuning_type'] != 'full' or config.get('adapter_name_or_path') or config['resume_from_checkpoint']:
        raise ValueError('Require a fresh base-model full SFT recipe')
    cpu = dict(config, use_cpu=True, bf16=False, deepspeed=None, dataloader_pin_memory=False)
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
    config['cutoff_len']=math.ceil((max(lengths)+64)/256)*256
    data_args.cutoff_len=config['cutoff_len']
    atomic(ROOT/'sft_config.json',config)
    if max(lengths) > config['cutoff_len']:
        raise ValueError('Prompt/answer would be truncated')
    actual = Counter()
    dataset = get_dataset(template, model_args, data_args, training_args, 'sft', **module)['train_dataset']
    for row in dataset:
        actual[json.dumps([row['input_ids'], row['labels']])] += 1
    if expected != actual or len(dataset) != len(examples):
        raise ValueError('LLaMA-Factory preprocessing changed or dropped examples')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation initialized CUDA')
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
    ds = HfTrainerDeepSpeedConfig(config['deepspeed'])
    ds_args = type('DSArgs', (), dict(vars(training_args), world_size=1, bf16=True))()
    ds.trainer_config_process(ds_args)
    if ds.mismatches or ds.config['train_batch_size'] != 16:
        raise ValueError('DeepSpeed effective-batch configuration mismatch')
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    kwargs = dict(lr=1e-5, betas=(.9,.999), eps=1e-8, weight_decay=.01)
    optimizer = DeepSpeedCPUAdam([parameter], adamw_mode=True, **kwargs)
    standard = torch.optim.AdamW([reference], **kwargs)
    for _ in range(3):
        parameter.grad = torch.tensor([.2,-.3]); reference.grad = parameter.grad.clone()
        optimizer.step(); standard.step()
    torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=1e-7)
    del optimizer
    atomic(ROOT/'evidence/preflight.json', dict(passed=True, at=time.time(), examples=len(examples),
        min_tokens=min(lengths), max_tokens=max(lengths), assistant_only_labels_verified=True,
        truncated_examples=0, config_sha256=sha(ROOT/'sft_config.json'), expected_steps=math.ceil(len(examples)/16)*3,
        full_language_model_sft=True, cpu_adam_verified=True, world_size=1,
        effective_batch_size=16, deepspeed_sha256=sha(ROOT/'deepspeed_config.json')))

def checkpoint_files(path, step):
    state = load(path/'trainer_state.json')
    tag = (path/'latest').read_text().strip()
    if state['global_step'] != step or tag != f'global_step{step}':
        raise ValueError('Checkpoint position mismatch')
    index = path/'model.safetensors.index.json'
    shards = set(load(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not shards or any(Path(name).name != name for name in shards):
        raise ValueError('Unsafe model shard names')
    required = [path/name for name in shards]
    for rank in range(1):
        required += [path/'rng_state.pth',
            path/tag/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt',
            path/tag/f'zero_pp_rank_{rank}_mp_rank_00_model_states.pt']
    if any(not f.is_file() or not f.stat().st_size for f in required):
        raise ValueError('Incomplete one-rank model/optimizer/RNG checkpoint')
    return shards

def train():
    import faulthandler
    trace=(ROOT/'logs/periodic_stacks.log').open('a')
    faulthandler.enable(file=trace, all_threads=True)
    # No asynchronous frame inspection: native GDB proved the recurring
    # faulthandler watchdog itself segfaults while inspecting compiled frames.
    verify_identity()
    import torch
    import deepspeed
    from functools import partial
    from transformers import TrainerCallback
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from llamafactory.train.tuner import run_exp
    rank = int(os.environ['LOCAL_RANK'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3' or int(os.environ['WORLD_SIZE']) != 1:
        raise ValueError('Require full SFT on GPU 3')
    os.sched_setaffinity(0, {42,43,44,45,46,47})
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(32*2**30/torch.cuda.get_device_properties(rank).total_memory, rank)
    torch.backends.cuda.enable_flash_sdp(True)
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError('Optimized Qwen3.5 linear attention unavailable')
    config = load(ROOT/'sft_config.json'); gate = load(ROOT/'evidence/preflight.json')
    if not gate['passed'] or gate['config_sha256'] != sha(ROOT/'sft_config.json') or gate['deepspeed_sha256'] != sha(ROOT/'deepspeed_config.json'):
        raise ValueError('Changed full-SFT preflight')
    output = Path(config['output_dir'])
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
    complete = []
    for path in checkpoints:
        try: checkpoint_files(path, int(path.name.split('-')[-1])); complete.append(path)
        except (OSError,ValueError,KeyError): pass
    resume = complete[-1] if complete else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError('Existing output has no complete one-rank checkpoint')
    config['resume_from_checkpoint'] = str(resume) if resume else None
    if resume:
        from transformers import Trainer
        from trusted_rng import restore_rng
        rng_sha = sha(resume/'rng_state.pth')
        def restore_own_rng(trainer, checkpoint):
            if trainer.args.world_size != 1:
                raise ValueError('RNG recovery requires one training rank')
            restore_rng(checkpoint, resume, rng_sha)
            atomic(ROOT/'evidence/rng_resume.json',dict(passed=True,at=time.time(),
                checkpoint=str(resume),rng_sha256=rng_sha,python_numpy_torch_cuda_restored=True))
        Trainer._load_rng_state = restore_own_rng
    status = dict(status='loading',pid=os.getpid(),rank=rank,gpu=3,framework='LLaMA-Factory',
        finetuning_type='full',parent_adapter=None,base_model=config['model_name_or_path'],
        resumed_from=config['resume_from_checkpoint'])
    def publish(**values):
        status.update(values,at=time.time()); atomic(ROOT/f'rank{rank}_status.json',status)
        if rank == 0: atomic(ROOT/'training_status.json',status)
    selected = []; initial = []; models = []
    def sample_hash():
        h = hashlib.sha256()
        for name, param in selected:
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                h.update(name.encode()); h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def verify_samples(path):
        from safetensors import safe_open
        index = path/'model.safetensors.index.json'
        mapping = load(index)['weight_map'] if index.exists() else {}
        for name, param in selected:
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                if rank == 0:
                    with safe_open(str(path/mapping.get(name,'model.safetensors')),framework='pt',device='cpu') as stream:
                        saved = stream.get_tensor(name)
                    if not torch.equal(saved,param.detach().cpu()):
                        raise ValueError('Saved full-model weights differ from live training sample')
    initial_step = load(resume/'trainer_state.json')['global_step'] if resume else 0
    class Audit(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            model = kw['model']; models.append(model)
            named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
            size = lambda p:getattr(p,'ds_numel',p.numel())
            trainable = sum(size(p) for _,p in named)
            if trainable != 8953803264 or any('lora_' in n or 'visual.' in n for n,_ in named):
                raise ValueError('Unexpected full-language-model training extent')
            candidates = [(n,p) for n,p in named if len(getattr(p,'ds_shape',p.shape)) == 2 and 1024 <= size(p) <= 4_000_000]
            selected.extend(sorted(candidates,key=lambda item:item[0])[:3])
            if len(selected) != 3: raise ValueError('No full-model update audit samples')
            initial.append(sample_hash())
            if resume: verify_samples(resume)
            if state.max_steps != EXPECTED_STEPS(): raise ValueError('Unexpected full-SFT step budget')
            model.save_pretrained = partial(model.save_pretrained,max_shard_size='1GB')
            optimizer = next(p for _,p in named)._z3_optimizer.optimizer
            if type(optimizer).__name__ != 'DeepSpeedCPUAdam': raise ValueError('Wrong offloaded optimizer')
            publish(status='training',step=state.global_step,epoch=state.epoch,max_steps=EXPECTED_STEPS(),trainable_parameters=trainable,
                vision_frozen=True,language_model_frozen=False,sample_names=[n for n,_ in selected],
                initial_weight_sample_hash=initial[0],optimizer=type(optimizer).__name__,
                optimizer_offload='cpu',attention=model.config._attn_implementation,
                flash_sdpa_enabled=torch.backends.cuda.flash_sdp_enabled(),optimized_linear_attention=True)
            if initial_step == EXPECTED_STEPS():
                proof = load(ROOT/f'evidence/checkpoint-{EXPECTED_STEPS()}.json')
                if not proof['passed'] or proof['sample_hash'] != initial[0]:
                    raise ValueError('Final-checkpoint recovery proof missing or changed')
                publish(final_weight_sample_hash=proof['sample_hash'],first_update_verified=True)
        def on_substep_end(self,args,state,control,**kw):
            publish(microbatches_completed=status.get('microbatches_completed',0)+1,peak_gpu_gib=torch.cuda.max_memory_allocated(rank)/2**30)
        def on_step_end(self,args,state,control,**kw):
            changed = sample_hash() != initial[0] if state.global_step <= initial_step+2 else status.get('first_update_verified',False)
            if state.global_step == initial_step+2 and not changed:
                raise ValueError('Full-model audit samples have not updated')
            publish(step=state.global_step,epoch=state.epoch,first_update_verified=changed,
                peak_gpu_gib=torch.cuda.max_memory_allocated(rank)/2**30)
            if state.global_step == EXPECTED_STEPS() or (not resume and state.global_step == 2): control.should_save=True
        def on_log(self,args,state,control,logs=None,**kw):
            if logs and 'loss' in logs:
                if not all(math.isfinite(float(v)) for v in logs.values() if isinstance(v,(int,float))):
                    raise ValueError('Nonfinite full-SFT metric')
                publish(latest_metrics=logs)
        def on_save(self,args,state,control,**kw):
            torch.distributed.barrier(device_ids=[rank])
            path = output/f'checkpoint-{state.global_step}'
            checkpoint_files(path,state.global_step); verify_samples(path)
            publish(latest_checkpoint=str(path),saved_weight_samples_match=True)
            if state.global_step == EXPECTED_STEPS():
                final_hash = sample_hash()
                if not status.get('first_update_verified'):
                    raise ValueError('No verified full-model update')
                if rank == 0:
                    atomic(ROOT/f'evidence/checkpoint-{EXPECTED_STEPS()}.json',dict(passed=True,step=EXPECTED_STEPS(),
                        sample_hash=final_hash,at=time.time()))
                publish(final_weight_sample_hash=final_hash)
    publish()
    run_exp(args=config,callbacks=[Audit()])
    # LLaMA-Factory destroys the distributed group before run_exp returns.
    # The last checkpoint was compared with live gathered weights in on_save.
    if status['step'] != EXPECTED_STEPS() or status['epoch'] < 2.999 or not status.get('first_update_verified') or not status.get('final_weight_sample_hash'):
        raise ValueError('Full SFT did not finish three epochs with updates')
    checkpoint_files(output/f'checkpoint-{EXPECTED_STEPS()}',EXPECTED_STEPS())
    if rank == 0:
        from safetensors import safe_open
        index = output/'model.safetensors.index.json'
        mapping = load(index)['weight_map'] if index.exists() else {}
        h = hashlib.sha256()
        for name,_ in selected:
            with safe_open(str(output/mapping.get(name,'model.safetensors')),framework='pt',device='cpu') as stream:
                tensor = stream.get_tensor(name)
            h.update(name.encode());h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        if h.hexdigest() != status['final_weight_sample_hash']:
            raise ValueError('Final saved full-model samples differ from the trained checkpoint')
        shards = sorted(set(mapping.values())) if mapping else ['model.safetensors']
        hashes = {name:sha(output/name) for name in shards}
        atomic(ROOT/'evidence/final_weights.json',dict(passed=True,at=time.time(),shards=hashes,
            final_weight_sample_hash=h.hexdigest(),optimizer_steps=EXPECTED_STEPS(),epochs=3,
            model_config_sha256=sha(output/'config.json')))
    publish(status='completed',model_dir=str(output),final_weights_evidence=str(ROOT/'evidence/final_weights.json'))

if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action',choices=['audit','prepare','train'])
    args = p.parse_args()
    try: globals()[args.action]()
    except BaseException as error:
        atomic(ROOT/(f'{args.action}_rank{os.environ.get("LOCAL_RANK","cpu")}_error.json'),dict(error=repr(error),at=time.time()))
        raise
