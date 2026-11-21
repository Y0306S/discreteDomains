#!/usr/bin/env python3
"""LLaMA-Factory ML ORPO with full-token preflight and verified checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def prepare(campaign):
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from collections import Counter
    import torch
    config = json.loads((campaign/'orpo_config.json').read_text())
    rows = [json.loads(line) for line in (campaign/'data/orpo_train.jsonl').open()]
    cpu = dict(config, use_cpu=True, bf16=False, dataloader_pin_memory=False, cutoff_len=131072)
    model_args, data_args, training_args, _, _ = get_train_args(cpu)
    module = load_tokenizer(model_args)
    tokenizer = module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    expected = Counter(); lengths = []
    for row in rows:
        system, user = row['conversations']
        arms = []
        for side in ('chosen', 'rejected'):
            answer = row[side]['value']
            source, target = template.encode_multiturn(tokenizer,
                [dict(role='user', content=user['value']), dict(role='assistant', content=answer)],
                system['value'], None)[0]
            if tokenizer.decode(target) != answer + '<|im_end|>\n':
                raise ValueError('ORPO target/EOS tokenization changed')
            arms.extend([source+target, [-100]*len(source)+target])
            lengths.append(len(source)+len(target))
        expected[json.dumps(arms)] += 1
    config['cutoff_len'] = max(6016, math.ceil((max(lengths)+256)/1024)*1024)
    data_args.cutoff_len = config['cutoff_len']
    actual = Counter()
    dataset = get_dataset(template, model_args, data_args, training_args, 'rm', **module)['train_dataset']
    for row in dataset:
        actual[json.dumps([row['chosen_input_ids'],row['chosen_labels'],
                          row['rejected_input_ids'],row['rejected_labels']])] += 1
    if expected != actual or len(rows) != len(dataset):
        raise ValueError('Training preprocessing dropped or truncated supervised tokens')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight initialized CUDA')
    atomic(campaign/'orpo_config.json',config)
    atomic(campaign/'orpo_preflight.json',dict(passed=True, examples=len(rows),
        min_tokens=min(lengths),max_tokens=max(lengths),cutoff_len=config['cutoff_len'],
        truncated_examples=0,assistant_only_labels_verified=True,
        config_sha256=sha(campaign/'orpo_config.json'),data_sha256=sha(campaign/'data/orpo_train.jsonl'),
        expected_steps=math.ceil(len(rows)/8),at=time.time()))
    print(json.dumps(dict(passed=True,examples=len(rows),cutoff_len=config['cutoff_len'])),flush=True)


def train(campaign):
    import torch
    from peft import get_peft_model_state_dict
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp
    # Identical loss-preserving answer-only projection used by Taskset ORPO.
    sys.path.insert(0,str(campaign/'code'))
    from taskset_answer_logits import install_orpo_answer_logits
    from transformers.models.qwen3_5 import modeling_qwen3_5
    install_orpo_answer_logits()
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError('Qwen3.5 optimized linear-attention kernels unavailable')
    config=json.loads((campaign/'orpo_config.json').read_text())
    gate=json.loads((campaign/'orpo_preflight.json').read_text())
    if not gate['passed'] or gate['config_sha256']!=sha(campaign/'orpo_config.json') or gate['data_sha256']!=sha(campaign/'data/orpo_train.jsonl'):
        raise ValueError('ORPO full-token preflight changed')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(48*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.enable_flash_sdp(True)
    output=Path(config['output_dir'])
    if (campaign/'orpo_status.json').exists():
        previous=json.loads((campaign/'orpo_status.json').read_text())
        if previous['status']=='completed':
            if sha(output/'adapter_model.safetensors')!=previous['adapter_sha256']:
                raise ValueError('Completed ORPO adapter changed')
            return
    checkpoints=sorted(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    required=('adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json')
    checkpoints=[p for p in checkpoints if all((p/name).exists() for name in required)]
    resume=checkpoints[-1] if checkpoints else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError('Output has no complete resumable checkpoint')
    config['resume_from_checkpoint']=str(resume) if resume else None
    status=dict(status='loading',pid=os.getpid(),host='LLMOptimizer2',gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        framework='LLaMA-Factory',parent_adapter=config.get('adapter_name_or_path'),resumed_from=config['resume_from_checkpoint'])
    def publish(**values):
        status.update(values,at=time.time());atomic(campaign/'orpo_status.json',status)
    def tensor_hash(model):
        h=hashlib.sha256()
        for key,value in sorted(get_peft_model_state_dict(model).items()):
            h.update(key.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def verify(model,path):
        from safetensors.torch import load_file
        saved=load_file(str(path/'adapter_model.safetensors'))
        actual=get_peft_model_state_dict(model)
        if set(saved)!=set(actual) or any(not torch.equal(saved[k],actual[k].detach().cpu()) for k in saved):
            raise ValueError('Saved adapter differs from live trained tensors')
        return len(saved)
    holder=[];initial=[]
    initial_step=json.loads((resume/'trainer_state.json').read_text())['global_step'] if resume else 0
    class Audit(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            model=kw['model'];holder.append(model);initial.append(tensor_hash(model))
            if resume:verify(model,resume)
            if state.max_steps!=gate['expected_steps']:raise ValueError('Unexpected optimizer step budget')
            publish(status='training',step=state.global_step,max_steps=state.max_steps,
                initial_adapter_hash=initial[0],attention=model.config._attn_implementation,
                flash_sdpa_enabled=torch.backends.cuda.flash_sdp_enabled(),optimized_linear_attention=True)
        def on_step_end(self,args,state,control,**kw):
            changed=tensor_hash(kw['model'])!=initial[0] if state.global_step<=initial_step+2 else status.get('first_update_verified',False)
            if state.global_step==initial_step+2 and not changed:raise ValueError('Adapter has not updated')
            publish(step=state.global_step,epoch=state.epoch,first_update_verified=changed,
                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            if state.global_step in (initial_step+1,initial_step+2):control.should_save=True
        def on_log(self,args,state,control,logs=None,**kw):
            if logs and 'loss' in logs:
                if not all(math.isfinite(float(v)) for v in logs.values() if isinstance(v,(int,float))):
                    raise ValueError('Nonfinite training metric')
                publish(latest_metrics=logs)
        def on_save(self,args,state,control,**kw):
            path=output/f'checkpoint-{state.global_step}'
            verify(kw['model'],path)
            if not all((path/name).exists() for name in required):raise ValueError('Incomplete checkpoint')
            publish(latest_checkpoint=str(path))
    publish()
    run_exp(args=config,callbacks=[Audit()])
    model=holder[0];count=verify(model,output)
    if tensor_hash(model)==initial[0]:raise ValueError('ORPO made no weight updates')
    if status['step']!=gate['expected_steps'] or status['epoch']<.999:raise ValueError('ORPO did not complete its epoch')
    publish(status='completed',adapter_sha256=sha(output/'adapter_model.safetensors'),verified_tensors=count,
        final_adapter_hash=tensor_hash(model))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train']);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args()
    try:(prepare if a.action=='prepare' else train)(a.campaign.resolve())
    except BaseException as error:
        if a.action=='train':atomic(a.campaign/'orpo_error.json',dict(error=repr(error),at=time.time()))
        raise
