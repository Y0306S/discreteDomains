"""Four-GPU ZeRO-2 assistant-only LoRA SFT with validated precomputed labels."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from protocol import *

def config(domain):
    return json.loads((ROOT/'configs'/(domain+'.json')).read_text())

def tokenize(tokenizer, messages, maximum):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    spans=[];cursor=0
    for message in messages:
        content=message['content']
        if any(token in content for token in ('<|im_start|>', '<|im_end|>', '<think>', '</think>')):
            raise ValueError('Chat control token in dataset')
        marker='<|im_start|>'+message['role']+'\n'
        start=text.find(marker,cursor)
        if start<0:raise ValueError('Native tokenizer template lacks expected message boundary')
        end=text.find('<|im_end|>',start+len(marker))
        content_start=text.find(content,start+len(marker),end)
        if content_start<0:raise ValueError('Native template changed message content')
        if message['role']=='assistant':spans.append((content_start,end+len('<|im_end|>')))
        cursor=end+len('<|im_end|>')
    encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
    if len(encoded['input_ids'])>maximum:
        raise ValueError(f'Trajectory exceeds context: {len(encoded["input_ids"])} > {maximum}; no silent truncation')
    labels=[];counts=[0]*len(spans)
    for token,(start,stop) in zip(encoded['input_ids'],encoded['offset_mapping']):
        overlap=[i for i,(left,right) in enumerate(spans) if start<right and stop>left]
        if overlap:
            left,right=spans[overlap[0]]
            if len(overlap)!=1 or not left<=start<stop<=right:raise ValueError('Token crosses supervision boundary')
            labels.append(token);counts[overlap[0]]+=1
        else:labels.append(-100)
    if not counts or not all(counts) or all(x!=-100 for x in labels):raise ValueError('Invalid assistant-only mask')
    return dict(input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'],labels=labels)

def preflight(domain, pilot=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preflight must be CPU-only')
    from transformers import AutoTokenizer
    cfg=config(domain);directory=ROOT/domain/'data'
    source=directory/('pilot_trajectories.jsonl' if pilot else 'trajectories.jsonl')
    manifest=json.loads(source.with_suffix('.manifest.json').read_text())
    if manifest['status']!='complete' or manifest['dataset_sha256']!=digest(source) or manifest['pilot']!=pilot:
        raise ValueError('Unfinished or changed dataset')
    if manifest['script_sha256']!=digest(Path(__file__).with_name('prepare.py')) or manifest['protocol_sha256']!=digest(Path(__file__).with_name('protocol.py')):
        raise ValueError('Annotation code changed after export')
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    destination=directory/('pilot_tokens.jsonl' if pilot else 'tokens.jsonl');temp=destination.with_suffix('.tmp')
    lengths=[];supervised=decisions=0;tasks=set()
    with source.open() as stream,temp.open('w') as out:
        for line in stream:
            row=json.loads(line);messages=row['messages'];actions=[m for m in messages if m['role']=='assistant']
            if len(actions)!=len(row['annotations']):raise ValueError('Annotation/action alignment mismatch')
            for message,annotation in zip(actions,row['annotations']):
                action=json.loads(message['content'])
                if any(action[k]!=annotation[k] for k in ('next_configuration','mean','std')):
                    raise ValueError('Training action differs from causal annotation')
                if not math.isfinite(action['mean']) or not math.isfinite(action['std']) or action['std']<0:
                    raise ValueError('Invalid predictive labels')
            data=tokenize(tokenizer,messages,cfg['max_seq_length'])
            out.write(canonical(data)+'\n');lengths.append(len(data['input_ids']))
            supervised+=sum(v!=-100 for v in data['labels']);decisions+=len(actions);tasks.add(row['task_id'])
    if len(lengths)!=manifest['trajectories'] or decisions!=manifest['supervised_decisions'] or tasks!=set(manifest['represented_tasks']):
        raise ValueError('Dataset coverage mismatch')
    os.replace(temp,destination)
    report=dict(passed=True,pilot=pilot,trajectories=len(lengths),tasks=len(tasks),supervised_decisions=decisions,
        supervised_tokens=supervised,min_tokens=min(lengths),max_tokens=max(lengths),total_tokens=sum(lengths),
        optimizer_steps=math.ceil(len(lengths)/64),dataset_sha256=digest(source),manifest_sha256=digest(source.with_suffix('.manifest.json')),
        tokens_sha256=digest(destination),tokenizer_vocab_sha256=value_hash(tokenizer.get_vocab()),
        config_sha256=value_hash(cfg),training_script_sha256=digest(__file__),
        assistant_only_loss_verified=True,truncated_trajectories=0,oracle_calls=0,at=time.time())
    atomic(directory/('pilot_preflight.json' if pilot else 'preflight.json'),report);print(canonical(report),flush=True)

def train(domain, smoke=False):
    import torch
    import torch.nn.functional as F
    from transformers import (AutoTokenizer, AutoModelForImageTextToText, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments, TrainerCallback, set_seed)
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    from safetensors.torch import load_file
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=4:raise ValueError('Recipe requires four GPUs, microbatch 1, accumulation 16, effective batch 64')
    torch.cuda.set_device(rank);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(48*2**30/torch.cuda.get_device_properties(rank).total_memory,rank)
    # SDPA dispatch is constrained to its FlashAttention CUDA backend, with no slow fallback.
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if not modeling_qwen3_5.is_fast_path_available:raise RuntimeError('Optimized Qwen3.5 linear attention kernels unavailable')
    cfg=config(domain);set_seed(42);directory=ROOT/domain/'data'
    report=json.loads((directory/'preflight.json').read_text())
    if (not report['passed'] or report['pilot'] or report['tokens_sha256']!=digest(directory/'tokens.jsonl')
            or report['config_sha256']!=value_hash(cfg) or report['training_script_sha256']!=digest(__file__)
            or report['dataset_sha256']!=digest(directory/'trajectories.jsonl')
            or report['manifest_sha256']!=digest(directory/'trajectories.manifest.json')):
        raise ValueError('CPU preflight is missing or stale')
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    if value_hash(tokenizer.get_vocab())!=report['tokenizer_vocab_sha256']:raise ValueError('Tokenizer changed')
    rows=[json.loads(line) for line in (directory/'tokens.jsonl').open()]
    if len(rows)!=report['trajectories']:raise ValueError('Token count changed')
    if smoke:
        longest=max(rows,key=lambda r:len(r['input_ids']));rows=[longest]*4
    output=ROOT/domain/('smoke_model' if smoke else 'model')
    state_path=ROOT/domain/('smoke_status.json' if smoke else 'training_status.json')
    status=dict(status='loading',domain=domain,smoke=smoke,world_size=world,gpus=[0,1,2,3],
        deepspeed='ZeRO-2',base_model=BASE,parent_adapter=None,attention='PyTorch Flash SDPA only',
        fast_linear_attention=True,effective_batch_size=4 if smoke else 64,at=time.time())
    def publish(**values):
        if rank==0:status.update(values,at=time.time());atomic(state_path,status)
    publish()
    args=TrainingArguments(output_dir=str(output),per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 16,num_train_epochs=1.,
        max_steps=1 if smoke else -1,learning_rate=1e-5,lr_scheduler_type='cosine',
        warmup_steps=0,weight_decay=0.,max_grad_norm=1.,bf16=True,tf32=True,
        optim='adamw_torch',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=1,save_steps=10,save_total_limit=2,save_strategy='no' if smoke else 'steps',
        report_to=[],dataloader_num_workers=0,dataloader_pin_memory=False,remove_unused_columns=False,
        deepspeed=str(ROOT/'configs/zero2.json'),ddp_timeout=1800,seed=42,data_seed=42,
        label_names=['labels'],ddp_find_unused_parameters=False)
    model=AutoModelForImageTextToText.from_pretrained(BASE,local_files_only=True,trust_remote_code=True,
        dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=16,lora_dropout=0.,bias='none',
        target_modules=r'.*language_model.*.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$',task_type='CAUSAL_LM'))
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or 'visual' in n for n,p in trainable):raise ValueError('Expected language-only LoRA parameters')
    model.enable_input_require_grads()
    initial={n:p.detach().cpu().clone() for n,p in trainable}
    class AnswerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels=inputs.pop('labels')
            if labels.shape[0]!=1:raise ValueError('Selective projection expects microbatch one')
            positions=(labels[0,1:]!=-100).nonzero().flatten()
            if not positions.numel():raise ValueError('No assistant target tokens')
            targets=labels[0,positions+1]
            outputs=model(**inputs,logits_to_keep=positions,use_cache=False)
            logits=outputs.logits[0]
            if logits.shape[0]!=targets.numel():raise ValueError('Answer projection alignment failed')
            # Chunk the FP32 vocabulary upcast; retain the exact token-mean SFT loss.
            loss=sum(F.cross_entropy(logits[i:i+128].float(),targets[i:i+128],reduction='sum')
                for i in range(0,len(targets),128))/len(targets)
            if not torch.isfinite(loss):raise ValueError('Nonfinite SFT loss')
            return (loss,outputs) if return_outputs else loss
    def weights_changed():
        return sum(float((p.detach().cpu().float()-initial[n].float()).square().sum()) for n,p in trainable)**.5
    class Audit(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            if not smoke and state.max_steps!=report['optimizer_steps']:raise ValueError('Unexpected epoch step budget')
            publish(status='training',step=state.global_step,total_steps=state.max_steps,
                trainable_parameters=sum(p.numel() for n,p in trainable))
        def on_step_end(self,args,state,control,**kw):
            delta=weights_changed() if state.global_step<=2 else status.get('initial_weight_delta_l2')
            if state.global_step<=2 and delta<=0:raise ValueError('Training did not update adapter weights')
            publish(step=state.global_step,epoch=state.epoch,initial_weight_delta_l2=delta,
                peak_allocated_gpu_gib=torch.cuda.max_memory_allocated(rank)/2**30)
            if not smoke and state.global_step in (1,2):control.should_save=True
        def on_log(self,args,state,control,logs=None,**kw):
            if logs:
                if any(not math.isfinite(float(v)) for v in logs.values() if type(v) in (int,float)):raise ValueError('Nonfinite training metric')
                publish(metrics=logs)
        def on_save(self,args,state,control,**kw):
            if rank==0:
                checkpoint=output/f'checkpoint-{state.global_step}'
                saved=load_file(str(checkpoint/'adapter_model.safetensors'))
                actual=get_peft_model_state_dict(model)
                if set(saved)!=set(actual) or any(not torch.equal(saved[k],actual[k].detach().cpu()) for k in saved):
                    raise ValueError('Saved checkpoint differs from live adapter')
                publish(latest_checkpoint=str(checkpoint),checkpoint_matches_live_weights=True)
    trainer=AnswerTrainer(model=model,args=args,train_dataset=rows,processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100),callbacks=[Audit()])
    trainer.model_accepts_loss_kwargs=False
    checkpoints=sorted(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    resume=checkpoints[-1] if checkpoints else None
    if output.exists() and any(output.iterdir()) and resume is None:raise ValueError('Nonempty output lacks resumable checkpoint')
    result=trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(output/'final_adapter'));trainer.save_state()
    if rank==0:
        tokenizer.save_pretrained(output/'final_adapter')
        saved=load_file(str(output/'final_adapter/adapter_model.safetensors'));actual=get_peft_model_state_dict(model)
        if set(saved)!=set(actual) or any(not torch.equal(saved[k],actual[k].detach().cpu()) for k in saved):
            raise ValueError('Final saved adapter differs from live model')
        if weights_changed()<=0 or (not smoke and (trainer.state.epoch<.999 or trainer.state.global_step!=report['optimizer_steps'])):
            raise ValueError('Training epoch or weight updates incomplete')
        publish(status='completed',step=trainer.state.global_step,epoch=trainer.state.epoch,
            final_weight_delta_l2=weights_changed(),metrics=result.metrics,
            adapter_sha256=digest(output/'final_adapter/adapter_model.safetensors'),
            saved_adapter_matches_live=True,trained_tensors=len(saved))
        atomic(output/'training_complete.json',status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train']);p.add_argument('domain',choices=['peptide','query'])
    p.add_argument('--pilot',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:
        if a.action=='preflight':preflight(a.domain,a.pilot)
        else:train(a.domain,a.smoke)
    except BaseException as error:
        if a.action=='train' and int(os.environ.get('RANK','0'))==0:
            atomic(ROOT/a.domain/('smoke_error.json' if a.smoke else 'training_error.json'),dict(error=repr(error),at=time.time()))
        raise
