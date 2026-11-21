"""Phase-wise TaskSet GRPO with selected groups of four and repeat rejection."""
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from rl_common import ROOT, adapter_digest, digest, parent_for, read, select_training_members, require_execution_enabled
from campaign_io import InfrastructureError, canonical, verify_campaign


def load_groups(phase, expected_sha):
    groups, excluded = [], []
    for job in [j for j in read('assets/jobs.json') if j['phase'] == phase]:
        directory = ROOT / 'rollouts' / phase / job['task_id'] / ('k' + str(job['prefix_k'])) / ('r' + str(job['repeat']))
        path = directory / ('rollouts_' + job['task_id'] + '.jsonl')
        if not path.exists():
            raise InfrastructureError('Collection is incomplete: ' + job['group_id'])
        ledger = directory / 'group.sqlite'
        if not ledger.exists():
            raise InfrastructureError('Rollout completion ledger is missing')
        with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as db:
            complete = db.execute("SELECT value FROM state WHERE key='complete'").fetchone()
        if not complete or json.loads(complete[0])['output_sha256'] != digest(path):
            raise InfrastructureError('Rollout export lacks a matching committed completion record')
        complete = json.loads(complete[0])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if complete.get('reason') == 'prefix_measurement_failed':
            if rows or complete.get('rows') != 0 or complete.get('trainable') is not False:
                raise InfrastructureError('Invalid failed-prefix exclusion')
            failures=complete.get('prefix_failures',[])
            if not failures:
                raise InfrastructureError('Failed-prefix exclusion lacks evidence')
            history=ROOT/'assets/histories'/(job['task_id']+'.json')
            observations=json.loads(history.read_text())
            for failure in failures:
                index=failure['index']
                expected=f"baselines/{job['task_id']}/{index:03d}.sqlite"
                if (type(index) is not int or not 0<=index<job['prefix_k']
                    or failure['path']!=expected or digest(ROOT/expected)!=failure['sha256']):
                    raise InfrastructureError('Failed-prefix evidence changed')
                with sqlite3.connect(f'file:{ROOT/expected}?mode=ro',uri=True) as db:
                    state={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM state')}
                result=state.get('evaluation_result',{})
                if (state.get('identity')!=dict(history_sha256=digest(history),index=index,split=[.25,.35,1])
                    or result.get('status')!='failure' or result.get('measured') is not True
                    or result.get('configuration')!=observations[index]['config']
                    or result.get('split')!=dict(test_size=.25,validation_size=.35,seed=1)):
                    raise InfrastructureError('Failed-prefix evidence is not a measured failure')
            excluded.append(dict(group=job['group_id'],reason=complete['reason']))
            continue
        if not rows:
            raise InfrastructureError('Empty rollout export')
        if any(row['group_id'] != job['group_id'] or row['policy_sha256'] != expected_sha
               or any(row[k] != job[k] for k in ('task_id', 'prefix_k', 'phase', 'scenario')) for row in rows):
            raise InfrastructureError('Rollout group/policy identity mismatch')
        if any(row['messages'] != rows[0]['messages'] for row in rows):
            raise InfrastructureError('Sibling prompts changed within a GRPO group')
        selection = select_training_members(rows, job['group_id'], read('settings.json')['training']['seed'])
        if (complete.get('selection') != selection or complete.get('rows') != len(rows)
                or complete.get('trainable') != selection['trainable'] or complete.get('reason') != selection['reason']
                or any(type(row.get('train_member')) is not bool
                       or row['train_member'] != (row['sample_i'] in selection['train_members'])
                       or row.get('group_status') != selection['reason'] for row in rows)):
            raise InfrastructureError('Rollout training membership differs from the committed four-member selection')
        if selection['trainable']:
            groups.append([row for row in rows if row['train_member']])
        else:
            excluded.append(dict(group=job['group_id'], reason=selection['reason']))
    return groups, excluded


def save_frozen_cache(path, inputs, cache):
    """One atomic commit for cache tensors and their input identity."""
    import torch
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('wb') as stream:
        torch.save(dict(inputs=inputs, cache=cache), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_frozen_cache(path, inputs):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('inputs') != inputs:
        raise InfrastructureError('Frozen policy/reference cache belongs to different rollouts')
    return saved['cache']


def validate_trainable_parameters(model):
    import torch
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in name or p.dtype != torch.float32 for name, p in named):
        raise InfrastructureError('Only FP32 LoRA master parameters may be optimized')
    return named


def restore_optimizer(optimizer, state, names):
    if state.get('parameter_names') != names:
        raise InfrastructureError('Optimizer parameter order differs from the saved adapter state')
    optimizer.load_state_dict(state['optimizer'])


def reference_cache(model, reference, batches, device):
    """Swap only adapter tensors so there is never a second base on the GPU."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from training_metrics import snapshot
    named = validate_trainable_parameters(model)
    saved = snapshot(named)
    source = load_file(str(reference / 'adapter_model.safetensors'), device='cpu')
    prefix = 'base_model.model.model.layers.'
    target = 'base_model.model.model.language_model.layers.'
    expected = {target + k[len(prefix):] if k.startswith(prefix) else k: v for k,v in source.items()}
    try:
        set_peft_model_state_dict(model, expected)
        actual = get_peft_model_state_dict(model)
        if set(actual) != set(expected) or any(not torch.equal(actual[k].detach().cpu(),v) for k,v in expected.items()):
            raise InfrastructureError('Reference adapter swap did not preserve exact saved weights')
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            values = [[token_logprobs(model,ids,n).cpu() for ids,n in group] for group in batches]
    finally:
        with torch.no_grad():
            for name,p in named:
                p.copy_(saved[name].to(p.device))
    return values


class KLGuardExceeded(InfrastructureError):
    pass


def load_training_state(directory):
    import torch
    metadata = json.loads((directory / 'complete.json').read_text())
    path = directory / 'training_state.pt'
    if not path.exists() or metadata.get('training_state_sha256') != digest(path):
        raise InfrastructureError('Committed optimizer/RNG state is missing or corrupt')
    return torch.load(path, map_location='cpu', weights_only=True)


def update_batch(model, optimizer, named_parameters, groups, batches, cache, indices, cfg, device):
    """The production update path, also exercised with a tiny CPU test model."""
    import torch
    from grpo_math import advantages, response_loss
    from training_metrics import snapshot, weight_change, policy_metrics
    optimizer.zero_grad(set_to_none=True)
    before_step = snapshot(named_parameters)
    sums = dict(reference_kl=0., clip_fraction=0., duplicate_rejection_loss=0., loss=0.)
    context_before = []
    for g in indices:
        if len(groups[g]) != 4 or len(batches[g]) != 4:
            raise InfrastructureError('Optimizer requires exactly four selected responses per group')
        adv = advantages([row['reward']['reward'] for row in groups[g]])
        context_kl = 0.
        for sample, (ids, prompt_len) in enumerate(batches[g]):
            from contextlib import nullcontext
            saved_tensors = torch.autograd.graph.save_on_cpu(pin_memory=True) if cfg.get('activation_offload') and device.type == 'cuda' else nullcontext()
            with saved_tensors, torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logp = token_logprobs(model, ids, prompt_len)
                old = cache['old'][g][sample].to(device)
                ref = cache['reference'][g][sample].to(device)
                loss, reject = response_loss(logp, old, ref, float(adv[sample]),
                    groups[g][sample]['evaluation_status'] == 'DUPLICATE', beta=cfg['kl_beta'],
                    epsilon=cfg['clip_epsilon'], duplicate_weight=cfg['duplicate_aux_weight'])
                weight = 1. / (4 * len(indices))
                loss = loss * weight
            if not torch.isfinite(loss):
                raise InfrastructureError('Nonfinite GRPO objective; last complete checkpoint retained')
            for key, value in policy_metrics(logp, old, ref, cfg['clip_epsilon']).items():
                sums[key] += weight * value
                if key == 'reference_kl':
                    context_kl += value / 4
            sums['duplicate_rejection_loss'] += weight * reject.detach().float().item()
            sums['loss'] += loss.detach().float().item()
            loss.backward()
        context_before.append(context_kl)
    if (not all(math.isfinite(v) for v in context_before) or sums['reference_kl'] > cfg['max_reference_kl']
            or cfg.get('per_context_kl_guard') and max(context_before) > cfg['max_reference_kl']):
        raise KLGuardExceeded('Reference KL guard exceeded before optimizer update; last complete checkpoint retained')
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in named_parameters], cfg['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    # Check the updated policy before publishing any checkpoint, including the
    # final step of a phase, where a pre-update-only guard would miss an overshoot.
    post_kl = 0.
    context_after = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for g in indices:
            group_kl = 0.
            for sample, (ids, n) in enumerate(batches[g]):
                logp = token_logprobs(model, ids, n)
                metrics = policy_metrics(logp, cache['old'][g][sample].to(device),
                                         cache['reference'][g][sample].to(device), cfg['clip_epsilon'])
                post_kl += metrics['reference_kl'] / (4 * len(indices))
                group_kl += metrics['reference_kl'] / 4
            context_after.append(group_kl)
    metrics = dict(sums, post_update_reference_kl=post_kl, grad_norm=float(norm),
                   step_weight_delta=weight_change(named_parameters, before_step),
                   phase_weight_delta=weight_change(named_parameters, cache['initial_lora']),
                   context_reference_kl_before=context_before, context_reference_kl_after=context_after)
    if (not math.isfinite(post_kl) or post_kl > cfg['max_reference_kl']
            or not all(math.isfinite(v) for v in context_after)
            or cfg.get('per_context_kl_guard') and max(context_after) > cfg['max_reference_kl']):
        raise KLGuardExceeded('Reference KL guard exceeded after optimizer update; no checkpoint published; resume from last complete checkpoint: '
                                  + json.dumps(metrics,sort_keys=True))
    if not math.isfinite(metrics['step_weight_delta']['l2']) or metrics['step_weight_delta']['l2'] <= 0:
        raise InfrastructureError('Optimizer step changed no finite adapter weights; no checkpoint published')
    return metrics


def transactional_update(model, optimizer, named, groups, batches, cache, indices, cfg, device):
    """Retry KL overshoots from exactly the same weights, moments and RNG."""
    import copy
    import torch
    from training_metrics import snapshot
    weights = snapshot(named)
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device.type == 'cuda' else None
    learning_rates = [group['lr'] for group in optimizer.param_groups]
    rejections = []
    for attempt in range(cfg.get('backoff_attempts', 5)):
        try:
            metrics = update_batch(model,optimizer,named,groups,batches,cache,indices,cfg,device)
            return dict(metrics, accepted_learning_rate=optimizer.param_groups[0]['lr'], rejected_attempts=rejections)
        except KLGuardExceeded as exc:
            rejections.append(dict(attempt=attempt+1, learning_rate=optimizer.param_groups[0]['lr'], error=str(exc)))
            with torch.no_grad():
                for name,p in named:
                    p.copy_(weights[name].to(p.device))
            optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            optimizer.zero_grad(set_to_none=True)
            torch.random.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            for group,lr in zip(optimizer.param_groups,learning_rates):
                group['lr'] = lr * cfg.get('backoff_factor',.5)**(attempt+1)
    raise InfrastructureError('KL backoff exhausted; weights and optimizer rolled back; no group skipped: '+canonical(rejections))


def find_checkpoint(directory, inputs):
    """Recover even if a crash occurred between atomic checkpoint and pointer commits."""
    candidates = []
    for path in directory.glob('checkpoint-*'):
        if '.tmp.' in path.name or not (path / 'complete.json').exists():
            continue
        metadata = json.loads((path / 'complete.json').read_text())
        if metadata['inputs'] != inputs:
            raise InfrastructureError('Training checkpoint belongs to different phase inputs')
        if (not (path / 'training_state.pt').exists()
                or digest(path / 'training_state.pt') != metadata.get('training_state_sha256')
                or digest(path / 'adapter_model.safetensors') != metadata['adapter_sha256']):
            raise InfrastructureError('Committed training checkpoint is corrupt')
        candidates.append((metadata['completed_groups'], metadata['steps'], path))
    return max(candidates)[2] if candidates else None


def token_batch(tokenizer, row, limit, device):
    import torch
    from mindware.domains.dlhpo.taskset_llm.spec import render_nothink
    prompt = tokenizer(render_nothink(row['messages']), add_special_tokens=False)['input_ids']
    if prompt != row['prompt_ids']:
        raise InfrastructureError('Rollout prompt tokenization differs from the training tokenizer')
    answer = row['answer_ids']
    if tokenizer.decode(answer, skip_special_tokens=True).strip() != row['completion']:
        raise InfrastructureError('Completion differs from the sampled token sequence')
    if not prompt or not answer or len(prompt) + len(answer) > limit:
        raise InfrastructureError('Prompt/completion exceeds the untruncated training envelope')
    return torch.tensor([prompt + answer], device=device), len(prompt)


def token_logprobs(model, input_ids, prompt_length):
    import torch
    import torch.nn.functional as F
    # Project only positions that predict answer tokens, avoiding full-history
    # vocabulary logits while keeping exactly the same token probabilities.
    indices = torch.arange(prompt_length - 1, input_ids.shape[1] - 1, device=input_ids.device)
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                   logits_to_keep=indices, use_cache=False, return_dict=True)
    logits = output.logits[0]
    targets = input_ids[0, prompt_length:]
    if logits.shape[0] != targets.numel():
        raise InfrastructureError('Model does not support answer-position logits correctly')
    return -F.cross_entropy(logits.float(), targets, reduction='none')


def load_model(path, trainable, settings):
    import torch
    from transformers import AutoModelForImageTextToText
    from mindware.domains.dlhpo.taskset_llm.llm_client import load_verified_adapter
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(settings.get('gpu_memory_gib', 40) * 2**30 / total)
    # Load and verify before placing the unused vision module / frozen lookup
    # on CPU. PEFT's redispatch of a mixed CPU map is incompatible with this
    # installed Accelerate version; explicit lookup hooks avoid redispatch.
    device_map = {'': 'cuda:0'}
    base = AutoModelForImageTextToText.from_pretrained(settings['base_model'], trust_remote_code=True,
        dtype=torch.bfloat16, device_map=device_map,
        attn_implementation=settings.get('attention_implementation', 'sdpa'))
    model, verification = load_verified_adapter(base, str(path))
    if settings.get('offload_visual'):
        base.model.visual.cpu()
        embedding=base.model.language_model.embed_tokens
        embedding.cpu()
        embedding.register_forward_pre_hook(lambda module,args:tuple(x.to('cpu') if torch.is_tensor(x) else x for x in args))
        embedding.register_forward_hook(lambda module,args,output:output.to('cuda:0'))
        model.hf_device_map={'model.visual':'cpu','model.language_model.embed_tokens':'cpu','':'cuda:0'}
    for name, param in model.named_parameters():
        param.requires_grad_(bool(trainable and 'lora_' in name))
    if trainable:
        model.peft_config['default'].inference_mode = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    # Dropout would give different old/current probabilities for unchanged weights.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.config.use_cache = False
    model.eval()
    return model, verification


def train(phase):
    require_execution_enabled()
    import fcntl
    import torch
    from transformers import AutoTokenizer
    from training_metrics import training_order, snapshot

    settings, signature = read('settings.json'), verify_campaign(ROOT)
    if not torch.cuda.is_available():
        raise InfrastructureError('Real TaskSet GRPO requires BF16 CUDA execution')
    if not torch.cuda.is_bf16_supported():
        raise InfrastructureError('Selected GPU does not support BF16')
    cfg = settings['training']
    directory = ROOT / 'models' / phase
    directory.mkdir(exist_ok=True)
    lock = (directory / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = parent_for(phase)
    reference = Path(settings['parent_adapter'])
    groups, excluded = load_groups(phase, adapter_digest(parent))
    if not groups:
        raise InfrastructureError('No complete GRPO groups with a reward signal; no training performed')
    inputs = dict(campaign=signature, phase=phase, parent_sha256=adapter_digest(parent),
        reference_sha256=adapter_digest(reference), groups=[g[0]['group_id'] for g in groups],
        rollout_sha256=__import__('hashlib').sha256(canonical(groups).encode()).hexdigest())
    final = directory / 'final'
    if (final / 'complete.json').exists():
        metadata = json.loads((final / 'complete.json').read_text())
        if metadata['inputs'] != inputs:
            raise InfrastructureError('Completed phase has different inputs')
        if digest(final / 'adapter_model.safetensors') != metadata['adapter_sha256']:
            raise InfrastructureError('Completed phase adapter is corrupt')
        load_training_state(final)
        return
    torch.manual_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(parent), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cache_path = directory / 'frozen_logprobs.pt'
    checkpoint_pointer = directory / 'latest.json'
    checkpoint = find_checkpoint(directory, inputs)
    if checkpoint:
        if not cache_path.exists():
            raise InfrastructureError('Training resume requires the original frozen policy log probabilities')
    model, verification = load_model(checkpoint or parent, True, settings)
    device = torch.device('cuda:0')
    batches = [[token_batch(tokenizer, row, settings['max_seq_len'], device)
                for row in group] for group in groups]
    if cache_path.exists():
        cache = load_frozen_cache(cache_path, inputs)
    else:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            old = [[token_logprobs(model, ids, n).cpu() for ids, n in group] for group in batches]
        ref_values = reference_cache(model, reference, batches, device)
        cache = dict(old=old, reference=ref_values,
                     initial_lora=snapshot([(n,p) for n,p in model.named_parameters() if p.requires_grad]))
        save_frozen_cache(cache_path, inputs, cache)
    named_parameters = validate_trainable_parameters(model)
    parameter_names = [n for n, _ in named_parameters]
    parameters = [p for n,p in named_parameters]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    completed, steps = 0, 0
    if checkpoint:
        state = load_training_state(checkpoint)
        if state['inputs'] != inputs:
            raise InfrastructureError('Training checkpoint belongs to different phase inputs')
        restore_optimizer(optimizer, state, parameter_names)
        completed, steps = state['completed_groups'], state['steps']
        torch.random.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
    elif phase != settings['phases'][0]['id']:
        # The optimizer moments continue across phases, matching the online ML path.
        previous = load_training_state(parent)
        restore_optimizer(optimizer, previous, parameter_names)
    # Keep an accepted backed-off rate across phase and cycle boundaries.
    for group in optimizer.param_groups:
        group['lr']=min(group['lr'],cfg['learning_rate'])
    order=training_order(len(groups),cfg['epochs_per_phase'],cfg['seed'])
    latest_metrics=json.loads((checkpoint/'complete.json').read_text()).get('metrics',{}) if checkpoint else {}

    def save_checkpoint(target, end):
        temporary = target.with_name(target.name + '.tmp.' + str(os.getpid()))
        if temporary.exists():
            raise InfrastructureError('Uncommitted checkpoint directory requires inspection: ' + str(temporary))
        temporary.mkdir()
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        state = dict(inputs=inputs, optimizer=optimizer.state_dict(), parameter_names=parameter_names, completed_groups=end, steps=steps,
            cpu_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        torch.save(state, temporary / 'training_state.pt')
        metadata = dict(inputs=inputs, completed_groups=end, steps=steps, excluded_groups=excluded,
            initial_adapter_verification=verification, bf16=True, at=time.time(),
            total_group_visits=len(order), metrics=latest_metrics,
            training_state_sha256=digest(temporary / 'training_state.pt'),
            adapter_sha256=digest(temporary / 'adapter_model.safetensors'))
        (temporary / 'complete.json').write_text(canonical(metadata))
        os.rename(temporary, target)
        pointer = directory / 'latest.tmp'
        pointer.write_text(canonical(dict(path=str(target))))
        pointer.replace(checkpoint_pointer)

    model.train()
    for start in range(completed, len(order), cfg['grad_accum_groups']):
        end = min(start + cfg['grad_accum_groups'], len(order))
        latest_metrics = transactional_update(model, optimizer, named_parameters, groups, batches, cache,
                                              order[start:end], cfg, device)
        steps += 1
        save_checkpoint(directory / ('checkpoint-' + str(steps)), end)
        event=dict(phase=phase,optimizer_step=steps,completed_group_visits=end,total_group_visits=len(order),**latest_metrics)
        with (directory/'updates.jsonl').open('a') as stream: stream.write(canonical(event)+'\n')
        print(canonical(event), flush=True)
        # Final phase/cycle checkpoints are separate and are always retained.
        import shutil
        old_checkpoints=sorted((p for p in directory.glob('checkpoint-*') if '.tmp.' not in p.name),
                               key=lambda p:int(p.name.split('-')[-1]))
        for old_checkpoint in old_checkpoints[:-2]:
            shutil.rmtree(old_checkpoint)
    # Replay fixed duplicate answers under the final policy. This measures their
    # likelihood change, without new model generation or oracle measurements.
    probes=[]
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for g,rows in enumerate(groups):
            for i,row in enumerate(rows):
                if row['evaluation_status']!='DUPLICATE' or len(probes)>=16: continue
                ids,prompt_len=batches[g][i]
                after=token_logprobs(model,ids,prompt_len).float().mean().item()
                before=cache['old'][g][i].float().mean().item()
                probes.append(dict(group=row['group_id'],sample_i=row['sample_i'],before_mean_logp=before,
                                   after_mean_logp=after,delta_mean_logp=after-before))
    (directory/'duplicate_probes.json').write_text(canonical(dict(probes=probes,interpretation='Negative delta means the fixed repeated answer became less likely; not a measured generation repeat rate')))
    # A final adapter plus optimizer/RNG state is the only next-phase handoff.
    if not final.exists():
        save_checkpoint(final, len(order))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.phase not in [x['id'] for x in read('settings.json')['phases']]:
        p.error('Unknown phase')
    if a.execute:
        train(a.phase)
    else:
        print(canonical(dict(configured=True, launched=False, phase=a.phase,
                             training=read('settings.json')['training'])))
