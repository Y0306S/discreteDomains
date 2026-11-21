"""Fixed observed-target retention validation and transactional VAE updates.

Target loss is reconstruction CE + 1e-4 KL, not a MIC prediction loss.
The initialization anchors were already observed; they are not a held-out set.
"""
import hashlib
import json
import math

import torch


class NumericalVAEError(FloatingPointError):
    """A numerical training/validation failure, eligible for rollback."""


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def finite_weights(model):
    return all(not v.is_floating_point() or torch.isfinite(v).all().item()
               for v in model.state_dict().values())


def initial_anchors(state):
    sequences = list(dict.fromkeys(r['sequence'] for r in state['rows']
                                  if r['phase'] == 'initialization'))
    if not sequences:
        raise ValueError('Target validation requires observed initialization anchors')
    return sequences


def target_probe(model, data, sequences, seeds=(8131, 8132, 8133)):
    """One fixed full batch, fp32, dropout disabled, paired posterior RNG.

    Preserve the search RNG and original train/eval mode. A full fixed batch
    preserves native token/padding reduction exactly across checkpoints.
    """
    from uniref_vae.data import collate_fn
    sequences = list(sequences)
    if not sequences or not seeds:
        raise ValueError('Empty target probe')
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        raise ValueError('Retention validation expects fp32 master VAE weights')
    device = parameter.device
    encoded = [data.encode(s).unsqueeze(0) for s in data.tokenize_sequence(sequences)]
    x = collate_fn(encoded).to(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    values = []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for seed in seeds:
                # Seed only the generators captured above, without changing other GPUs.
                torch.random.default_generator.manual_seed(seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(seed)
                output = model(x)
                row = {k: float(output[k]) for k in ('loss', 'recon_loss', 'kldiv')}
                if not all(math.isfinite(v) for v in row.values()):
                    raise NumericalVAEError('Nonfinite target validation loss')
                if not math.isclose(row['loss'], row['recon_loss'] + .0001*row['kldiv'], rel_tol=1e-5, abs_tol=1e-10):
                    raise ValueError('Unexpected VAE loss definition')
                values.append(row)
    finally:
        model.train(was_training)
    return dict(mean_target_loss=math.fsum(r['loss'] for r in values)/len(values),
        reconstruction_loss=math.fsum(r['recon_loss'] for r in values)/len(values),
        kl_divergence=math.fsum(r['kldiv'] for r in values)/len(values),
        by_seed=values, seeds=list(seeds), sequences=len(sequences),
        sequence_sha256=hashlib.sha256(json.dumps(sequences, separators=(',', ':')).encode()).hexdigest(),
        precision='fp32_eval', device=str(device), independent_heldout=False)


def target_decision(before, after, baseline, tolerance=1e-12):
    """No relative degradation allowance; epsilon only handles numerical equality."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid loss comparison tolerance')
    for candidate in (after, baseline):
        for key in ('sequence_sha256', 'seeds', 'sequences', 'precision', 'device'):
            if candidate[key] != before[key]:
                raise ValueError('Target probe identity changed: '+key)
    losses = [v['mean_target_loss'] for v in (before, after, baseline)]
    if not all(math.isfinite(v) for v in losses):
        return False, ['nonfinite_target_loss']
    reasons = []
    if losses[1] > losses[0] + tolerance:
        reasons.append('target_loss_worse_than_pre_update')
    if losses[1] > losses[2] + tolerance:
        reasons.append('target_loss_worse_than_initial_baseline')
    return not reasons, reasons


def checked_training_step(model, x, optimizer, scaler):
    """Reject nonfinite loss/gradient/weights instead of silently skipped steps."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
        output = model(x)
    loss = output['loss']
    if not torch.isfinite(loss).item():
        raise NumericalVAEError('Nonfinite VAE training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm_squared = 0.
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise NumericalVAEError('Nonfinite VAE gradient')
            norm_squared += float(parameter.grad.detach().double().square().sum())
    if not math.isfinite(norm_squared):
        raise NumericalVAEError('Nonfinite VAE gradient norm')
    scaler.step(optimizer)
    scaler.update()
    if not finite_weights(model):
        raise NumericalVAEError('Nonfinite VAE weights after optimizer step')
    return dict(loss=float(loss.detach()), reconstruction_loss=float(output['recon_loss'].detach()),
        kl_divergence=float(output['kldiv'].detach()), gradient_norm=math.sqrt(norm_squared),
        scaler=float(scaler.get_scale()), optimizer_step_executed=True)


def guarded_retrain(model, data, state, train_candidate, source_check, author_baseline,
                    seeds=(8131,8132,8133), tolerance=1e-12):
    """Publish weights only after source AND target checks; never mutate BO models.

    train_candidate appends per-step records. Optimizer/scaler are update-local
    and discarded on rejection. Infrastructure exceptions restore then propagate.
    """
    initial = cpu_state(model)
    anchors = initial_anchors(state)
    target_before = target_probe(model, data, anchors, seeds)
    baseline = state.setdefault('target_retention_initial_baseline', target_before)
    before = source_check(model, data)
    after = target_after = None
    records, reasons = [], []
    accepted = False
    try:
        train_candidate(records)
        model.eval()
        if not finite_weights(model):
            raise NumericalVAEError('Nonfinite candidate VAE weights')
        target_after = target_probe(model, data, anchors, seeds)
        target_ok, reasons = target_decision(target_before, target_after, baseline, tolerance)
        after = source_check(model, data)
        if not math.isfinite(after['reconstruction_loss']):
            after = None
            raise NumericalVAEError('Nonfinite source validation loss')
        source_ok = (
            after['feasible'] >= .8*max(before['feasible'], author_baseline['feasible']) and
            after['reconstruction_loss'] <= 1.25*min(before['reconstruction_loss'], author_baseline['reconstruction_loss']))
        if not source_ok:
            reasons.append('source_retention_failed')
        accepted = target_ok and source_ok
    except NumericalVAEError as error:
        reasons = ['numerical_vae_failure: '+str(error)]
    except Exception:
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        model.eval()
        raise
    if not accepted:
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
    model.eval()
    kept = cpu_state(model)
    delta = math.sqrt(math.fsum(float((v-initial[k]).double().square().sum())
                               for k,v in kept.items() if v.is_floating_point()))
    if not accepted:
        assert all(torch.equal(v, initial[k]) for k,v in kept.items()), 'Incomplete VAE rollback'
    return dict(accepted=accepted, rolled_back=not accepted, rejection_reasons=reasons,
        target_loss_before=target_before, target_loss_after=target_after,
        target_loss_initial_baseline=baseline, target_loss_absolute_tolerance=tolerance,
        source_decoder_before=before, source_decoder_after=after,
        vae_change_l2=delta, optimizer_steps=len(records), training_steps=records,
        first_loss=records[0]['loss'] if records else None, last_loss=records[-1]['loss'] if records else None,
        objective='reconstruction+0.0001*KL', gate_version=1)
