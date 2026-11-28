"""Audited proposal-only continuation for queries that exhaust latent retries."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import time


def write_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as handle:
        pickle.dump(value, handle, protocol=4)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def collect(runner, task, state, checkpoint, path, sample, max_batches):
    """Retain accepted plans and continue retry seeds without any SQL execution."""
    seen = {runner.plan_key(row['plan']) for row in state['rows']}
    seen.update(runner.plan_key(item['plan']) for item in checkpoint['accepted'])
    accepted, rejected = checkpoint['accepted'], checkpoint['rejected']
    if len(accepted) == 10:
        return dict(items=accepted, rejected=rejected, fit=checkpoint['fit'])
    for retry in range(checkpoint['next_retry'], max_batches):
        latents, decoded = sample(retry)
        if len(latents) != 10 or len(decoded) != 10:
            raise ValueError('Expected ten latent vectors and ten decoder responses')
        for latent, item in zip(latents, decoded):
            try:
                plan = runner.canonicalize(task, item['plan'])
                plan = runner.canonicalize(task, plan, strict=True)
                key = runner.plan_key(plan)
                if item.get('fallback'):
                    raise ValueError('decoder fallback')
                if key in seen:
                    raise ValueError('duplicate canonical plan')
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                rejected.append(dict(retry=retry, error=str(exc), decoded=item))
                continue
            accepted.append(dict(plan=plan, proposed_z=latent, raw_vae_tokens=item['plan'],
                decode_retries=item.get('decode_retries'), native_canonicalized=item['plan'] != plan))
            seen.add(key)
            if len(accepted) == 10:
                break
        checkpoint['next_retry'] = retry + 1
        write_checkpoint(path, checkpoint)
        runner.atomic(runner.ROOT/'runs'/task/'proposal_progress.json', dict(task=task,
            round=checkpoint['round'], latent_batches=retry+1, latent_attempts=(retry+1)*10,
            accepted=len(accepted), rejected=len(rejected), last_rejections=rejected[-10:],
            recovery=True, checkpoint=str(path), max_latent_batches=max_batches, at=time.time()))
        if len(accepted) == 10:
            return dict(items=accepted, rejected=rejected, fit=checkpoint['fit'])
    raise RuntimeError(f'Could only decode {len(accepted)}/10 unique feasible plans after {max_batches} batches; partial proposals saved in {path}')


def next_batch(runner, task, state, vae, round_number, policy):
    import torch
    # This amendment applies only to explicitly listed queries, starting at the failed round.
    path = runner.ROOT/'runs'/task/f'proposal_recovery_{round_number:03}.pickle'
    fingerprint = hashlib.sha256(pickle.dumps(dict(rows=state['rows'], turbo=state['turbo']), protocol=4)).hexdigest()
    identity = dict(task=task, round=round_number, state_sha256=fingerprint,
        recipe_sha256=runner.digest(runner.ROOT/'recipe.json'), policy=policy)
    checkpoint = None
    if path.exists():
        with path.open('rb') as handle:
            checkpoint = pickle.load(handle)
        if checkpoint['identity'] != identity:
            raise ValueError('Proposal checkpoint identity changed')
        if len(checkpoint['accepted']) == 10:
            return dict(items=checkpoint['accepted'], rejected=checkpoint['rejected'], fit=checkpoint['fit'])
    observed, keys = [], set()
    for row in state['rows']:
        key = runner.plan_key(row['plan'])
        if not row['censored'] and key not in keys:
            observed.append(row)
            keys.add(key)
    if not observed:
        raise ValueError('LOLBO needs an uncensored incumbent')
    x = torch.tensor([row['z'] for row in observed], dtype=torch.float32)
    y = torch.tensor([-row['runtime'] for row in observed], dtype=torch.float32).view(-1, 1)
    seed = runner.seed_for(task, 'lolbo', round_number, runner.recipe()['seed'])
    # Seed initialization as well as fitting so a recovery can be reproduced.
    torch.manual_seed(seed % (2**63 - 1))
    device = torch.device('cpu')
    model, mll = runner.lolbo.make_surrogate(x, device)
    if checkpoint is None:
        model = runner.fit_lolbo(model, mll, x, y, 20 if round_number == 1 else 2)
        checkpoint = dict(identity=identity, round=round_number, next_retry=0,
            accepted=[], rejected=[], model_state=model.state_dict(),
            fit=dict(train_rows=len(observed), epochs=20 if round_number == 1 else 2, seed=seed,
                proposal_recovery_policy=policy, surrogate_initialization_seeded=True))
        write_checkpoint(path, checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
    turbo = runner.lolbo.TurboState(**state['turbo'])
    def sample(retry):
        torch.manual_seed((seed + retry) % (2**63 - 1))
        with torch.no_grad():
            z = runner.lolbo.generate_batch(turbo, model, x, y, batch_size=10,
                n_candidates=2000, device=device, unit_cube=False)
        if z.shape != (10, 64) or not torch.isfinite(z).all():
            raise ValueError('LOLBO returned an invalid latent batch')
        latents = z.tolist()
        return latents, vae.decode(latents, seed=(seed + retry) % (2**31 - 1))
    return collect(runner, task, state, checkpoint, path, sample, policy['max_latent_batches'])


def install(runner, task, operations):
    policy_path = Path(operations)/'bolt_proposal_recovery_policy.json'
    if not policy_path.exists():
        return
    all_policies = json.loads(policy_path.read_text())
    policy = all_policies.get('tasks', {}).get(task)
    if policy is None:
        return
    original = runner.next_batch
    def amended(task, state, vae, round_number):
        if round_number < policy['start_round']:
            return original(task, state, vae, round_number)
        return next_batch(runner, task, state, vae, round_number, policy)
    runner.next_batch = amended
