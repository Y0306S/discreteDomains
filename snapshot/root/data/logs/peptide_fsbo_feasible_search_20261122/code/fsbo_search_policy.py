"""FSBO peptide search with exact decoded feasibility and accepted-update recenter.

Coordinates here are absolute decoder inputs. Canonical transfer coordinates
belong to the objective backend, never to the constraint training data.
Derived from tools/peptide_local_policy.py; isolated from historical campaigns.
"""
from __future__ import annotations

import numpy as np
import torch


DEFAULTS = dict(recenter_epochs=1, top_k=200,
                tr_length=0.8, tr_length_min=0.5**7, tr_length_max=1.6,
                failure_tolerance=32, success_tolerance=10, perturb_dimensions=20.)


def tpe_constraints(trial):
    # Stable module identity keeps CLI-created sampler checkpoints portable.
    return trial.user_attrs['constraints']


def incumbent(rows):
    valid = [i for i, row in enumerate(rows) if row['feasible']]
    if not valid:
        raise ValueError('Local peptide search requires a measured feasible incumbent')
    return max(valid, key=lambda i: rows[i]['score'])


def new_region(rows, constraints, batch_size, settings, turbo_class):
    i = incumbent(rows)
    return turbo_class(dim=256, batch_size=batch_size,
        length=settings['tr_length'], length_min=settings['tr_length_min'],
        length_max=settings['tr_length_max'],
        failure_tolerance=settings['failure_tolerance'],
        success_tolerance=settings['success_tolerance'],
        best_value=rows[i]['score'],
        best_constraint_values=torch.as_tensor(constraints[i]).reshape(-1).clone())


def bounds(center, region):
    # Native generate_batch with absolute_bounds=None: weights=8.
    return center - 4.*region.length, center + 4.*region.length


def sparse_mask(count, dim=256, perturb_dimensions=20.):
    mask = torch.rand(count, dim) <= min(perturb_dimensions/dim, 1.)
    empty = torch.where(mask.sum(1) == 0)[0]
    # Match native BOLT's fallback coordinate range.
    mask[empty, torch.randint(0, dim-1, (len(empty),))] = True
    return mask


def candidate_pool(center, region, count, perturb_dimensions=20.):
    from torch.quasirandom import SobolEngine
    low, high = bounds(center, region)
    perturbed = low + (high-low)*SobolEngine(len(center), scramble=True).draw(count)
    mask = sparse_mask(count, len(center), perturb_dimensions)
    candidates = center.expand(count, -1).clone()
    candidates[mask] = perturbed[mask]
    return candidates


def adaptive_pool(center, region, count, decode, constraints, *,
                  lower=.1, upper=.5, max_adjustments=3, decode_batch_size=128):
    """Reuse one Sobol draw AND sparse mask across at most three rescalings.

    Decode the final pool too, including after the last permitted adjustment.
    Return those exact sequences for scoring, without a second stochastic decode.
    No oracle calls are made here.
    """
    from torch.quasirandom import SobolEngine
    if count < 1 or decode_batch_size < 1 or max_adjustments < 0:
        raise ValueError('Invalid candidate pool budget')
    if not 0 <= lower <= upper <= 1:
        raise ValueError('Invalid feasibility thresholds')
    unit = SobolEngine(len(center), scramble=True).draw(count).to(center)
    mask = sparse_mask(count, len(center)).to(center.device)
    region.length = min(region.length_max, max(region.length_min, region.length))
    history = []
    for attempt in range(max_adjustments+1):
        low, high = bounds(center, region)
        perturbed = low + (high-low)*unit
        candidates = center.expand(count, -1).clone()
        candidates[mask] = perturbed[mask]
        sequences = []
        with torch.no_grad():
            for batch in candidates.split(decode_batch_size):
                decoded = list(decode(batch))
                if len(decoded) != len(batch):
                    raise ValueError('Decoder returned a misaligned pool')
                sequences.extend(decoded)
        values = np.asarray(constraints(sequences)).reshape(count, -1)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite decoded constraints')
        feasible = (values <= 0).all(axis=1)
        rate = float(feasible.mean())
        history.append(dict(length=float(region.length), feasible_rate=rate))
        if attempt == max_adjustments:
            break
        length = region.length
        if rate < lower:
            length = max(region.length_min, length/2.)
        elif rate > upper:
            length = min(region.length_max, length*2.)
        if length == region.length:
            break
        region.length = length
    return candidates, sequences, feasible, dict(
        adjustments=len(history)-1, probes=history, length=float(region.length),
        feasible_rate=rate, decoded_candidates=count*len(history))


def select_batch(candidates, sequences, feasible, count, rank_indices):
    """Rank exact feasible candidates first; fill shortfalls from infeasible ones.

    rank_indices(pool, count) returns local row indices without replacement.
    Keep row identity even when distinct latent rows decode to the same peptide.
    """
    feasible = np.asarray(feasible, dtype=bool)
    if len(sequences) != len(candidates) or feasible.shape != (len(candidates),):
        raise ValueError('Misaligned candidate pool')
    if not 0 < count <= len(candidates):
        raise ValueError('Invalid selection count')
    chosen = []
    for valid in (True, False):
        ids = np.flatnonzero(feasible == valid)
        take = min(count-len(chosen), len(ids))
        if take:
            local = np.asarray(rank_indices(candidates[ids], take), dtype=int).reshape(-1)
            if (len(local) != take or len(set(local)) != take or
                    (local < 0).any() or (local >= len(ids)).any()):
                raise ValueError('Ranking must return unique pool indices')
            chosen.extend(ids[local].tolist())
        if len(chosen) == count:
            break
    return candidates[chosen], [sequences[i] for i in chosen], dict(
        feasible_selected=int(feasible[chosen].sum()),
        infeasible_fill=int((~feasible[chosen]).sum()))


def thompson_indices(model, candidates, count, sampler_class):
    """Use native objective TS, sampling row IDs to preserve decoded identity.

    Exact decoded constraints have already split the pool, so a predicted
    constraint is not allowed to veto known feasible candidates.
    """
    class IndexedModel:
        def posterior(self, ids, observation_noise=False):
            return model.posterior(candidates[ids[..., 0].long()],
                                   observation_noise=observation_noise)

    ids = torch.arange(len(candidates), device=candidates.device,
                       dtype=candidates.dtype).unsqueeze(-1)
    with torch.no_grad():
        selected = sampler_class(model=IndexedModel(), constrained=False,
                                 replacement=False)(ids, num_samples=count)
    return selected.flatten().long().cpu().numpy()


def recenter_plan(rows, recent, settings):
    # Native V5 uses last bsz observations + unique feasible top-k, in score order.
    order = sorted(range(len(rows)), key=lambda i: rows[i]['score'], reverse=True)
    top, seen = [], set()
    for i in order:
        row = rows[i]
        if row['feasible'] and row['sequence'] not in seen:
            seen.add(row['sequence'])
            top.append(row['sequence'])
            if len(top) == settings['top_k']:
                break
    if not top:
        raise ValueError('Recenter requires feasible seed sequences')
    inputs = [r['sequence'] for r in rows[-recent:]] + top
    return dict(inputs=inputs*settings['recenter_epochs'], cursor=0,
                batch=max(1, 2560//max(map(len, inputs))))


def next_phase(state, recipe):
    plan = state['recenter_plan']
    if plan is not None and plan['cursor'] >= len(plan['inputs']):
        state['recenter_plan'] = None
    return 'recenter' if state['recenter_plan'] is not None else 'acquisition'


def after_retrain(state, event, recipe):
    """Call before atomic publication of the retained VAE and new coordinates."""
    if not event['accepted'] or event.get('rolled_back', False):
        return False
    if state['recenter_plan'] is not None:
        raise ValueError('Cannot replace an unfinished recenter plan')
    state['recenter_plan'] = recenter_plan(
        state['rows'], recipe['batch_size'], recipe['local'])
    return True


def constrain_indices(model, candidates):
    """Native constraint-TS rule: sampled feasible, otherwise least violation."""
    with torch.no_grad():
        sample = model.posterior(candidates).rsample(torch.Size([1])).reshape(-1)
    valid = torch.where(sample <= 0)[0]
    return valid if len(valid) else sample.argmin().reshape(1)


def phase_metrics(rows):
    result = {}
    for phase in ('acquisition', 'recenter'):
        group = [r for r in rows if r['phase'] == phase]
        fresh = [r for r in group if not r['cached']]
        result[phase] = dict(proposals=len(group), fresh=len(fresh),
            fresh_feasible=sum(r['feasible'] for r in fresh),
            feasible=sum(r['feasible'] for r in group),
            unique_feasible=len({r['sequence'] for r in fresh if r['feasible']}),
            fresh_feasible_rate=sum(r['feasible'] for r in fresh)/len(fresh) if fresh else None)
    return result


def commit(state, pending, recipe, update_region, turbo_class):
    """Commit a durably scored event once; callers transact state+pending together."""
    if pending['event'] != state['progress']['events']+1:
        raise ValueError('Pending event is out of order')
    before = state['rows'][incumbent(state['rows'])]['score']
    phase, scored = pending['phase'], pending['scored']
    constraints = np.asarray(pending['constraints'], dtype=np.float32).reshape(-1, 1)
    z = np.asarray(pending['z'], dtype=np.float32)
    canonical = np.asarray(pending['canonical_z'], dtype=np.float32)
    n = len(pending['sequences'])
    if (z.shape != (n, 256) or canonical.shape != z.shape or constraints.shape != (n, 1)
            or not all(np.isfinite(v).all() for v in (z, canonical, constraints))):
        raise ValueError('Misaligned event coordinates or constraints')
    progress = state['progress']
    acquired = phase == 'acquisition'
    progress['events'] += 1
    progress['rounds'] += int(acquired)
    progress['proposals' if acquired else 'recenter_proposals'] += n
    progress['fresh_evaluations'] += scored['fresh_evaluations']
    progress['acquisition_fresh' if acquired else 'recenter_fresh'] += scored['fresh_evaluations']
    progress['cache_hits'] += sum(scored['cache_hits'])
    progress['seconds'] += pending.get('seconds', 0.)
    for i, sequence in enumerate(pending['sequences']):
        score = float(scored['scores'][i])
        state['rows'].append(dict(sequence=sequence, score=score, mic=-score,
            feasible=bool(constraints[i, 0] <= 0), cached=bool(scored['cache_hits'][i]),
            phase=phase, round=progress['rounds'], elapsed=progress['seconds'],
            proposed_z=z[i].tolist()))
    state['constraint_z'] = np.concatenate((state['constraint_z'], z))
    state['features'] = np.concatenate((state['features'], canonical))
    state['constraints'] = np.concatenate((state['constraints'], constraints))
    after = state['rows'][incumbent(state['rows'])]['score']
    if after > before:
        state['stagnation'] = 0
    elif acquired:
        state['stagnation'] += 1
    if acquired:
        # Retain native success/failure adaptation, but enforce the floor and
        # remove region restarts. Decoder feasibility also adjusts length before
        # acquisition; neither path may take it below the configured minimum.
        update_region(state['trust_region'], torch.tensor(scored['scores']).reshape(-1, 1),
                      torch.as_tensor(constraints))
        region = state['trust_region']
        region.length = min(region.length_max, max(region.length_min, region.length))
        region.restart_triggered = False
    else:
        state['recenter_plan']['cursor'] = pending['next_cursor']
    state['last_event_size'] = n
