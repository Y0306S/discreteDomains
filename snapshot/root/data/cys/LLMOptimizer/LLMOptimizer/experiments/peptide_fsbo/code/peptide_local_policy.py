"""Frozen-VAE peptide policy, using BOLT's existing local/recenter mechanisms.

Coordinates here are absolute decoder inputs. Canonical transfer coordinates
belong to the objective backend, never to the constraint training data.
"""
from __future__ import annotations

import numpy as np
import torch


DEFAULTS = dict(recenter_every=10, recenter_epochs=2, top_k=1000,
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
    if state['recenter_plan'] is None and state['stagnation'] >= recipe['local']['recenter_every']:
        state['recenter_plan'] = recenter_plan(state['rows'], recipe['batch_size'], recipe['local'])
        state['stagnation'] = 0
    return 'recenter' if state['recenter_plan'] is not None else 'acquisition'


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
        update_region(state['trust_region'], torch.tensor(scored['scores']).reshape(-1, 1),
                      torch.as_tensor(constraints))
        if state['trust_region'].restart_triggered:
            # Include the just-committed batch and the actual incumbent constraint.
            state['trust_region'] = new_region(state['rows'], state['constraints'],
                recipe['batch_size'], recipe['local'], turbo_class)
    else:
        state['recenter_plan']['cursor'] = pending['next_cursor']
    state['last_event_size'] = n
