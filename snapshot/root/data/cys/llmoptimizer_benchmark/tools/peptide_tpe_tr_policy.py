"""Sequential constrained TPE in an adaptive box. No GP or posterior sampling."""
from __future__ import annotations

import copy
import math

import numpy as np


DEFAULTS = dict(length=0.8, length_min=0.5**7, length_max=1.6,
                success_tolerance=3, failure_tolerance=32,
                relative_improvement=1e-3, n_startup_trials=0,
                n_ei_candidates=24)


def constraints(trial):
    return trial.user_attrs['constraints']


def validate(settings):
    if set(settings) != set(DEFAULTS):
        raise ValueError('Incomplete or unknown TPE-TR setting')
    if not all(math.isfinite(float(v)) for v in settings.values()):
        raise ValueError('Nonfinite TPE-TR setting')
    if not 0 < settings['length_min'] <= settings['length'] <= settings['length_max']:
        raise ValueError('Invalid trust-region lengths')
    for key in ('success_tolerance', 'failure_tolerance', 'n_ei_candidates'):
        if settings[key] < 1 or int(settings[key]) != settings[key]:
            raise ValueError('Invalid '+key)
    if settings['n_startup_trials'] != 0 or settings['relative_improvement'] < 0:
        raise ValueError('TPE-TR must use its imported warm starts immediately')


def incumbent(records):
    feasible = [i for i, r in enumerate(records) if max(r['constraints']) <= 0]
    if not feasible:
        raise ValueError('TPE-TR needs a measured feasible starting point')
    return max(feasible, key=lambda i: records[i]['score'])


def new_region(records, settings):
    validate(settings)
    best = incumbent(records)
    return dict(length=settings['length'], center_index=best,
                best_value=float(records[best]['score']), successes=0, failures=0,
                revision=0, resets=0, last_action='initialize')


def bounds(records, region, global_bounds):
    center = np.asarray(records[region['center_index']]['z'], dtype=np.float64)
    lower, upper = map(float, global_bounds)
    if (center.ndim != 1 or not np.isfinite(center).all() or not lower < upper
            or np.any(center < lower) or np.any(center > upper)):
        raise ValueError('Invalid region center/global bounds')
    radius = (upper-lower)*region['length']/2
    low, high = np.maximum(lower, center-radius), np.minimum(upper, center+radius)
    if not np.all(low < high):
        raise ValueError('Degenerate trust region')
    return low, high


def observe(region, records, settings):
    """Apply exactly one newly measured (or cached) proposal, never an unscored one."""
    latest = records[-1]
    score = float(latest['score'])
    if not math.isfinite(score) or not all(math.isfinite(c) for c in latest['constraints']):
        raise ValueError('Invalid measurement; region left unchanged')
    previous = region['best_value']
    feasible = max(latest['constraints']) <= 0
    significant = feasible and score > previous + settings['relative_improvement']*abs(previous)
    region['last_action'] = 'unchanged'
    changed = False
    # Infeasible high scores never move the center or count as successes.
    if feasible and score > previous:
        region.update(center_index=len(records)-1, best_value=score)
        changed = True
        region['last_action'] = 'move_center'
    if significant:
        region['successes'] += 1
        region['failures'] = 0
    else:
        region['successes'] = 0
        region['failures'] += 1
    old_length = region['length']
    if region['successes'] >= settings['success_tolerance']:
        region['length'] = min(settings['length_max'], old_length*2)
        region['successes'] = 0
        region['last_action'] = 'expand'
    elif region['failures'] >= settings['failure_tolerance']:
        region['length'] = old_length/2
        region['failures'] = 0
        region['last_action'] = 'shrink'
        if region['length'] < settings['length_min']:
            # Re-open the initial box about the best feasible point; no extra evaluations.
            region['length'] = settings['length']
            region['resets'] += 1
            region['last_action'] = 'reset_at_incumbent'
    if changed or old_length != region['length'] or region['last_action'] == 'reset_at_incumbent':
        region['revision'] += 1


class SequentialTPE:
    """Rebuild a normalized local study only after a box change or process restart.

    Historical points outside the current box remain in the durable global archive,
    but are not relabelled/clipped into local training observations. Their true scores
    are retained for future boxes. Sampler RNG state is carried across study rebuilds.
    """
    def __init__(self, settings, global_bounds, seed, sampler=None):
        import optuna
        from optuna.distributions import FloatDistribution
        validate(settings)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.settings, self.global_bounds, self.seed = settings, global_bounds, seed
        self.sampler = copy.deepcopy(sampler) if sampler is not None else optuna.samplers.TPESampler(
            seed=seed, constraints_func=constraints, n_startup_trials=0,
            n_ei_candidates=settings['n_ei_candidates'], multivariate=False)
        self.study = None
        self.revision = None
        self.pending = None
        self.distribution = FloatDistribution(0., 1.)

    def build(self, records, region):
        import optuna
        from optuna.trial import create_trial
        if self.study is not None and self.revision == region['revision']:
            return
        if self.pending is not None:
            raise ValueError('Cannot change region while an evaluation is pending')
        self.low, self.high = bounds(records, region, self.global_bounds)
        self.names = [f'u{i:03d}' for i in range(len(self.low))]
        self.distributions = {name:self.distribution for name in self.names}
        self.study = optuna.create_study(direction='maximize', sampler=self.sampler)
        for record in records:
            z = np.asarray(record['z'], dtype=np.float64)
            if np.any(z < self.low) or np.any(z > self.high):
                continue
            # Preserve exact trial parameters on a restart in the same region.
            u = record['u'] if record.get('region_revision') == region['revision'] else (
                (z-self.low)/(self.high-self.low)).tolist()
            self.study.add_trial(create_trial(params=dict(zip(self.names, map(float, u))),
                distributions=self.distributions, value=record['score'],
                user_attrs={'constraints':record['constraints']},
                system_attrs={'constraints':record['constraints']}))
        self.local_observations = len(self.study.trials)
        if not self.local_observations:
            raise ValueError('The feasible incumbent must be in its own region')
        self.revision = region['revision']

    def ask(self, records, region):
        if self.pending is not None:
            raise ValueError('Only one recommendation may be pending at a time')
        self.build(records, region)
        trial = self.study.ask(fixed_distributions=self.distributions)
        u = [trial.params[name] for name in self.names]
        z = self.low + np.asarray(u)*(self.high-self.low)
        self.pending = trial
        return dict(number=trial.number, u=u, z=z.tolist(),
                    region_revision=region['revision'], local_observations=self.local_observations)

    def restore_pending(self, records, region, proposal):
        from optuna.trial import Trial, TrialState, create_trial
        self.build(records, region)
        if proposal['region_revision'] != region['revision']:
            raise ValueError('Pending proposal/region revision mismatch')
        if proposal['number'] != len(self.study.trials):
            raise ValueError('Pending trial sequence changed')
        expected = self.low + np.asarray(proposal['u'])*(self.high-self.low)
        if not np.array_equal(expected, proposal['z']):
            raise ValueError('Pending proposal coordinates changed')
        self.study.add_trial(create_trial(params=dict(zip(self.names, proposal['u'])),
            distributions=self.distributions, state=TrialState.RUNNING))
        self.pending = Trial(self.study, self.study.trials[-1]._trial_id)

    def tell(self, score, values):
        if self.pending is None:
            raise ValueError('No recommendation awaiting a measurement')
        if not math.isfinite(score) or not all(math.isfinite(c) for c in values):
            raise ValueError('Invalid objective/constraint')
        self.pending.set_user_attr('constraints', list(values))
        self.study.tell(self.pending, float(score))
        self.pending = None
        self.local_observations += 1

    def checkpoint(self):
        # The constraints callback lives in this importable module (not __main__).
        return copy.deepcopy(self.study.sampler if self.study is not None else self.sampler)
