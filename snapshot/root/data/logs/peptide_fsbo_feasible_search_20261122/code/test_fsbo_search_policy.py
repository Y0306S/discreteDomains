"""Behavioral regression tests; no MIC calls or VAE training."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import fsbo_search_policy as policy

sys.path.insert(0, '/root/data/cys/BOLT/optimization/peptides')
from lolbo.utils.bo_utils.turbo import TurboState, update_state
from lolbo.utils.bo_utils.constrained_max_posterior_sampling import MaxPosteriorSampling


def row(sequence, score, feasible=True):
    return dict(sequence=sequence, score=score, feasible=feasible)


def recipe():
    return dict(local=dict(policy.DEFAULTS), batch_size=50)


def region(length=.8):
    return SimpleNamespace(length=length, length_min=.0078125, length_max=1.6)


class RecenterTests(unittest.TestCase):
    def state(self):
        return dict(rows=[row(str(i), i) for i in range(300)],
                    recenter_plan=None, stagnation=10000)

    def test_stagnation_alone_never_recenters(self):
        self.assertEqual(policy.next_phase(self.state(), recipe()), 'acquisition')

    def test_rejected_or_rolled_back_update_does_not_recenter(self):
        for event in (dict(accepted=False, rolled_back=True),
                      dict(accepted=True, rolled_back=True)):
            state = self.state()
            self.assertFalse(policy.after_retrain(state, event, recipe()))
            self.assertEqual(policy.next_phase(state, recipe()), 'acquisition')

    def test_accepted_update_recent50_unique_feasible_top200_once(self):
        state = self.state()
        state['rows'].insert(0, row('infeasible', 1e9, False))
        state['rows'].insert(1, row('299', 1e8))
        self.assertTrue(policy.after_retrain(state, dict(accepted=True, rolled_back=False), recipe()))
        self.assertEqual(policy.next_phase(state, recipe()), 'recenter')
        plan = state['recenter_plan']
        self.assertEqual(len(plan['inputs']), 250)
        self.assertEqual(plan['inputs'][:50], [str(i) for i in range(250, 300)])
        self.assertEqual(plan['inputs'][50:], [str(i) for i in range(299, 99, -1)])

    def test_checkpointed_plan_resumes_and_finishes_without_retrigger(self):
        state = self.state()
        policy.after_retrain(state, dict(accepted=True), recipe())
        state['recenter_plan']['cursor'] = 17
        restored = json.loads(json.dumps(state))
        self.assertEqual(policy.next_phase(restored, recipe()), 'recenter')
        self.assertEqual(restored['recenter_plan']['cursor'], 17)
        restored['recenter_plan']['cursor'] = len(restored['recenter_plan']['inputs'])
        self.assertEqual(policy.next_phase(restored, recipe()), 'acquisition')
        self.assertIsNone(restored['recenter_plan'])


class PoolTests(unittest.TestCase):
    def pool(self, rates, length=.8, count=100, batch=100):
        calls = []
        def decode(x):
            calls.append(x.clone())
            return [str(i) for i in range(len(x))]
        def constraints(sequences):
            rate = rates[min(len(calls)-1, len(rates)-1)]
            return [-1 if i < round(rate*len(sequences)) else 1 for i in range(len(sequences))]
        tr = region(length)
        output = policy.adaptive_pool(torch.zeros(256), tr, count, decode, constraints,
                                      decode_batch_size=batch)
        return tr, calls, output

    def test_halve_below_ten_percent(self):
        tr, calls, output = self.pool([.09, .2])
        self.assertEqual(tr.length, .4)
        self.assertEqual(output[3]['adjustments'], 1)

    def test_double_above_fifty_percent(self):
        tr, calls, output = self.pool([.51, .2], length=.4)
        self.assertEqual(tr.length, .8)

    def test_exact_thresholds_do_not_adjust(self):
        for rate in (.1, .5):
            tr, calls, output = self.pool([rate])
            self.assertEqual(tr.length, .8)
            self.assertEqual(len(calls), 1)

    def test_floor_and_ceiling_stop_adjustments(self):
        for length, rate in ((.0078125, 0.), (1.6, 1.)):
            tr, calls, output = self.pool([rate], length)
            self.assertEqual(tr.length, length)
            self.assertEqual(len(calls), 1)
        tr, calls, output = self.pool([0.], length=.01)
        self.assertEqual(tr.length, .0078125)

    def test_oscillation_stops_after_three_changes_and_decodes_final_length(self):
        tr, calls, output = self.pool([0., 1., 0., 1.])
        self.assertEqual([v['length'] for v in output[3]['probes']], [.8, .4, .8, .4])
        self.assertEqual(output[3]['adjustments'], 3)
        self.assertEqual(len(calls), 4)
        self.assertEqual(tr.length, .4)
        self.assertTrue(output[2].all())
        torch.testing.assert_close(output[0], calls[-1])

    def test_reuses_sobol_points_and_mask(self):
        _, calls, _ = self.pool([0., 0., .2])
        torch.testing.assert_close(calls[1], calls[0]/2)
        torch.testing.assert_close(calls[2], calls[0]/4)

    def test_5000_candidates_are_decoded_in_bounded_batches(self):
        calls = []
        def decode(x):
            calls.append(len(x))
            return ['AC' for _ in x]
        _, seqs, feasible, info = policy.adaptive_pool(
            torch.zeros(256), region(.0078125), 5000, decode,
            lambda seqs: np.ones((len(seqs), 1)))
        self.assertEqual(sum(calls), 5000)
        self.assertLessEqual(max(calls), 128)
        self.assertEqual(len(seqs), 5000)
        self.assertFalse(feasible.any())
        self.assertEqual(info['decoded_candidates'], 5000)


class SelectionTests(unittest.TestCase):
    def select(self, feasible_count):
        # Infeasible rows have higher objective values: feasibility still wins.
        x = torch.arange(100.).reshape(-1, 1)
        seqs = ['decoded-'+str(i) for i in range(100)]
        valid = np.arange(100) < feasible_count
        calls = []
        def rank(pool, count):
            calls.append(pool.clone())
            return pool[:, 0].argsort(descending=True)[:count].numpy()
        return policy.select_batch(x, seqs, valid, 50, rank), calls

    def test_selects_only_feasible_when_at_least_fifty_exist(self):
        for feasible_count in (50, 60, 100):
            (z, seqs, info), calls = self.select(feasible_count)
            self.assertEqual(info['feasible_selected'], 50)
            self.assertEqual(info['infeasible_fill'], 0)
            self.assertTrue((z[:, 0] < feasible_count).all())
            self.assertEqual(len(calls), 1)
            self.assertEqual(seqs, ['decoded-'+str(int(v)) for v in z[:, 0]])

    def test_shortfall_filled_from_infeasible_pool(self):
        for feasible_count in (0, 1, 23, 49):
            (z, seqs, info), calls = self.select(feasible_count)
            self.assertEqual(len(z), 50)
            self.assertEqual(info['feasible_selected'], feasible_count)
            self.assertEqual(info['infeasible_fill'], 50-feasible_count)
            self.assertEqual(len(z.unique()), 50)

    def test_native_TS_preserves_row_identity_with_duplicate_latents(self):
        class Model:
            def posterior(self, x, observation_noise=False):
                class Posterior:
                    def rsample(self, sample_shape):
                        return torch.arange(len(x)).float().reshape(1, -1, 1).expand(*sample_shape, -1, -1)
                return Posterior()
        # Distinct stochastic decodes may share identical latent values.
        candidates = torch.zeros(10, 256)
        ids = policy.thompson_indices(Model(), candidates, 4, MaxPosteriorSampling)
        self.assertEqual(set(ids), {6, 7, 8, 9})


class CommitTests(unittest.TestCase):
    def test_native_shrink_clamps_at_floor_without_restarting(self):
        state = dict(rows=[row('AC', -5.)], constraint_z=np.zeros((1, 256)),
            features=np.zeros((1, 256)), constraints=np.array([[-.1]]),
            stagnation=0, recenter_plan=None,
            progress=dict(events=0, rounds=0, proposals=0, fresh_evaluations=0,
                acquisition_fresh=0, recenter_fresh=0, recenter_proposals=0,
                cache_hits=0, seconds=0.))
        tr = TurboState(dim=256, batch_size=50, length=.0078125,
            failure_tolerance=1, best_value=-5., best_constraint_values=torch.tensor([-.1]))
        state['trust_region'] = tr
        for i in range(1, 5):
            pending = dict(event=i, phase='acquisition', sequences=['CA'],
                z=np.ones((1, 256)), canonical_z=np.ones((1, 256)), constraints=[[-.1]],
                scored=dict(scores=[-6.], fresh_evaluations=0, cache_hits=[True]))
            policy.commit(state, pending, recipe(), update_state, TurboState)
            self.assertIs(state['trust_region'], tr)
            self.assertEqual(tr.length, .0078125)
            self.assertFalse(tr.restart_triggered)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
