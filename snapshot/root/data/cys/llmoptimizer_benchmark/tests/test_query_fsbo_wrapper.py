"""Query transfer wrapper: censoring, frozen coordinates, stalls and recovery."""
import copy
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import query_fsbo_eval as wrapper

PLAN_A, PLAN_B = [1, 1, 2, 1, 1], [2, 1, 1, 1, 1]


class FrozenVAE:
    def __init__(self, *args, **kwargs):
        self.task, self.fallback = 'CEB_1A1', PLAN_A
        self.encode_calls = []
        self.fail_encode = False

    def encode(self, plans):
        self.encode_calls.append(copy.deepcopy(plans))
        if self.fail_encode:
            raise RuntimeError('Encoding interrupted after SQL result was committed')
        return np.ones((len(plans), 64)).tolist()

    def decode(self, vectors, seed=None):
        return [dict(plan=PLAN_B, fallback=False, decode_retries=0) for _ in vectors]


class PosteriorClient:
    def __init__(self, *args, **kwargs):
        self.last_sample = None

    def request(self, op, **values):
        if op == 'fit':
            return dict(target_observations=len(values['score']))
        x, count = np.asarray(values['candidates']), values['count']
        return dict(draws=np.tile(np.arange(len(x)), (count, 1)), info=dict(correlated=True))


class QueryFSBOWrapperTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.recipe = dict(peptide_cpu=str(wrapper.PEPTIDE_CPU), batch_size=1, seed=123,
            lower=[0.]*64, upper=[1.]*64, n_candidates=16, timeout_seconds=100., vae_pool='unused')
        self.initial = [dict(plan=PLAN_A, runtime=2., censored=0, z=[0.]*64, status='bao_import')]

    def test_timeout_then_success_keeps_first_distinct_exact_measurement(self):
        rows = [dict(self.initial[0], censored=1, runtime=100.), self.initial[0],
                dict(self.initial[0], runtime=0.5),
                dict(self.initial[0], plan=PLAN_B, runtime=1.)]
        kept = wrapper.distinct_completed(rows)
        self.assertEqual([r['runtime'] for r in kept], [2., 1.])

    def test_fresh_task_seed_controls_proposals_and_decode_stream(self):
        import torch
        import random
        wrapper.seed_fresh_task(123)
        first = (random.random(), np.random.rand(), torch.rand(4))
        wrapper.seed_fresh_task(123)
        second = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_query_proposals_use_configured_fifty_second_limit(self):
        recipe = dict(self.recipe, timeout_seconds=50.)
        engine = wrapper.FSBOEngine(Path('recipe.json'), recipe, self.initial, FrozenVAE())
        with patch.object(wrapper, 'ModelClient', PosteriorClient), patch.object(wrapper, 'memory_guard'):
            proposals = engine.ask(1)
        self.assertEqual(proposals[0]['timeout'], 50.)

    def test_observed_plan_uses_encoder_mean_and_only_encodes_new_plans(self):
        vae = FrozenVAE()
        engine = wrapper.FSBOEngine(Path('recipe.json'), self.recipe, self.initial, vae)
        rows = [dict(plan=PLAN_B, runtime=1., censored=0, z=[77.]*64)]
        engine.tell([], rows)
        engine.tell([], rows)
        self.assertEqual(vae.encode_calls, [[PLAN_B]])
        self.assertEqual(engine.rows[-1]['z'], [1.]*64)
        self.assertEqual(rows[0]['z'], [77.]*64)
        self.assertEqual(rows[0]['canonical_z'], [1.]*64)

    def test_stalled_and_censored_batches_adjust_only_trust_region(self):
        vae = FrozenVAE()
        engine = wrapper.FSBOEngine(Path('recipe.json'), self.recipe, self.initial, vae)
        for _ in range(32):
            engine.tell([], [dict(self.initial[0], runtime=100., censored=1)])
        self.assertEqual(engine.rounds, 32)
        self.assertEqual(engine.turbo.length, 0.4)
        self.assertEqual(engine.turbo.best_value, -2.)
        self.assertEqual(vae.encode_calls, [])

    def test_checkpoint_excludes_live_model_and_vae_locks(self):
        engine = wrapper.FSBOEngine(Path('recipe.json'), self.recipe, self.initial, FrozenVAE())
        engine.client = lambda: None  # Deliberately not picklable.
        with patch.object(wrapper, 'QueryVAEBackend', FrozenVAE):
            restored = pickle.loads(pickle.dumps(engine))
        self.assertIsNone(restored.client)
        self.assertIsInstance(restored.vae, FrozenVAE)
        self.assertEqual(restored.rows, engine.rows)

    def test_resume_after_encoding_failure_never_repeats_committed_sql(self):
        vae = FrozenVAE()
        vae.fail_encode = True
        engine = wrapper.FSBOEngine(Path('recipe.json'), self.recipe, self.initial, vae)
        backend = wrapper.CanonicalBackend(vae, lambda p: p)
        calls = []

        def oracle(plan, timeout):
            calls.append(list(plan))
            return dict(runtime=1., censored=0, status='completed')

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(wrapper, 'ModelClient', PosteriorClient), \
                patch.object(wrapper, 'QueryVAEBackend', FrozenVAE), \
                patch.object(wrapper, 'memory_guard'):
            with self.assertRaisesRegex(RuntimeError, 'Encoding interrupted'):
                wrapper.query.run_loop(Path(temp), dict(task_id='CEB_1A1', method='fsbo'),
                    self.initial, engine, backend, oracle, 3)
            result = wrapper.query.run_loop(Path(temp), dict(task_id='CEB_1A1', method='fsbo'),
                self.initial, None, backend, oracle, 3)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['proposals'], 2)
            self.assertEqual(result['database_calls'], 1)
            self.assertEqual(result['cache_hits'], 1)
            self.assertEqual(calls, [PLAN_B])
            self.assertEqual(result['unresolved_database_calls'], 0)

    def test_unknown_sql_result_is_not_repeated_on_resume(self):
        proposal = dict(plan=PLAN_B, z=[0.]*64, timeout=100., trial=None)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            journal = wrapper.query.JsonJournal(directory)
            journal.update({'proposal_0000.pending.json': proposal,
                'proposal_0000.database_attempts.json': [dict(proposal=proposal, status='in_flight')]})
            with self.assertRaisesRegex(RuntimeError, 'Unresolved database call'):
                wrapper.query.evaluate_pending(directory, [proposal], self.initial, [],
                    lambda *args: self.fail('An unknown SQL execution was repeated'))


if __name__ == '__main__':
    unittest.main()
