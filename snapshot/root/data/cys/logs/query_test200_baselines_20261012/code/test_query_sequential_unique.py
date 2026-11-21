"""Sequential feedback, recovered-plan uniqueness, and durable rejection recovery."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_query_ceb_baselines_v2 as runner
import query_test200_fleet_v2 as fleet
import query_test200_pipeline_v2 as pipeline
from run_artifact_store import JsonJournal


def initial_rows():
    return [dict(plan=[i], runtime=float(i + 1), censored=0, status='bao_import',
                 z=[i / 49, -i / 49]) for i in range(49)]


class ScriptedRandom:
    method = 'random'

    def __init__(self, values):
        self.values = list(values)
        self.position = 0
        self.discarded = []
        self.results = []

    def ask(self, remaining):
        value = self.values[self.position]
        self.position += 1
        return [dict(z=[float(value)], timeout=100.0, trial=self.position)]

    def discard_duplicate(self, proposal):
        self.discarded.append(proposal['trial'])

    def tell(self, proposals, rows):
        self.results.extend(row['runtime'] for row in rows)


class AliasBackend:
    def decode(self, vectors):
        return [dict(plan=[int(z[0])], decode_retries=0, fallback=False) for z in vectors]

    def plan_key(self, plan):
        return str(plan[0] % 1000)


class Oracle:
    def __init__(self):
        self.calls = []

    def __call__(self, plan, timeout):
        self.calls.append(list(plan))
        return dict(runtime=2.0, censored=0, status='completed')


class SequentialTPE(runner.Engine):
    def ask(self, remaining):
        assert not any(t.state.name == 'RUNNING' for t in self.study.trials)
        proposals = super().ask(remaining)
        assert len(proposals) == 1
        return proposals


class SequentialUniqueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.meta = dict(task_id='CEB_10A32', method='random',
                         unique_plan_policy='exclude_bao_and_previous_recovered_queries',
                         max_unique_draws_per_slot=10)

    def test_random_excludes_bao_and_alias_duplicates_without_charging(self):
        oracle = Oracle()
        engine = ScriptedRandom([1000, 101, 1101, 102])
        summary = runner.run_loop(self.root, self.meta, initial_rows(), engine,
                                  AliasBackend(), oracle, 51)
        self.assertEqual(oracle.calls, [[101], [102]])
        self.assertEqual(summary['proposals'], 2)
        self.assertEqual(summary['cache_hits'], 0)
        self.assertEqual(summary['unique_additional_plans'], 2)
        self.assertEqual(summary['random_sampling']['rejected_duplicates'], 2)
        self.assertEqual(engine.discarded, [1, 3])
        self.assertEqual(len(JsonJournal(self.root).names('random_draw_*.json')), 2)

    def test_resume_preserves_seen_plans_and_sampling_progress(self):
        oracle = Oracle()
        engine = ScriptedRandom([1000, 101, 1101, 102])
        first = runner.run_loop(self.root, self.meta, initial_rows(), engine,
                                AliasBackend(), oracle, 51, stop_after_batches=1)
        self.assertEqual(first['status'], 'paused')
        summary = runner.run_loop(self.root, self.meta, initial_rows(), None,
                                  AliasBackend(), oracle, 51)
        self.assertEqual(oracle.calls, [[101], [102]])
        self.assertEqual(summary['random_sampling']['draws'], 4)

    def test_rejection_evidence_is_recovered_from_checkpoint(self):
        oracle = Oracle()
        runner.run_loop(self.root, self.meta, initial_rows(), ScriptedRandom([1000, 101, 102]),
                        AliasBackend(), oracle, 51, stop_after_batches=1)
        journal = JsonJournal(self.root)
        journal.delete('random_draw_00000001.json')
        runner.run_loop(self.root, self.meta, initial_rows(), None, AliasBackend(), oracle, 51)
        self.assertTrue(journal.exists('random_draw_00000001.json'))
        self.assertEqual(oracle.calls, [[101], [102]])

    def test_exhaustion_never_claims_completion_or_evaluates_a_duplicate(self):
        oracle = Oracle()
        meta = dict(self.meta, max_unique_draws_per_slot=3)
        with self.assertRaisesRegex(RuntimeError, 'sampling exhausted'):
            runner.run_loop(self.root, meta, initial_rows(), ScriptedRandom([1000] * 3),
                            AliasBackend(), oracle, 50)
        summary = json.loads((self.root / 'status.json').read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['charged_observations'], 49)
        self.assertEqual(oracle.calls, [])
        self.assertTrue((self.root / 'checkpoint.pt').exists())

    def test_tpe_receives_each_result_before_asking_again(self):
        engine = SequentialTPE('tpe', initial_rows(), 0, 1, 100.0, 4.0)
        summary = runner.run_loop(self.root, dict(task_id='CEB_10A32', method='tpe'),
                                  initial_rows(), engine, AliasBackend(), Oracle(), 52)
        self.assertEqual(summary['proposals'], 3)
        self.assertEqual(len(engine.study.trials), 52)
        self.assertTrue(all(t.state.name == 'COMPLETE' for t in engine.study.trials))
        with self.assertRaisesRegex(ValueError, 'batch size 1'):
            runner.Engine('tpe', initial_rows(), 0, 10, 100.0, 4.0)

    def test_real_random_discard_does_not_complete_an_unevaluated_trial(self):
        engine = runner.Engine('random', initial_rows(), 0, 1, 100.0, 4.0)
        proposal = engine.ask(1)[0]
        engine.discard_duplicate(proposal)
        self.assertEqual(engine.study.trials[-1].state.name, 'FAIL')
        self.assertIsNone(engine.study.trials[-1].value)

    def test_native_recovery_produces_same_key_for_equivalent_invalid_tokens(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native'))
        backend = runner.FrozenBackend.__new__(runner.FrozenBackend)
        backend.spec = SimpleNamespace(all_tables=['a', 'b'], query_tables=[('a', 1), ('b', 1)],
            query_template='SELECT count(*) FROM a a1, b b1 WHERE a1.id=b1.id',
            pg_hint_plan_join_order=True)
        valid = [0, 1, 1, 1, 2]
        malformed = [2, 99, 99, 99, 4]
        self.assertEqual(backend.plan_key(valid), backend.plan_key(malformed))
        self.assertEqual(malformed, [2, 99, 99, 99, 4])

    def test_crossjoin_resampling_and_bao_fallback_are_preserved(self):
        backend = runner.FrozenBackend.__new__(runner.FrozenBackend)
        backend.torch, backend.device = torch, 'cpu'
        backend.fallback = [99]
        backend.has_crossjoin = lambda plan: plan == [1]
        with self.subTest(recovery='resample'):
            samples = iter([[[1]], [[42]]])
            backend.vae = SimpleNamespace(sample=lambda **kwargs: next(samples))
            result = backend.decode([[0.0] * 64])[0]
            self.assertEqual(result, dict(plan=[42], decode_retries=1, fallback=False))
        with self.subTest(recovery='historical fallback'):
            backend.vae = SimpleNamespace(sample=lambda **kwargs: [[1]])
            result = backend.decode([[0.0] * 64])[0]
            self.assertEqual(result, dict(plan=[99], decode_retries=51, fallback=True))

    def test_launch_uses_method_batch_size_and_revised_runner(self):
        plan = dict(split='test', batch_size=1, seed=0, random_tpe_timeout_seconds=100,
                    random_tpe_latent_bound=4, max_unique_draws_per_slot=10000,
                    runner='code/run_query_ceb_baselines_v2.py')
        fleet.write(self.root / 'plan.json', plan)
        for method in ('tpe', 'random'):
            command = fleet.run_command(self.root, {'method': method}, 'CEB_10A32', self.root)
            self.assertEqual(command[1], str(self.root / plan['runner']))
            self.assertEqual(command[command.index('--batch-size') + 1], '1')

    def test_pipeline_adopts_active_lolbo_without_relaunching_it(self):
        for folder in ('claims', 'workers'):
            (self.root / folder).mkdir()
        runtime = dict(runner='code/original.py', batch_size=10, files_sha256={})
        plan = dict(methods=pipeline.ORDER, hosts=fleet.HOSTS, tasks=['CEB_10A32'],
                    workers=[dict(name='w0', host='LLMOptimizer2', method='lolbo')],
                    budget=1049, method_runtime={m:copy.deepcopy(runtime) for m in pipeline.ORDER})
        fleet.write(self.root / 'plan.json', plan)
        fleet.write(self.root / 'workers/w0.json', dict(mode='production', method='lolbo', status='complete'))
        class NextStage(Exception):
            pass
        with patch.object(pipeline.os, 'sched_setaffinity'), \
             patch.object(pipeline.socket, 'gethostname', return_value=fleet.HOSTS['LLMOptimizer2']), \
             patch.object(pipeline, 'finished', return_value=True), \
             patch.object(pipeline.fleet, 'validate_complete', return_value={'proposals':1000, 'database_calls':100}), \
             patch.object(pipeline, 'ensure_smoke', side_effect=NextStage) as smoke, \
             patch.object(pipeline, 'launch_host') as launch:
            with self.assertRaises(NextStage):
                pipeline.run(self.root)
            self.assertEqual(smoke.call_args.args[-1], 'tpe')
            launch.assert_not_called()
        self.assertTrue((self.root / 'stage.lolbo.complete.json').exists())


if __name__ == '__main__':
    unittest.main()
