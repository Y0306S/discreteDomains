"""Budget, provenance and crash recovery for the CEB runtime campaign."""

from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from examples.benchmark.domains import run_query_ceb_baselines as runner


def initial_rows():
    return [dict(plan=[i], runtime=float(i + 1), censored=0, status='bao_import',
                 z=[i / 49, -i / 49]) for i in range(49)]


class FakeBackend:
    def decode(self, vectors):
        return [dict(plan=[1000 + round(abs(z[0]) * 10000), round(z[1] * 10000)],
                     decode_retries=0, fallback=False) for z in vectors]


class CountingOracle:
    def __init__(self):
        self.calls = []

    def __call__(self, plan, timeout):
        self.calls.append(list(plan))
        return dict(runtime=1.0 + sum(abs(x) for x in plan) % 113,
                    censored=int(plan[0] % 3 == 0), status='completed')


class SequentialEngine:
    batch_size = 10

    def __init__(self):
        self.next = 100
        self.batch_sizes = []

    def ask(self, remaining):
        count = min(self.batch_size, remaining)
        result = [dict(z=[float(i)], timeout=100.0, trial=None)
                  for i in range(self.next, self.next + count)]
        self.next += count
        return result

    def tell(self, proposals, rows):
        self.batch_sizes.append(len(rows))


class SequentialBackend:
    def decode(self, vectors):
        return [dict(plan=[int(z[0])], fallback=False, decode_retries=0) for z in vectors]


class CEBBaselineTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.meta = dict(task_id='CEB_10A32', method='random')

    def tearDown(self):
        self.tmp.cleanup()

    def test_final_batch_is_capped_and_initialization_is_charged(self):
        engine = SequentialEngine()
        oracle = CountingOracle()
        summary = runner.run_loop(self.root, self.meta, initial_rows(), engine,
                                  SequentialBackend(), oracle, total=74)
        self.assertEqual(engine.batch_sizes, [10, 10, 5])
        self.assertEqual(summary['charged_observations'], 74)
        self.assertEqual(summary['proposals'], 25)
        self.assertEqual(len(oracle.calls), 25)
        self.assertEqual(summary['status'], 'complete')

    def test_production_budget_has_no_extra_batch(self):
        engine = SequentialEngine()
        oracle = CountingOracle()
        summary = runner.run_loop(self.root, self.meta, initial_rows(), engine,
                                  SequentialBackend(), oracle, total=1049)
        self.assertEqual(len(engine.batch_sizes), 100)
        self.assertEqual(summary['charged_observations'], 1049)
        self.assertEqual(summary['proposals'], 1000)

    def test_cache_hits_are_charged_and_never_query_database(self):
        proposals = [dict(plan=[0], z=[0.0], timeout=100.0, trial=None),
                     dict(plan=[100], z=[1.0], timeout=100.0, trial=None),
                     dict(plan=[100], z=[2.0], timeout=100.0, trial=None)]
        oracle = CountingOracle()
        rows = runner.evaluate_pending(self.root, proposals, initial_rows(), [], oracle)
        self.assertEqual(len(rows), 3)
        self.assertEqual(oracle.calls, [[100]])
        self.assertEqual([row['status'] for row in rows], ['cache_hit', 'completed', 'cache_hit'])
        replay = runner.evaluate_pending(self.root, proposals, initial_rows(), [], oracle)
        self.assertEqual(replay, rows)
        self.assertEqual(oracle.calls, [[100]])

    def test_ambiguous_inflight_call_is_not_silently_repeated(self):
        proposal = dict(plan=[100], z=[1.0], timeout=100.0, trial=None)
        runner.atomic_json(self.root / 'proposal_0000.pending.json', proposal)
        oracle = CountingOracle()
        with self.assertRaisesRegex(RuntimeError, 'Unresolved database call'):
            runner.evaluate_pending(self.root, [proposal], initial_rows(), [], oracle)
        self.assertEqual(oracle.calls, [])

    def test_saved_partial_batch_is_reused_before_further_calls(self):
        proposals = [dict(plan=[100 + i], z=[float(i)], timeout=100.0, trial=None)
                     for i in range(3)]
        oracle = CountingOracle()
        runner.evaluate_pending(self.root, proposals[:1], initial_rows(), [], oracle)
        runner.evaluate_pending(self.root, proposals, initial_rows(), [], oracle)
        self.assertEqual(oracle.calls, [[100], [101], [102]])

    def test_confirmed_io_failure_resumes_same_proposal_without_fake_score(self):
        import json
        import psycopg
        proposal = dict(plan=[100], z=[1.0], timeout=100.0, trial=None)

        def broken(plan, timeout):
            raise psycopg.errors.IoError('could not read block: Input/output error')

        with self.assertRaises(psycopg.errors.IoError):
            runner.evaluate_pending(self.root, [proposal], initial_rows(), [], broken)
        self.assertFalse((self.root / 'proposal_0000.json').exists())
        attempts = json.loads((self.root / 'proposal_0000.database_attempts.json').read_text())
        self.assertEqual(attempts[0]['status'], 'io_error')
        oracle = CountingOracle()
        rows = runner.evaluate_pending(self.root, [proposal], initial_rows(), [], oracle)
        self.assertEqual(oracle.calls, [[100]])
        self.assertEqual(rows[0]['database_attempts'], 2)
        self.assertEqual(rows[0]['proposal'], proposal)
        summary = runner.write_history(self.root, self.meta, initial_rows(), rows, 50, 'complete')
        self.assertEqual(summary['proposals'], 1)
        self.assertEqual(summary['database_calls'], 2)
        runner.evaluate_pending(self.root, [proposal], initial_rows(), [], oracle)
        self.assertEqual(oracle.calls, [[100]])

    def test_unknown_error_still_blocks_replay(self):
        proposal = dict(plan=[100], z=[1.0], timeout=100.0, trial=None)

        def broken(plan, timeout):
            raise RuntimeError('connection outcome unknown')

        with self.assertRaisesRegex(RuntimeError, 'connection outcome unknown'):
            runner.evaluate_pending(self.root, [proposal], initial_rows(), [], broken)
        oracle = CountingOracle()
        with self.assertRaisesRegex(RuntimeError, 'Unresolved database call'):
            runner.evaluate_pending(self.root, [proposal], initial_rows(), [], oracle)
        self.assertEqual(oracle.calls, [])

    def test_resume_rejects_changed_budget(self):
        runner.run_loop(self.root, self.meta, initial_rows(), SequentialEngine(),
                        SequentialBackend(), CountingOracle(), total=49)
        with self.assertRaisesRegex(ValueError, 'Resume settings'):
            runner.run_loop(self.root, self.meta, initial_rows(), None,
                            SequentialBackend(), CountingOracle(), total=50)

    def test_runner_upgrade_requires_explicit_old_hash_and_records_migration(self):
        old = dict(self.meta, runner_sha256='old')
        new = dict(self.meta, runner_sha256='new')
        oracle = CountingOracle()
        runner.run_loop(self.root, old, initial_rows(), SequentialEngine(),
                        SequentialBackend(), oracle, total=69, stop_after_batches=1)
        with self.assertRaisesRegex(ValueError, 'Resume settings'):
            runner.run_loop(self.root, new, initial_rows(), None,
                            SequentialBackend(), oracle, total=69)
        summary = runner.run_loop(self.root, new, initial_rows(), None,
                                  SequentialBackend(), oracle, total=69,
                                  resume_runner_sha256='old')
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(len(oracle.calls), 20)
        self.assertEqual(summary['runner_migrations'][0]['from_sha256'], 'old')
        self.assertEqual(summary['runner_migrations'][0]['to_sha256'], 'new')
        again = runner.run_loop(self.root, new, initial_rows(), None,
                                SequentialBackend(), oracle, total=69)
        self.assertEqual(len(again['runner_migrations']), 1)
        self.assertEqual(len(oracle.calls), 20)

    def test_runner_upgrade_does_not_allow_other_provenance_changes(self):
        old = dict(self.meta, runner_sha256='old')
        runner.run_loop(self.root, old, initial_rows(), SequentialEngine(),
                        SequentialBackend(), CountingOracle(), total=49)
        for new, total, approved_hash in [
                (dict(old, runner_sha256='new', task_id='CEB_10A33'), 49, 'old'),
                (dict(old, runner_sha256='new'), 50, 'old'),
                (dict(old, runner_sha256='new'), 49, 'wrong')]:
            with self.subTest(new=new, total=total, approved_hash=approved_hash):
                with self.assertRaisesRegex(ValueError, 'Resume settings'):
                    runner.run_loop(self.root, new, initial_rows(), None,
                                    SequentialBackend(), CountingOracle(), total=total,
                                    resume_runner_sha256=approved_hash)

    def test_completed_run_is_idempotent(self):
        oracle = CountingOracle()
        runner.run_loop(self.root, self.meta, initial_rows(), SequentialEngine(),
                        SequentialBackend(), oracle, total=59)
        runner.run_loop(self.root, self.meta, initial_rows(), None,
                        SequentialBackend(), oracle, total=59)
        self.assertEqual(len(oracle.calls), 10)

    def test_batch_boundary_pause_restores_without_repeating_results(self):
        oracle = CountingOracle()
        first = runner.run_loop(self.root, self.meta, initial_rows(), SequentialEngine(),
                                SequentialBackend(), oracle, total=69, stop_after_batches=1)
        self.assertEqual(first['status'], 'paused')
        self.assertEqual(first['proposals'], 10)
        second = runner.run_loop(self.root, self.meta, initial_rows(), None,
                                 SequentialBackend(), oracle, total=69)
        self.assertEqual(second['status'], 'complete')
        self.assertEqual(len(oracle.calls), 20)
        self.assertEqual(len({tuple(plan) for plan in oracle.calls}), 20)

    def test_real_optuna_random_and_tpe_restore_pending_batch(self):
        for method in ('random', 'tpe'):
            with self.subTest(method=method):
                directory = self.root / method
                directory.mkdir()
                initial = initial_rows()
                engine = runner.Engine(method, initial, 0, 10, 100.0, 4.0)
                pending = engine.ask(5)
                decoded = FakeBackend().decode([p['z'] for p in pending])
                pending = [{**p, **plan} for p, plan in zip(pending, decoded)]
                meta = {**self.meta, 'method': method}
                runner.checkpoint(directory / 'checkpoint.pt', dict(
                    meta=meta, total=54, engine=engine, initial=initial,
                    completed=[], pending=pending))
                oracle = CountingOracle()
                runner.evaluate_pending(directory, pending[:2], initial, [], oracle)
                summary = runner.run_loop(directory, meta, initial, None,
                                          FakeBackend(), oracle, 54)
                self.assertEqual(summary['proposals'], 5)
                self.assertEqual(len(oracle.calls), 5)
                state = torch.load(directory / 'checkpoint.pt', weights_only=False)
                self.assertTrue(all(t.state.name == 'COMPLETE' for t in state['engine'].study.trials))
                self.assertEqual(len(state['engine'].study.trials), 54)

    def test_best_runtime_excludes_timeouts(self):
        rows = initial_rows()
        rows[0].update(runtime=0.001, censored=1)
        summary = runner.write_history(self.root, self.meta, rows, [], 49, 'complete')
        self.assertEqual(summary['best_uncensored_runtime'], 2.0)

    def test_exact_sql_path_never_falls_back(self):
        expected = self.root / 'workload/ceb-3k/10a/10a32.sql'
        expected.parent.mkdir(parents=True)
        expected.write_text('SELECT 1')
        self.assertEqual(runner.sql_path(self.root, 'CEB_10A32'), expected)
        with self.assertRaises(FileNotFoundError):
            runner.sql_path(self.root, 'CEB_10A33')

    def test_split_rejects_overlap(self):
        train = self.root / 'train.txt'
        test = self.root / 'test.txt'
        train.write_text('\n'.join(f'CEB_1A{i}' for i in range(800)))
        test.write_text('\n'.join(f'CEB_1A{i}' for i in range(700, 900)))
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            runner.load_split(train, test)


if __name__ == '__main__':
    unittest.main()
