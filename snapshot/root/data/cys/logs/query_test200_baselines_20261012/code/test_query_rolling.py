"""Worker handoff, runtime isolation, and interrupted-claim recovery."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import query_test200_fleet_v3 as fleet
import query_test200_rolling as rolling


class RollingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('claims', 'workers', 'rolling'):
            (self.root / folder).mkdir()
        self.worker = dict(name='w0', host='LLMOptimizer2', method='lolbo', bo_cpu=8)
        self.plan = dict(workers=[self.worker], tasks=['q0', 'q1'], methods=rolling.ORDER,
            split='test', seed=0, batch_size=10, runner='old.py', files_sha256={'old.py': 'old'},
            random_tpe_timeout_seconds=100, random_tpe_latent_bound=4,
            max_unique_draws_per_slot=10000, method_runtime={
                method: dict(runner='lolbo.py' if method == 'lolbo' else 'sequential.py',
                             batch_size=10 if method == 'lolbo' else 1,
                             files_sha256={method + '.py': method}) for method in rolling.ORDER})
        fleet.write(self.root / 'plan.json', self.plan)
        fleet.write(self.root / 'rolling_manifest.json', dict(
            plan_sha256=fleet.digest(self.root / 'plan.json'), files_sha256={'dispatcher.py': 'new'}))

    def claim(self, method, task, worker='w0', status='running'):
        fleet.write(self.root / 'claims' / f'{method}.{task}.json',
                    dict(method=method, task=task, worker=worker, status=status))

    def test_overlapping_methods_use_pinned_settings_without_changing_root_plan(self):
        before = (self.root / 'plan.json').read_bytes()
        for method in rolling.ORDER:
            runtime = fleet.method_plan(self.root, method)
            worker = runtime['workers'][0]
            command = fleet.run_command(self.root, worker, 'q0', self.root / 'outputs')
            self.assertEqual(command[1], str(self.root / runtime['runner']))
            self.assertEqual(command[command.index('--batch-size') + 1],
                             '10' if method == 'lolbo' else '1')
            self.assertEqual(command[command.index('--budget') + 1], '1049')
            self.assertEqual(worker['method'], method)
            if method == 'random':
                self.assertIn('--max-unique-draws-per-slot', command)
        self.assertEqual((self.root / 'plan.json').read_bytes(), before)
        self.assertEqual(fleet.method_plan(self.root, 'lolbo')['batch_size'], 10)

    def test_changed_plan_is_rejected(self):
        changed = copy.deepcopy(self.plan)
        changed['batch_size'] = 1
        fleet.write(self.root / 'plan.json', changed)
        with self.assertRaisesRegex(RuntimeError, 'plan changed'):
            fleet.method_plan(self.root, 'tpe')

    def test_active_task_on_other_slot_does_not_block_next_method(self):
        self.claim('lolbo', 'q0', status='complete')
        self.claim('lolbo', 'q1', worker='w1')
        self.assertFalse(rolling.available(self.root, self.plan, self.worker))
        self.assertTrue(rolling.available(self.root, self.plan, dict(self.worker, method='tpe')))

    def test_interrupted_owned_claim_is_resumed_before_advancing(self):
        self.claim('lolbo', 'q0', status='complete')
        self.claim('lolbo', 'q1')
        self.assertTrue(rolling.available(self.root, self.plan, self.worker))
        self.assertEqual(fleet.claim_task(self.root, self.plan, self.worker), 'q1')

    def test_shared_queue_never_reassigns_another_workers_task(self):
        first = dict(self.worker, method='tpe')
        second = dict(first, name='w1')
        self.assertEqual(fleet.claim_task(self.root, self.plan, first), 'q0')
        self.assertEqual(fleet.claim_task(self.root, self.plan, second), 'q1')
        self.assertEqual(fleet.claim_task(self.root, self.plan, first), 'q0')
        self.assertIsNone(fleet.claim_task(self.root, self.plan, dict(first, name='w2')))

    def test_worker_lock_protects_legacy_evaluator(self):
        self.assertTrue(rolling.slot_free(self.root, 'w0'))
        with fleet.lock(self.root / 'workers/w0.lock'):
            self.assertFalse(rolling.slot_free(self.root, 'w0'))

    def dispatch_patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(rolling, 'verify'))
        stack.enter_context(patch.object(fleet, 'allocation', return_value=(self.plan, self.worker)))
        stack.enter_context(patch.object(rolling.os, 'sched_setaffinity'))
        stack.enter_context(patch.object(fleet, 'db_start'))
        return stack

    def test_dispatch_waits_for_legacy_then_smokes_and_runs_each_next_method(self):
        self.claim('lolbo', 'q0', status='complete')
        self.claim('lolbo', 'q1', worker='w1')
        stack = self.dispatch_patches()
        stack.enter_context(patch.object(rolling, 'slot_free', side_effect=[False, True]))
        sleep = stack.enter_context(patch.object(rolling.time, 'sleep'))
        stack.enter_context(patch.object(fleet, 'validate_smoke',
                                        side_effect=[RuntimeError('fresh smoke'), {}, RuntimeError('fresh smoke'), {}]))
        run = stack.enter_context(patch.object(fleet, 'run_worker'))
        rolling.dispatch(self.root, 'w0')
        sleep.assert_called_once_with(15)
        self.assertEqual([(call.kwargs.get('method'), call.kwargs.get('smoke', False))
                          for call in run.call_args_list],
                         [('tpe', True), ('tpe', False), ('random', True), ('random', False)])
        self.assertEqual(fleet.read(self.root / 'rolling/w0.json')['status'], 'complete')

    def test_failed_smoke_never_launches_production(self):
        stack = self.dispatch_patches()
        stack.enter_context(patch.object(fleet, 'validate_smoke', side_effect=RuntimeError('fresh smoke')))
        run = stack.enter_context(patch.object(fleet, 'run_worker', side_effect=RuntimeError('smoke failed')))
        with self.assertRaisesRegex(RuntimeError, 'smoke failed'):
            rolling.dispatch(self.root, 'w0')
        self.assertEqual(run.call_count, 1)
        self.assertTrue(run.call_args.kwargs['smoke'])
        self.assertEqual(fleet.read(self.root / 'rolling/w0.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
