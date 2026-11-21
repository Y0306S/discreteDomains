import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ml_deferred_campaign as queue
from ml_fixed_campaign import claim, command, digest, load, save


class DeferredCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = {'task_type': 'cls', 'dataset': 'fixture', 'algorithm': 'random_forest', 'metric': 'bal_acc'}
        save(self.root / 'tasks_132.json', [self.task])
        self.protocol = {'manifest_sha256': digest(self.root / 'tasks_132.json'),
                         'task_count': 1, 'methods': ['bolt_ml'], 'seed': 1,
                         'hosts': {'fixture': {'hostname': 'fixture'}}}
        save(self.root / 'protocol.json', self.protocol)
        save(self.root / 'runtime_policy.json', {'attempts_per_search': 100})
        self.trajectory = self.root / 'outputs' / 'trajectory.csv'
        self.progress = self.root / 'sessions/bolt_ml/cls/fixture/random_forest/progress.json'

    def complete(self):
        save(self.trajectory.parent / 'config.json', {
            'optimizer': 'bolt_ml', 'seed': 1, 'metric': 'bal_acc',
            'include_algorithms': ['random_forest'], 'task_id': 'fixture_random_forest_bolt_ml'})
        with self.trajectory.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['iter_idx', 'oracle_called', 'evaluation_status'])
            writer.writeheader()
            for index in range(100):
                writer.writerow({'iter_idx': index, 'oracle_called': True,
                                 'evaluation_status': 'SUCCESS' if index < 98 else ('FAILED' if index == 98 else 'TIMEOUT')})
        save(self.progress, {'attempts': 100, 'budget_exhausted': True, 'pending': False,
             'identity': 'verified-task', 'successful': 98, 'failed': 2, 'trajectory': str(self.trajectory)})

    def test_missing_partial_and_pending_tasks_block(self):
        self.assertTrue(queue.predecessor_readiness(self.root, 'fixture'))
        self.complete()
        original = load(self.progress)
        for changes in ({'attempts': 99}, {'pending': True}, {'identity': None}, {'successful': 97}):
            save(self.progress, {**original, **changes})
            self.assertTrue(queue.predecessor_readiness(self.root, 'fixture'))

    def test_failures_and_timeouts_count_as_actual_evaluations(self):
        self.complete()
        self.assertEqual(queue.predecessor_readiness(self.root, 'fixture'), [])

    def test_wrong_method_or_task_cannot_satisfy_dependency(self):
        self.complete()
        path = self.trajectory.parent / 'config.json'
        original = load(path)
        for changes in ({'optimizer': 'semanticopt'}, {'seed': 2}, {'task_id': 'other_random_forest_bolt_ml'}):
            save(path, {**original, **changes})
            self.assertIn('identity mismatch', str(queue.predecessor_readiness(self.root, 'fixture')))

    def test_cached_or_incorrect_ledger_does_not_count(self):
        self.complete()
        original = self.trajectory.read_text()
        for changed in (original.replace('True', 'False', 1),
                        original.replace('99,', '98,'), original.replace('FAILED', 'SUCCESS')):
            self.trajectory.write_text(changed)
            self.assertIn('ledger is invalid', str(queue.predecessor_readiness(self.root, 'fixture')))

    def test_workers_and_scheduler_must_exit(self):
        self.complete()
        save(self.root / 'state/worker.json', {'host': 'fixture'})
        with patch.object(queue, 'process_alive', return_value=True):
            self.assertIn('still active', str(queue.predecessor_readiness(self.root, 'fixture')))
        lock = claim(self.root / 'locks/fixture.lock')
        try:
            self.assertIn('scheduler still owns', str(queue.predecessor_readiness(self.root, 'fixture')))
        finally:
            lock.close()

    def test_restarting_watcher_does_not_repeat_launch(self):
        save(self.root / 'launch.json', {'status': 'launched', 'scheduler_pid': 123})
        with patch.object(queue.socket, 'gethostname', return_value='fixture'), \
                patch.object(queue, 'launch') as launch, patch.object(queue, 'predecessor_readiness') as readiness:
            queue.run(self.root, 'fixture')
        launch.assert_not_called()
        readiness.assert_not_called()

    def test_ordered_gpu_pair_reaches_worker_and_guard(self):
        protocol = dict(self.protocol, baseline_python='/python', method_args={'tosfit': [
            '--tosfit-device', 'cuda:0', '--tosfit-reference-device', 'cuda:1']})
        save(self.root / 'protocol.json', protocol)
        args, env, session = command(self.root, 'tosfit', self.task, [4, 5, 6], '1,0')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1,0')
        self.assertEqual(env['EVAL_N_JOBS'], '3')
        self.assertEqual(args[args.index('--tosfit-reference-device') + 1], 'cuda:1')
        self.assertEqual(queue.gpu_devices(['0,1', '1,0', '2,3', '3,2']), {0, 1, 2, 3})

    def test_changed_external_asset_blocks_prepared_campaign(self):
        asset = self.root / 'upstream.py'
        asset.write_text('original')
        save(self.root / 'protocol.json', dict(self.protocol, external_checksums={str(asset): digest(asset)}))
        save(self.root / 'data_manifest.json', {})
        save(self.root / 'checksums.json', {'protocol.json': digest(self.root / 'protocol.json')})
        save(self.root / 'prepared.json', {'checksums_sha256': digest(self.root / 'checksums.json')})
        queue.verify_campaign(self.root)
        asset.write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'external asset changed'):
            queue.verify_campaign(self.root)


if __name__ == '__main__':
    unittest.main()
