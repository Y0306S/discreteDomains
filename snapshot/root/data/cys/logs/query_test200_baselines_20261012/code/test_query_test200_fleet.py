"""Cross-method queue isolation and exact held-out launch settings."""

from pathlib import Path
import tempfile
import unittest

import query_test200_fleet as fleet
import query_test200_pipeline as pipeline


class Test200FleetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'claims').mkdir()
        self.plan = dict(tasks=['CEB_10A32', 'CEB_10A49'], split='test', seed=0,
                         batch_size=10, random_tpe_timeout_seconds=100,
                         random_tpe_latent_bound=4)
        fleet.write(self.root / 'plan.json', self.plan)

    def worker(self, name, method, host='LLMOptimizer2'):
        return dict(name=name, method=method, host=host)

    def test_same_task_runs_once_for_each_method(self):
        for method in ('random', 'tpe', 'lolbo'):
            worker = self.worker(method, method)
            self.assertEqual(fleet.claim_task(self.root, self.plan, worker), 'CEB_10A32')
            path = self.root / 'claims' / f'{method}.CEB_10A32.json'
            claim = fleet.read(path)
            fleet.write(path, dict(claim, status='complete'))
            self.assertEqual(fleet.claim_task(self.root, self.plan, worker), 'CEB_10A49')

    def test_workers_share_queue_without_double_claiming(self):
        first = self.worker('w0', 'tpe')
        second = self.worker('w1', 'tpe', 'LLMOptimizer3')
        self.assertEqual(fleet.claim_task(self.root, self.plan, first), 'CEB_10A32')
        self.assertEqual(fleet.claim_task(self.root, self.plan, second), 'CEB_10A49')
        self.assertEqual(fleet.claim_task(self.root, self.plan, first), 'CEB_10A32')

    def test_command_uses_test_split_and_selected_method(self):
        for method in ('random', 'tpe', 'lolbo'):
            command = fleet.run_command(self.root, self.worker('w0', method),
                                        'CEB_10A32', self.root / 'outputs')
            for key, value in [('--method', method), ('--split', 'test'),
                               ('--budget', '1049'), ('--batch-size', '10'),
                               ('--seed', '0'), ('--timeout', '100')]:
                self.assertEqual(command[command.index(key) + 1], value)

    def test_completion_from_another_method_is_rejected(self):
        plan = dict(self.plan, vae_sha256='vae', bao_sha256='bao')
        meta = dict(task_id='CEB_10A32', method='random', split='test', seed=0,
                    status='complete', budget_total=49, charged_observations=49,
                    initial_observations=49, proposals=0, vae_sha256='vae',
                    bao_sha256='bao', unresolved_database_calls=0)
        rows = [dict(config={f'z_{i}': 0.0 for i in range(64)}, objectives=[1.0],
                     extra_info={'charged_index': j}) for j in range(49)]
        fleet.write(self.root / 'history.json', dict(meta_info=meta, observations=rows))
        fleet.validate_complete(self.root, plan, 'CEB_10A32', 49, 'random')
        with self.assertRaisesRegex(RuntimeError, 'mismatched'):
            fleet.validate_complete(self.root, plan, 'CEB_10A32', 49, 'tpe')

    def test_stage_waits_for_every_task_and_worker_to_finish(self):
        (self.root / 'workers').mkdir()
        worker = self.worker('w0', 'lolbo')
        plan = dict(self.plan, workers=[worker])
        for task in plan['tasks']:
            fleet.write(self.root / 'claims' / f'lolbo.{task}.json',
                        dict(task=task, method='lolbo', status='complete'))
        path = self.root / 'workers/w0.json'
        fleet.write(path, dict(method='lolbo', status='running'))
        self.assertFalse(pipeline.finished(self.root, plan, 'lolbo'))
        fleet.write(path, dict(method='lolbo', status='complete'))
        self.assertTrue(pipeline.finished(self.root, plan, 'lolbo'))
        with fleet.lock(self.root / 'workers/w0.lock'):
            self.assertFalse(pipeline.finished(self.root, plan, 'lolbo'))
        self.assertFalse(pipeline.finished(self.root, plan, 'tpe'))


if __name__ == '__main__':
    unittest.main()
