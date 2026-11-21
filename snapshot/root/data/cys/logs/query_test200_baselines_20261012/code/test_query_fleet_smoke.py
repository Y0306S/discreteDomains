"""A stale or failed smoke invocation must never authorize production."""

import copy
from pathlib import Path
import tempfile
import time
import unittest

import query_test200_fleet as fleet


class SmokeVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.worker = dict(name='LLMOptimizer3.w06', host='LLMOptimizer3', gpu=3,
                           bo_cpu=3, db_cpus=[32, 33, 34], port=5446,
                           method='lolbo',
                           pgdata='/dev/shm/current/pg_imdb_d')
        self.plan = dict(files_sha256={'code/runner.py': 'current-source'},
                         split='test', seed=0, smoke_task='CEB_11A21',
                         vae_sha256='author-vae', bao_sha256='author-bao')
        self.request_path = self.root / 'smoke/LLMOptimizer3.w06.request.json'
        self.verified_path = self.root / 'smoke/LLMOptimizer3.w06.verified.json'

    def complete_smoke(self):
        request = fleet.begin_smoke(self.root, self.plan, self.worker)
        summary = dict(task_id='CEB_11A21', method='lolbo', split='test', seed=0,
                       status='complete', budget_total=69, charged_observations=69,
                       initial_observations=49, proposals=20, vae_sha256='author-vae',
                       bao_sha256='author-bao', unresolved_database_calls=0)
        directory = Path(request['output']) / 'CEB_11A21/lolbo/seed0'
        observations = [dict(config={f'z_{i}': 0.0 for i in range(64)},
                             objectives=[1.0], extra_info={'charged_index': index})
                        for index in range(69)]
        fleet.write(directory / 'history.json', dict(meta_info=summary, observations=observations))
        verified = dict(run_id=request['run_id'], worker=self.worker,
                        files_sha256=self.plan['files_sha256'], output=request['output'],
                        verified_epoch=time.time(), summary=summary,
                        checkpoint_reload=True, archived_proposals=True,
                        phases=[dict(stop_after_batches=1, returncode=0, status='paused', proposals=10),
                                dict(stop_after_batches=None, returncode=0, status='complete', proposals=20)])
        fleet.write(self.verified_path, verified)
        return verified

    def test_current_success_and_history_are_accepted(self):
        expected = self.complete_smoke()
        self.assertEqual(fleet.validate_smoke(self.root, self.plan, self.worker), expected)

    def test_new_smoke_uses_empty_directory_and_preserves_old_evidence(self):
        previous = self.complete_smoke()
        request = fleet.begin_smoke(self.root, self.plan, self.worker)
        self.assertNotEqual(request['output'], previous['output'])
        self.assertEqual(fleet.read(Path(request['output']) / 'previous_verification.json'), previous)
        self.assertFalse(self.verified_path.exists())
        self.assertFalse((Path(request['output']) / 'CEB_11A21').exists())
        self.assertTrue((Path(previous['output']) / 'CEB_11A21/lolbo/seed0/history.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'fresh smoke'):
            fleet.validate_smoke(self.root, self.plan, self.worker)

    def test_stale_success_after_failed_new_invocation_is_rejected(self):
        previous = self.complete_smoke()
        fleet.begin_smoke(self.root, self.plan, self.worker)
        fleet.write(self.verified_path, previous)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            fleet.validate_smoke(self.root, self.plan, self.worker)

    def test_old_worker_allocation_and_code_cannot_authorize_launch(self):
        self.complete_smoke()
        for change in ({'gpu': 2}, {'bo_cpu': 14}, {'pgdata': '/dev/shm/old/pg_imdb_d'}):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'stale'):
                fleet.validate_smoke(self.root, self.plan, {**self.worker, **change})
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            fleet.validate_smoke(self.root, {**self.plan, 'files_sha256': {}}, self.worker)

    def test_invocation_before_requested_launch_is_rejected(self):
        self.complete_smoke()
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            fleet.validate_smoke(self.root, self.plan, self.worker, not_before=time.time() + 1)

    def test_failed_or_missing_reload_cannot_authorize_launch(self):
        verified = self.complete_smoke()
        for field, value in [('phases', []), ('checkpoint_reload', False), ('archived_proposals', False)]:
            fleet.write(self.verified_path, {**verified, field: value})
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'pause/reload'):
                fleet.validate_smoke(self.root, self.plan, self.worker)
        failed = copy.deepcopy(verified)
        failed['phases'][1]['returncode'] = 1
        fleet.write(self.verified_path, failed)
        with self.assertRaisesRegex(RuntimeError, 'pause/reload'):
            fleet.validate_smoke(self.root, self.plan, self.worker)

    def test_saved_history_must_still_match(self):
        verified = self.complete_smoke()
        history = Path(verified['output']) / 'CEB_11A21/lolbo/seed0/history.json'
        data = fleet.read(history)
        data['observations'].pop()
        fleet.write(history, data)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete or mismatched'):
            fleet.validate_smoke(self.root, self.plan, self.worker)


if __name__ == '__main__':
    unittest.main()
