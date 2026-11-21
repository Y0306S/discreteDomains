import csv
import io
import json
import os
import pickle
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmark'))
import inventory_ml_fixed_tasks as inventory
import ml_deferred_campaign as queue
import ml_fixed_campaign as campaign
import ml_run_archive as archive
from audit_ml_evaluation_records import audit


class RunArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'campaign'
        self.task = dict(task_type='cls', dataset='fixture', algorithm='adaboost', metric='bal_acc')
        self.method = 'bolt_ml'
        key = archive.task_key(self.task)
        self.session = self.root / 'sessions' / self.method / key
        self.output = self.root / 'outputs' / self.method / key / 'bolt_ml/CASH-fixture'
        self.trajectory = self.output / 'example_traj.csv'
        self.state = self.root / 'state' / self.method / (key + '.json')
        self.history = self.root / 'history/bolt_ml/cls/bal_acc/fixture/bolt_ml/adaboost.json'
        cpus = sorted(os.sched_getaffinity(0))[:1]
        campaign.save(self.root / 'tasks_132.json', [self.task])
        campaign.save(self.root / 'runtime_policy.json', dict(attempts_per_search=3, policy_id='fixture'))
        campaign.save(self.root / 'protocol.json', dict(methods=[self.method], seed=1, task_count=1,
            ckpt32_policy='rerun-all', archive_completed_runs=True,
            manifest_sha256=campaign.digest(self.root / 'tasks_132.json'),
            policy_sha256=campaign.digest(self.root / 'runtime_policy.json'),
            hosts={'fixture': dict(hostname=socket.gethostname(), cpus=cpus, gpu_slots=[],
                                  limits={self.method: 1}, memory_gib=10000)}))
        campaign.save(self.root / 'queues/bolt_ml.json', [self.task])
        campaign.save(self.root / 'reused/bolt_ml.json', [])
        campaign.save(self.output / 'config.json', dict(optimizer='bolt_ml', seed=1, metric='bal_acc',
            include_algorithms=['adaboost'], task_id='fixture_adaboost_bolt_ml'))
        with self.trajectory.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['iter_idx', 'algorithm', 'perf', 'oracle_called',
                                                       'evaluation_status', 'eval_time', 'config_json'])
            writer.writeheader()
            for i in range(3):
                writer.writerow(dict(iter_idx=i, algorithm='adaboost', perf=.5 if i < 2 else '-inf',
                                     oracle_called=True, evaluation_status='SUCCESS' if i < 2 else 'TIMEOUT',
                                     eval_time=i + 1, config_json=json.dumps({'algorithm': 'adaboost', 'depth': i + 1})))
        self.progress = dict(identity='fixture-run', attempts=3, successful=2, failed=1,
                             pending=False, budget_exhausted=True, clean_complete=False,
                             status='complete_with_failures', trajectory=str(self.trajectory))
        campaign.save(self.session / 'progress.json', self.progress)
        self.session.joinpath('checkpoint.pkl').write_bytes(b'opaque checkpoint\0preserved')
        for i in range(3):
            campaign.save(self.session / f'evaluations/{i}/execution.json', dict(status='completed', elapsed_seconds=i))
            result = dict(result={'objectives': [-.5]} if i < 2 else None, timeout=i == 2,
                          traceback=None, elapsed_time=i + 1)
            (self.session / f'evaluations/{i}/result.pkl').write_bytes(pickle.dumps(result))
        campaign.save(self.history, dict(observations=[dict(config={'x': i}, objectives=[i]) for i in range(3)]))
        campaign.save(self.state, dict(self.task, method=self.method, status='complete_with_failures',
            host='fixture', pid=-1, boot_id='not-this-boot', start_ticks='0', session=str(self.session),
            progress=self.progress))
        log = self.root / 'logs' / self.method / (key + '.log')
        log.parent.mkdir(parents=True)
        log.write_text('all original logs\n')
        self.original = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def pack(self, cleanup=True):
        return archive.pack_run(self.root, self.method, self.task, cleanup=cleanup)

    def test_round_trip_logical_readers_status_inventory_and_queue(self):
        before = queue.predecessor_readiness(self.root, 'fixture')
        self.assertEqual(before, [])
        errors = []
        policy = dict(policy_id='fixture', per_evaluation_seconds=2400)
        original_inventory = inventory.scan(self.root / 'outputs/bolt_ml', 'bolt_ml', 'fixture',
                                           {archive.task_key(self.task): self.task}, policy, errors)
        self.assertEqual(errors, [])
        result = self.pack()
        self.assertGreater(result['removed_entries'], 10)
        self.assertFalse(self.session.exists())
        self.assertFalse(self.trajectory.exists())
        self.assertFalse(self.state.exists())
        self.assertEqual(archive.load_json(self.session / 'progress.json'), self.progress)
        self.assertEqual(archive.read_bytes(self.history), self.original[self.history.relative_to(self.root)])
        self.assertEqual(queue.predecessor_readiness(self.root, 'fixture'), before)
        self.assertEqual(campaign.status(self.root, quiet=True)['bolt_ml']['complete_with_failures'], 1)
        self.assertEqual(inventory.scan(self.root / 'outputs/bolt_ml', 'bolt_ml', 'fixture',
                         {archive.task_key(self.task): self.task}, policy, errors), original_inventory)
        destination = self.root.parent / 'restored'
        data = archive.restore(result['archive'], destination)
        for name, content in self.original.items():
            if name.parts[0] in ('sessions', 'outputs', 'history', 'state', 'logs'):
                self.assertEqual((destination / name).read_bytes(), content)
        self.assertEqual(data['run_id'], 'fixture-run')
        self.assertEqual(self.pack()['removed_entries'], 0)
        self.assertTrue((self.root / 'protocol.json').exists())
        self.assertTrue((self.root / 'locks/bolt_ml/cls/fixture/adaboost.lock').exists())

    def test_completed_archive_never_launches_a_worker(self):
        self.pack()
        with patch.object(campaign.subprocess, 'Popen') as launch, \
                patch.object(campaign, 'start_archiver') as pack:
            campaign.run(self.root, 'fixture')
        launch.assert_not_called()
        pack.assert_called_once_with(self.root, 'fixture')

    def test_raw_evaluation_export_is_identical_after_cleanup(self):
        before, after = self.root.parent / 'audit-before', self.root.parent / 'audit-after'
        self.assertTrue(audit(self.root, before))
        self.pack()
        self.assertTrue(audit(self.root, after))
        self.assertEqual((before / 'current_snapshot/bolt_ml.jsonl').read_bytes(),
                         (after / 'current_snapshot/bolt_ml.jsonl').read_bytes())

    def test_completion_enqueues_archive_after_releasing_task_claim(self):
        self.state.unlink()
        campaign.save(self.session / 'progress.json', dict(self.progress, attempts=0, successful=0,
                      failed=0, budget_exhausted=False, status='pending'))
        (self.root / 'code').mkdir()
        requests = []

        def command(root, method, task, cpus, gpu):
            program = ("from pathlib import Path; "
                       f"p=Path({str(self.session / 'progress.json')!r}); "
                       "q=p.with_name('next.json'); "
                       f"q.write_text({json.dumps(self.progress)!r}); q.replace(p)")
            return [sys.executable, '-c', program, '--session-dir', str(self.session)], os.environ.copy(), self.session

        def start(root, host):
            requests.append(host)
            if archive.load_json(self.session / 'progress.json')['budget_exhausted']:
                self.assertEqual(self.pack()['status'], 'compacted')
            return SimpleNamespace(poll=lambda: 0, returncode=0)

        original_sleep = time.sleep
        with patch.object(campaign, 'command', command), patch.object(campaign, 'start_archiver', start), \
                patch.object(campaign.time, 'sleep', lambda seconds: original_sleep(.02)):
            campaign.run(self.root, 'fixture')
        self.assertEqual(len(requests), 2)
        self.assertFalse(self.session.exists())
        self.assertEqual(archive.load_json(self.session / 'progress.json')['attempts'], 3)

    def test_pending_and_incomplete_budgets_are_rejected(self):
        for change in ({'pending': True}, {'attempts': 2}, {'identity': ''}):
            campaign.save(self.session / 'progress.json', dict(self.progress, **change))
            with self.assertRaises(archive.ArchiveError):
                self.pack()
            self.assertTrue(self.trajectory.exists())

    def test_lock_and_live_worker_prevent_archiving(self):
        lock = campaign.claim(self.root / 'locks/bolt_ml/cls/fixture/adaboost.lock')
        try:
            self.assertEqual(self.pack()['status'], 'locked')
        finally:
            lock.close()
        with patch.object(archive, '_alive', return_value=True):
            with self.assertRaisesRegex(archive.ArchiveError, 'Worker still owns'):
                self.pack()

    def test_new_source_content_aborts_cleanup_before_unlinking(self):
        result = self.pack(cleanup=False)
        extra = self.session / 'unexpected.txt'
        extra.write_text('new evidence')
        with self.assertRaisesRegex(archive.ArchiveError, 'Source changed'):
            self.pack()
        self.assertTrue(self.trajectory.exists())
        self.assertTrue(extra.exists())
        self.assertTrue(Path(result['archive']).exists())

    def test_corrupt_archive_does_not_look_pending_or_delete_sources(self):
        result = self.pack(cleanup=False)
        Path(result['archive']).write_bytes(b'corrupt archive')
        with self.assertRaises(archive.ArchiveError):
            archive.load_json(self.session / 'progress.json', {})
        with self.assertRaises(archive.ArchiveError):
            self.pack()
        self.assertTrue(self.trajectory.exists())

    def test_interrupted_cleanup_restarts_from_published_archive(self):
        def interrupted(root, data):
            (self.session / 'checkpoint.pkl').unlink()
            self.state.unlink()
            raise OSError('simulated interrupted cleanup')
        with patch.object(archive, '_cleanup', interrupted):
            with self.assertRaisesRegex(OSError, 'simulated'):
                self.pack()
        self.assertEqual(archive.load_json(self.session / 'progress.json'), self.progress)
        result = self.pack()
        self.assertGreater(result['removed_entries'], 0)
        self.assertFalse(self.session.exists())

    def test_symlink_dependencies_are_preserved_without_following_them(self):
        external = self.root.parent / 'external'
        external.mkdir()
        (external / 'weights').write_bytes(b'shared weights')
        (self.session / 'weights').symlink_to(external, target_is_directory=True)
        result = self.pack()
        data = archive.verify(result['archive'])
        self.assertFalse(any(r['name'].endswith('/weights/weights') for r in data['entries']))
        self.assertEqual((external / 'weights').read_bytes(), b'shared weights')

    def test_wrong_run_identity_is_not_overwritten(self):
        self.pack(cleanup=False)
        campaign.save(self.session / 'progress.json', dict(self.progress, identity='different-run'))
        with self.assertRaisesRegex(archive.ArchiveError, 'different run'):
            self.pack()

    def test_large_diagnostic_fields_are_read_by_a_fresh_inventory(self):
        with self.trajectory.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['experience_retrieval'] = json.dumps({'evidence': 'x' * 180000})
        with self.trajectory.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        previous = csv.field_size_limit(131072)
        self.addCleanup(csv.field_size_limit, previous)
        errors = []
        found = inventory.scan(self.root / 'outputs/bolt_ml', 'bolt_ml', 'fixture',
            {archive.task_key(self.task): self.task}, dict(policy_id='fixture', per_evaluation_seconds=2400), errors)
        self.assertEqual(errors, [])
        self.assertEqual(found[archive.task_key(self.task)][0]['rows'], 3)


if __name__ == '__main__':
    unittest.main()
