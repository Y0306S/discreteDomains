"""Check that query training keeps measured outcomes and excludes censored data."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import query_transfer as transfer


class QuerySourcesTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(vae_sha256='vae', bao_sha256='bao', train_sha256='train', test_sha256='test')
        self.history = dict(task_id='CEB_1A1', meta_info=dict(task_id='CEB_1A1', method='lolbo',
            split='train', seed=0, status='complete', charged_observations=1049,
            vae_sha256='vae', bao_sha256='bao', train_split_sha256='train',
            test_split_sha256='test', unresolved_database_calls=0, update_e2e=False), observations=[])
        for i in range(1049):
            plan = [1, 1, 2, 1, 1] if i < 3 else [2, 1, 1, 1, 1]
            censored = int(i == 0)
            runtime = 10.0 if i == 0 else (3.0 if i == 1 else 2.0)
            self.history['observations'].append(dict(objectives=[runtime], trial_state=censored,
                extra_info=dict(charged_index=i, oracle='postgres_runtime', runtime=runtime,
                                censored=censored, plan=plan)))

    def test_first_uncensored_per_plan_preserves_measured_runtime(self):
        rows, audit = transfer.source_rows(self.history, 'CEB_1A1', self.plan)
        self.assertEqual(list(rows.values()), [(1, 3.0), (3, 2.0)])
        self.assertEqual(audit, dict(total=1049, censored=1, duplicates=1046, retained=2))

    def test_rejects_wrong_split_and_unresolved_database_calls(self):
        for key, value in [('split', 'test'), ('unresolved_database_calls', 1), ('update_e2e', True)]:
            with self.subTest(key=key):
                history = copy.deepcopy(self.history)
                history['meta_info'][key] = value
                with self.assertRaises(ValueError):
                    transfer.source_rows(history, 'CEB_1A1', self.plan)

    def test_rejects_misaligned_runtime(self):
        self.history['observations'][10]['objectives'] = [999.0]
        with self.assertRaises(ValueError):
            transfer.source_rows(self.history, 'CEB_1A1', self.plan)

    def test_canonical_duplicates_keep_first_uncensored_outcome(self):
        rows, audit = transfer.source_rows(self.history, 'CEB_1A1', self.plan,
            canonicalize=lambda encoded: [1, 1, 2, 1, 1])
        self.assertEqual(list(rows.values()), [(1, 3.0)])
        self.assertEqual(audit['retained'], 1)
        self.assertEqual(audit['canonicalized'], 1046)


if __name__ == '__main__':
    unittest.main()
