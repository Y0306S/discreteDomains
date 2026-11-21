import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from examples.benchmark.domains.query_train800_fleet import archive_proposals, claim_task


class FleetTests(unittest.TestCase):
    def test_completed_archive_retains_exact_results_and_can_be_rechecked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = []
            for i in range(3):
                content = json.dumps(dict(proposal_index=i, plan=[i], runtime=i + 0.5)).encode()
                (root / f'proposal_{i:04d}.json').write_bytes(content)
                expected.append(content)
            archive_proposals(root, 3)
            self.assertEqual(list(root.glob('proposal_*.json')), [root / 'proposal_archive.json'])
            with tarfile.open(root / 'proposals.tar.gz') as tar:
                self.assertEqual([tar.extractfile(p).read() for p in tar.getmembers()], expected)
            archive_proposals(root, 3)

    def test_claims_never_move_partial_tasks_to_another_host_or_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'claims').mkdir()
            plan = dict(tasks=['a', 'b', 'c'], task_hosts=dict(a='h1', b='h2', c='h1'))
            worker1 = dict(host='h1', name='h1.w0')
            self.assertEqual(claim_task(root, plan, worker1), 'a')
            self.assertEqual(claim_task(root, plan, worker1), 'a')
            self.assertEqual(claim_task(root, plan, dict(host='h1', name='h1.w1')), 'c')
            self.assertEqual(claim_task(root, plan, dict(host='h2', name='h2.w0')), 'b')


if __name__ == '__main__':
    unittest.main()
