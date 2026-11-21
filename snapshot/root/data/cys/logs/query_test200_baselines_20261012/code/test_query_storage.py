from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from run_artifact_store import JsonJournal
from run_query_ceb_baselines import evaluate_pending, atomic_json


class QueryStorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proposal = {'plan': [1, 2, 3, 4, 5], 'z': [0.] * 64, 'timeout': 10}

    def test_result_recovery_does_not_repeat_sql(self):
        oracle = Mock(return_value={'runtime': 1., 'censored': 0, 'status': 'success'})
        first = evaluate_pending(self.root, [self.proposal], [], [], oracle)
        second = evaluate_pending(self.root, [self.proposal], [], [], oracle)
        self.assertEqual(first, second)
        self.assertEqual(oracle.call_count, 1)
        self.assertEqual(JsonJournal(self.root).names('*.pending.json'), [])

    def test_unresolved_call_still_refuses_replay_after_migration(self):
        atomic_json(self.root / 'proposal_0000.pending.json', self.proposal)
        atomic_json(self.root / 'proposal_0000.database_attempts.json',
                    [{'proposal': self.proposal, 'status': 'in_flight'}])
        JsonJournal(self.root).migrate()
        oracle = Mock()
        with self.assertRaisesRegex(RuntimeError, 'Unresolved database call'):
            evaluate_pending(self.root, [self.proposal], [], [], oracle)
        oracle.assert_not_called()

    def test_confirmed_io_failure_can_resume_and_retains_evidence(self):
        journal = JsonJournal(self.root)
        journal.update({'proposal_0000.pending.json': self.proposal,
                        'proposal_0000.database_attempts.json': [
                            {'proposal': self.proposal, 'status': 'io_error'}]})
        oracle = Mock(return_value={'runtime': 1., 'censored': 0, 'status': 'success'})
        rows = evaluate_pending(self.root, [self.proposal], [], [], oracle)
        self.assertEqual(rows[0]['database_attempts'], 2)
        self.assertEqual(rows[0]['database_attempt_history'][0]['status'], 'io_error')
        self.assertEqual(rows[0]['database_elapsed_unknown_attempts'], 1)

    def test_reviewed_storage_replay_keeps_original_outcome_unknown(self):
        journal = JsonJournal(self.root)
        recovery = dict(replay_authorized=True, active_database_calls=0,
                        original_result_unavailable=True, evidence_report='verified-outage.json')
        journal.update({'proposal_0000.pending.json': self.proposal,
                        'proposal_0000.database_attempts.json': [
                            {'proposal': self.proposal, 'status': 'in_flight', 'storage_recovery': recovery}]})
        oracle = Mock(return_value={'runtime': 1., 'censored': 0, 'status': 'success'})
        rows = evaluate_pending(self.root, [self.proposal], [], [], oracle)
        self.assertEqual(rows[0]['database_attempt_history'][0]['status'], 'in_flight')
        self.assertEqual(rows[0]['database_attempts'], 2)


if __name__ == '__main__':
    unittest.main()
