"""Regression checks for the empty-journal startup race and safe recovery."""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

CODE = Path('/root/data/cys/logs/query_test200_baselines_20261012/code')
sys.path.insert(0, str(CODE))
import query_test200_fleet_v3 as fleet
from run_artifact_store import DATABASE, JsonJournal
from run_query_ceb_baselines_v2 import evaluate_pending


class StartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_reader_during_initialization_retries_and_then_recovers(self):
        with sqlite3.connect(self.root / DATABASE):
            pass
        progress = fleet.monitor_progress(self.root)
        self.assertEqual(progress['saved_proposals'], 0)
        self.assertEqual(progress['monitor_warning'], 'no such table: artifacts')
        fleet.initialize_journal(self.root)
        recovered = fleet.monitor_progress(self.root, progress)
        self.assertEqual(recovered, dict(summary={}, saved_proposals=0))

    def test_existing_evidence_survives_initialization(self):
        journal = JsonJournal(self.root)
        journal.write('proposal_0000.json', {'runtime': 1.25})
        fleet.initialize_journal(self.root)
        self.assertEqual(journal.read('proposal_0000.json'), {'runtime': 1.25})
        self.assertEqual(fleet.monitor_progress(self.root)['saved_proposals'], 1)

    def test_missing_schema_in_nonempty_database_is_not_repaired(self):
        with sqlite3.connect(self.root / DATABASE) as connection:
            connection.execute('CREATE TABLE unrelated(value)')
        with self.assertRaisesRegex(sqlite3.OperationalError, 'no such table: artifacts'):
            fleet.initialize_journal(self.root)

    def test_corrupt_database_is_not_masked(self):
        with sqlite3.connect(self.root / DATABASE) as connection:
            connection.execute('CREATE TABLE artifacts(name)')
        with self.assertRaises(sqlite3.OperationalError):
            fleet.initialize_journal(self.root)
        with patch.object(fleet.JsonJournal, 'names', side_effect=sqlite3.DatabaseError('malformed')):
            with self.assertRaises(sqlite3.DatabaseError):
                fleet.monitor_progress(self.root)

    def test_temporary_lock_retains_last_count(self):
        previous = dict(summary={'proposals': 7}, saved_proposals=7)
        error = sqlite3.OperationalError('database is locked')
        error.sqlite_errorcode = sqlite3.SQLITE_BUSY
        with patch.object(fleet.JsonJournal, 'names', side_effect=error):
            progress = fleet.monitor_progress(self.root, previous)
        self.assertEqual(progress['saved_proposals'], 7)
        self.assertEqual(progress['summary'], previous['summary'])
        self.assertIn('monitor_warning', progress)

    def test_pending_cached_proposal_does_not_execute_sql(self):
        fleet.initialize_journal(self.root)
        proposal = dict(plan=[1, 2, 3], z=[0.0] * 64, timeout=100.0)
        initial = [dict(plan=proposal['plan'], runtime=2.5, censored=0, status='completed')]
        oracle = Mock()
        first = evaluate_pending(self.root, [proposal], initial, [], oracle)
        replay = evaluate_pending(self.root, [proposal], initial, [], oracle)
        oracle.assert_not_called()
        self.assertEqual(first, replay)
        self.assertEqual(first[0]['status'], 'cache_hit')
        self.assertEqual(JsonJournal(self.root).names('proposal_*.pending.json'), [])


if __name__ == '__main__':
    unittest.main()
