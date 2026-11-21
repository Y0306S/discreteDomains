import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / 'examples/benchmark/domains'
sys.path.insert(0, str(SCRIPTS))
import prepare_taskset_transfer as exporter
from taskset_transfer_common import load_corpus, read_split


class TaskSetSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = sorted(read_split()['train'])[:2]
        cls.raw = {task: exporter.history_bytes(task) for task in cls.ids}

    def test_test_split_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'train838'):
                exporter.prepare(Path(td), [read_split()['test'][0], self.ids[0]])

    def test_partial_history_cannot_be_exported(self):
        changed = dict(self.raw)
        history = json.loads(changed[self.ids[0]])
        history['observations'] = history['observations'][:99]
        changed[self.ids[0]] = json.dumps(history).encode()
        with tempfile.TemporaryDirectory() as td, patch.object(exporter, 'history_bytes', side_effect=changed.__getitem__):
            with self.assertRaisesRegex(ValueError, '100-evaluation'):
                exporter.prepare(Path(td), self.ids)
            self.assertFalse((Path(td) / 'manifest.json').exists())

    def test_failed_penalty_is_excluded_with_provenance_preserved(self):
        changed = dict(self.raw)
        history = json.loads(changed[self.ids[0]])
        history['observations'][0]['trial_state'] = 1
        history['observations'][0]['objectives'] = [1e10]
        changed[self.ids[0]] = json.dumps(history).encode()
        with tempfile.TemporaryDirectory() as td:
            with patch.object(exporter, 'history_bytes', side_effect=changed.__getitem__), \
                    patch('taskset_transfer_common.history_bytes', side_effect=changed.__getitem__):
                manifest = exporter.prepare(Path(td), self.ids)
                _, tasks = load_corpus(Path(td))
            self.assertEqual(manifest['observations'], 199)
            self.assertEqual(manifest['excluded_failed_observations'], 1)
            record = tasks[self.ids[0]]
            self.assertEqual(record['excluded_rows'], [1])
            self.assertEqual(record['source_rows'][0], 2)
            self.assertNotIn(1e10, record['y_minimize'])
            self.assertEqual(record['source_sha256'], hashlib.sha256(changed[self.ids[0]]).hexdigest())
            with self.assertRaisesRegex(ValueError, 'Changed source'):
                load_corpus(Path(td))


if __name__ == '__main__':
    unittest.main()
