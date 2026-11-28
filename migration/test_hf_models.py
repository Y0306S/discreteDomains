"""Small CPU tests of migration integrity and conflict handling; no HF access."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from hf_models import export_bank, restore, safe_child, verify_payload


class ModelTransferTests(unittest.TestCase):
    def test_rgpe_round_trip_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'bank.sqlite'
            rows = []
            with sqlite3.connect(source) as conn:
                conn.execute('CREATE TABLE models(task TEXT PRIMARY KEY, metadata TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL)')
                for task, payload in [('pep_000', b'GP\x00payload'*200), ('pep_001', b'other-model')]:
                    sha = hashlib.sha256(payload).hexdigest()
                    metadata = json.dumps({'task': task, 'normalization': [1, 2]})
                    conn.execute('INSERT INTO models VALUES(?,?,?,?)', (task, metadata, sha, payload))
                    rows.append(dict(task=task, metadata=metadata, sha256=sha, bytes=len(payload), name=f'models/{task}.pt'))
            model = dict(id='fixture', source=str(source), mode='sqlite', rows=rows)
            folder = root/'stage'
            folder.mkdir()
            files = export_bank(model, folder)
            manifest = dict(model=model, files=files)
            (folder/'payload-manifest.json').write_text(json.dumps(manifest))
            destination = root/'restored'
            restore(folder, destination, apply=True)
            restored = destination/str(source).lstrip('/')
            with sqlite3.connect(source) as original, sqlite3.connect(restored) as actual:
                self.assertEqual(original.execute('SELECT * FROM models ORDER BY task').fetchall(),
                                 actual.execute('SELECT * FROM models ORDER BY task').fetchall())
            with self.assertRaises(FileExistsError):
                restore(folder, destination, apply=True)
            (folder/rows[0]['name']).write_bytes(b'corrupted')
            with self.assertRaises(RuntimeError):
                verify_payload(folder, manifest)

    def test_file_restore_conflict_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'bundle'
            folder.mkdir()
            payload = b'final model'
            (folder/'weights').write_bytes(payload)
            item = dict(source='/root/models/weights', name='weights', bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest())
            (folder/'payload-manifest.json').write_text(json.dumps(dict(model=dict(mode='file'), files=[item])))
            destination = root/'destination'
            restore(folder, destination)
            self.assertFalse(destination.exists())
            restore(folder, destination, apply=True)
            target = destination/'root/models/weights'
            self.assertEqual(target.read_bytes(), payload)
            target.write_bytes(b'different model')
            with self.assertRaises(FileExistsError):
                restore(folder, destination, apply=True)

    def test_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ['../outside', '/absolute', 'nested/../../outside']:
                with self.assertRaises(ValueError):
                    safe_child(Path(directory), value)


if __name__ == '__main__':
    unittest.main()
