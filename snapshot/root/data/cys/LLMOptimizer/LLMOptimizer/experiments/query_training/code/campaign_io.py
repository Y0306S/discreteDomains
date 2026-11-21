"""One durable database per task, with exclusive cross-host task ownership."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3


class InfrastructureError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_campaign(root):
    checksums = json.loads((root / 'checksums.json').read_text())
    effective = dict(checksums)
    # Keep the original campaign identity for existing task ledgers. Explicit,
    # hash-chained maintenance revisions may fix infrastructure, not the recipe.
    revisions_path = root / 'maintenance_revisions.json'
    if revisions_path.exists():
        for revision in json.loads(revisions_path.read_text())['revisions']:
            if revision['protocol_sha256'] != checksums['protocol.json']:
                raise InfrastructureError('Maintenance revision changed the experiment protocol')
            for name, change in revision['files'].items():
                if effective.get(name) != change['before']:
                    raise InfrastructureError('Broken maintenance revision chain: ' + name)
                effective[name] = change['after']
    for name, checksum in effective.items():
        if sha(root / name) != checksum:
            raise InfrastructureError('Prepared campaign changed: ' + name)
    external = root / 'external_checksums.json'
    if external.exists():
        for name, checksum in json.loads(external.read_text()).items():
            if sha(name) != checksum:
                raise InfrastructureError('Pinned external dependency changed: ' + name)
    return hashlib.sha256(canonical(checksums).encode()).hexdigest()


def require_live_gates(root):
    gates = json.loads((root / 'launch_gates.json').read_text())
    checks = gates.get('checks', {})
    if not checks or any(check.get('status') != 'passed' or not check.get('evidence')
                         for check in checks.values()):
        raise InfrastructureError('Live validation gates are pending; no job may launch')


class TaskStore:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Network filesystems may emulate flock with POSIX locks, which would
        # conflict with SQLite's own locks if both used the database inode.
        self.lock = self.path.with_suffix(self.path.suffix + '.lock').open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        try:
            self.db = sqlite3.connect(self.path, timeout=120)
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS rounds (attempt INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            self.db.commit()
            previous = self.get('identity')
            if previous is not None and previous != identity:
                raise InfrastructureError('Task checkpoint belongs to a different campaign/recipe')
            if previous is None:
                self.put('identity', identity)
        except BaseException:
            if hasattr(self, 'db'):
                self.db.close()
            self.lock.close()
            raise

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (key, canonical(value)))

    def finish(self, attempt, row, snapshot):
        with self.db:
            self.db.execute('INSERT INTO rounds VALUES (?,?)', (attempt, canonical(row)))
            self.db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', ('snapshot', canonical(snapshot)))
            for key in ('pending', 'evaluation_started', 'evaluation_result', 'draft'):
                self.db.execute('DELETE FROM state WHERE key=?', (key,))

    def rows(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT payload FROM rounds ORDER BY attempt')]

    def close(self):
        self.db.close()
        self.lock.close()


def save_child_result(path, result):
    # The supervisor owns the task lock; its child only commits the measured result.
    from result_integrity import prepare_result
    result = prepare_result(result)
    with sqlite3.connect(path, timeout=120) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', ('evaluation_result', canonical(result)))


def save_child_scratch(path, directory):
    with sqlite3.connect(path, timeout=120) as db:
        db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', ('active_scratch', canonical(str(directory))))


def process_group_cleanup(pid):
    import signal
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
