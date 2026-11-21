"""A bounded-file, transactional store for run evidence and logical result paths."""

from contextlib import contextmanager
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import zlib

DATABASE = 'artifacts.sqlite3'


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Unsafe artifact name: ' + repr(name))
    return name


def connect(root, create=False):
    path = Path(root) / DATABASE
    if not create and not path.exists():
        raise FileNotFoundError(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not create:
        return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
    connection = sqlite3.connect(str(path), timeout=60)
    connection.execute('PRAGMA synchronous=FULL')
    # A persistent rollback journal works on the shared filesystem; WAL does not.
    connection.execute('PRAGMA journal_mode=TRUNCATE')
    connection.execute('PRAGMA temp_store=MEMORY')
    connection.execute('CREATE TABLE IF NOT EXISTS artifacts '
        '(name TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL, metadata TEXT NOT NULL)')
    connection.commit()
    return connection


@contextmanager
def database(root, create=False):
    connection = connect(root, create)
    try:
        yield connection
    finally:
        connection.close()


def decode(row):
    raw = zlib.decompress(row[0])
    if hashlib.sha256(raw).hexdigest() != row[1]:
        raise RuntimeError('Artifact checksum mismatch')
    return raw


def fetch(connection, name):
    row = connection.execute('SELECT payload,sha256 FROM artifacts WHERE name=?', (safe_name(name),)).fetchone()
    if row is None:
        raise FileNotFoundError(name)
    return decode(row)


def put(connection, name, raw, metadata=None, immutable=False):
    safe_name(name)
    digest = hashlib.sha256(raw).hexdigest()
    old = connection.execute('SELECT sha256 FROM artifacts WHERE name=?', (name,)).fetchone()
    if immutable and old is not None and old[0] != digest:
        raise RuntimeError('Completed artifact changed: ' + name)
    connection.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?)',
                       (name, zlib.compress(raw, 1), digest, json.dumps(metadata or {}, sort_keys=True)))


def names(connection, pattern='*'):
    return [row[0] for row in connection.execute('SELECT name FROM artifacts ORDER BY name')
            if fnmatch.fnmatchcase(row[0], pattern)]


def read_bytes(root, name):
    path = Path(root) / safe_name(name)
    if (Path(root) / DATABASE).exists():
        with database(root) as connection:
            try:
                return fetch(connection, name)
            except FileNotFoundError:
                pass
    return path.read_bytes()


def logical_read(path):
    path = Path(path).absolute()
    for parent in path.parents:
        if (parent / DATABASE).exists():
            return read_bytes(parent, str(path.relative_to(parent)))
    raise FileNotFoundError(path)


def serialized_read(raw, name=None):
    connection = sqlite3.connect(':memory:')
    try:
        connection.deserialize(raw)
        return names(connection) if name is None else fetch(connection, name)
    finally:
        connection.close()


class JsonJournal:
    def __init__(self, root):
        self.root = Path(root)

    def read(self, name):
        return json.loads(read_bytes(self.root, name))

    def exists(self, name):
        try:
            read_bytes(self.root, name)
            return True
        except FileNotFoundError:
            return False

    def write(self, name, value):
        self.update({name: value})

    def update(self, values, delete=()):
        """Publish a result and retire its pending marker in one transaction."""
        with database(self.root, create=True) as connection, connection:
            for name, value in values.items():
                raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
                put(connection, name, raw)
            for name in delete:
                connection.execute('DELETE FROM artifacts WHERE name=?', (safe_name(name),))
        for name in delete:
            (self.root / name).unlink(missing_ok=True)

    def delete(self, name):
        with database(self.root, create=True) as connection, connection:
            connection.execute('DELETE FROM artifacts WHERE name=?', (safe_name(name),))
        (self.root / name).unlink(missing_ok=True)

    def names(self, pattern):
        result = {p.name for p in self.root.glob(pattern) if p.is_file()}
        if (self.root / DATABASE).exists():
            with database(self.root) as connection:
                result.update(names(connection, pattern))
        return sorted(result)

    def migrate(self, pattern='proposal_*.json'):
        files = sorted(p for p in self.root.glob(pattern) if p.is_file())
        if files:
            fold(self.root, files)


def fold(root, paths):
    """Commit and verify immutable files before deleting any loose originals."""
    root = Path(root)
    files, directories = [], []
    for path in paths:
        path = Path(path)
        entries = [path] + sorted(path.rglob('*')) if path.is_dir() else [path]
        for item in entries:
            if item.is_symlink():
                raise RuntimeError('Unexpected symlink in round evidence: ' + str(item))
            if item.is_dir():
                directories.append(item)
            elif item.is_file():
                files.append(item)
    if not files:
        return 0
    descriptions = []
    with database(root, create=True) as connection:
        with connection:
            for path in files:
                info = path.stat()
                raw = path.read_bytes()
                name = str(path.relative_to(root))
                metadata = dict(mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns, size=len(raw))
                put(connection, name, raw, metadata, immutable=True)
                descriptions.append((path, name, hashlib.sha256(raw).hexdigest()))
        for path, name, digest in descriptions:
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest or hashlib.sha256(fetch(connection, name)).hexdigest() != digest:
                raise RuntimeError('Artifact changed while consolidating: ' + name)
        for path, _, _ in descriptions:
            path.unlink()
    for directory in sorted(set(directories), key=lambda p: len(p.parts), reverse=True):
        directory.rmdir()
    return len(files)


def compact_ml(root, committed_attempts):
    root = Path(root)
    paths = []
    for parent, prefix in ((root / 'evaluations', ''), (root / 'decisions', 'trial-')):
        if parent.exists():
            for path in parent.iterdir():
                suffix = path.name[len(prefix):] if path.name.startswith(prefix) else ''
                if path.is_dir() and suffix.isdigit() and int(suffix) < committed_attempts:
                    paths.append(path)
    return fold(root, paths)
