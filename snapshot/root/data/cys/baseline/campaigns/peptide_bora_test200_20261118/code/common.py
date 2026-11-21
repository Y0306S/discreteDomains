"""Campaign-owned state and immutable asset verification."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            payload = pickle.dumps(value, protocol=5) if binary else (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def config():
    return read(ROOT/'config.json')


def verify():
    manifest = read(ROOT/'evidence/assets.json')
    for path, expected in manifest['sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Pinned asset changed: '+path)
    return config()


def valid(sequence, reference):
    from Levenshtein import distance
    return (isinstance(sequence, str) and 0 < len(sequence) <= 30 and
            not set(sequence)-set('ACDEFGHIKLMNPQRSTVWY') and sequence.count('C') <= 1 and
            1-distance(sequence, reference)/len(reference) >= .75)


def proc_identity(pid):
    try:
        p = Path('/proc')/str(pid)
        fields = (p/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=int(fields[19]),
                    argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v])
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(identity):
    current = proc_identity(identity['pid'])
    return bool(current and current['start_ticks'] == identity['start_ticks'] and
                current['argv'] == identity['argv'])


def status(path, **values):
    atomic(path, dict(at=time.time(), pid=os.getpid(), **values))
