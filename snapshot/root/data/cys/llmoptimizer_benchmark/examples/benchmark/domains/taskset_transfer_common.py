"""TaskSet-only training contracts, compatible with the isolated GP environments."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile

import numpy as np

CAMPAIGN = Path('/root/data/logs/taskset_major_families_cpu_20260929')
HISTORIES = Path('/root/data/logs/taskset_corpus_train_1000s_100t')
SPLIT_SHA256 = '4cf49ac352313823e8e356809fd987d22723d55ab868faaee79e531ae55401ef'
PROTOCOL = dict(training_steps=1000, eval_every_n=1000, num_replicas=1)
METRIC = 'min_normalized_valid_outer'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    with temp.open('w') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_split():
    path = CAMPAIGN / 'split.json'
    split = json.loads(path.read_text())
    if (digest(path) != SPLIT_SHA256 or len(split['train']) != 838 or len(split['test']) != 210
            or len(set(split['train'])) != 838 or len(set(split['test'])) != 210
            or set(split['train']) & set(split['test'])):
        raise ValueError('TaskSet split identity changed')
    return split


def history_bytes(task):
    if Path(task).name != task or task in ('.', '..'):
        raise ValueError('Invalid task ID')
    path = HISTORIES / task / 'tpe/seed1/history.json'
    archive = path.parent.with_name('seed1.tar.gz')
    if archive.exists():
        with tarfile.open(str(archive), 'r:gz') as stream:
            manifest = json.load(stream.extractfile('manifest.json'))
            raw = stream.extractfile('run/history.json').read()
            entry = next(r for r in manifest['entries'] if r['name'] == 'run/history.json')
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Archived history checksum mismatch: ' + task)
            return raw
    try:
        return path.read_bytes()
    except FileNotFoundError:
        if archive.exists():
            return history_bytes(task)
        raise


def load_corpus(directory, check_sources=True):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    split = read_split()
    if (manifest['domain'] != 'taskset_cash' or manifest['status'] != 'complete'
            or manifest['split_sha256'] != SPLIT_SHA256 or manifest['protocol'] != PROTOCOL
            or manifest['objective_metric'] != METRIC or manifest['source_policy'] != 'successful_real_tpe_observations'):
        raise ValueError('TaskSet corpus contract mismatch')
    for name, expected in manifest['files'].items():
        if digest(directory / name) != expected:
            raise ValueError('Changed corpus file: ' + name)
    records = json.loads((directory / 'tasks.json').read_text())
    ids = [row['task_id'] for row in records]
    if (ids != manifest['source_tasks'] or len(ids) != len(set(ids))
            or not set(ids) <= set(split['train']) or set(ids) & set(split['test'])):
        raise ValueError('TaskSet source tasks violate train838 restriction')
    if manifest['purpose'] == 'production_training' and set(ids) != set(split['train']):
        raise ValueError('Full TaskSet training requires all 838 source tasks')
    train, valid = manifest['fsbo_train_tasks'], manifest['fsbo_validation_tasks']
    if not train or not valid or set(train) & set(valid) or set(train) | set(valid) != set(ids):
        raise ValueError('Invalid FSBO internal validation split')
    with np.load(str(directory / 'features.npz'), allow_pickle=False) as arrays:
        x, offsets = arrays['X_fsbo'], arrays['offsets']
    if (x.shape != (manifest['observations'], manifest['fsbo_dim']) or not np.isfinite(x).all()
            or offsets[0] != 0 or offsets[-1] != len(x) or len(offsets) != len(ids) + 1
            or not np.all(np.diff(offsets) >= 2)):
        raise ValueError('Invalid TaskSet feature matrix')
    tasks = {}
    for i, record in enumerate(records):
        task = record['task_id']
        if check_sources and hashlib.sha256(history_bytes(task)).hexdigest() != record['source_sha256']:
            raise ValueError('Changed source history: ' + task)
        y = np.asarray(record['y_minimize'], dtype=float)
        features = x[offsets[i]:offsets[i + 1]]
        rows = record['source_rows']
        if (len(features) != len(y) or len(record['configs']) != len(y)
                or len(rows) != len(y) or rows != sorted(set(rows))
                or not np.isfinite(y).all() or any(r < 1 or r > 100 for r in rows)):
            raise ValueError('Misaligned TaskSet observations: ' + task)
        tasks[task] = dict(record, X=features, y_minimize=y, y_maximize=-y)
    return manifest, tasks
