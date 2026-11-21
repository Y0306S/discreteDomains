"""CEB-only source contracts shared by the isolated GP environments."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np

QP_ROOT = Path('/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization')
HISTORY_ROOT = QP_ROOT / 'lolbo_scripts/optimization_all_collected_data'
SPLIT_ROOT = Path('/root/data/cys/logs/bolt_resume')
TRAIN_FILE = SPLIT_ROOT / 'train800_bsz10.txt'
TEST_FILE = SPLIT_ROOT / 'test200_holdout.txt'
HISTORICAL_VAE_FILE = QP_ROOT / 'vae/CEB_64.ckpt'
VAE_FILE = QP_ROOT / 'vae/CEB_64_author.ckpt'
AUTHOR_VAE_SHA256 = '83a4942701bffaaf8311887e530e3157b6843134e52e9f53d59f957c24aebab1'
OFFICIAL_BAO_SHA256 = 'caaec38f3f68684ba2494c165f70c41107a396d4ceea1565c716efbeb37ee375'
POLICY = 'uncensored_first_distinct_plan_min_two'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_assets(root=QP_ROOT, checkpoint=None):
    checkpoint = Path(checkpoint or Path(root) / VAE_FILE.relative_to(QP_ROOT)).resolve()
    bao = Path(root) / 'initialization_data/ceb_3k_bao_initialization.csv'
    if digest(checkpoint) != AUTHOR_VAE_SHA256:
        raise ValueError('This query campaign requires the pinned BayesQO author VAE')
    if digest(bao) != OFFICIAL_BAO_SHA256:
        raise ValueError('This query campaign requires the restored official Bao initialization')
    return checkpoint


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
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
    train, test = TRAIN_FILE.read_text().split(), TEST_FILE.read_text().split()
    if (len(train) != 800 or len(set(train)) != 800 or len(test) != 200
            or len(set(test)) != 200 or set(train) & set(test)
            or any(not re.fullmatch(r'CEB_\d+[A-Z]\d+', t) for t in train + test)):
        raise ValueError('Invalid authoritative CEB train800/test200 split')
    return train, test


def source_path(task):
    if not re.fullmatch(r'CEB_\d+[A-Z]\d+', task):
        raise ValueError('Invalid query ID')
    return HISTORY_ROOT / ('optimimze-db_nowandb_%s_all-data-collected.csv' % task)


def source_digest(path):
    """Validate retired local-VAE sources without restoring thousands of files."""
    path = Path(path)
    if path.exists():
        return digest(path)
    index = json.loads((path.parent / 'retired_train800_local_vae.json').read_text())
    entry = index['files'][path.name]
    with zipfile.ZipFile(path.parent / index['archive']) as archive:
        content = archive.read(path.name)
    actual = hashlib.sha256(content).hexdigest()
    if actual != entry['sha256']:
        raise ValueError('Archived source digest mismatch: ' + str(path))
    return actual


def scale_latents(values, lower, upper):
    values, lower, upper = np.asarray(values), np.asarray(lower), np.asarray(upper)
    if (values.ndim != 2 or values.shape[1] != 64 or lower.shape != (64,)
            or upper.shape != (64,) or not np.all(upper > lower)
            or not all(np.isfinite(a).all() for a in (values, lower, upper))):
        raise ValueError('Invalid latent features or bounds')
    scaled = (values - lower) / (upper - lower)
    if np.any(scaled < 0) or np.any(scaled > 1):
        raise ValueError('Latents outside the fixed source-derived space; no clipping')
    return scaled


def window_indices(count, size=32):
    if count < 2 or size < 2:
        raise ValueError('Windows require at least two observations')
    for start in range(0, count, size):
        end = min(start + size, count)
        # A one-point tail reuses its immediate predecessor to retain a transition.
        yield list(range(start if end - start > 1 else start - 1, end))


def load_corpus(directory, check_sources=True):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    train, test = read_split()
    if (manifest.get('domain') != 'ceb_query_runtime'
            or manifest.get('status') != 'complete' or manifest.get('policy') != POLICY
            or manifest.get('representation') != 'frozen_vae_encoder_mean_64'
            or manifest['train_split_sha256'] != digest(TRAIN_FILE)
            or manifest['test_split_sha256'] != digest(TEST_FILE)
            or manifest['vae_sha256'] != digest(VAE_FILE)):
        raise ValueError('Query corpus policy, split or VAE identity mismatch')
    for name in ('tasks.json', 'features.npz'):
        if digest(directory / name) != manifest['files'][name]:
            raise ValueError('Changed query corpus file: ' + name)
    if manifest.get('plan_representation') == 'oracle_canonical_alias_encoding':
        if (manifest['codec_sha256'] != digest(QP_ROOT / 'codec/codec.py')
                or manifest['oracle_sha256'] != digest(QP_ROOT / 'oracle/oracle.py')):
            raise ValueError('Historical query codec changed')
    records = json.loads((directory / 'tasks.json').read_text())
    ids = [r['task_id'] for r in records]
    if (ids != manifest['source_tasks'] or len(ids) != len(set(ids))
            or not set(ids).issubset(train) or set(ids) & set(test)
            or set(ids) | {r['task_id'] for r in manifest['excluded_tasks']} != set(train)):
        raise ValueError('Source tasks do not match the permitted split')
    with np.load(str(directory / 'features.npz'), allow_pickle=False) as data:
        raw, x, offsets = data['encoder_means'], data['X'], data['offsets']
    expected = scale_latents(raw, manifest['lower'], manifest['upper'])
    if (x.shape != expected.shape or not np.allclose(x, expected, rtol=1e-6, atol=1e-7)
            or len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(x)
            or not np.all(np.diff(offsets) >= 2)):
        raise ValueError('Invalid normalized source features or offsets')
    tasks = {}
    for index, record in enumerate(records):
        task = record['task_id']
        path = source_path(task)
        if (Path(record['source_path']).resolve() != path.resolve()
                or (check_sources and source_digest(path) != record['source_sha256'])):
            raise ValueError('Changed or noncanonical source history: ' + task)
        if 'sql_path' in record and digest(record['sql_path']) != record['sql_sha256']:
            raise ValueError('Query SQL changed: ' + task)
        y = np.asarray(record['runtime_seconds'], dtype=float)
        plans, rows = record['plans'], record['source_rows']
        features = x[offsets[index]:offsets[index + 1]]
        if (len(y) != len(features) or len(plans) != len(y) or len(rows) != len(y)
                or len({tuple(p) for p in plans}) != len(plans)
                or rows != sorted(set(rows)) or not np.isfinite(y).all() or np.any(y <= 0)):
            raise ValueError('Invalid distinct completed observations: ' + task)
        tasks[task] = dict(record, X=features, y_minimize=y, y_maximize=-y)
    if sum(len(t['X']) for t in tasks.values()) != manifest['observations']:
        raise ValueError('Corpus observation count mismatch')
    return manifest, tasks
