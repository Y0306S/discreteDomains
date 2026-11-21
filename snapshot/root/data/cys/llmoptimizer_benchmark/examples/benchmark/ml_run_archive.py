"""Verified, immutable completed ML runs with readers for their original paths."""

from contextlib import contextmanager
import csv
import fcntl
import fnmatch
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import stat
import sys
import tempfile
import time
from urllib.parse import quote
import zipfile


FORMAT = 'ml_completed_run_v1'


class ArchiveError(RuntimeError):
    pass


def task_key(task):
    return '/'.join(task[k] for k in ('task_type', 'dataset', 'algorithm'))


def archive_path(root, method, task, seed=None):
    root = Path(root)
    if seed is None:
        seed = json.loads((root / 'protocol.json').read_text()).get('seed', 1)
    parts = [task[k] for k in ('task_type', 'dataset', 'algorithm')]
    name = '__'.join(quote(str(p), safe='') for p in parts) + f'__seed{seed}.zip'
    return root / 'archives' / method / name


def locate(path):
    """Map an old logical result path to its run, without scanning result trees."""
    path = Path(os.path.abspath(path))
    for parent in path.parents:
        if parent.name not in ('sessions', 'outputs', 'state', 'logs', 'history'):
            continue
        root = parent.parent
        if not (root / 'protocol.json').is_file():
            continue
        parts = path.relative_to(parent).parts
        if len(parts) < 4:
            continue
        method, task_type = parts[:2]
        if (parent.name in ('outputs', 'history') and method in ('fsbo', 'rgpe')
                and len(parts) >= 5 and parts[2] in ('cls', 'rgs')):
            # Older transfer campaigns insert the algorithm before task_type.
            task_type = parts[2]
            dataset = parts[4] if parent.name == 'history' else parts[3]
            algorithm = Path(parts[-1]).stem if parent.name == 'history' else parts[4]
            if algorithm != parts[1]:
                continue
        elif parent.name == 'history':
            if len(parts) != 6:
                continue
            dataset, algorithm = parts[3], Path(parts[5]).stem
        else:
            dataset = parts[2]
            algorithm = Path(parts[3]).stem if parent.name in ('state', 'logs') else parts[3]
        task = dict(task_type=task_type, dataset=dataset, algorithm=algorithm)
        return archive_path(root, method, task), str(path.relative_to(root))
    return None, None


def _safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ArchiveError('Unsafe archive member: ' + repr(name))
    return path


def _manifest(archive):
    try:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['format'] != FORMAT:
            raise ArchiveError('Unrecognized ML archive format')
        entries = manifest['entries']
        names = [record['name'] for record in entries]
        if len(set(names)) != len(names) or 'manifest.json' in names:
            raise ArchiveError('Duplicate manifest entries')
        if len({str(PurePosixPath(name)) for name in names}) != len(names):
            raise ArchiveError('Overlapping archive paths')
        for name in names:
            _safe_name(name)
        actual = archive.namelist()
        if len(actual) != len(names) + 1 or set(actual) != set(names) | {'manifest.json'}:
            raise ArchiveError('Archive member inventory mismatch')
        links = {r['name'] for r in entries if r['kind'] == 'symlink'}
        for name in names:
            if any(str(p) in links for p in PurePosixPath(name).parents):
                raise ArchiveError('Member nested below an archived symlink')
        return manifest
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ArchiveError('Invalid ML archive manifest') from exc


def manifest(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return _manifest(archive)
    except zipfile.BadZipFile as exc:
        raise ArchiveError('Corrupt ML archive: ' + str(path)) from exc


def _hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def _member_bytes(archive, record):
    data = archive.read(record['name'])
    if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ArchiveError('Member checksum mismatch: ' + record['name'])
    return data


def read_bytes(path):
    packed, name = locate(path)
    if packed is None or not packed.exists():
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            # The packer may publish and unlink between the existence check and read.
            if packed is None or not packed.exists():
                raise
    try:
        with zipfile.ZipFile(packed) as archive:
            records = {r['name']: r for r in _manifest(archive)['entries']}
            record = records.get(name)
            if record is None:
                raise FileNotFoundError(path)
            if record['kind'] != 'file':
                raise ArchiveError('Logical result is not a regular file: ' + name)
            return _member_bytes(archive, record)
    except zipfile.BadZipFile as exc:
        raise ArchiveError('Corrupt ML archive: ' + str(packed)) from exc


def load_json(path, default=None):
    try:
        return json.loads(read_bytes(path))
    except FileNotFoundError:
        return default


def exists(path):
    packed, name = locate(path)
    if packed is not None and packed.exists():
        return any(r['name'] == name for r in manifest(packed)['entries'])
    return Path(path).exists()


@contextmanager
def open_text(path, *, newline=None, encoding='utf-8'):
    stream = io.StringIO(read_bytes(path).decode(encoding), newline=newline)
    try:
        yield stream
    finally:
        stream.close()


def iter_files(root, pattern='*'):
    root = Path(os.path.abspath(root))
    paths = {p for p in root.rglob(pattern) if p.is_file()} if root.exists() else set()
    campaign = next((p for p in (root, *root.parents) if (p / 'protocol.json').is_file()), None)
    if campaign is not None:
        for packed in (campaign / 'archives').glob('*/*.zip'):
            for record in manifest(packed)['entries']:
                path = campaign / record['name']
                if record['kind'] == 'file' and path.is_relative_to(root) and fnmatch.fnmatchcase(path.name, pattern):
                    paths.add(path)
    return iter(sorted(paths))


def iter_trajectories(root):
    return iter_files(root, '*_traj.csv')


def _describe(path, name):
    info = path.lstat()
    record = dict(name=name, mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns)
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(path)
        data = os.fsencode(target)
        record.update(kind='symlink', target=target, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    elif stat.S_ISDIR(info.st_mode):
        record.update(name=name.rstrip('/') + '/', kind='directory', size=0)
    elif stat.S_ISREG(info.st_mode):
        with path.open('rb') as stream:
            record.update(kind='file', size=info.st_size, sha256=_hash(stream))
    else:
        raise ArchiveError('Unexpected special file: ' + str(path))
    return record


def _gather(root, names):
    records = []

    def visit(path, name):
        record = _describe(path, name)
        records.append(record)
        if record['kind'] == 'directory':
            for child in sorted(path.iterdir()):
                visit(child, name + '/' + child.name)

    for name in names:
        _safe_name(name)
        path = root / name
        if path.exists() or path.is_symlink():
            visit(path, name)
    return records


def _alive(state):
    try:
        if state.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            return False
        proc = Path('/proc') / str(state['pid'])
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z' or fields[19] != state['start_ticks']:
            return False
        args = (proc / 'cmdline').read_bytes().decode().split('\0')
        return '--session-dir' in args and args[args.index('--session-dir') + 1] == state['session']
    except (OSError, KeyError, ValueError, IndexError):
        return False


def _quiescent(root, method, task, progress, state):
    protocol = json.loads((root / 'protocol.json').read_text())
    owner = protocol.get('hosts', {}).get(state.get('host'), {}).get('hostname')
    if owner and owner != socket.gethostname():
        raise ArchiveError('Archive on the owning host to verify worker exit: ' + owner)
    if _alive(state):
        raise ArchiveError('Worker still owns this run')
    if state.get('status') not in ('complete', 'complete_with_failures'):
        raise ArchiveError('Task has not exited with a completed budget')
    if state.get('method') != method or task_key(state) != task_key(task):
        raise ArchiveError('Task identity mismatch')
    budget = json.loads((root / 'runtime_policy.json').read_text())['attempts_per_search']
    if not (progress.get('budget_exhausted') and progress.get('pending') is False
            and progress.get('identity') and progress.get('attempts') == budget
            and progress.get('successful', 0) + progress.get('failed', 0) == budget):
        raise ArchiveError('Incomplete or pending evaluation ledger')
    trajectory = Path(progress['trajectory'])
    _task_output(root, method, task, trajectory)
    config = load_json(trajectory.parent / 'config.json')
    expected = protocol.get('ckpt32_optimizer', 'ptllm_tpe_qwen') if method == 'ckpt32' else method
    expected = {'ptllm_tpe_qwen': 'ptllm_tpe', 'ptllm_rgpe_qwen': 'ptllm_rgpe'}.get(expected, expected)
    if (not config or config.get('seed') != protocol.get('seed', 1)
            or config.get('optimizer') != expected
            or config.get('include_algorithms') != [task['algorithm']]
            or config.get('metric') != task['metric']):
        raise ArchiveError('Result configuration identity mismatch')
    csv.field_size_limit(sys.maxsize)
    with open_text(trajectory, newline='') as stream:
        rows = list(csv.DictReader(stream))
    legacy_success = (method in ('fsbo', 'rgpe') and progress['successful'] == budget
                      and progress['failed'] == 0 and rows
                      and all('evaluation_status' not in r and r.get('trial_state') == '0'
                              and math.isfinite(float(r['perf']))
                              and r.get('algorithm') == task['algorithm']
                              and json.loads(r['config_json']).get('algorithm') == task['algorithm']
                              for r in rows))
    if (len(rows) != budget or [int(r['iter_idx']) for r in rows] != list(range(budget))
            or (not legacy_success and (
                sum(r.get('evaluation_status') == 'SUCCESS' for r in rows) != progress['successful']
                or any(r.get('evaluation_status') not in ('SUCCESS', 'FAILED', 'TIMEOUT') for r in rows)))
            or any('oracle_called' in r and r['oracle_called'] != 'True' for r in rows)):
        raise ArchiveError('Invalid completed evaluation ledger')
    return protocol


def _task_output(root, method, task, trajectory):
    paths = [root / 'outputs' / method / task_key(task)]
    if method in ('fsbo', 'rgpe'):
        paths.append(root / 'outputs' / method / task['algorithm'] / task_key(task))
    for path in paths:
        if trajectory.is_relative_to(path):
            return path
    raise ArchiveError('Trajectory points outside the task output')


def verify(path):
    try:
        with zipfile.ZipFile(path) as archive:
            data = _manifest(archive)
            for record in data['entries']:
                member = archive.getinfo(record['name'])
                if member.file_size != record['size']:
                    raise ArchiveError('Member size mismatch: ' + record['name'])
                if record['kind'] != 'directory':
                    with archive.open(member) as stream:
                        if _hash(stream) != record['sha256']:
                            raise ArchiveError('Member checksum mismatch: ' + record['name'])
            return data
    except (zipfile.BadZipFile, EOFError) as exc:
        raise ArchiveError('Corrupt ML archive: ' + str(path)) from exc


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _compare_remaining(root, data):
    expected = {r['name']: r for r in data['entries']}
    remaining = _gather(root, data['source_roots'])
    for record in remaining:
        old = expected.get(record['name'])
        keys = ('kind', 'size', 'sha256', 'target', 'mode')
        if old is None or any(old.get(k) != record.get(k) for k in keys):
            raise ArchiveError('Source changed; refusing cleanup: ' + record['name'])
    return remaining


def _cleanup(root, data):
    remaining = _compare_remaining(root, data)
    retained = {PurePosixPath(name) for name in data.get('retained_completion_records', [])}
    protected = retained | {parent for name in retained for parent in name.parents}
    removed = 0
    # Verify all surviving originals before the first unlink; never follow links.
    for record in sorted(remaining, key=lambda r: len(PurePosixPath(r['name']).parts), reverse=True):
        if PurePosixPath(record['name']) in protected:
            continue
        path = root / record['name']
        if record['kind'] == 'directory':
            path.rmdir()
        else:
            path.unlink()
        removed += 1
    return removed


def pack_run(root, method, task, *, cleanup=False, held_lock=None, retain_completion_records=False):
    root = Path(os.path.abspath(root))
    task = {k: task[k] for k in ('task_type', 'dataset', 'algorithm', 'metric')}
    key = task_key(task)
    lock = held_lock
    if lock is None:
        lock_path = root / 'locks' / method / (key + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return dict(status='locked', task=key, method=method)
    try:
        target = archive_path(root, method, task)
        progress_path = root / 'sessions' / method / key / 'progress.json'
        state_path = root / 'state' / method / (key + '.json')
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else load_json(progress_path, {})
        state = json.loads(state_path.read_text()) if state_path.exists() else load_json(state_path, {})
        protocol = _quiescent(root, method, task, progress, state)
        if target.exists():
            data = verify(target)
            if (data['run_id'] != progress['identity'] or data['task'] != task
                    or data['method'] != method or data['campaign'] != str(root)):
                raise ArchiveError('Existing archive belongs to a different run')
        else:
            output = _task_output(root, method, task, Path(progress['trajectory']))
            names = ['sessions/' + method + '/' + key, str(output.relative_to(root)),
                     'logs/' + method + '/' + key + '.log', 'state/' + method + '/' + key + '.json']
            history = root / 'history' / method / task['task_type'] / task['metric'] / task['dataset']
            names += [str(p.relative_to(root)) for p in sorted(history.glob('*/' + task['algorithm'] + '.json'))]
            if method in ('fsbo', 'rgpe'):
                history = (root / 'history' / method / task['algorithm'] / task['task_type']
                           / task['metric'] / task['dataset'])
                names += [str(p.relative_to(root)) for p in sorted(history.glob('*/' + task['algorithm'] + '.json'))]
            records = _gather(root, names)
            provenance = []
            for name in ('protocol.json', 'runtime_policy.json', 'checksums.json', 'code_sha256.json',
                         'prepared.json', 'tasks_132.json', 'data_manifest.json'):
                path = root / name
                if path.is_file():
                    provenance.append((path, _describe(path, 'provenance/' + name)))
            data = dict(format=FORMAT, campaign=str(root), method=method, task=task,
                        seed=protocol.get('seed', 1), run_id=progress['identity'], progress=progress,
                        created_at=time.time(), source_roots=names,
                        retained_completion_records=([str(p.relative_to(root)) for p in (progress_path, state_path)]
                                                     if retain_completion_records else []),
                        trajectory=str(Path(progress['trajectory']).relative_to(root)),
                        entries=records + [r for _, r in provenance],
                        external_dependencies='Shared code, datasets, environments and model weights are referenced by provenance.')
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.partial', dir=target.parent)
            os.close(descriptor)
            try:
                with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                    archive.writestr('manifest.json', json.dumps(data, indent=2))
                    for path, record in [(root / r['name'], r) for r in records] + provenance:
                        info = zipfile.ZipInfo(record['name'])
                        info.create_system = 3
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info._compresslevel = 1
                        kind = {'file': stat.S_IFREG, 'directory': stat.S_IFDIR, 'symlink': stat.S_IFLNK}[record['kind']]
                        info.external_attr = (kind | record['mode']) << 16
                        with archive.open(info, 'w', force_zip64=True) as stream:
                            if record['kind'] == 'file':
                                with path.open('rb') as source:
                                    shutil.copyfileobj(source, stream, 1024 * 1024)
                            elif record['kind'] == 'symlink':
                                stream.write(os.fsencode(record['target']))
                verify(temporary)
                if _gather(root, names) != records:
                    raise ArchiveError('Run changed while packing')
                if any(_describe(p, r['name']) != r for p, r in provenance):
                    raise ArchiveError('Campaign provenance changed while packing')
                with open(temporary, 'rb') as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                _sync_directory(target.parent)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        removed = _cleanup(root, data) if cleanup else 0
        return dict(status='compacted' if cleanup else 'verified', task=key, method=method,
                    archive=str(target), run_id=data['run_id'], evaluations=progress['attempts'],
                    failed=progress['failed'], files=sum(r['kind'] == 'file' for r in data['entries']),
                    archive_bytes=target.stat().st_size, removed_entries=removed)
    finally:
        if held_lock is None:
            lock.close()


def restore(path, destination):
    data = verify(path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        for record in sorted(data['entries'], key=lambda r: len(PurePosixPath(r['name']).parts)):
            target = destination.joinpath(*_safe_name(record['name']).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if record['kind'] == 'directory':
                target.mkdir(exist_ok=True)
            elif record['kind'] == 'symlink':
                target.symlink_to(record['target'])
            else:
                with archive.open(record['name']) as source, target.open('xb') as stream:
                    shutil.copyfileobj(source, stream, 1024 * 1024)
                with target.open('rb') as stream:
                    if _hash(stream) != record['sha256']:
                        raise ArchiveError('Restored content mismatch: ' + record['name'])
            if record['kind'] != 'symlink':
                target.chmod(record['mode'])
    return data
