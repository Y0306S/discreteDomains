"""Verified completed-run archives. Active runs keep their rotating checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile


class ArchiveError(RuntimeError):
    """Fail closed: archive errors must never be interpreted as a fresh run."""


def archive_path(history):
    run = Path(history).parent
    return run.with_name(run.name + ".tar.gz")


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return digest(stream)


def _manifest(archive):
    manifest = json.load(archive.extractfile("manifest.json"))
    if manifest.get("format_version") != 1:
        raise ArchiveError("Unsupported archive format")
    records = {}
    for record in manifest["entries"]:
        name = record["name"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or name == ".":
            raise ArchiveError(f"Unsafe archive path: {name}")
        if name in records or name == "manifest.json":
            raise ArchiveError(f"Duplicate archive path: {name}")
        if record["kind"] not in ("file", "directory", "symlink"):
            raise ArchiveError(f"Unsupported member: {name}")
        records[name] = record
    members = archive.getmembers()
    if len(members) != len(records) + 1 or {m.name for m in members} != set(records) | {"manifest.json"}:
        raise ArchiveError("Archive membership differs from manifest")
    for name in records:
        for parent in PurePosixPath(name).parents:
            if str(parent) in records and records[str(parent)]["kind"] != "directory":
                raise ArchiveError(f"Archive member traverses a non-directory: {name}")
    return manifest, records


def _member_bytes(archive, name, records):
    record = records[name]
    member = archive.getmember(name)
    if record["kind"] != "file" or not member.isfile() or member.size != record["size"]:
        raise ArchiveError(f"Wrong archive member type/size: {name}")
    data = archive.extractfile(member).read()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ArchiveError(f"Archive checksum mismatch: {name}")
    return data


def read_archive_member(path, name="run/history.json"):
    try:
        with tarfile.open(path, "r:gz") as archive:
            _, records = _manifest(archive)
            return _member_bytes(archive, name, records)
    except Exception as exc:
        if isinstance(exc, ArchiveError):
            raise
        raise ArchiveError(f"Cannot read {path}: {exc}") from exc


def _validate_history(history, job, result, budget=100):
    meta = history.get("meta_info", {})
    rows = history["observations"]
    if (history.get("task_id") != job["task_id"] or meta.get("domain") != "taskset"
            or meta.get("optimizer") != job["method"] or meta.get("seed") != 1):
        raise ArchiveError("Archive history identity mismatch")
    successful = sum(row["trial_state"] == 0 and math.isfinite(row["objectives"][0]) for row in rows)
    if (len(rows) != budget or result["n_observations"] != budget
            or successful != result["n_successful"] or result["job"] != job
            or result.get("returncode") != 0
            or result.get("status", "complete" if successful == budget else "complete_with_failures")
            not in ("complete", "complete_with_failures")):
        raise ArchiveError("Only verified completed runs can be archived")


def verify_archive(path):
    """Verify all evidence, including files unrelated to history.json."""
    try:
        with tarfile.open(path, "r:gz") as archive:
            manifest, records = _manifest(archive)
            for name, record in records.items():
                member = archive.getmember(name)
                if member.mode != record["mode"]:
                    raise ArchiveError(f"Archive mode mismatch: {name}")
                if record["kind"] == "file":
                    if not member.isfile() or member.size != record["size"]:
                        raise ArchiveError(f"Archive type/size mismatch: {name}")
                    if digest(archive.extractfile(member)) != record["sha256"]:
                        raise ArchiveError(f"Archive checksum mismatch: {name}")
                elif record["kind"] == "symlink":
                    if not member.issym() or member.linkname != record["target"]:
                        raise ArchiveError(f"Archive symlink mismatch: {name}")
                elif not member.isdir():
                    raise ArchiveError(f"Archive directory mismatch: {name}")
            history = json.loads(_member_bytes(archive, "run/history.json", records))
            result = json.loads(_member_bytes(archive, "completion.json", records))
            _validate_history(history, manifest["job"], result, manifest["observations"])
        return manifest
    except Exception as exc:
        if isinstance(exc, ArchiveError):
            raise
        raise ArchiveError(f"Cannot verify {path}: {exc}") from exc


def _describe(path, name):
    meta = path.lstat()
    record = {"name": name, "mode": stat.S_IMODE(meta.st_mode), "mtime_ns": meta.st_mtime_ns}
    if stat.S_ISLNK(meta.st_mode):
        record.update(kind="symlink", target=os.readlink(path))
    elif stat.S_ISDIR(meta.st_mode):
        record.update(kind="directory")
    elif stat.S_ISREG(meta.st_mode):
        record.update(kind="file", size=meta.st_size, sha256=file_digest(path))
    else:
        raise ArchiveError(f"Unexpected special file: {path}")
    return record


def _gather(roots):
    entries = []

    def visit(path, name):
        record = _describe(path, name)
        entries.append((path, record))
        if record["kind"] == "directory":
            for child in sorted(path.iterdir()):
                visit(child, name + "/" + child.name)

    for path, name in roots:
        visit(path, name)
    return entries


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _job_paths(campaign, job):
    for key in ("id", "task_id", "method"):
        if Path(job[key]).name != job[key] or job[key] in (".", ".."):
            raise ArchiveError(f"Unsafe job {key}")
    run = Path(job["output_root"]) / job["task_id"] / job["method"] / "seed1"
    roots = [(run, "run"), (campaign / "runtimes" / job["id"], "runtime"),
             (campaign / "job_logs" / (job["id"] + ".log"), "job.log")]
    for path, _ in roots:
        if path.resolve() != path.absolute():
            raise ArchiveError(f"Cleanup root must not traverse symlinks: {path}")
    return run, roots


@contextmanager
def job_lock(campaign, job_id):
    if Path(job_id).name != job_id or job_id in (".", ".."):
        raise ArchiveError("Invalid job ID")
    with (Path(campaign) / "locks" / (job_id + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock


def _check_lock(campaign, job, lock):
    expected = (campaign / "locks" / (job["id"] + ".lock")).stat()
    actual = os.fstat(lock.fileno())
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ArchiveError("Wrong job lock")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def pack_job(campaign, job, *, cleanup=False, lock=None):
    """Publish first, then remove only verified job-owned originals under flock.

    A crash during removal is recoverable by calling this again. Completion JSON
    remains a small operational index; locks and shared dependencies are retained.
    """
    campaign = Path(campaign).resolve()
    if lock is None:
        with job_lock(campaign, job["id"]) as held:
            return pack_job(campaign, job, cleanup=cleanup, lock=held)
    _check_lock(campaign, job, lock)
    run, owned_roots = _job_paths(campaign, job)
    target = archive_path(run / "history.json")
    result_path = campaign / "results" / (job["id"] + ".json")
    if not target.exists():
        result = json.loads(result_path.read_text())
        history = json.loads((run / "history.json").read_text())
        _validate_history(history, job, result)
        roots = owned_roots + [(result_path, "completion.json")]
        for relative in ("split.json", "source_hashes.json", "task_baselines.json",
                         "distributed/plan.json", "distributed/source_hashes.json", "distributed/runner.py"):
            path = campaign / relative
            if path.exists():
                roots.append((path, "campaign/" + relative))
        entries = _gather(roots)
        manifest = {"format_version": 1, "campaign": str(campaign), "job": job, "seed": 1,
                    "observations": 100, "source_roots": [
                        {"path": str(path), "archive_path": name} for path, name in roots],
                    "scope": "All run evidence, runtime, completion, log and campaign metadata; shared datasets and environments are external dependencies.",
                    "entries": [record for _, record in entries]}
        temporary = target.with_name(target.name + ".partial")
        # A previous unpublished archive is disposable; all originals still exist.
        temporary.unlink(missing_ok=True)
        try:
            with tarfile.open(temporary, "x:gz", dereference=False) as archive:
                for path, record in entries:
                    archive.inodes.clear()  # Store hard-linked source files as independent verified bytes.
                    archive.add(path, arcname=record["name"], recursive=False)
                data = (json.dumps(manifest, indent=2) + "\n").encode()
                member = tarfile.TarInfo("manifest.json")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            verify_archive(temporary)
            if [record for _, record in _gather(roots)] != manifest["entries"]:
                raise ArchiveError("Sources changed during archival")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _sync_directory(target.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    manifest = verify_archive(target)
    if manifest["job"] != job or manifest["campaign"] != str(campaign):
        raise ArchiveError("Archive belongs to a different job/campaign")
    result = json.loads(read_archive_member(target, "completion.json"))
    report = {"format": "taskset-run-tar-v1", "path": str(target),
              "sha256": file_digest(target), "bytes": target.stat().st_size,
              "all_contents_verified": True, "cleanup_complete": False,
              "removed_entries_this_call": 0}
    if result_path.exists():
        old = json.loads(result_path.read_text()).get("archive", {})
        if old.get("sha256") and old["sha256"] != report["sha256"]:
            raise ArchiveError("Archive differs from published checksum")
    if cleanup:
        records = {record["name"]: record for record in manifest["entries"]}
        remaining = _gather([(path, name) for path, name in owned_roots if path.exists()])
        # Validate everything before removing anything; directory mtimes change
        # after interrupted cleanup, while surviving file contents must still match.
        for path, record in remaining:
            expected = records.get(record["name"])
            current = dict(record)
            if expected is None:
                raise ArchiveError(f"Unarchived source appeared: {path}")
            expected = dict(expected)
            if record["kind"] == "directory":
                current.pop("mtime_ns")
                expected.pop("mtime_ns")
            if current != expected:
                raise ArchiveError(f"Source differs from archive; refusing cleanup: {path}")
        for path, record in reversed(remaining):
            if record["kind"] == "directory":
                path.rmdir()
            else:
                path.unlink()
        report.update(cleanup_complete=True, removed_entries_this_call=len(remaining))
    result["archive"] = report
    _write_json(result_path, result)
    return report


def restore_archive(path, destination):
    """Restore into a NEW directory, never into live run paths or symlink targets."""
    path, destination = Path(path), Path(destination)
    manifest = verify_archive(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".restore-", dir=destination.parent))
    try:
        records = manifest["entries"]
        with tarfile.open(path, "r:gz") as archive:
            # Symlinks are restored last and are never traversed during extraction.
            for record in sorted(records, key=lambda r: (r["kind"] == "symlink", r["name"].count("/"))):
                output = temporary / record["name"]
                output.parent.mkdir(parents=True, exist_ok=True)
                if record["kind"] == "directory":
                    output.mkdir(exist_ok=True)
                elif record["kind"] == "symlink":
                    output.symlink_to(record["target"])
                else:
                    with archive.extractfile(record["name"]) as source, output.open("xb") as stream:
                        shutil.copyfileobj(source, stream)
            for record in sorted(records, key=lambda r: r["name"].count("/"), reverse=True):
                output = temporary / record["name"]
                if record["kind"] != "symlink":
                    os.chmod(output, record["mode"])
                os.utime(output, ns=(record["mtime_ns"], record["mtime_ns"]), follow_symlinks=False)
                if _describe(output, record["name"]) != record:
                    raise ArchiveError(f"Restored member differs: {record['name']}")
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(temporary, destination)
        _sync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination
