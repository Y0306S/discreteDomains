#!/usr/bin/env python3
"""Prepare, publish, verify and restore exact final-model bundles on Hugging Face.

Preparation uses local symlinks for immutable files and streams SQLite payloads.
No credentials are accepted as arguments or written to the manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from model_catalog import REPO, digest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def safe_child(root, relative):
    p = Path(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    # Do not resolve the final component: prepared weight files are symlinks.
    target = root/p
    if not target.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {relative}')
    return target


def export_bank(model, folder):
    """Read each immutable blob in chunks, never load the entire bank into RAM."""
    source = Path(model['source'])
    files = []
    with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as conn:
        for row in model['rows']:
            target = safe_child(folder, row['name'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == row['bytes'] and digest(target) == row['sha256']:
                pass
            else:
                identity = conn.execute('SELECT rowid,metadata,sha256,length(payload) FROM models WHERE task=?', (row['task'],)).fetchone()
                if identity is None or identity[1:] != (row['metadata'], row['sha256'], row['bytes']):
                    raise RuntimeError(f"Source bank changed: {row['task']}")
                temporary = target.with_suffix('.partial')
                h = hashlib.sha256()
                with conn.blobopen('models', 'payload', identity[0], readonly=True) as blob, temporary.open('wb') as stream:
                    while block := blob.read(8 * 1024 * 1024):
                        h.update(block)
                        stream.write(block)
                if h.hexdigest() != row['sha256'] or temporary.stat().st_size != row['bytes']:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"Source bank digest mismatch: {row['task']}")
                temporary.replace(target)
            files.append(dict(name=row['name'], bytes=row['bytes'], sha256=row['sha256']))
            print(f"{model['id']}: exported {row['task']}", flush=True)
    return files


def prepare(model, stage, export_rgpe=False):
    folder = stage/model['id']
    folder.mkdir(parents=True, exist_ok=True)
    if model['mode'] == 'sqlite':
        if not export_rgpe:
            print(f"DEFERRED {model['id']}: use --export-rgpe to export {len(model['rows'])} source GPs.")
            return
        files = export_bank(model, folder)
    else:
        files = []
        for item in model['files']:
            source = Path(item['source'])
            stat = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
                raise RuntimeError(f'Source changed since catalog hashing: {source}')
            target = safe_child(folder, item['name'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != source.resolve():
                    raise FileExistsError(target)
            else:
                target.symlink_to(source)
            files.append(item)
    manifest = dict(schema_version=1, model=model, files=files)
    write_json(folder/'payload-manifest.json', manifest)
    # Preserve an upstream README rather than overwrite an immutable source link.
    card = folder/'README.md'
    if not card.exists():
        card.write_text(f"---\ntags:\n- discreteDomains\n- reproducibility\n---\n\n# {model['id']}\n\n"
                        f"{model['note']}\n\nExact selected final artifacts for the {model['domain']} domain. "
                        "Code and original licenses: https://github.com/Y0306S/discreteDomains .\n\n"
                        f"Dependencies: {', '.join(model['dependencies']) or 'See original campaign recipe'}.\n\n"
                        "`payload-manifest.json` records original paths, sizes, SHA256 and restoration metadata. "
                        "Use migration/hf_models.py to verify and restore. No optimizer or intermediate checkpoints are included. "
                        "Scientific datasets, SQL databases and resume journals are transferred separately. "
                        "Original third-party license terms continue to apply.\n")
    print(f"PREPARED {model['id']}: {len(files)} files, {sum(f['bytes'] for f in files):,} bytes", flush=True)


def verify_payload(folder, manifest):
    for f in manifest['files']:
        p = safe_child(folder, f['name'])
        if not p.is_file() or p.stat().st_size != f['bytes'] or digest(p) != f['sha256']:
            raise RuntimeError(f"Payload verification failed: {f['name']}")


def verify_remote(api, repo_id, revision, manifest):
    entries = {x.rfilename: x for x in api.model_info(repo_id, revision=revision, files_metadata=True).siblings}
    for f in manifest['files']:
        remote = entries.get(f['name'])
        if remote is None or remote.size != f['bytes']:
            raise RuntimeError(f"Remote missing or size mismatch: {f['name']}")
        if remote.lfs:
            if remote.lfs.sha256 != f['sha256']:
                raise RuntimeError(f"Remote LFS digest mismatch: {f['name']}")
        else:
            # Small Git objects use SHA1(blob header + contents), not file SHA256.
            from huggingface_hub import hf_hub_download
            p = Path(hf_hub_download(repo_id, f['name'], revision=revision))
            if digest(p) != f['sha256']:
                raise RuntimeError(f"Remote small-file digest mismatch: {f['name']}")
    from huggingface_hub import hf_hub_download
    remote_manifest = Path(hf_hub_download(repo_id, 'payload-manifest.json', revision=revision))
    if json.loads(remote_manifest.read_text()) != manifest:
        raise RuntimeError('Remote provenance manifest differs')


def upload(folder, namespace, receipts):
    from huggingface_hub import HfApi
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', namespace):
        raise ValueError('Supply a Hugging Face user or organization namespace')
    manifest = json.loads((folder/'payload-manifest.json').read_text())
    verify_payload(folder, manifest)
    api = HfApi()
    api.whoami()  # Uses hf auth login / HF_TOKEN; never log the token.
    repo_id = f"{namespace}/discrete-domains-{manifest['model']['id']}"
    try:
        existing = api.model_info(repo_id)
    except Exception as exc:
        from huggingface_hub.errors import RepositoryNotFoundError
        if not isinstance(exc, RepositoryNotFoundError):
            raise
        api.create_repo(repo_id, repo_type='model', private=True, exist_ok=False)
    else:
        if not existing.private:
            raise RuntimeError(f'Refusing to upload into existing public repository {repo_id}; choose a private destination.')
    allowed = [f['name'] for f in manifest['files']] + ['README.md', 'payload-manifest.json']
    result = api.upload_folder(repo_id=repo_id, folder_path=str(folder),
                               allow_patterns=allowed, ignore_patterns=['**/*.partial', '.cache/**'],
                               commit_message='Preserve exact final discrete-domain experiment models')
    revision = result.oid
    verify_remote(api, repo_id, revision, manifest)
    receipt = dict(id=manifest['model']['id'], repo_id=repo_id, revision=revision,
                   private=True, verified=True, bytes=sum(f['bytes'] for f in manifest['files']))
    receipts.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(receipts.read_text()) if receipts.exists() else []
    previous = [r for r in previous if r['id'] != receipt['id']] + [receipt]
    write_json(receipts, previous)
    print(f"UPLOADED AND VERIFIED {repo_id}@{revision}", flush=True)


def restore_bank(folder, manifest, target):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.restoring')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with sqlite3.connect(temporary) as conn:
            conn.execute('CREATE TABLE models(task TEXT PRIMARY KEY,metadata TEXT NOT NULL,sha256 TEXT NOT NULL,payload BLOB NOT NULL)')
            for row in manifest['model']['rows']:
                cur = conn.execute('INSERT INTO models VALUES(?,?,?,zeroblob(?))',
                                   (row['task'], row['metadata'], row['sha256'], row['bytes']))
                with conn.blobopen('models', 'payload', cur.lastrowid) as blob, safe_child(folder, row['name']).open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        blob.write(block)
            conn.commit()
            if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Restored SQLite bank failed quick_check')
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def restore(folder, destination, apply=False):
    manifest = json.loads((folder/'payload-manifest.json').read_text())
    model = manifest['model']
    if model['mode'] == 'sqlite':
        target = safe_child(destination, model['source'].lstrip('/'))
        print(f"RESTORE {len(model['rows'])} RGPE models -> {target}")
        if apply:
            verify_payload(folder, manifest)
            restore_bank(folder, manifest, target)
        return
    targets = [(f, safe_child(destination, f['source'].lstrip('/'))) for f in manifest['files']]
    for f, target in targets:
        print(f"RESTORE {f['name']} -> {target}")
        if target.exists() and (target.stat().st_size != f['bytes'] or digest(target) != f['sha256']):
            raise FileExistsError(f'Conflicting destination: {target}')
    if apply:
        verify_payload(folder, manifest)
        for f, target in targets:
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=target.name+'.', dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as stream, safe_child(folder, f['name']).open('rb') as source:
                    shutil.copyfileobj(source, stream, 8 * 1024 * 1024)
                if digest(temporary) != f['sha256']:
                    raise RuntimeError(f'Copy verification failed: {target}')
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--catalog', type=Path, default=REPO/'migration/inventory/final-models.json')
    prep.add_argument('--stage', type=Path, default=Path('/root/data/discreteDomains-final-models/hf'))
    prep.add_argument('--id', action='append')
    prep.add_argument('--export-rgpe', action='store_true')
    up = sub.add_parser('upload')
    up.add_argument('--stage', type=Path, default=Path('/root/data/discreteDomains-final-models/hf'))
    up.add_argument('--namespace', required=True)
    up.add_argument('--catalog', type=Path, default=REPO/'migration/inventory/final-models.json')
    up.add_argument('--id', action='append')
    up.add_argument('--receipts', type=Path, default=Path('/root/data/discreteDomains-final-models/hf-publication.json'))
    res = sub.add_parser('restore')
    res.add_argument('--folder', type=Path)
    res.add_argument('--repo-id')
    res.add_argument('--revision', help='Required for downloads: use verified publication commit SHA.')
    res.add_argument('--destination', type=Path, required=True)
    res.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        models = json.loads(args.catalog.read_text())['models']
        if args.id and set(args.id) - {m['id'] for m in models}:
            parser.error('Unknown model id')
        for m in models:
            if not args.id or m['id'] in args.id:
                prepare(m, args.stage, args.export_rgpe)
    elif args.action == 'upload':
        folders = sorted(args.stage.glob('*/payload-manifest.json'))
        selected = [p.parent for p in folders if not args.id or p.parent.name in args.id]
        if not selected or (args.id and set(args.id) - {p.name for p in selected}):
            parser.error('Requested prepared bundle missing')
        expected = set(args.id or [m['id'] for m in json.loads(args.catalog.read_text())['models']])
        missing = expected - {p.name for p in selected}
        if missing:
            parser.error('Final-model bundles not prepared: '+', '.join(sorted(missing)))
        for folder in selected:
            upload(folder, args.namespace, args.receipts)
    else:
        if bool(args.folder) == bool(args.repo_id):
            parser.error('Choose exactly one of --folder or --repo-id')
        if args.repo_id:
            if not args.revision or not re.fullmatch('[a-f0-9]{40}', args.revision):
                parser.error('Use a full immutable --revision commit SHA')
            from huggingface_hub import snapshot_download
            folder = Path(snapshot_download(args.repo_id, revision=args.revision))
        else:
            folder = args.folder
        restore(folder, args.destination, args.apply)


if __name__ == '__main__':
    main()
