"""Consistent SQLite backup and retained-record hashes; no inference or SQL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
from common import *

p=argparse.ArgumentParser();p.add_argument('label');args=p.parse_args()
out=ROOT/'snapshots'/args.label;out.mkdir(parents=True,exist_ok=True)
sources=list((SEM/'query/runs').glob('*/observations.sqlite'))+list((BOLT/'runs').glob('*/runtime.sqlite'))


def one(path):
    kind='semanticopt' if path.name=='observations.sqlite' else 'bolt'
    target=out/kind/path.parent.name/path.name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise RuntimeError('Refuse to overwrite snapshot '+str(target))
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30) as source:
        with sqlite3.connect(target) as dest:source.backup(dest)
    with sqlite3.connect(target) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        digest=hashlib.sha256()
        if kind=='semanticopt':
            rows=db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
            assert [r[0] for r in rows]==list(range(len(rows)))
            for n,v in rows:digest.update(str(n).encode()+b'\0'+v.encode()+b'\0')
            uncertain=dict(db.execute("SELECT key,value FROM state WHERE key IN ('in_flight','pending')").fetchall())
            metadata=dict(records=len(rows),record_prefix_sha256=digest.hexdigest(),
                          pending_keys=[k for k,v in uncertain.items() if json.loads(v) is not None])
        else:
            rows=db.execute('SELECT key,value FROM kv ORDER BY key').fetchall()
            for k,v in rows:digest.update(k.encode()+b'\0'+v+b'\0')
            metadata=dict(kv_entries=len(rows),kv_sha256=digest.hexdigest())
    return dict(kind=kind,task=path.parent.name,source=str(path),backup=str(target),sha256=sha(target),**metadata)

with ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(one,sources))
atomic(out/'manifest.json',dict(at=time.time(),records=result))
print(json.dumps(dict(backups=len(result),label=args.label,semanticopt=sum(r['kind']=='semanticopt' for r in result),bolt=sum(r['kind']=='bolt' for r in result))))
