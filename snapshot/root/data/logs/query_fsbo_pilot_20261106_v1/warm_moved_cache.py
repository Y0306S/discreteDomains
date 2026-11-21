#!/usr/bin/env python3
"""Warm the two remote private file-backed databases before timing them."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
deadline=time.monotonic()+900
while True:
    state=json.loads((ROOT/'host_LLMOptimizer4.json').read_text())
    if state['status']=='ready' and all(any(w['name']==name and w['status']=='ready' for w in state['workers'])
                                      for name in ['w05','w06']):break
    if time.monotonic()>deadline:raise TimeoutError('Moved workers did not become ready')
    time.sleep(2)
results=[]
for name in ['w05','w06']:
    worker=next(w for w in state['workers'] if w['name']==name)
    data=Path(worker['pgdata'])
    assert data==ROOT/'databases'/name
    removed=[]
    for log in ['pg.log','query_train800_author_20261002.log']:
        file=data/log
        if file.exists():removed.append(dict(path=str(file),bytes=file.stat().st_size));file.unlink()
    started=time.monotonic();total=0
    for file in data.rglob('*'):
        if not file.is_file() or file.is_symlink():continue
        with file.open('rb') as handle:
            while True:
                chunk=handle.read(8*2**20)
                if not chunk:break
                total+=len(chunk)
    result=dict(worker=name,bytes_read=total,seconds=time.monotonic()-started,
                removed_copied_logs=removed,storage='private_file_backed_with_full_cache_warmup')
    results.append(result)
    print(json.dumps(result),flush=True)
(ROOT/'cache_warmup.json').write_text(json.dumps(dict(status='complete',workers=results),indent=2))
