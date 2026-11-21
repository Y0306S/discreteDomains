#!/usr/bin/env python3
"""Make consistent SQLite backups; no oracle, workload pause or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    campaign=args.campaign.resolve();output=args.output.resolve()
    if output==campaign or campaign in output.parents:
        raise SystemExit('Backups must be outside the campaign.')
    sources=sorted(campaign.glob('*/runs/*/runtime.sqlite'))
    print(f'{len(sources)} runtime journals in {campaign}')
    if not args.apply:
        print('Preview only. Add --apply to use the SQLite online backup API.')
        return
    manifest=[]
    for source in sources:
        target=output/source.relative_to(campaign)
        if target.exists():raise SystemExit('Backup exists: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        # read-only prevents accidental creation/mutation of the live database.
        with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=120) as src:
            with sqlite3.connect(target) as dst:
                src.backup(dst,pages=512,sleep=.1)
                result=dst.execute('PRAGMA quick_check').fetchall()
                if result != [('ok',)]:raise RuntimeError(str(result))
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append(dict(source=str(source),backup=str(target.relative_to(output)),sha256=digest))
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Backed up {len(manifest)} journals. A running campaign may advance after this snapshot.')


if __name__=='__main__':
    main()
