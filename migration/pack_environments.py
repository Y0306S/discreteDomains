#!/usr/bin/env python3
"""Prepare current environments as separate archives, outside Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--names', nargs='+', default=['openbox_qwen_eval','bolt_clean_venv',
                        'llamafactory_qwen35','bl_fsbo_exact','pg16'])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        raise SystemExit('Choose an archive directory outside this Git repository.')
    records = json.loads((REPO/'environments/index.json').read_text())
    prefixes = {r['name']:r['prefix'] for r in records}
    prefixes['pg16']='/root/data/cys/envs/pg16'
    commands = []
    for name in args.names:
        prefix = Path(prefixes[name])
        archive = output/(name+'.tar.gz')
        if archive.exists():
            raise SystemExit('Archive already exists: '+str(archive))
        if (prefix/'pyvenv.cfg').exists():
            cmd = ['tar','-czf',str(archive),'-C',str(prefix),'.']
            kind = 'venv'
        else:
            executable = shutil.which('conda-pack') or '/root/miniconda3/bin/conda-pack'
            cmd = [executable,'-p',str(prefix),'-o',str(archive),
                   '--ignore-editable-packages','--n-threads','2']
            kind = 'conda'
        commands.append((name,prefix,archive,kind,cmd))
        print(shlex.join(cmd))
    if not args.apply:
        print('Preview only. Add --apply to build the archives; allow tens of GB.')
        return
    output.mkdir(parents=True,exist_ok=True)
    manifest = []
    for name,prefix,archive,kind,cmd in commands:
        subprocess.run(cmd,check=True)
        h=hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):
                h.update(chunk)
        manifest.append(dict(name=name,prefix=str(prefix),kind=kind,
                             archive=archive.name,bytes=archive.stat().st_size,sha256=h.hexdigest()))
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
