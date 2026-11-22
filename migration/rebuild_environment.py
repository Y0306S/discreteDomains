#!/usr/bin/env python3
"""Rebuild one environment from the exported package specifications."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    entries=json.loads((REPO/'environments/index.json').read_text())
    prefixes={r['name']:Path(r['prefix']) for r in entries}
    prefixes['pg16']=Path('/root/data/cys/envs/pg16')
    if args.name not in prefixes:raise SystemExit('Unknown environment: '+args.name)
    prefix=prefixes[args.name];spec=REPO/'environments'/args.name
    if args.apply and prefix.exists():raise SystemExit('Refusing to replace an existing environment: '+str(prefix))
    commands=[]
    entry=next((r for r in entries if r['name']==args.name), {})
    if entry and not entry.get('conda', True):
        base=Path(entry['base_prefix'])
        if not (base/'bin/python').exists():raise SystemExit('Restore/rebuild parent first: '+str(base))
        commands.append([str(base/'bin/python'),'-m','venv','--system-site-packages',str(prefix)])
    else:
        conda=shutil.which('conda')
        if not conda:raise SystemExit('Install Miniforge/Conda and put conda on PATH first.')
        commands.append([conda,'create','-y','-p',str(prefix),'--file',str(spec/'conda-explicit.txt')])
    python=str(prefix/'bin/python')
    pip_spec = spec/('overlay-only.txt' if entry and not entry.get('conda', True) else 'pip-only.txt')
    if pip_spec.exists() and pip_spec.read_text().strip():
        commands.append([python,'-m','pip','install','--extra-index-url',
                         'https://download.pytorch.org/whl/cu124','-r',str(pip_spec)])
    direct=json.loads((spec/'direct-installs.json').read_text()) if (spec/'direct-installs.json').exists() else []
    for package in direct:
        if package['installer']!='pip':continue
        url=package['direct_url']['url']
        if package['name'].lower()=='llamafactory':
            commands.append([python,'-m','pip','install','--no-deps','-e',
                             '/root/data/cys/LLMOptimizer/LlamaFactory'])
        elif package['direct_url'].get('vcs_info'):
            vcs=package['direct_url']['vcs_info']
            if vcs['vcs']!='git': raise SystemExit('Unsupported VCS: '+vcs['vcs'])
            commands.append([python,'-m','pip','install','--no-deps','git+'+url+'@'+vcs['commit_id']])
        elif url.startswith('file:///') and package['direct_url'].get('archive_info'):
            local=Path(unquote(urlparse(url).path))
            if args.apply and not local.is_file(): raise SystemExit('Restore archive first: '+str(local))
            commands.append([python,'-m','pip','install','--no-deps',str(local)])
        elif url.startswith('file:///root/data/cys/baseline/repos/'):
            local=Path(unquote(urlparse(url).path))
            if args.apply and not local.is_dir(): raise SystemExit('Restore package source first: '+str(local))
            cmd=[python,'-m','pip','install','--no-deps']
            if package['direct_url'].get('dir_info',{}).get('editable'): cmd.append('-e')
            commands.append(cmd+[str(local)])
        elif url.startswith(('http://','https://')):
            commands.append([python,'-m','pip','install','--no-deps',url])
        elif package['name'].lower()=='hebo':
            commands.append([python,'-m','pip','install','--no-deps','HEBO=='+package['version']])
        else:
            raise SystemExit('Restore local package source before rebuilding: '+package['name'])
    for command in commands:print(shlex.join(command))
    if not args.apply:
        print('Preview only. Add --apply on the NEW server to install.')
        return
    for command in commands:subprocess.run(command,check=True)


if __name__=='__main__':
    main()
