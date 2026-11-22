#!/usr/bin/env python3
"""Export installed environment metadata without importing torch or using GPUs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    'bolt_clean_venv': '/root/data/cys/envs/bolt_clean_venv',
    'openbox_qwen_eval': '/root/data/cys/envs/openbox_qwen_eval',
    'llamafactory_qwen35': '/root/data/cys/blockdata_archive/miniconda3/envs/llamafactory_qwen35',
    'bl_fsbo_exact': '/root/data/cys/baseline/envs/bl_fsbo_exact',
    'bl_lolbo_exact': '/root/data/cys/baseline/envs/bl_lolbo_exact',
    'bl_bora': '/root/data/cys/baseline/envs/bl_bora',
    'bl_rgpe': '/root/data/cys/baseline/envs/bl_rgpe',
    'bl_semanticopt': '/root/data/cys/baseline/envs/bl_semanticopt',
    'taskset_qwen35_fast_20261030': '/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030',
    'bl_tosfit': '/root/data/cys/baseline/envs/bl_tosfit',
    'bl_optformer': '/root/data/cys/baseline/envs/bl_optformer',
    'bl_lolbo_harness': '/root/data/cys/baseline/envs/bl_lolbo_harness',
    'bl_lolbo_harness_cu118': '/root/data/cys/baseline/envs/bl_lolbo_harness_cu118',
    'bolt_qp': '/root/data/cys/envs/bolt_qp',
}
PROBE = r'''
import json,platform,sys
try:
    import importlib.metadata as m
except ImportError:
    import importlib_metadata as m
packages=[]
for d in m.distributions():
    packages.append(dict(name=d.metadata['Name'],version=d.version,
        installer=(d.read_text('INSTALLER') or '').strip(),
        direct_url=json.loads(d.read_text('direct_url.json') or 'null')))
print(json.dumps(dict(python=sys.version,executable=sys.executable,
    prefix=sys.prefix,base_prefix=sys.base_prefix,platform=platform.platform(),packages=packages)))
'''


def export(item):
    name, value = item
    prefix = Path(value)
    out = REPO/'environments'/name
    out.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([str(prefix/'bin/python'), '-c', PROBE], text=True,
                            capture_output=True, timeout=180)
    if result.returncode:
        return {'name': name, 'prefix': value, 'error': result.stderr[-500:]}
    info = json.loads(result.stdout)
    info['original_prefix'] = value
    info['resolved_prefix'] = str(prefix.resolve())
    # Sort and deduplicate inherited distributions, preferring the first
    # visible distribution (the overlay venv precedes its base environment).
    packages = {}
    for p in info['packages']:
        if p['name']:
            packages.setdefault(re.sub(r'[-_.]+', '-', p['name'].lower()), p)
    info['packages'] = [packages[k] for k in sorted(packages)]
    (out/'installed.json').write_text(json.dumps(info, indent=2)+'\n')
    (out/'requirements.txt').write_text(''.join(f'{p["name"]}=={p["version"]}\n' for p in info['packages']))
    pip_packages = [p for p in info['packages'] if p['installer']=='pip' and not p['direct_url']]
    (out/'pip-only.txt').write_text(''.join(f'{p["name"]}=={p["version"]}\n' for p in pip_packages))
    local = [p for p in info['packages'] if p['direct_url']]
    (out/'direct-installs.json').write_text(json.dumps(local, indent=2)+'\n')
    if (prefix/'pyvenv.cfg').exists():
        (out/'pyvenv.cfg').write_text((prefix/'pyvenv.cfg').read_text())
    meta = prefix/'conda-meta'
    if meta.exists():
        records = [json.loads(p.read_text()) for p in sorted(meta.glob('*.json'))]
        urls = [p['url'] for p in records if p.get('url')]
        (out/'conda-explicit.txt').write_text('# platform: linux-64\n@EXPLICIT\n'+'\n'.join(urls)+'\n')
        (out/'conda-packages.json').write_text(json.dumps([
            {k:p[k] for k in ['name','version','build','channel','url','md5','sha256'] if k in p}
            for p in records], indent=2)+'\n')
    return {'name': name, 'prefix': value, 'python': info['python'],
            'packages': len(info['packages']), 'conda': meta.exists(),
            'base_prefix': info['base_prefix']}


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(export, ENVIRONMENTS.items()))
    # installed.json includes inherited distributions for --system-site-packages
    # overlays. Only local overrides belong in a specification rebuild.
    for record in records:
        if record.get('error') or record['conda']:
            continue
        parent = next((r for r in records if r['prefix'] == record['base_prefix']), None)
        if not parent or parent.get('error'):
            continue
        child_dir = REPO/'environments'/record['name']
        child = json.loads((child_dir/'installed.json').read_text())
        base = json.loads((REPO/'environments'/parent['name']/'installed.json').read_text())
        inherited = {p['name'].lower(): p['version'] for p in base['packages']}
        delta = [p for p in child['packages'] if p['installer'] == 'pip'
                 and not p['direct_url']
                 and inherited.get(p['name'].lower()) != p['version']]
        (child_dir/'overlay-only.txt').write_text(
            ''.join(f'{p["name"]}=={p["version"]}\n' for p in delta))
    (REPO/'environments/index.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
