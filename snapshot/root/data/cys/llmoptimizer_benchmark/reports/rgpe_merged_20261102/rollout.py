"""Resume a completely drained queue with a pinned acquisition-only amendment.

This refuses live legacy workers, changed recipes, or incomplete validation.
It never signals or kills a worker and never changes optimizer state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT.parents[1]/'tools'
CAMPAIGN = Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')
DISPATCH = CAMPAIGN/'dispatch'
ARCHIVE = DISPATCH/'rollout_20261102_merged'
HOSTS = ('LLMOptimizer2', 'LLMOptimizer4')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def remote(host, script, *args):
    command=['/usr/bin/python3','-']+list(args)
    if host == 'LLMOptimizer4': command=['ssh','-o','BatchMode=yes',host]+command
    result=subprocess.run(command,input=script,text=True,capture_output=True,check=True,timeout=30)
    return json.loads(result.stdout)


SCAN = '''import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
parent=json.loads((root/'recipe.json').read_text())['parent_recipe']
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        args=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':continue
        legacy=any(a.endswith('/peptide_local_bo.py') or a.endswith('/peptide_local_dispatch.py') for a in args)
        model=any(a.endswith('/peptide_transfer_models.py') for a in args) and parent in args
        if model or (legacy and (str(root) in args or str(root/'dispatch/plan.json') in args)):
            rows.append(dict(pid=int(p.name),args=args))
    except (OSError,ValueError):pass
print(json.dumps(dict(host=os.uname().nodename,legacy=rows)))
'''

LAUNCH = '''import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);host=sys.argv[2];tools=Path(sys.argv[3])
command=[sys.executable,'-u',str(tools/'peptide_rgpe_merged_dispatch.py'),
         '--plan',str(root/'dispatch/plan.json'),'--host',host,'--execute']
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
with (root/'dispatch'/(host+'.log')).open('ab') as log:
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,env=env)
record=dict(pid=child.pid,command=command,at=time.time())
(root/'dispatch'/(host+'.launch.json')).write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    report=json.loads((ROOT/'validation.json').read_text())
    assert report['passed'] and report['all_800_sources_passed'] and report['launcher_passed']
    for path, expected in report['code_sha256'].items(): assert digest(path)==expected
    recipe_hash=digest(CAMPAIGN/'recipe.json')
    states={host:remote(host,SCAN,str(CAMPAIGN)) for host in HOSTS}
    active={host:json.loads((DISPATCH/(host+'.json')).read_text())['active'] for host in HOSTS}
    assert all(not state['legacy'] for state in states.values()),states
    assert all(not group for group in active.values()),active
    assert (DISPATCH/'stop.json').exists() and not (DISPATCH/'pause.json').exists()
    plan=json.loads((DISPATCH/'plan.json').read_text())
    assert recipe_hash==plan['recipe_sha256']==report['recipe_sha256']
    assert not ARCHIVE.exists(), 'This rollout has already started; inspect it before recovery'
    if not args.execute:
        print(json.dumps(dict(ready=True,hosts=states,recipe_sha256=recipe_hash)));return
    ARCHIVE.mkdir()
    # Back up durable task state only after coordinators have exited. Never
    # unpickle/rewrite state, pending requests, cached scores, or RNG here.
    checkpoints=[]
    for source in sorted((CAMPAIGN/'runs').glob('*/runtime.sqlite')):
        destination=ARCHIVE/'checkpoints'/source.parent.name/'runtime.sqlite'
        destination.parent.mkdir(parents=True)
        with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as db:
            assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
            values={key:digest_bytes(value) for key,value in db.execute('SELECT key,value FROM meta')}
            with sqlite3.connect(destination) as backup:db.backup(backup)
        checkpoints.append(dict(task=source.parent.name,meta_sha256=values,
            backup=str(destination),backup_sha256=digest(destination)))
    for name in ['plan.json','stop.json']+[host+suffix for host in HOSTS for suffix in ('.json','.launch.json')]:
        (ARCHIVE/name).write_bytes((DISPATCH/name).read_bytes())
    amendment=dict(format='rgpe-merged-acquisition-v1',campaign=str(CAMPAIGN),
        recipe_sha256=recipe_hash,code=report['code_sha256'],
        validation=dict(path=str(ROOT/'validation.json'),sha256=digest(ROOT/'validation.json')))
    amendment_path=CAMPAIGN/'rgpe_acquisition_amendment.json'
    assert not amendment_path.exists()
    write(amendment_path,amendment)
    plan.update(format='peptide-rgpe-merged-dispatch-v1',at=time.time(),
        dispatcher_sha256=digest(TOOLS/'peptide_rgpe_merged_dispatch.py'),
        acquisition_amendment=str(amendment_path),acquisition_amendment_sha256=digest(amendment_path),
        previous_plan_sha256=digest(ARCHIVE/'plan.json'))
    write(DISPATCH/'plan.json',plan)
    sys.path.insert(0,str(TOOLS))
    from peptide_rgpe_merged_policy import verify
    verify(amendment_path)
    dryruns={}
    for host in HOSTS:
        command=['/usr/bin/python3',str(TOOLS/'peptide_rgpe_merged_dispatch.py'),
                 '--plan',str(DISPATCH/'plan.json'),'--host',host]
        if host=='LLMOptimizer4':command=['ssh','-o','BatchMode=yes',host]+command
        result=subprocess.run(command,text=True,capture_output=True,check=True,timeout=30)
        dryruns[host]=json.loads(result.stdout)
    for host in HOSTS:(DISPATCH/(host+'.json')).unlink()
    (DISPATCH/'stop.json').unlink()
    launched={host:remote(host,LAUNCH,str(CAMPAIGN),host,str(TOOLS)) for host in HOSTS}
    write(ROOT/'rollout.json',dict(at=time.time(),recipe_sha256=recipe_hash,
        plan_sha256=digest(DISPATCH/'plan.json'),amendment_sha256=digest(amendment_path),
        checkpoints=checkpoints,dryruns=dryruns,launched=launched,legacy_workers_before=states,
        original_state_untouched=True,original_recipe_untouched=True,vae_updates=False))
    print(json.dumps(dict(launched=launched,backed_up_tasks=len(checkpoints),recipe_unchanged=True)))


def digest_bytes(value):return hashlib.sha256(value).hexdigest()


if __name__=='__main__':main()
