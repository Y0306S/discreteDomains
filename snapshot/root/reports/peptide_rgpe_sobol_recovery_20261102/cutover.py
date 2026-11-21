"""Replace only the dispatch controller, adopting live optimization processes."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
R=Path('/root/reports/peptide_rgpe_sobol_recovery_20261102')
sys.path.insert(0,str(R/'code'))
import repair_dispatch as dispatch
from sobol_repair import verify,digest
p=argparse.ArgumentParser();p.add_argument('--host',required=True);args=p.parse_args()
verify(R/'manifest.json')
C=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')
status=C/'dispatch'/(args.host+'.json');plan=C/'dispatch/sobol_repair_plan.json'
old=json.loads(status.read_text());pid=old['pid'];snap=dispatch.processes()
assert os.uname().nodename==json.loads(plan.read_text())['hosts'][args.host]['hostname']
assert snap[pid]['args'][2]=='/root/data/cys/llmoptimizer_benchmark/tools/peptide_rgpe_merged_dispatch.py',snap[pid]
assert (C/'dispatch/pause.json').exists()
os.kill(pid,signal.SIGSTOP)
try:
    old=json.loads(status.read_text());assert old['pid']==pid
    dispatch.atomic_json(R/(args.host+'.status.before.json'),old)
    snap=dispatch.processes();active=[];lost=[]
    for rec in old['active']:
        row=snap.get(rec['pid'])
        if row and row['start']==rec['start']:
            active.append(rec)
        elif dispatch.recoverable_sobol_exit(rec):
            lost.append(rec)
        elif (C/'runs'/(rec['task']+'.tar.gz')).exists():pass
        else:raise RuntimeError('Unexplained worker exit during cutover: '+rec['task'])
    staged=dict(old,active=active,plan_sha256=digest(plan),status='paused',
                sobol_repair_manifest_sha256=digest(R/'manifest.json'))
    dispatch.atomic_json(status,staged)
    # SIGTERM is intentionally avoided: its handler would stop healthy workers.
    os.kill(pid,signal.SIGKILL)
except Exception:
    os.kill(pid,signal.SIGCONT)
    raise
command=['/usr/bin/python3','-u',str(R/'code/repair_dispatch.py'),'--plan',str(plan),'--host',args.host,'--execute']
log=R/(args.host+'.dispatcher.log')
with log.open('ab') as f:
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
dispatch.atomic_json(R/(args.host+'.cutover.json'),dict(at=time.time(),old_dispatcher=pid,new_dispatcher=child.pid,adopted=len(active),legacy_exits=lost,command=command))
print(json.dumps(dict(host=args.host,pid=child.pid,adopted=len(active),legacy_exits=len(lost))))
