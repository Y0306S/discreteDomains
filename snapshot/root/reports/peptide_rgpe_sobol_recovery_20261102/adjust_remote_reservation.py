import json,os,signal,subprocess,sys,time
from pathlib import Path
R=Path('/root/reports/peptide_rgpe_sobol_recovery_20261102')
sys.path.insert(0,str(R/'code'));import repair_dispatch as d
from sobol_repair import digest,verify
verify(R/'manifest.json')
C=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')
plan=C/'dispatch/sobol_repair_plan.json';status=C/'dispatch/LLMOptimizer4.json'
s=json.loads(status.read_text());ps=d.processes();pid=s['pid']
assert os.uname().nodename=='cs-fa498-8793f-server'
assert str(R/'code/repair_dispatch.py') in ps[pid]['args']
assert s['observed_worker_peak_gib']*1.75<7
assert not (C/'dispatch/pause.json').exists()
os.kill(pid,signal.SIGSTOP)
try:
 s=json.loads(status.read_text());assert s['pid']==pid
 old=plan.read_bytes();oldsha=digest(plan);new=json.loads(old)
 assert new['hosts']['LLMOptimizer4']['memory_per_worker_gib']==8
 (R/'plan.before_memory_adjustment.json').write_bytes(old)
 new['hosts']['LLMOptimizer4']['memory_per_worker_gib']=7
 new['previous_plan_sha256']=oldsha
 new['memory_revision']=dict(at=time.time(),host='LLMOptimizer4',previous_gib=8,new_gib=7,
   observed_worker_peak_gib=s['observed_worker_peak_gib'],dynamic_peak_multiplier=1.75,headroom_gib=16,
   reason='Admit the remaining two of the original ten failed tasks with measured growth reservation and unchanged 26-worker cap')
 d.atomic_json(plan,new)
 d.atomic_json(R/'LLMOptimizer4.status.before_memory_adjustment.json',s)
 s['plan_sha256']=digest(plan);d.atomic_json(status,s)
 os.kill(pid,signal.SIGKILL)
except Exception:
 os.kill(pid,signal.SIGCONT);raise
command=['/usr/bin/python3','-u',str(R/'code/repair_dispatch.py'),'--plan',str(plan),'--host','LLMOptimizer4','--execute']
with (R/'LLMOptimizer4.dispatcher.log').open('ab') as f:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d.atomic_json(R/'memory_adjustment.json',dict(at=time.time(),old_pid=pid,new_pid=child.pid,old_plan_sha256=oldsha,new_plan_sha256=digest(plan),revision=new['memory_revision']))
d.atomic_json(C/'dispatch/active_plan.json',dict(at=time.time(),path=str(plan),sha256=digest(plan),sobol_repair_manifest=str(R/'manifest.json')))
print(json.dumps(dict(new_pid=child.pid,adopted=len(s['active']),memory_per_worker_gib=7)))
