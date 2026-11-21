"""Read-only audit for the repaired RGPE dispatchers and original ten tasks."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path('/root/reports/peptide_rgpe_sobol_recovery_20261102')
C=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')
sys.path.insert(0,str(R/'code'))
import repair_dispatch as dispatch

def local():
 snap=dispatch.processes()
 return {str(pid):{k:row[k] for k in ('args','start','seconds','cpus','rss')} for pid,row in snap.items()
         if any(str(a).endswith(('/repair_launch.py','/repair_worker.py','/sobol_helper.py','/peptide_rgpe_merged_launch.py','/peptide_rgpe_merged_worker.py','/run_taskset_bora.py')) for a in row['args'])}

if '--local' in sys.argv:
 print(json.dumps(local()));raise SystemExit(0)
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','LLMOptimizer4','/usr/bin/python3',str(R/'live_audit.py'),'--local']
remote=json.loads(subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=30).stdout)
processes={'LLMOptimizer2':local(),'LLMOptimizer4':remote}
initial={x['task']:x for x in json.loads((R/'before.json').read_text())['tasks']}
report={'at':time.time(),'hosts':{},'restarts':[],'paused':(C/'dispatch/pause.json').exists()}
all_tasks=[]
for host in processes:
 status=json.loads((C/'dispatch'/(host+'.json')).read_text())
 procs=processes[host]
 report['hosts'][host]={'active':len(status['active']),'status':status['status'],
   'heartbeat_age_seconds':round(time.time()-status['at'],2),
   'repaired_model_workers':sum(any(a.endswith('/repair_worker.py') for a in row['args']) for row in procs.values()),
   'bora_workers':sum(any(a.endswith('/run_taskset_bora.py') for a in row['args']) for row in procs.values()),
   'cpu_used':status['cgroup_used_cores'],'cpu_quota':status['quota_cores'],'memory_gib':status['memory_gib']}
 for rec in status['active']:
  all_tasks.append(rec['task'])
  if rec['task'] not in initial:continue
  row=procs.get(str(rec['pid']));assert row and row['start']==rec['start']
  assert any(a.endswith('/repair_launch.py') for a in rec['command'])
  run=C/'runs'/rec['task'];progress=json.loads((run/'progress.json').read_text())
  worker=run/'sobol_repair.json'
  log=Path(rec['log']).read_text(errors='replace')
  report['restarts'].append({'task':rec['task'],'host':host,'pid':rec['pid'],
    'cpu':rec['cpu'],'cpu_seconds':row['seconds'],'rounds':progress['rounds'],
    'repair_worker_activated':worker.exists(),
    'extended_draw_seen':'[sobol-repair] completed extended QMC draw' in log,
    'traceback':'Traceback (most recent call last)' in log})
assert len(all_tasks)==len(set(all_tasks))
report['all_ten_restarted']=len(report['restarts'])==10
report['all_ten_repair_activated']=report['all_ten_restarted'] and all(x['repair_worker_activated'] for x in report['restarts'])
report['passed']=report['all_ten_repair_activated'] and not report['paused'] and not any(x['traceback'] for x in report['restarts'])
(R/'live_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
