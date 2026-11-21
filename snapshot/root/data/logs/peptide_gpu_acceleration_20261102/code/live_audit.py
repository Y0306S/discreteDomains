"""Read-only health, allocation, batch cadence and checkpoint continuation audit."""
import json,os,subprocess,sys,time,urllib.request,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
C=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')
F=Path('/root/data/logs/peptide_batch50_gpu_test200_20261102_v1/fsbo')
NAMES={'rgpe_launch.py','rgpe_worker.py','fsbo_launch.py','fsbo_dispatch.py','rgpe_dispatch.py','repair_launch.py','repair_worker.py','peptide_rgpe_merged_launch.py','peptide_rgpe_merged_worker.py'}
def local():
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   a=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
   if not a or 'python' not in Path(a[0]).name:continue
   script=next((Path(s).name for s in a[1:] if s.endswith('.py')),None)
   if script not in NAMES:continue
   stat=(p/'stat').read_text().rsplit(')',1)[1].split()
   if stat[0]=='Z':continue
   rows.append(dict(pid=int(p.name),script=script,ppid=int(stat[1]),cpus=sorted(os.sched_getaffinity(int(p.name)))))
  except (OSError,ValueError):pass
 return rows
if '--local' in sys.argv:print(json.dumps(local()));raise SystemExit(0)
remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','LLMOptimizer4','/usr/bin/python3',str(Path(__file__)),'--local'],capture_output=True,text=True,check=True,timeout=30)
report=dict(at=time.time(),hosts={},fsbo={},rgpe={},gpu_services=[])
tasks=[]
for host,ps in [('LLMOptimizer2',local()),('LLMOptimizer4',json.loads(remote.stdout))]:
 d=json.loads((C/'dispatch'/(host+'.json')).read_text());tasks.extend(r['task'] for r in d['active'])
 model=[v for v in ps if v['script']=='rgpe_worker.py'];coord=[v for v in ps if v['script']=='rgpe_launch.py'];fsbo=[v for v in ps if v['script']=='fsbo_launch.py']
 old=[v for v in ps if v['script'] in ('repair_launch.py','repair_worker.py','peptide_rgpe_merged_launch.py','peptide_rgpe_merged_worker.py')]
 overlap=[dict(fsbo=f['pid'],rgpe=r['pid'],cpus=sorted(set(f['cpus'])&set(r['cpus']))) for f in fsbo for r in coord if set(f['cpus'])&set(r['cpus'])]
 errors=[]
 for rec in d['active']:
  if 'Traceback (most recent call last)' in Path(rec['log']).read_text(errors='replace')[-20000:]:errors.append(rec['task'])
 report['hosts'][host]=dict(rgpe_coordinators=len(coord),rgpe_models=len(model),old_cpu_workers=len(old),fsbo_coordinators=len(fsbo),dispatch_active=len(d['active']),dispatch_target=d['max_active'],heartbeat_age=time.time()-d['at'],cpu_used=d['cgroup_used_cores'],cpu_quota=d['quota_cores'],memory_gib=d['memory_gib'],cpu_overlaps=overlap,tracebacks=errors)
stages=collections.Counter();workers=[]
for task in tasks:
 p=C/'runs'/task/'rgpe_gpu_stage.json';s=json.loads(p.read_text()) if p.exists() else {};stages[s.get('stage','initializing')]+=1
 workers.append(dict(task=task,stage=s.get('stage','initializing'),source_models_completed=s.get('source_models_completed'),endpoint=s.get('endpoint')))
progress=[json.loads(p.read_text()) for p in (C/'runs').glob('*/progress.json')]
report['rgpe']=dict(active=len(tasks),unique_tasks=len(tasks)==len(set(tasks)),stages=dict(stages),workers=workers,total_proposals=sum(d['proposals'] for d in progress),completed=len(list((C/'runs').glob('*.tar.gz'))),paused=(C/'dispatch/pause.json').exists())
fs=[];bad=[]
for p in (F/'runs').glob('*/progress.json'):
 d=json.loads(p.read_text());fs.append(dict(task=p.parent.name,rounds=d['rounds'],proposals=d['proposals'],fresh=d['fresh_evaluations']))
 h=json.loads(p.with_name('history.json').read_text());counts=collections.Counter(v['round'] for v in h['observations'] if v.get('phase')=='acquisition')
 if any(v!=50 for v in counts.values()):bad.append(dict(task=p.parent.name,counts=dict(counts)))
report['fsbo']=dict(batch_size=50,completed=len(list((F/'runs').glob('*.tar.gz'))),workers=fs,non50_acquisition_batches=bad,paused=(F/'dispatch/pause.json').exists())
for port in [18941,18942,18943]:
 with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=5) as r:d=json.load(r)
 report['gpu_services'].append({k:d[k] for k in ['pid','gpu','calls','errors','busy']})
report['passed']=len(tasks)==53 and report['rgpe']['unique_tasks'] and not report['rgpe']['paused'] and not report['fsbo']['paused'] and not bad and all(not d['tracebacks'] and not d['cpu_overlaps'] and not d['old_cpu_workers'] and d['heartbeat_age']<20 and d['rgpe_coordinators']==d['dispatch_active'] and d['rgpe_models']==d['dispatch_active'] for d in report['hosts'].values()) and all(d['errors']==0 for d in report['gpu_services'])
(ROOT/'live_audit.json').write_text(json.dumps(report,indent=2)+'\n')
brief=json.loads(json.dumps(report));brief['rgpe'].pop('workers');print(json.dumps(brief,indent=2))
