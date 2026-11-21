"""Read-only host resource and campaign ownership audit."""
import json,os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'operations/recovery'))
from recovery import probe
host=sys.argv[1]
def cpu():
 return dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
a=cpu();t=time.monotonic();time.sleep(5);b=cpu();elapsed=time.monotonic()-t
p=probe(host)
foreign=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=[x.decode(errors='replace') for x in f.read_bytes().split(b'\0') if x]
  if any(x.startswith(str(root.parent/c)) for x in args for c in ['peptide_bora_test200_20261118','peptide_rgpe_screen200_test200_20261119']):foreign.append(dict(pid=int(f.parent.name),args=args))
 except OSError:pass
out=dict(host=host,at=time.time(),cpu_cores_used=(int(b['usage_usec'])-int(a['usage_usec']))/elapsed/1e6,cpu_throttled_usec=int(b['throttled_usec'])-int(a['throttled_usec']),memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),workers=len(p['workers']),live_workers=sum(w['alive'] for w in p['workers']),databases=len(p['databases']),ready_databases=sum(d['ready'] for d in p['databases']),active_databases=sum(bool(d.get('active')) for d in p['databases']),transport=p['transport'],unrecorded_workers=p['unrecorded_workers'],stopped_campaign_processes=foreign)
(root/'operations/reallocate_20261119'/f'capacity_audit.{host}.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
