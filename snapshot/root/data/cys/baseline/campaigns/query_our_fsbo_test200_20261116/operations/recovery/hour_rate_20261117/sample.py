"""Read-only one-hour campaign throughput measurement."""
from pathlib import Path
import json,os,time,datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
base=read(HERE/'baseline.json')
end=base['started_at']+3600
atomic(HERE/'process.json',dict(pid=os.getpid(),started_at=time.time(),end_at=end,start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]))
def sample():
 now=time.time();s=read(ROOT/'status.json')
 out=dict(at=now,elapsed_seconds=now-base['started_at'],status=s,recovery=read(ROOT/'operations/recovery/local_latest.json'))
 if s:
  out['new_measurements']=s['counted_measurements']-base['status']['counted_measurements']
  out['rate_per_hour']=out['new_measurements']*3600/out['elapsed_seconds'] if out['elapsed_seconds']>0 else None
 atomic(HERE/'latest.json',out)
 with (HERE/'samples.jsonl').open('a') as f:f.write(json.dumps(out)+'\n')
 return out
while True:
 value=sample()
 if time.time()>=end:
  value['remaining_evaluations']=value['status']['measurement_target']-value['status']['counted_measurements']
  rate=value['rate_per_hour']
  value['eta_hours_at_observed_rate']=value['remaining_evaluations']/rate if rate and rate>0 else None
  atomic(HERE/'result.json',value)
  break
 due=min(end,base['started_at']+(int((time.time()-base['started_at'])//60)+1)*60)
 while time.time()<due:time.sleep(min(10,max(0,due-time.time())))
