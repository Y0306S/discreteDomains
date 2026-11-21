"""Persist minute-level evidence on each host; never restart or replay work."""
import argparse,fcntl,json,os,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(path):
 try:return Path(path).read_text().strip()
 except OSError:return None

def sample(host):
 result=dict(at=time.time(),host=host,pid1_stat=read('/proc/1/stat'),boot_id=read('/proc/sys/kernel/random/boot_id'),loadavg=read('/proc/loadavg'))
 result['cgroup']={name:read('/sys/fs/cgroup/'+name) for name in ['memory.current','memory.max','memory.peak','memory.events','memory.stat','cpu.stat','memory.pressure','io.pressure']}
 result['processes']={}
 for p in (ROOT/'services').glob('*.process.json'):
  try:
   saved=json.loads(p.read_text())
   if saved.get('host','LLMOptimizer2')!=host:continue
   stat=read(f"/proc/{saved['pid']}/stat");fields=stat.rsplit(') ',1)[1].split() if stat else []
   result['processes'][p.name]=dict(pid=saved['pid'],alive=bool(fields and fields[0]!='Z' and saved.get('start_ticks')==fields[19]))
  except (ValueError,KeyError,IndexError):continue
 result['status']=json.loads((ROOT/'status.json').read_text())
 fs=os.statvfs(ROOT);result['disk_free_bytes']=fs.f_bavail*fs.f_frsize;result['disk_free_inodes']=fs.f_favail
 if host=='LLMOptimizer2':
  result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=15).strip()
 else:
  try:
   active=json.loads((ROOT/'services/active_inference_stage.json').read_text());port=21810 if active['stage']=='warm30' else 21811
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=15) as response:h=json.load(response)
   result['local_inference_transport']=dict(ready=h.get('ready'),queued=h.get('queued'),replicas=h.get('replicas'))
  except Exception as exc:result['local_inference_transport']=dict(ready=False,error=repr(exc))
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--host',required=True,choices=['LLMOptimizer2','LLMOptimizer4']);a=parser.parse_args()
 lock=(ROOT/'services'/f'diagnostics.{a.host}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
 (ROOT/'services'/f'diagnostics.{a.host}.process.json').write_text(json.dumps(dict(pid=os.getpid(),host=a.host,start_ticks=stat[19],at=time.time())))
 start=time.monotonic();tick=0
 while True:
  try:result=sample(a.host)
  except Exception as exc:result=dict(host=a.host,at=time.time(),diagnostic_error=repr(exc))
  data=json.dumps(result,allow_nan=False)
  with (ROOT/'logs'/f'diagnostics.{a.host}.jsonl').open('a') as log:log.write(data+'\n');log.flush();os.fsync(log.fileno())
  p=ROOT/'services'/f'diagnostics.{a.host}.latest.json';temp=p.with_suffix('.tmp');temp.write_text(data);temp.replace(p)
  tick=max(tick+1,int((time.monotonic()-start)//60)+1);time.sleep(max(0,start+tick*60-time.monotonic()))

if __name__=='__main__':main()
