"""Remote persistent ten-minute scheduler; forced SSH key can only invoke tick."""
import fcntl,json,os,subprocess,time
from pathlib import Path
OPS=Path(__file__).resolve().parent
LOOP='query-training-hourly-20261112'
def atomic(path,value):
 p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(path)
def main():
 lock=(OPS/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 record=dict(pid=os.getpid(),host='LLMOptimizer4',start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],at=time.time())
 atomic(OPS/'watchdog.process.json',record);start=time.monotonic();tick=0
 while True:
  cfg=json.loads((OPS/'config.json').read_text());interval=cfg['interval_seconds']
  if not cfg['enabled'] or (OPS/'STOP').exists():
   atomic(OPS/'state.json',dict(loop_id=LOOP,state='stopped',process=record,at=time.time(),next_due=None));return
  at=time.time();previous=json.loads((OPS/'latest.json').read_text()) if (OPS/'latest.json').exists() else None
  cmd=['ssh','-i','/root/.ssh/query_recovery_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/root/.ssh/query_recovery_known_hosts','-p',str(cfg['target_port']),'root@'+cfg['target_host'],'tick']
  atomic(OPS/'state.json',dict(loop_id=LOOP,state='checking',process=record,last_start=at,next_due=None,interval_seconds=interval,automatic_restarts=True,mechanism='independent persistent Python scheduler on LLMOptimizer4; restricted SSH invokes checkpoint recovery on LLMOptimizer2',chat_notifications=False))
  try:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=1500)
   if p.returncode:result=dict(result='target_unreachable_or_command_failed',returncode=p.returncode,error=p.stderr[-4000:],at=time.time())
   else:result=json.loads(p.stdout.strip().splitlines()[-1])
  except Exception as exc:result=dict(result='watchdog_check_failed',error=repr(exc),at=time.time())
  result.update(check_started=at,check_finished=time.time(),watchdog_host='LLMOptimizer4')
  if previous and isinstance(previous.get('counted_measurements'),int) and isinstance(result.get('counted_measurements'),int):result['measurement_delta']=result['counted_measurements']-previous['counted_measurements']
  atomic(OPS/'latest.json',result)
  with (OPS/'history.jsonl').open('a') as log:log.write(json.dumps(result)+'\n');log.flush();os.fsync(log.fileno())
  tick=max(tick+1,int((time.monotonic()-start)//interval)+1);delay=max(0,start+tick*interval-time.monotonic())
  atomic(OPS/'state.json',dict(loop_id=LOOP,state='active',process=record,last_start=at,last_finish=time.time(),last_result=result['result'],next_due=time.time()+delay,interval_seconds=interval,automatic_restarts=True,mechanism='independent persistent Python scheduler on LLMOptimizer4; restricted SSH invokes checkpoint recovery on LLMOptimizer2',chat_notifications=False))
  # Short sleeps make STOP/config changes responsive without another check.
  due=time.monotonic()+delay
  while time.monotonic()<due:
   cfg=json.loads((OPS/'config.json').read_text())
   if not cfg['enabled'] or (OPS/'STOP').exists():break
   time.sleep(min(10,max(0,due-time.monotonic())))
if __name__=='__main__':main()
