"""Stop only the two user-selected peptide campaigns, preserving durable state."""
import hashlib,json,os,pickle,signal,socket,sys,time
from pathlib import Path

BASE=Path('/root/data/cys/baseline/campaigns')
BORA=BASE/'peptide_bora_test200_20261118'
RGPE=BASE/'peptide_rgpe_screen200_test200_20261119'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BORA/'vendor'))

def inventory():
    result={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
            fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
            result[int(p.name)]=dict(args=args,ppid=int(fields[1]),ticks=fields[19],state=fields[0])
        except (OSError,ValueError):pass
    return result

def send(pid,sig,record):
    current=inventory().get(pid)
    if current and current['ticks']==record['ticks'] and current['args']==record['args']:
        os.kill(pid,sig)

for root in [BORA,RGPE]:
    (root/'STOP').write_text('User requested stop and reallocation to Query ORPO + FSBO.\n')
before=inventory()
controllers={pid:r for pid,r in before.items() if any(a in [str(BORA/'code/scheduler.py'),str(RGPE/'code/dispatch.py')] for a in r['args'])}
for pid,r in controllers.items():send(pid,signal.SIGSTOP,r)
runners={pid:r for pid,r in before.items() if any(a in [str(BORA/'code/worker.py'),str(RGPE/'code/launch.py')] for a in r['args'])}
for pid,r in runners.items():send(pid,signal.SIGSTOP,r)
time.sleep(.3)
audit=[]
for pid,r in runners.items():
    task=r['args'][r['args'].index('--task')+1]
    if str(BORA/'code/worker.py') in r['args']:
        checkpoint=BORA/'runs'/task/'checkpoint.pkl'
        for _ in range(30):
            state=pickle.loads(checkpoint.read_bytes())
            if not state.get('pending') or state['pending']['status']!='in_flight':break
            send(pid,signal.SIGCONT,r);time.sleep(1);send(pid,signal.SIGSTOP,r);time.sleep(.05)
        assert not state.get('pending') or state['pending']['status']!='in_flight',task+' oracle still unresolved'
        audit.append(dict(campaign='BORA',task=task,observations=len(state['rows']),pending=state.get('pending',{}).get('status') if state.get('pending') else None,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    else:
        import sqlite3
        checkpoint=RGPE/'rgpe/runs'/task/'runtime.sqlite'
        for _ in range(30):
            db=sqlite3.connect('file:'+str(checkpoint)+'?mode=ro',uri=True,timeout=10)
            row=db.execute("SELECT value FROM meta WHERE key='pending'").fetchone();pending=pickle.loads(row[0]) if row else None;db.close()
            if pending is None:break
            send(pid,signal.SIGCONT,r);time.sleep(1);send(pid,signal.SIGSTOP,r);time.sleep(.05)
        assert pending is None,task+' pending batch requires drain'
        audit.append(dict(campaign='RGPE',task=task,pending=None,checkpoint=str(checkpoint)))
selected={pid for pid,r in before.items() if any(a.startswith(str(root)+'/') for root in [BORA,RGPE] for a in r['args'])}
while True:
    extra={pid for pid,r in before.items() if r['ppid'] in selected}-selected
    if not extra:break
    selected.update(extra)
assert os.getpid() not in selected
for pid in selected:
    if pid in controllers:continue
    send(pid,signal.SIGKILL,before[pid])
for pid,r in controllers.items():send(pid,signal.SIGKILL,r)
time.sleep(.3)
remaining=[pid for pid in selected if (x:=inventory().get(pid)) and x['state']!='Z' and x['ticks']==before[pid]['ticks']]
result=dict(host=socket.gethostname(),at=time.time(),stopped_processes=len(selected),remaining=remaining,checkpoints=audit)
(OUT/('stop_'+socket.gethostname()+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(dict(host=result['host'],stopped=len(selected),remaining=remaining,checkpoints=len(audit))))
assert not remaining
