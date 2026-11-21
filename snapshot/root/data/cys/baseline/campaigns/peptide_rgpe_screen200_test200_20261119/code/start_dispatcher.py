from pathlib import Path
import json,os,socket,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
HOSTS={'cs-94bac-0693e-server':'dispatch_local',
       'cs-0b2cd-18deb-server':'dispatch_cs-0b2cd-18deb-server',
       'cs-fa498-8793f-server':'dispatch_cs-fa498-8793f-server'}
name=HOSTS[socket.gethostname()]
record=ROOT/'evidence'/(name+'.process.json')
if record.exists():
    old=json.loads(record.read_text());proc=Path('/proc',str(old['pid']))
    if proc.exists():
        stat=proc.joinpath('stat').read_text().split()
        if stat[21]==old['start_ticks'] and stat[2]!='Z':raise RuntimeError('Dispatcher already alive')
    (ROOT/'evidence/dispatch_cap44'/record.name).write_text(record.read_text())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
         OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
cmd=['/root/data/cys/envs/bolt_clean_venv/bin/python','-u',str(ROOT/'code/dispatch.py')]
with (ROOT/'evidence'/(name+'.log')).open('a') as f:
    child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],cmd=cmd,at=time.time())
record.write_text(json.dumps(row,indent=2));print(json.dumps(row))
