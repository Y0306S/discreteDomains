"""Start owned, bounded inference services on the audited hosts."""
from pathlib import Path
import hashlib,json,os,signal,socket,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
host=socket.gethostname()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         PYTHONDONTWRITEBYTECODE='1')

def start(name,cmd,extra=None):
    path=ROOT/'services'/(name+'.process.json')
    if path.exists():
        old=json.loads(path.read_text())
        proc=Path('/proc')/str(old['pid'])
        if proc.exists() and proc.joinpath('stat').read_text().split()[21]==old['start_ticks']:
            raise RuntimeError('Service already exists: '+name)
    with (ROOT/'services'/(name+'.log')).open('a') as f:
        p=subprocess.Popen(cmd,env=dict(env,**(extra or {})),stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],
                host=host,cmd=cmd,started=time.time())
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if host=='cs-94bac-0693e-server':
    for gpu in range(4):
        start('math'+str(gpu),[PY,'-u','/root/data/cys/rgpe_exact_acceleration_20261104/code/cuda_service.py',
            '--port',str(22440+gpu),'--cpu',str(36+gpu),'--token-file',str(ROOT/'services/token')],
            dict(CUDA_VISIBLE_DEVICES=str(gpu)))
elif host=='cs-0b2cd-18deb-server':
    old=Path('/root/data/cys/baseline/artifacts/taskset_base_qwen_fsbo_test200_20261117')
    pool=json.loads((old/'workers/cpu_new.pool.json').read_text())
    histories=list((old/'test').glob('*/llm_fsbo/seed1/history.json'))
    if pool['state']!='finished' or len(histories)!=200 or any(
            len(json.loads(h.read_text())['observations'])<80 for h in histories):
        raise ValueError('TaskSet campaign not complete')
    retired=[]
    for i in range(6):
        row=json.loads((old/('service%d.process.json'%i)).read_text())
        pid=row['pid'];proc=Path('/proc')/str(pid)
        if not proc.exists():continue
        command=proc.joinpath('cmdline').read_bytes()
        if str(old).encode() not in command or b'serve_taskset_llm.py' not in command:
            raise ValueError('PID identity mismatch')
        # Do not retire a service with established clients.
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            f=line.split()
            if int(f[1].split(':')[1],16)==row['port'] and f[3]=='01':
                raise ValueError('TaskSet service still has an active client')
        retired.append(dict(row,cmdline=command.decode().replace('\0',' '),
                            start_ticks=proc.joinpath('stat').read_text().split()[21]))
    (ROOT/'evidence/retired_taskset_services.json').write_text(json.dumps(retired,indent=2))
    for row in retired:os.kill(row['pid'],signal.SIGTERM)
    vae=json.loads((ROOT/'rgpe/recipe.json').read_text())['vae']
    for i,gpu in enumerate([1,2,4,6,8,10,12,13]):
        start('vae'+str(i),[PY,'-u',str(ROOT/'code/vae_server.py'),'--gpu',str(gpu),
            '--cpu',str([0,1,2,3,4,5,60,61][i]),'--port',str(22460+i),'--vae',vae,
            '--gpu-memory-fraction','.20','--max-weight-banks','64',
            '--skip-identical-weights','--http-keepalive'])
else:raise ValueError('No service allocation for '+host)
