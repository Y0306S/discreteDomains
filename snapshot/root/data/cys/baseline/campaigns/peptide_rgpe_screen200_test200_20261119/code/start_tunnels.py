from pathlib import Path
import json,os,socket,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
assert socket.gethostname()=='cs-94bac-0693e-server'
plans=[('vae_to_local','LLMOptimizer3', [('L',p) for p in range(22460,22468)]),
       ('math_to_host3','LLMOptimizer3',[('R',p) for p in range(22440,22444)]),
       ('all_to_host4','LLMOptimizer4',[('R',p) for p in list(range(22440,22444))+list(range(22460,22468))])]
for name,host,forwards in plans:
    cmd=['ssh','-NT','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',
         '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
    for direction,port in forwards:
        cmd+=['-'+direction,'127.0.0.1:%d:127.0.0.1:%d'%(port,port)]
    cmd+=[host]
    with (ROOT/'services'/(name+'.log')).open('a') as f:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'services'/(name+'.process.json')).write_text(json.dumps(dict(pid=p.pid,
        start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],cmd=cmd,at=time.time()),indent=2))
    print(name,p.pid)
