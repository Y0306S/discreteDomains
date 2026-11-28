"""Cancel only this campaign's multiplexed forwards owned by Optimizer2."""
from common import *
assert (SEM/'PAUSE').exists()
assert load(ROOT/'bolt.status.json')['live']>=12
results=[]
for host in ['LLMOptimizer4','LLMOptimizer3']:
    record=load(SEM/'query/services'/f'{host}.tunnel.process.json')
    argv=record['command'];forward_args=[]
    for i,v in enumerate(argv):
        if v in ('-L','-R'):forward_args += [v,argv[i+1]]
    port=34979 if host=='LLMOptimizer4' else 36449
    cmd=['ssh','-S',f'/tmp/.ssh-mux-root@ssh-cn-huabei1.ebcloud.com:{port}',
         '-O','cancel']+forward_args+[host]
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
    results.append(dict(host=host,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
                        forwarded_pairs=len(forward_args)//2))
atomic(ROOT/'retired_optimizer2_forwards.json',dict(at=time.time(),results=results))
print(json.dumps(results))
