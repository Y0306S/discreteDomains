"""Identity-checked operations on the explicitly named local host."""
import argparse
import signal
from common import *


def semanticopt_drained():
    states=[load(p) for p in (SEM/'query').glob('worker*.status.json')]
    return len(states)==20 and all(d.get('status') in ('paused','idle') for d in states)


def stop_llm(host):
    require_host(host)
    if not (SEM/'PAUSE').exists() or not semanticopt_drained():
        raise RuntimeError('SemanticOPT has not drained all twenty workers')
    stopped=[]
    specs=load(SEM/'allocation.json')['replicas']
    labels=[f'replica{s["index"]:02}' for s in specs if s['host']==host]
    if host=='LLMOptimizer2':labels.append('broker')
    for label in labels:
        path=SEM/'query/services'/f'{label}.process.json';record=load(path)
        if alive(record):
            cmd=command(record['pid'])
            if str(SEM/'services/qwen_inference_server.py') not in cmd:
                raise RuntimeError('Unexpected model process '+label)
            os.kill(record['pid'],signal.SIGTERM)
            stopped.append(dict(label=label,**record))
    atomic(ROOT/f'stopped_llm_{host}.json',dict(at=time.time(),host=host,processes=stopped))
    print(json.dumps(dict(host=host,stopped=len(stopped))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stop-llm','launch']);p.add_argument('host');p.add_argument('--role');a=p.parse_args()
    if a.command=='stop-llm':stop_llm(a.host)
    else:
        expected='LLMOptimizer3' if a.role=='vae' else 'LLMOptimizer4'
        if a.host!=expected:raise RuntimeError('Wrong service host')
        require_host(expected)
        record=load(ROOT/'processes'/f'{a.role}_supervisor.json')
        if not alive(record):record=launch([PYTHON,'-u',str(ROOT/'runtime.py'),a.role],f'{a.role}_supervisor',environment())
        print(json.dumps(record))
