import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('host');p.add_argument('--seconds',type=int,default=20);a=p.parse_args()
def cpu():return {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}
before=cpu();start=time.monotonic();time.sleep(a.seconds);after=cpu();elapsed=time.monotonic()-start
r=dict(host=a.host,seconds=elapsed,cpu_equivalents=(after['usage_usec']-before['usage_usec'])/1e6/elapsed,
       throttled_periods=after['nr_throttled']-before['nr_throttled'],
       memory_gib=int(Path('/sys/fs/cgroup/memory.current').read_text())/1024**3,
       memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),at=time.time())
atomic(ROOT/'samples'/f'{a.host}_{int(time.time())}.json',r);print(json.dumps(r))
