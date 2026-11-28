"""Read-only host inventory and isolated database ownership checks."""
import argparse
import json
from pathlib import Path
import time
import psycopg2
from common import *

p=argparse.ArgumentParser();p.add_argument('host');args=p.parse_args()
cfg=load(SEM/'allocation.json')
specs=[s for s in cfg['workers']+cfg['bolt_workers'] if s['host']==args.host]
if args.host=='LLMOptimizer3':specs.append(load(BOLT/'control/extra_worker.json'))
results=[]
for s in specs:
    try:
        with psycopg2.connect(host='127.0.0.1',port=s['db_port'],dbname='imdb',user='imdb',password='imdb',connect_timeout=4) as c:
            with c.cursor() as q:
                q.execute('SHOW data_directory');data=q.fetchone()[0]
                q.execute('SELECT oid,spcname,pg_tablespace_location(oid) FROM pg_tablespace');tablespaces=q.fetchall()
                q.execute("SELECT pid,state,wait_event_type,wait_event FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND state=%s",('active',));active=q.fetchall()
                settings={}
                for name in ['temp_tablespaces','temp_file_limit','work_mem','max_parallel_workers_per_gather','shared_buffers']:
                    q.execute('SHOW '+name);settings[name]=q.fetchone()[0]
                q.execute("SELECT temp_files,temp_bytes FROM pg_stat_database WHERE datname=current_database()");temp=q.fetchone()
        pid=int((Path(data)/'postmaster.pid').read_text().splitlines()[0])
        results.append(dict(spec=s,data=data,tablespaces=tablespaces,active=active,settings=settings,
                            temp_stats=temp,postmaster_pid=pid,postmaster_ticks=ticks(pid)))
    except Exception as exc:results.append(dict(spec=s,error=str(exc)))
result=dict(host=args.host,at=time.time(),databases=results,
            cgroup={name:(Path('/sys/fs/cgroup')/name).read_text() for name in ['cpu.max','memory.max','memory.current','memory.stat']})
atomic(ROOT/f'inventory_{args.host}.json',result)
print(json.dumps(result,indent=2))
