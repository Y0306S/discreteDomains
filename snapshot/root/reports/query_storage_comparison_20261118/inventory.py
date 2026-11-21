import collections,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAM=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'

def category(path):
    return 'RAM/tmpfs' if '/dev/shm/' in path else 'local block storage' if '/root/blockdata/' in path else 'Lustre' if path.startswith('/root/data/') else 'unknown'
def main():
    evidence=[]
    for path in [
        '/root/data/cys/logs/query_test200_baselines_20261012/plan.before_sequential_unique.json',
        '/root/data/cys/logs/query_test200_baselines_20261012/plan.before_blockdata_migration_20261027.json',
        '/root/data/cys/logs/query_test200_baselines_20261012/plan.json',
        '/root/data/logs/query_fsbo_pilot_20261106_v1/active_pool.json',
        '/root/data/logs/query_fsbo_local_expansion_20261107_v1/plan.json',
        str(CAM/'resources.json'),
    ]:
        d=json.loads(Path(path).read_text());workers=d.get('workers',[])
        if not workers:workers=d.get('active_workers',[])
        evidence.append(dict(path=path,workers=workers,counts=dict(collections.Counter(category(w.get('pgdata','')) for w in workers))))
    (ROOT/'historical_inventory.json').write_text(json.dumps(evidence,indent=2))
    cfg=json.loads((CAM/'resources.json').read_text())
    probe='''import json,os,subprocess,psycopg
from pathlib import Path
cfg=json.loads(Path(CAMPAIGN+'/resources.json').read_text())
out=[]
for w in cfg['workers']:
 if w['host']!=HOST:continue
 try:
  with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',autocommit=True,connect_timeout=5) as c:
   p=c.execute('show data_directory').fetchone()[0]
   out.append(dict(index=w['index'],host=HOST,port=w['db_port'],pgdata=p,realpath=os.path.realpath(p),filesystem=subprocess.check_output(['findmnt','-T',p,'-rn','-o','FSTYPE'],text=True).strip(),temp_tablespaces=c.execute('show temp_tablespaces').fetchone()[0],tablespaces=c.execute('select spcname,pg_tablespace_location(oid) from pg_tablespace').fetchall(),active_other_clients=c.execute("select count(*) from pg_stat_activity where pid<>pg_backend_pid() and backend_type='client backend' and state='active'").fetchone()[0]))
 except Exception as e:out.append(dict(index=w['index'],error=repr(e)))
print(json.dumps(out))
'''
    rows=[]
    for host in ['LLMOptimizer2','LLMOptimizer4']:
        code='CAMPAIGN='+repr(str(CAM))+'\nHOST='+repr(host)+'\n'+probe
        command=[PY,'-'] if host=='LLMOptimizer2' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,PY+' -']
        r=subprocess.run(command,input=code,text=True,capture_output=True,timeout=90,check=True)
        rows.extend(json.loads(r.stdout))
    (ROOT/'live_inventory.json').write_text(json.dumps(dict(at=time.time(),workers=rows),indent=2))
    print(json.dumps(dict(live_counts=dict(collections.Counter(x.get('filesystem','error') for x in rows)),active_other_clients=sum(x.get('active_other_clients',0) for x in rows)),indent=2))
if __name__=='__main__':main()
