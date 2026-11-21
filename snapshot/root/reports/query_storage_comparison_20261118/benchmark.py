"""Isolated paired replay of saved test200 plans; production results are read-only."""
import hashlib,json,math,os,random,statistics,sys,time,traceback
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parent
CAM=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
sys.path.insert(0,str(CAM/'batch_code'))
from query_data import definition
sys.path.insert(0,'/root/data/cys/logs/query_train800_author_fixed_20261004/code')
from run_query_ceb_baselines import RuntimeOracle
from workload.workloads import WorkloadSpec,OracleCodec
from oracle.oracle import _decode_query

os.environ.update(DB_HOST='127.0.0.1',DB_NAME='imdb',DB_USER='imdb',DB_PASSWORD='imdb')
os.sched_setaffinity(0,{18})
PORTS={'ram':22450,'lustre':22451}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,default=str))
def append(name,value):
    with (ROOT/name).open('a') as f:f.write(json.dumps(value,default=str)+'\n');f.flush()
class ReadOnlyOracle(RuntimeOracle):
    @staticmethod
    def connect():
        c=RuntimeOracle.connect()
        c.execute('SET default_transaction_read_only = on')
        if os.environ.get('STORAGE_PROBE_TEMP_TABLESPACE'):
            c.execute("SELECT set_config('temp_tablespaces', %s, false)",(os.environ['STORAGE_PROBE_TEMP_TABLESPACE'],))
        return c
def connect(label):
    os.environ['DB_PORT']=str(PORTS[label]);return ReadOnlyOracle.connect()
def plan_shape(node):
    keys=('Node Type','Join Type','Relation Name','Alias','Index Name','Scan Direction','Strategy','Sort Key','Hash Cond','Merge Cond','Join Filter','Filter','Index Cond')
    return dict({k:node[k] for k in keys if k in node},Plans=[plan_shape(p) for p in node.get('Plans',[])])
def sample():
    audit=json.loads(Path('/root/reports/query_convergence_fsbo_tpe_lolbo_20261109/audit.json').read_text())
    sources={(x['task'],x['method']):x for x in audit['sources']}
    tasks=['CEB_10A32','CEB_11A100','CEB_11B147','CEB_1A1020','CEB_2A19','CEB_2B170','CEB_2C202','CEB_3A1351','CEB_3B23','CEB_5A109','CEB_6A10','CEB_7A102','CEB_8A120','CEB_9A112','CEB_9B37']
    rows=[]
    for i,task in enumerate(tasks):
        method=['FSBO','LOLBO','TPE'][i%3];source=sources[task,method];raw=Path(source['source']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==source['sha256']
        doc=json.loads(raw)
        # Choose the median of distinct, actually executed, successful plans
        # whose historical runtimes were <=8 s; exclude Bao imports/cache hits.
        unique={}
        for obs in doc['observations'][49:]:
            e=obs['extra_info']
            if e.get('database_call') and e.get('status')=='completed' and not e.get('censored') and .05<=e['runtime']<=8:
                unique.setdefault(tuple(e['plan']),e)
        candidates=sorted(unique.values(),key=lambda x:x['runtime'])
        assert candidates,task
        row=candidates[len(candidates)//2]
        d,_,sql,_=definition(task)
        if 'sql_sha256' in doc['meta_info']:
            assert hashlib.sha256(sql.encode()).hexdigest()==doc['meta_info']['sql_sha256']
        else:
            historical_recipe=json.loads(Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1/recipe.json').read_text())
            assert Path(historical_recipe['native']).resolve()==Path('/root/data/cys/logs/query_train800_author_fixed_20261004/native').resolve()
        rows.append(dict(task=task,plan=row['plan'],historical_seconds=row['runtime'],historical_method=method,source=source,proposal_index=row['proposal_index'],historical_database=doc['meta_info'].get('database'),historical_settings=doc['meta_info'].get('session_settings')))
    pool=json.loads((CAM/'pools/batch10_frozen_pool.json').read_text())['tasks']
    for task in ['CEB_11A121','CEB_11A122','CEB_11A123']:
        row=min((r for r in pool[task] if not r['censored']),key=lambda x:x['runtime'])
        rows.append(dict(task=task,plan=row['plan'],historical_seconds=row['runtime'],historical_method='BOLT30_warm_start',source=str(CAM/'pools/batch10_frozen_pool.json'),failure_affected_task=True))
    for i,row in enumerate(rows):
        d,_,_,_=definition(row['task']);spec=WorkloadSpec.from_definition(d,OracleCodec.Aliases)
        row.update(id=i,timeout_seconds=60,sql=_decode_query(spec,row['plan']),sql_sha256=hashlib.sha256(_decode_query(spec,row['plan']).encode()).hexdigest())
    save('samples.json',rows);return rows
def snapshot(label):
    with connect(label) as c:
        settings=c.execute('select name,setting,unit,source from pg_settings order by name').fetchall()
        counts={t:c.execute('select count(*) from '+t).fetchone()[0] for t in ['title','name','movie_info','cast_info']}
        return dict(settings=settings,counts=counts,version=c.execute('select version()').fetchone()[0],affinity=sorted(os.sched_getaffinity(c.execute('select pg_backend_pid()').fetchone()[0])))
def execute(row,label,phase,repetition):
    os.environ['DB_PORT']=str(PORTS[label]);d,_,_,_=definition(row['task']);oracle=ReadOnlyOracle(WorkloadSpec.from_definition(d,OracleCodec.Aliases),60)
    start=time.time()
    try:result=oracle(row['plan'].copy(),60)
    except psycopg.Error as exc:result=dict(status='infrastructure_error',error=str(exc),sqlstate=exc.sqlstate,censored=None,runtime=None)
    out=dict(sample=row['id'],task=row['task'],storage=label,phase=phase,repetition=repetition,started=start,finished=time.time(),result=result)
    append('timings.jsonl',out);print(json.dumps(out),flush=True)
    return out
def explain(row,label):
    try:
        with connect(label) as c:
            c.execute('SET statement_timeout = 60000')
            start=time.time();plan=c.execute('EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON) '+row['sql']).fetchone()[0][0]
            elapsed=time.time()-start
            # Separately verify query result values, independent of row ordering.
            values=c.execute(row['sql']).fetchall();result_hash=digest(sorted(map(lambda r:json.dumps(r,default=str),values)))
            out=dict(sample=row['id'],storage=label,plan=plan,shape_sha256=digest(plan_shape(plan['Plan'])),elapsed=elapsed,result_sha256=result_hash,result_rows=len(values))
    except psycopg.Error as exc:out=dict(sample=row['id'],storage=label,error=str(exc),sqlstate=exc.sqlstate)
    append('explain.jsonl',out);print('EXPLAIN',row['id'],label,'error' if 'error' in out else 'ok',flush=True)
def main():
    assert not (ROOT/'timings.jsonl').exists(),'Never overwrite a timing run'
    assert (CAM/'PAUSE').exists()
    start=time.time();rows=sample()
    # First replay precedes large count scans, but the OS cache is not flushed.
    for row in rows:
        order=['ram','lustre'] if row['id']%2==0 else ['lustre','ram']
        for label in order:execute(row,label,'first_pass',0)
    snapshots={label:snapshot(label) for label in PORTS};save('environment.json',snapshots)
    assert snapshots['ram']['counts']==snapshots['lustre']['counts']
    a={k:(v,u) for k,v,u,s in snapshots['ram']['settings']};b={k:(v,u) for k,v,u,s in snapshots['lustre']['settings']}
    differences={k:[a[k],b[k]] for k in a if a[k]!=b[k]};save('setting_differences.json',differences)
    assert set(differences)<= {'data_directory','port','unix_socket_directories','config_file','hba_file','ident_file'},differences
    for repetition in range(1,6):
        ordered=rows.copy();random.Random(181100+repetition).shuffle(ordered)
        for row in ordered:
            order=['ram','lustre'] if (row['id']+repetition)%2==0 else ['lustre','ram']
            for label in order:execute(row,label,'repeated',repetition)
        save('benchmark_state.json',dict(state='repeated',completed_repetitions=repetition,total_repetitions=5,elapsed_seconds=time.time()-start))
    for row in rows:
        for label in ['ram','lustre']:explain(row,label)
    save('benchmark_state.json',dict(state='complete',samples=len(rows),repetitions=5,elapsed_seconds=time.time()-start,finished=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('benchmark_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
