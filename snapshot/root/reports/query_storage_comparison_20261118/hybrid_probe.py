"""Match the earlier RAM-table plus Lustre-temporary-file arrangement."""
import json,os,random,time
from pathlib import Path
import psycopg
import benchmark as bench
ROOT=Path(__file__).resolve().parent
NAME='storage_probe_lustre_temp'
TEMP=Path('/root/data/cys/baseline/storage_tests/query_storage_comparison_20261118/hybrid_temp')
def main():
    assert json.loads((ROOT/'cold_probe_state.json').read_text())['state']=='complete'
    assert not (ROOT/'hybrid_probe_state.json').exists() and not TEMP.exists()
    examples=[json.loads(line) for line in (ROOT/'explain.jsonl').read_text().splitlines()]
    ids={r['sample'] for r in examples if r.get('plan',{}).get('Plan',{}).get('Temp Written Blocks',0)>0}
    rows=[r for r in json.loads((ROOT/'samples.json').read_text()) if r['id'] in ids]
    TEMP.mkdir();os.chown(TEMP,10000,10000)
    with psycopg.connect(host='127.0.0.1',port=22450,dbname='imdb',user='imdb',autocommit=True) as db:
        assert db.execute('show data_directory').fetchone()[0]=='/dev/shm/query_storage_comparison_20261118/ram'
        db.execute(psycopg.sql.SQL('CREATE TABLESPACE {} LOCATION {}').format(psycopg.sql.Identifier(NAME),psycopg.sql.Literal(str(TEMP))))
    bench.PORTS['ram_lustre_temp']=22450
    started=time.time()
    for repetition in range(4):
        ordered=rows.copy();random.Random(181200+repetition).shuffle(ordered)
        for row in ordered:
            order=['ram_lustre_temp','lustre'] if (row['id']+repetition)%2 else ['lustre','ram_lustre_temp']
            for label in order:
                if label=='ram_lustre_temp':os.environ['STORAGE_PROBE_TEMP_TABLESPACE']=NAME
                else:os.environ.pop('STORAGE_PROBE_TEMP_TABLESPACE',None)
                bench.execute(row,label,'hybrid_warmup' if repetition==0 else 'hybrid_repeated',repetition)
        bench.save('hybrid_probe_state.json',dict(state='running',completed_repetitions=repetition,plan_ids=sorted(ids),elapsed_seconds=time.time()-started))
    # Verify actual temporary reads/writes in the hybrid arrangement too.
    os.environ['STORAGE_PROBE_TEMP_TABLESPACE']=NAME
    for row in rows:bench.explain(row,'ram_lustre_temp')
    os.environ.pop('STORAGE_PROBE_TEMP_TABLESPACE',None)
    with psycopg.connect(host='127.0.0.1',port=22450,dbname='imdb',user='imdb',autocommit=True) as db:
        db.execute(psycopg.sql.SQL('DROP TABLESPACE {}').format(psycopg.sql.Identifier(NAME)))
    bench.save('hybrid_probe_state.json',dict(state='complete',repetitions=3,plan_ids=sorted(ids),temporary_directory=str(TEMP),elapsed_seconds=time.time()-started))
if __name__=='__main__':main()
