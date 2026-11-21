"""Release only the two isolated investigation clusters after evidence is saved."""
import json,os,shutil,subprocess,time
from pathlib import Path
from provision import PG,PREFIX
ROOT=Path(__file__).resolve().parent
def main():
    assert (ROOT/'REPORT.md').exists() and (ROOT/'summary.json').exists()
    assert json.loads((ROOT/'cold_probe_state.json').read_text())['state']=='complete'
    assert json.loads((ROOT/'hybrid_probe_state.json').read_text())['state']=='complete'
    clusters=json.loads((ROOT/'clusters.json').read_text())['clusters'];done=[]
    allowed={'/dev/shm/query_storage_comparison_20261118/ram','/root/data/cys/baseline/storage_tests/query_storage_comparison_20261118/lustre'}
    import psycopg
    for c in clusters:
        assert c['path'] in allowed and Path(c['path']).resolve()==Path(c['path'])
        with psycopg.connect(host='127.0.0.1',port=c['port'],dbname='imdb',user='imdb',autocommit=True) as db:
            assert db.execute('show data_directory').fetchone()[0]==c['path']
            assert db.execute("select count(*) from pg_stat_activity where backend_type='client backend' and pid<>pg_backend_pid()").fetchone()[0]==0
        subprocess.run(PREFIX+[str(PG/'pg_ctl'),'-D',c['path'],'-w','-m','fast','stop'],check=True)
        assert not (Path(c['path'])/'postmaster.pid').exists()
        for name in ['storage_test.log','backup_manifest','postmaster.opts','postgresql.conf','postgresql.auto.conf']:
            p=Path(c['path'])/name
            if p.exists():shutil.copyfile(p,ROOT/(c['label']+'.'+name))
        shutil.rmtree(c['path']);done.append(dict(label=c['label'],path=c['path'],port=c['port'],stopped=True,removed=True))
    temp=Path('/root/data/cys/baseline/storage_tests/query_storage_comparison_20261118/hybrid_temp')
    if temp.exists():shutil.rmtree(temp)
    (ROOT/'cleanup.json').write_text(json.dumps(dict(at=time.time(),clusters=done,production_migrated=False,production_resumed=False),indent=2))
if __name__=='__main__':main()
