"""Move only PostgreSQL query work files to disk-backed, private tablespaces."""
import argparse
import json
import os
from pathlib import Path
import pwd
import socket
import time

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parent


def configure(worker):
    path = ROOT/'tablespaces'/worker['name']
    path.mkdir(parents=True,exist_ok=True)
    user = pwd.getpwnam('pgimdb')
    os.chown(path,user.pw_uid,user.pw_gid)
    path.chmod(0o700)
    name = 'fsbo_tmp_'+worker['name']
    kwargs = dict(host='127.0.0.1',port=worker['db_port'],dbname='imdb',user='imdb',
                  password='imdb',autocommit=True,connect_timeout=5)
    with psycopg.connect(**kwargs) as connection:
        connection.execute('SET statement_timeout=15000')
        previous = connection.execute('SHOW temp_tablespaces').fetchone()[0]
        existing = connection.execute('SELECT pg_tablespace_location(oid) FROM pg_tablespace WHERE spcname=%s',(name,)).fetchone()
        if existing:
            assert existing[0] == str(path)
        else:
            connection.execute(sql.SQL('CREATE TABLESPACE {} LOCATION {}').format(sql.Identifier(name),sql.Literal(str(path))))
        connection.execute(sql.SQL('ALTER DATABASE imdb SET temp_tablespaces = {}').format(sql.Identifier(name)))
    with psycopg.connect(**kwargs) as connection:
        actual = connection.execute('SHOW temp_tablespaces').fetchone()[0]
        assert actual == name
        settings = dict(connection.execute("SELECT name,setting FROM pg_settings WHERE name IN ('work_mem','shared_buffers','max_parallel_workers','max_parallel_workers_per_gather','dynamic_shared_memory_type')").fetchall())
    return dict(worker=worker['name'],port=worker['db_port'],previous=previous,
                temp_tablespaces=actual,path=str(path),settings=settings,at=time.time())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-after-pilot',action='store_true')
    parser.add_argument('--worker')
    args=parser.parse_args()
    assert socket.gethostname() == 'cs-fa498-8793f-server'
    recipe=json.loads((ROOT/'recipe.json').read_text())
    fs=os.statvfs(ROOT)
    assert fs.f_bavail*fs.f_frsize > 256*2**30
    if args.all_after_pilot:
        test=json.loads((ROOT/'storage_pilot_validation.json').read_text())
        assert test['passed'] and test['committed_results_unchanged']
        workers=recipe['workers']
    else:
        assert args.worker
        workers=[next(w for w in recipe['workers'] if w['name']==args.worker)]
    results=[]
    for worker in workers:
        result=configure(worker)
        results.append(result)
        print(json.dumps(result),flush=True)
        (ROOT/'storage_configuration.json').write_text(json.dumps(dict(at=time.time(),workers=results,
            scope='Database defaults for new connections; SQL, plan hints, work_mem, parallelism and frozen FSBO assets unchanged.',
            comparability='Changing temporary-file storage can affect runtime; historical measurements retain their original provenance.'),indent=2)+'\n')
