"""Recover Random Search on machines 2/4 and reserve machine 3 for its owner."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

import psycopg

ROOT = Path(__file__).resolve().parent
QUERY = Path('/root/data/cys/logs/query_test200_baselines_20261012')
HANDOFF = Path('/root/data/cys/logs/query_random_optformer_handoff_20261014')
EXPANSION = Path('/root/data/cys/logs/query_random_expand_20261023')
sys.path[:0] = [str(QUERY/'code'), str(HANDOFF)]
import handoff as h
import query_test200_fleet_v3 as fleet
import query_test200_rolling as rolling
from run_artifact_store import JsonJournal, database, decode

d = h.d
HOST = next(k for k,v in d.HOSTS.items() if v == socket.gethostname())
ALLOWED = ['LLMOptimizer2','LLMOptimizer4']
load, write = h.load, fleet.write


def connect(worker):
    return psycopg.connect(host='127.0.0.1',port=worker['port'],dbname='imdb',user='imdb',
        connect_timeout=10,autocommit=True,application_name='query_hosts24_recovery')


def all_workers():
    return load(ROOT/'inventory.json')['workers']


def claims():
    return [load(p) for p in sorted((QUERY/'claims').glob('random.*.json'))]


def inventory():
    if (ROOT/'inventory.json').exists():
        return
    rolling.verify(QUERY)
    for folder in ('tasks','logs','processes','state','locks','storage','checkpoints','retired'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    plan=load(QUERY/'plan.json')
    workers=[dict(w,method='random') for w in plan['workers']]
    workers += [load(p) for p in sorted((HANDOFF/'allocations').glob('*.json'))]
    workers += [load(p) for p in sorted((EXPANSION/'allocations').glob('*.json'))]
    assert len(workers)==29 and len({w['name'] for w in workers})==29
    write(ROOT/'before_plan.json',plan)
    write(ROOT/'before_rolling_manifest.json',load(QUERY/'rolling_manifest.json'))
    write(ROOT/'inventory.json',dict(time=fleet.stamp(),workers=workers,
        claims=claims(),plan_sha256=fleet.digest(QUERY/'plan.json')))


def args_for(pid):
    if d.process(pid) is None:
        return []
    return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')


def evidence(directory):
    result={}
    with database(directory) as connection:
        if connection.execute('PRAGMA quick_check').fetchone()[0]!='ok':
            raise RuntimeError('Artifact store is corrupt: '+str(directory))
        rows=connection.execute("SELECT name,payload,sha256 FROM artifacts WHERE name GLOB "
            "'proposal_[0-9][0-9][0-9][0-9].json' ORDER BY name").fetchall()
        for i,(name,payload,sha) in enumerate(rows):
            raw=decode((payload,sha))
            record=json.loads(raw)
            assert name==f'proposal_{i:04d}.json' and record['proposal_index']==i
            result[name]=sha
    return result


def quiesce():
    workers=[w for w in all_workers() if w['host']==HOST]
    known_scripts={str(QUERY/'code/query_test200_rolling.py'),str(EXPANSION/'expand.py'),
                   str(HANDOFF/'handoff.py')}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():
            continue
        args=args_for(int(path.name))
        if known_scripts.intersection(args):
            d.stop(d.process(int(path.name)))
        if str(QUERY/'code/run_query_ceb_baselines_v3.py') in args:
            raise RuntimeError('A query runner is unexpectedly alive; inspect before recovery')
    owned={w['name']:w for w in workers}
    checked=[]
    for worker in workers:
        if fleet.postgres_pid(worker) is None:
            raise RuntimeError('Expected private PostgreSQL instance is missing')
        with connect(worker) as connection:
            actual=connection.execute('SHOW data_directory').fetchone()[0]
            assert actual==worker['pgdata']
            clients=connection.execute("SELECT count(*) FROM pg_stat_activity WHERE "
                "backend_type='client backend' AND pid<>pg_backend_pid()").fetchone()[0]
            if clients:
                raise RuntimeError('Private database still has clients: '+worker['name'])
            connection.execute("SET statement_timeout='10s'")
            connection.execute('SELECT 1 FROM title LIMIT 1').fetchone()
            checked.append(dict(worker=worker['name'],pid=fleet.postgres_pid(worker),clients=0,
                database_started_at=float(connection.execute(
                    'SELECT extract(epoch FROM pg_postmaster_start_time())').fetchone()[0])))
    for claim in claims():
        if claim['status']=='complete' or claim['worker'] not in owned:
            continue
        worker=owned[claim['worker']]
        task=claim['task']
        directory=QUERY/'outputs'/task/'random/seed0'
        with fleet.lock(QUERY/'workers'/f'{worker["name"]}.lock'), fleet.lock(directory/'.lock'):
            journal=JsonJournal(directory)
            marker=ROOT/'tasks'/f'{task}.json'
            if not marker.exists():
                write(marker,dict(task=task,claim=claim,worker=worker,time=fleet.stamp(),
                    completed_sha256=evidence(directory),checkpoint_sha256=fleet.digest(directory/'checkpoint.pt'),
                    previous_status=load(directory/'status.json'),
                    pending={n:dict(proposal=journal.read(n),attempts=journal.read(
                        n.replace('.pending.json','.database_attempts.json')))
                        for n in journal.names('proposal_*.pending.json')}))
            before=load(marker)
            assert evidence(directory)==before['completed_sha256']
            for name,record in before['pending'].items():
                assert not journal.exists(name.replace('.pending.json','.json'))
                attempts=record['attempts']
                last=attempts[-1]
                if last['status'] not in ('in_flight','error','io_error'):
                    raise RuntimeError('Unrecognized pending call: '+task)
                if last['status']=='error' and not ('Disk quota exceeded' in last.get('error','')
                        or last.get('sqlstate') in ('53100','58030')):
                    raise RuntimeError('Pending call was not interrupted by the storage outage')
                assert all(a['proposal']==record['proposal'] for a in attempts)
                last['storage_recovery']=dict(replay_authorized=True,active_database_calls=0,
                    original_result_unavailable=True,evidence_report=str(marker),verified_at=time.time(),
                    task_lock_verified_free=True,original_worker=worker['name'],
                    reason='Quota outage ended the runner; source database has zero other clients, '
                           'and no result exists in the verified journal. Replay the same pending plan.')
                journal.write(name.replace('.pending.json','.database_attempts.json'),attempts)
            print(json.dumps(dict(reviewed=task,saved=len(before['completed_sha256']),
                                  interrupted=len(before['pending']))),flush=True)
    write(ROOT/f'{HOST}.quiesced.json',dict(time=fleet.stamp(),host=HOST,databases=checked))


def release_machine3():
    if HOST!='LLMOptimizer3' or not (ROOT/f'{HOST}.quiesced.json').exists():
        raise RuntimeError('Quiesce machine 3 first')
    backup=load(h.BACKUP)
    if fleet.digest(Path(backup['archive']))!=backup['sha256']:
        raise RuntimeError('Read-only IMDb restore archive checksum mismatch')
    released=[]
    for worker in [w for w in all_workers() if w['host']==HOST]:
        target=Path(worker['pgdata'])
        marker=ROOT/'retired'/f'{worker["name"]}.json'
        if marker.exists() and not target.exists():
            released.append(load(marker))
            continue
        with fleet.lock(QUERY/'workers'/f'{worker["name"]}.lock'):
            if not target.resolve().is_relative_to('/dev/shm') or target.stat().st_uid!=10000:
                raise RuntimeError('Refusing to remove a nonvolatile or unowned database')
            h.stop_database(worker)
            control=subprocess.check_output([str(fleet.PG_BIN/'pg_controldata'),str(target)],
                text=True,env=dict(os.environ,LC_ALL='C'))
            state=next(s.split(':',1)[1].strip() for s in control.splitlines()
                       if s.startswith('Database cluster state:'))
            if state!='shut down' or fleet.postgres_pid(worker) is not None:
                raise RuntimeError('Database did not shut down cleanly')
            log=target/'query_train800_author_20261002.log'
            if log.exists():
                shutil.copyfile(log,ROOT/'retired'/f'{worker["name"]}.postgres.log')
            record=dict(worker=worker,time=fleet.stamp(),cluster_state=state,
                restore_archive=backup['archive'],restore_sha256=backup['sha256'],
                result_location=str(QUERY/'outputs'),action='Remove stopped, reproducible tmpfs IMDb clone')
            write(marker,record)
            shutil.rmtree(target)
            released.append(record)
            print(json.dumps(dict(released=worker['name'])),flush=True)
    write(ROOT/'LLMOptimizer3.released.json',dict(time=fleet.stamp(),workers=released))


def prepare():
    if HOST!='LLMOptimizer2':
        raise RuntimeError('Prepare from machine 2')
    if (ROOT/'manifest.json').exists():
        return verify()
    assert (ROOT/'LLMOptimizer3.released.json').exists()
    for host in d.HOSTS:
        assert (ROOT/f'{host}.quiesced.json').exists()
    selected=[w for w in all_workers() if w['host'] in ALLOWED]
    targets=sorted([w for w in selected if w['host']=='LLMOptimizer2'],key=lambda w:w['name'])
    source=[c for c in claims() if c['host']=='LLMOptimizer3' and c['status']!='complete']
    assert len(selected)==20 and len(targets)==9 and len(source)==9
    assignments=[]
    with fleet.lock(QUERY/'claims/.queue.lock',blocking=True):
        for claim,worker in zip(source,targets):
            path=QUERY/'claims'/f'random.{claim["task"]}.json'
            assert load(path)==claim
            updated=dict(claim,worker=worker['name'],host=worker['host'],status='running',
                updated_at=fleet.stamp(),reassignment_report=str(ROOT/'assignments.json'))
            assignments.append(dict(task=claim['task'],before=claim,after=updated))
        write(ROOT/'assignments.json',assignments)
        for item in assignments:
            write(QUERY/'claims'/f'random.{item["task"]}.json',item['after'])
    plan=load(QUERY/'plan.json')
    assert fleet.digest(QUERY/'plan.json')==load(ROOT/'inventory.json')['plan_sha256']
    plan['workers']=[w for w in plan['workers'] if w['host']!='LLMOptimizer3']
    plan['reserved_hosts']={'LLMOptimizer3':dict(reason='Reserved by user on 2026-10-26',
                                             replacement_deployment=str(ROOT))}
    plan['active_random_deployment']=str(ROOT)
    write(QUERY/'plan.json',plan)
    rolling_manifest=load(QUERY/'rolling_manifest.json')
    rolling_manifest['plan_sha256']=fleet.digest(QUERY/'plan.json')
    write(QUERY/'rolling_manifest.json',rolling_manifest)
    enabled=load(HANDOFF/'enabled.json')
    write(ROOT/'previous_handoff_enabled.json',enabled)
    write(HANDOFF/'enabled.json',dict(enabled=False,time=time.time(),superseded_by=str(ROOT)))
    files=[Path(__file__),QUERY/'plan.json',QUERY/'rolling_manifest.json',
           QUERY/'code/query_test200_fleet_v3.py',QUERY/'code/run_query_ceb_baselines_v3.py']
    write(ROOT/'manifest.json',dict(time=fleet.stamp(),allowed_hosts=ALLOWED,reserved_hosts=['LLMOptimizer3'],
        workers=selected,sources={str(p):fleet.digest(p) for p in files},
        protocol='Existing random search and numerical runner unchanged; machine-3 partials move to machine-2 GPU workers',
        retained_completed_tasks=78,retained_proposals=94342))
    verify()


def verify():
    manifest=load(ROOT/'manifest.json')
    for p,sha in manifest['sources'].items():
        if fleet.digest(Path(p))!=sha:
            raise RuntimeError('Pinned source changed: '+p)
    rolling.verify(QUERY)
    return manifest


def migrate_checkpoint(worker,task,directory):
    marker=ROOT/'tasks'/f'{task}.json'
    if not marker.exists():
        return
    old=load(marker)
    if old['worker']['host']!='LLMOptimizer3':
        return
    changed=ROOT/'checkpoints'/f'{task}.migration.json'
    if changed.exists():
        assert load(changed)['target_worker']==worker['name']
        return
    import torch
    import run_query_ceb_baselines_v3 as runner
    # Historical checkpoints pickle Engine under the executable module.
    sys.modules['__main__'].Engine=runner.Engine
    runner.Engine.__module__='__main__'
    path=directory/'checkpoint.pt'
    with fleet.lock(directory/'.lock'):
        assert evidence(directory)==old['completed_sha256']
        assert fleet.digest(path)==old['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        previous=copy.deepcopy(saved['meta'])
        expected_device=worker.get('device','cuda:0')
        assert previous['method']=='random' and previous['device']==expected_device=='cuda:0'
        assert previous['task_id']==task and saved['total']==1049
        assert previous['database']['DB_PORT']==str(old['worker']['port'])
        backup=ROOT/'checkpoints'/f'{task}.before.pt'
        if not backup.exists():
            shutil.copy2(path,backup)
        saved['meta']['database']['DB_PORT']=str(worker['port'])
        assert {**saved['meta'],'database':previous['database']}==previous
        temporary=path.with_suffix('.migration.tmp')
        with temporary.open('wb') as stream:
            torch.save(saved,stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        record=dict(time=fleet.stamp(),task=task,target_worker=worker['name'],
            from_database=previous['database'],to_database=saved['meta']['database'],
            before_sha256=old['checkpoint_sha256'],after_sha256=fleet.digest(path),
            backup=str(backup),preserved='Optimizer, sampled RNG state, initializations, pending plan, and completed scores')
        write(changed,record)
        write(directory/'execution_migrations.json',[record])


def bind_fleet():
    selected={w['name']:w for w in load(ROOT/'manifest.json')['workers']}
    original_command=fleet.run_command
    original_claim=fleet.claim_task
    moved={row['task'] for row in load(ROOT/'assignments.json')}
    def allocation(root,name,method=None):
        if HOST not in ALLOWED or name not in selected or selected[name]['host']!=HOST:
            raise RuntimeError('Worker is not authorized on this host')
        if Path(root)!=QUERY or method not in (None,'random'):
            raise RuntimeError('Only existing Random Search is authorized')
        return fleet.method_plan(QUERY,'random'),selected[name]
    def command(root,worker,task,output,*args,**kwargs):
        if Path(output)==QUERY/'outputs':
            migrate_checkpoint(worker,task,Path(output)/task/'random/seed0')
        result=original_command(root,worker,task,output,*args,**kwargs)
        result[result.index('--device')+1]=worker.get('device','cuda:0')
        return result
    def claim(root,plan,worker):
        with fleet.lock(root/'claims/.queue.lock',blocking=True):
            for task in plan['tasks']:
                if task not in moved:
                    continue
                existing=load(root/'claims'/f'random.{task}.json')
                if existing['worker']==worker['name'] and existing['status']!='complete':
                    write(root/'claims'/f'random.{task}.json',dict(existing,updated_at=fleet.stamp()))
                    return task
        return original_claim(root,plan,worker)
    fleet.allocation,fleet.run_command=allocation,command
    fleet.claim_task=claim


def storage(worker):
    module_path=Path('/root/data/cys/logs/query_random_storage_20261015/recover.py')
    spec=importlib.util.spec_from_file_location('quota_storage',module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with connect(worker) as c:
        row=c.execute('SELECT pg_tablespace_location(oid) FROM pg_tablespace WHERE spcname=%s',
                      (module.TABLESPACE,)).fetchone()
    parent=(Path(row[0]).parent if row else Path('/root/data/cys/queryopt_temp/query_hosts24_20261026'))
    if row and Path(row[0]).name!=worker['name']:
        raise RuntimeError('Unexpected private spill tablespace')
    module.ROOT=ROOT/'storage'/worker['name']
    module.ROOT.mkdir(parents=True,exist_ok=True)
    module.TEMP_ROOT=parent
    module.HANDOFF=ROOT
    previous_verify,previous_allocations=h.verify,h.allocations
    h.verify,h.allocations=verify,lambda:[worker]
    try:
        module.configure()
    finally:
        h.verify,h.allocations=previous_verify,previous_allocations


def worker_main(name):
    if (ROOT / 'launch_hold.json').exists():
        raise RuntimeError('Random Search is paused; see launch_hold.json before resuming')
    verify()
    bind_fleet()
    plan,worker=fleet.allocation(QUERY,name,'random')
    with fleet.lock(ROOT/'locks'/f'{name}.lock'):
        try:
            fleet.db_start(QUERY,name)
            storage(worker)
            try:
                fleet.validate_smoke(QUERY,plan,worker)
            except RuntimeError:
                write(ROOT/'state'/f'{name}.json',dict(status='smoke',time=fleet.stamp()))
                fleet.run_worker(QUERY,name,smoke=True,method='random')
                fleet.validate_smoke(QUERY,plan,worker)
            write(ROOT/'state'/f'{name}.json',dict(status='production',time=fleet.stamp()))
            fleet.run_worker(QUERY,name,method='random')
            h.stop_database(worker)
            write(ROOT/'state'/f'{name}.json',dict(status='complete',time=fleet.stamp()))
        except BaseException as exc:
            write(ROOT/'state'/f'{name}.json',dict(status='failed',error=str(exc),time=fleet.stamp()))
            raise


def launch(names=None):
    if (ROOT / 'launch_hold.json').exists():
        raise RuntimeError('Random Search is paused; see launch_hold.json before resuming')
    manifest=verify()
    if HOST not in ALLOWED:
        raise RuntimeError('Machine 3 is reserved for the user')
    bind_fleet()
    workers=[w for w in manifest['workers'] if w['host']==HOST]
    occupied=set()
    for w in workers:
        for cpu in [w['bo_cpu'],*w['db_cpus']]:
            key=h.core_key(cpu)
            assert key not in occupied
            occupied.add(key)
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert len(occupied)<=int(quota)/int(period)
    launched=[]
    for worker in workers:
        name=worker['name']
        if names and name not in names:
            continue
        previous=load(ROOT/'processes'/f'{name}.json')
        if previous and d.process(previous['pid'])==previous:
            launched.append(dict(worker=name,process=previous,existing=True))
            continue
        if load(ROOT/'state'/f'{name}.json',{}).get('status')=='failed':
            raise RuntimeError('Inspect the failed recovery before retry: '+name)
        with fleet.lock(QUERY/'workers'/f'{name}.lock'):
            pass
        env=fleet.environment(worker)
        command=['taskset','-c',str(worker['bo_cpu']),fleet.PYTHON,str(Path(__file__)),
                 'worker','--name',name]
        with (ROOT/'logs'/f'{name}.log').open('a') as stream:
            process=subprocess.Popen(command,env=env,cwd=QUERY,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record=d.process(process.pid)
        assert record is not None
        write(ROOT/'processes'/f'{name}.json',record)
        launched.append(dict(worker=name,process=record))
    write(ROOT/f'{HOST}.launch.json',dict(time=fleet.stamp(),workers=launched))
    print(json.dumps(launched),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','quiesce','release-machine3','prepare','launch','worker'])
    parser.add_argument('--name')
    args=parser.parse_args()
    if args.action=='release-machine3':
        release_machine3()
    elif args.action=='worker':
        worker_main(args.name)
    elif args.action=='launch':
        launch([args.name] if args.name else None)
    else:
        globals()[args.action]()
