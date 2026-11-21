#!/usr/bin/env python3
"""Pin a query FSBO-only, no-acquisition readiness and capacity pilot."""
import hashlib
import json
from pathlib import Path
import re
import secrets
import sys
import time

ROOT=Path(__file__).resolve().parent
TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
SOURCE=Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
TRANSFER=Path('/root/data/logs/query_transfer_train800_20261106_v1')
NATIVE=SOURCE/'native'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (ROOT/'plan.json').exists():
        raise ValueError('Plan already exists')
    sys.path.insert(0,str(SOURCE/'code'))
    sys.path.insert(0,str(NATIVE))
    from run_query_ceb_baselines import load_initializations
    from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
    from codec.codec import AliasesCodec
    from oracle.oracle import _decode_query, plan_has_crossjoin
    manifest=json.loads((TRANSFER/'corpus/manifest.json').read_text())
    status=json.loads((TRANSFER/'fsbo/status.json').read_text())
    assert status['status']=='complete' and status['step']==100000 and status['reload_verified']
    assert status['weights_sha256']==digest(TRANSFER/'fsbo/weights')
    assert status['identity']['corpus_sha256']==digest(TRANSFER/'corpus/manifest.json')
    reference=json.loads(Path('/root/data/logs/query_vae_cpu_capacity_20261106_v1/cpu_benchmark.json').read_text())
    selected=[q['task'] for q in reference['selected_queries']]
    train=(SOURCE/'train800.txt').read_text().split();test=(SOURCE/'test200.txt').read_text().split()
    assert set(selected)<=set(train) and not set(selected)&set(test)
    rows=load_initializations(NATIVE/'initialization_data/ceb_3k_bao_initialization.csv',selected)
    workload=get_workload_set('CEB_3K');tasks={}
    for task in selected:
        match=re.fullmatch(r'CEB_(\d+[A-Z])(\d+)',task)
        sql=NATIVE/'workload/ceb-3k'/match[1].lower()/((match[1]+match[2]).lower()+'.sql')
        spec=WorkloadSpec.from_definition(workload.extract_query(sql.read_text(),task),OracleCodec.Aliases)
        codec=AliasesCodec(spec.all_tables)
        def canonical(plan):
            tree=codec.decode(spec.query_tables,plan.copy());encoded=codec.encode(tree)
            if not tree.equal(codec.decode(spec.query_tables,encoded.copy())):
                raise ValueError('Canonical plan round trip differs')
            return encoded
        warm=[];seen=set()
        for row in rows[task]:
            if row['censored']:continue
            plan=canonical(row['plan']);key=tuple(plan)
            if key in seen:continue
            seen.add(key);warm.append(dict(plan=plan,runtime=row['runtime']))
        if not warm:raise ValueError('No usable Bao warmup: '+task)
        known=next(row for row in rows[task] if not row['censored'] and 0.02<=row['runtime']<=3.)
        fallback=canonical(known['plan'])
        assert not plan_has_crossjoin(spec,fallback.copy())
        tasks[task]=dict(warmup=warm,imported_bao_observations=49,fallback=fallback,
                         plan=fallback,sql=_decode_query(spec,fallback.copy()),sql_sha256=digest(sql),
                         historical_runtime=known['runtime'])
    (ROOT/'tasks.json').write_text(json.dumps(tasks,indent=2))
    token=ROOT/'service_token';token.write_text(secrets.token_hex(32));token.chmod(0o600)
    workers=[]
    for index in range(20):
        local=index<7
        start=80+index*3 if local else (index-7)*3
        name='w%02d'%index
        disk=index==6
        data=('/root/blockdata' if disk else '/dev/shm')+'/query_fsbo_pilot_20261106_v1/'+name
        workers.append(dict(name=name,host='LLMOptimizer2' if local else 'LLMOptimizer4',
                            cpus=list(range(start,start+3)),optimizer_cpu=start,
                            db_port=19700+index,service_port=19600+index,pgdata=data,
                            storage='local_disk' if disk else 'tmpfs',task=selected[index%len(selected)]))
        (ROOT/'workers'/name).mkdir(parents=True,exist_ok=True)
    pinned=[TRANSFER/'fsbo/weights',TRANSFER/'corpus/manifest.json',TRANSFER/'fsbo/status.json',ROOT/'tasks.json',
            SOURCE/'train800.txt',SOURCE/'test200.txt',NATIVE/'initialization_data/ceb_3k_bao_initialization.csv',
            Path('/root/data/cys/baseline/repos/FSBO/fsbo_modules.py'),
            Path('/root/data/cys/baseline/repos/FSBO/fsbo_utils.py')]
    pinned += [TOOLS/name for name in ['fsbo_target_refit.py','query_fsbo_pilot_worker.py',
                                     'query_fsbo_pilot_db.py','query_fsbo_pilot_host.py','query_vae_client.py']]
    plan=dict(domain='ceb_query_runtime',method='fsbo',purpose='readiness_and_bounded_pilot_only',
              source=str(SOURCE),transfer=str(TRANSFER),weights=str(TRANSFER/'fsbo/weights'),
              seed=123,representation=manifest['representation'],lower=manifest['lower'],upper=manifest['upper'],
              fsbo_repo='/root/data/cys/baseline/repos/FSBO',
              model_python='/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python',
              query_python='/root/data/cys/envs/bolt_clean_venv/bin/python',
              vae_pool='/root/data/logs/query_vae_cpu_capacity_20261106_v1/pool.json',
              tasks_file=str(ROOT/'tasks.json'),token_file=str(token),
              database_archive='/root/data/cys/logs/semanticopt_recovery_20261007/pg_imdb_d.clean_shutdown.tar',
              postgres_settings=json.loads((SOURCE/'plan.json').read_text())['postgres_settings'],
              memory_headroom_gib={'LLMOptimizer2':30,'LLMOptimizer4':32},
              allocation={'LLMOptimizer2':7,'LLMOptimizer4':13},workers=workers,
              pinned_files={str(p):digest(p) for p in pinned},
              acquisition_wrapper=None,optimization_campaign_started=False,
              tmpfs_resize=dict(attempted=True,succeeded=False,reason='Container denied tmpfs remount'),
              created=time.time())
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(status='prepared',workers=len(workers),allocation=plan['allocation'],
                         warmup={task:len(value['warmup']) for task,value in tasks.items()},
                         acquisition_wrapper=None,optimization_campaign_started=False)),flush=True)


if __name__=='__main__':
    main()
