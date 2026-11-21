"""Real-model integration, approximate GP numerics and saved-score recovery checks."""
import io
import json
import subprocess
import time
from common import *
import numpy as np
import torch


def main():
    started=time.monotonic()
    setup()
    cfg=prepare()
    checks={}
    checks['train800_test200_disjoint']=len(cfg['train_tasks'])==800 and not set(cfg['tasks'])&set(cfg['train_tasks'])
    smoke=read(ROOT/'smoke_offline/status.json')
    checks['real_native_lolbo_joint_update']=smoke['status']=='complete' and smoke['vae_parameter_change_l2']>0
    path=ROOT/'validation_fullscale/adapted_online/pep_219'
    result=read(path/'result.json')
    checks['online_joint_update_and_following_acquisition']=(result['e2e_updates']==1 and result['rounds']>=2
        and all(e['vae_change_l2']>0 and e['fsbo_change_l2']>0 for e in result['adaptation_events']))
    ledger=[json.loads(s) for s in (path/'oracle_requests.jsonl').read_text().splitlines()]
    events=[r['event'] for r in ledger]
    checks['fresh_process_scored_pending_no_rescore']=events.count(1)==1 and len(events)==len(set(events)) and not (path/'pending.pt').exists()
    checks['logical_physical_fresh_counts']=sum(x['physical_requested'] for x in ledger)==result['fresh_evaluations']
    checks['full_5000_candidate_pool']=all(e['maximum_updates']==1000 for e in result['fit_events'])
    from Levenshtein import distance
    seen=set()
    fresh=0
    for r in result['observations']:
        if r['phase']!='initialization':
            assert r['cached']==(r['sequence'] in seen)
            assert r['feasible']==(1-distance(r['sequence'],result['reference'])/len(result['reference'])>=.75)
            fresh+=not r['cached']
        seen.add(r['sequence'])
    checks['recenter_infeasible_duplicate_budget_accounting']=fresh==result['fresh_evaluations']
    # Validate that refreshed source-embedding tables work with the exact old trainer.
    source_manifest=read(OLD/'corpus/manifest.json')
    tasks=source_manifest['fsbo_train_tasks'][:2]+source_manifest['fsbo_validation_tasks'][:2]
    vae,data=load_vae(ROOT/'smoke_offline/weights.pt')
    db=corpus_db()
    directory=ROOT/'validation_corpus'
    directory.mkdir(exist_ok=True)
    target=sqlite3.connect(str(directory/'corpus.sqlite'))
    target.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,sha TEXT NOT NULL,payload BLOB NOT NULL)')
    low,high=np.full(256,np.inf),np.full(256,-np.inf)
    hashes={}
    for task in tasks:
        d=task_data(db,task)
        x=encode(vae,data,d['sequences'][:100].tolist())
        low=np.minimum(low,x.min(0));high=np.maximum(high,x.max(0))
        stream=io.BytesIO();np.savez_compressed(stream,X=x,y=d['y'][:100])
        payload=stream.getvalue();digest=hashlib.sha256(payload).hexdigest();hashes[task]=digest
        with target:
            target.execute('INSERT OR REPLACE INTO tasks VALUES(?,?,?)',(task,digest,payload))
    manifest=dict(source_manifest,source_tasks=tasks,fsbo_train_tasks=tasks[:2],fsbo_validation_tasks=tasks[2:],
        representation='real_joint_adapted_vae_component_only',lower=low.tolist(),upper=high.tolist(),
        database=str(directory/'corpus.sqlite'),task_hashes=hashes)
    write(directory/'manifest.json',manifest)
    target.close()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (ROOT/'validation_meta.log').open('ab') as output:
        subprocess.run(['/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python','-u',str(ROOT/'code/train_fsbo.py'),'--component'],
            stdout=output,stderr=subprocess.STDOUT,env=env,check=True)
    status=read(ROOT/'validation_meta/status.json')
    checks['exact_fsbo_trainer_new_embeddings_and_reload']=status['status']=='complete' and status['reload_verified']
    # Exercise production-size iterative GP likelihood and posterior, not just N<=800 Cholesky.
    from evaluate import FSBO
    d=task_data(db,cfg['train_tasks'][0])
    fsbo=FSBO(OLD/'fsbo/weights',source_manifest,max_updates=2)
    fit=fsbo.fit(d['X'],d['y'],123)
    center=torch.as_tensor(d['X'][0],device='cuda')
    candidates=center[None,:]+.01*torch.randn(5000,256,device='cuda')
    with torch.no_grad():
        draws=fsbo.posterior(candidates).rsample(torch.Size([50]))
    checks['production_5100_observation_iterative_likelihood']=(fit['likelihood']=='seeded_probe_lanczos'
        and abs(fit['restored_loss']-fit['best_loss'])<1e-5)
    checks['production_5000_candidate_joint_posterior']=draws.shape==(50,5000,1) and bool(torch.isfinite(draws).all())
    del fsbo,draws,candidates
    saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
    vae.load_state_dict(saved['vae'],strict=True)
    state=saved['state']
    # Every target feature must use the current VAE after its last update.
    actual=encode(vae,data,[r['sequence'] for r in state['rows']])
    checks['all_target_embeddings_refresh_and_resume']=bool(np.allclose(actual,state['features'],rtol=1e-5,atol=1e-5))
    validation=dict(passed=all(checks.values()),checks=checks,elapsed_seconds=time.monotonic()-started,
        numerical_fit=fit,online_validation_code_sha256=result['identity']['code_sha256'],
        code_sha256={str(p):sha(p) for p in (ROOT/'code').glob('*.py')})
    write(ROOT/'validation.json',validation)
    print(json.dumps(validation),flush=True)
    if not validation['passed']:
        raise RuntimeError('Real integration validation failed')
    db.close()


if __name__=='__main__':
    main()
