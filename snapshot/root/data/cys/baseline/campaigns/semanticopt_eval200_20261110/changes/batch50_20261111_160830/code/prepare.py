"""Freeze held-out splits, training identities, protocols and service configs."""
import shutil
from common import *

def prepare():
    if (ROOT/'identity.json').exists():raise RuntimeError('Campaign already frozen')
    source_query=load('/root/data/logs/query_fsbo_test200_50s_20261106_v1/recipe.json')
    source_peptide=load('/root/data/logs/peptide_tpe_tr_test200_20261107_v1/recipe.json')
    recipe=dict(method='SemanticOPT',format='semanticopt-discrete-eval-v1',seed=42,
        stages=['query','peptide'],dependency=str(DEPENDENCY/'orpo_rl/evaluation_complete.json'),
        replicas=4,gpus=[0,1,2,3],replicas_per_gpu=1,reserved_replica_slots_per_gpu=2,candidate_retries=5,
        acquisition='two_validated_candidate_responses_success_weighted_expected_improvement_one_evaluation',
        fallback=False,latent_space=False,thompson_sampling=False,
        generation=dict(temperature=.7,max_new_tokens=1024,top_p=.95,top_k=50,enable_thinking=False),
        cpu_policy=dict(LLMOptimizer2=dict(quota=40,reserved=6),LLMOptimizer4=dict(quota=64,reserved=2),
            query_cores_per_worker=3,peptide_cores_per_worker=1,memory_reserve_gib=12,adaptive=True))
    files={}
    for domain,source in [('query',source_query),('peptide',source_peptide)]:
        training=load(TRAIN/domain/'training_status.json')
        adapter=TRAIN/domain/'model/final_adapter'
        if training['status']!='completed' or not training['saved_adapter_matches_live'] or \
            digest(adapter/'adapter_model.safetensors')!=training['adapter_sha256']:
            raise ValueError('Training adapter verification failed: '+domain)
        identity=model_identity(training['base_model'],adapter)
        test,train=source['test'],source['train']
        if len(test)!=200 or len(set(test))!=200 or len(train)!=800 or set(test)&set(train):
            raise ValueError('Invalid 800/200 split')
        settings=dict(test=test,train=train,smoke_task=train[0],base=training['base_model'],
            adapter=str(adapter),adapter_sha256=training['adapter_sha256'],identity=identity,
            initialization=source['initialization'],proposals=source['proposals'])
        if domain=='query':
            settings.update(timeout_seconds=50,database_archive='/root/data/cys/logs/semanticopt_recovery_20261007/pg_imdb_d.clean_shutdown.tar')
            paths=[QUERY_SOURCE/'test200.txt',QUERY_SOURCE/'train800.txt',
                QUERY_SOURCE/'native/initialization_data/ceb_3k_bao_initialization.csv']
            paths += [QUERY_SOURCE/'native'/name for name in ['codec/codec.py','workload/workloads.py','oracle/oracle.py']]
        else:
            settings.update(pool=source['prepared_pool'],cpu_upstream=source['cpu_upstream'])
            # Source smoke task is train800: use its audited initialization pool, not test200.
            train_pool=load('/root/data/logs/peptide_bo_train800_20261027_v5b/initialization.json')
            test_pool=load(source['prepared_pool'])
            combined=dict(test_pool)
            combined[train[0]]=train_pool[train[0]]
            atomic(ROOT/domain/'initialization.json',combined)
            settings['pool']=str(ROOT/domain/'initialization.json')
            paths=[Path(source['prepared_pool']),Path(settings['pool'])]
            paths += [Path(p) for p in source['code'] if '/upstream_cpu/apex_oracle/' in p]
            paths += [Path(p) for p in source['assets'] if 'APEX_pathogen_models' in p]
        recipe[domain]=settings
        svc=ROOT/domain/'services';svc.mkdir(parents=True,exist_ok=True)
        for i in range(4):
            atomic(svc/f'replica{i:02}.json',dict(name=f'semanticopt_{domain}_gpu{i}_replica0',
                base=settings['base'],adapter=str(adapter),identities=[identity],gpu=i,cpus=[i],
                memory_gib=21,max_input_tokens=15360,max_new_tokens=1024,port=19360+i))
        atomic(svc/'broker.json',dict(name=f'semanticopt_{domain}_broker',port=19990,cpus=[4],identities=[identity],
            replicas=[f'http://127.0.0.1:{19360+i}' for i in range(4)],journal=str(svc/'requests.sqlite')))
        atomic(svc/'client.json',dict(url='http://127.0.0.1:19990',identities=[identity],request_timeout_seconds=3600))
        paths += list(adapter.glob('*.json'))+list(adapter.glob('*.jinja'))+[adapter/'adapter_model.safetensors']
        files.update({str(path):digest(path) for path in paths})
    atomic(ROOT/'recipe.json',recipe)
    paths=list((ROOT/'code').glob('*.py'))+list((ROOT/'services').glob('*.py'))+[ROOT/'recipe.json']
    paths += [p for domain in ['query','peptide'] for p in (ROOT/domain/'services').glob('*.json')]
    files.update({str(path):digest(path) for path in paths})
    atomic(ROOT/'identity.json',dict(created_at=time.time(),files=files))
    print(canonical(dict(prepared=True,stages=recipe['stages'],tasks_per_stage=200,replicas=4,reserved_slots_per_gpu=2)))

if __name__=='__main__':prepare()
