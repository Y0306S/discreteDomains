import json,sys,time,secrets,subprocess,os,hashlib
from pathlib import Path
from query_protocol import atomic,digest
from qwen_inference_transport import model_identity
ROOT=Path(__file__).resolve().parents[1]
WARM=Path('/root/data/cys/baseline/campaigns/query_warmstart_full800_20261112')
TRAIN=Path('/root/data/cys/baseline/campaigns/query_our_orpo_rl800_20261112')
PARENT=Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1/recipe.json')
BASE='/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B'
if (ROOT/'recipe.json').exists(): raise RuntimeError('Immutable recipe already exists')
for d in ['services','logs','evidence','pools']:(ROOT/d).mkdir(parents=True,exist_ok=True)
source=json.loads(PARENT.read_text())
models={}
for stage,b,a in [('warm_start',str(WARM/'model/checkpoint-1224'),None),('orpo',BASE,str(TRAIN/'models/orpo/checkpoint-103'))]:
 models[stage]=dict(identity=model_identity(b,a),checkpoint=a or b)
assert models['orpo']['identity']['adapter_sha256']=='0b331b7c11f03717742aaf735c782ffe311079d2df6678c1e5cc7f7fa3690bf3'
weights=json.loads((WARM/'evidence/final_weights.json').read_text());assert weights['passed']
print('Rehashing full BOLT checkpoint',flush=True)
for n,h in weights['shards'].items():
 assert digest(Path(models['warm_start']['checkpoint'])/n)==h,n
assert digest(Path(models['warm_start']['checkpoint'])/'config.json')==weights['model_config_sha256']
assert digest(source['fsbo_weights'])=='a7eddba4ba1de5b52904a6dcdf5589578643b6ae39774f475886c18882900e22'
assert digest(Path(source['native'])/'vae/CEB_64_author.ckpt')=='83a4942701bffaaf8311887e530e3157b6843134e52e9f53d59f957c24aebab1'
source.update(batch_size=1,initialization=50,n_candidates=5000,seed=1,vae_pool=str(ROOT/'services/vae_pool.json'))
source['pinned_files']={p:digest(p) for p in source['pinned_files']}
atomic(ROOT/'fsbo_recipe.json',source)
recipe=dict(format='query-our-bolt50-orpo-fsbo-v1',name=ROOT.name,tasks=source['test'],train_tasks=source['train'],seed=1,
 models=models,timeout_seconds=50,warm_start=dict(count=50,maximum_samples=500,random_fallback=False,acceptance='first_valid_unique_generation_order'),
 optimization=dict(attempts=80,llm_generation_limit=5,fsbo_decode_limit=5,initial_llm_attempts=8,
 selector='PUCB_p1=.7_q=arm_best_c2=1.4',schedule_denominator=91,cooldown='attempts10..20_after_selected_LLM',
 random_fallback=False,fsbo_region='fresh_native_TuRBO_region_at_current_incumbent_each_FSBO_selection',
 censored='excluded_from_exact_GP_fit_and_incumbents',warm_updates_selector=False),
 sampling=dict(temperature=1.,top_p=.9,top_k=50,max_new_tokens=1024),
 stages=['warm_start','orpo'],rl_enabled=False,fsbo_recipe=str(ROOT/'fsbo_recipe.json'),
 warm_shards=weights['shards'],parent_recipe_sha256=digest(PARENT),workers=8,created=time.time())
assert len(set(recipe['tasks']))==200 and len(set(recipe['train_tasks']))==800
assert not set(recipe['tasks'])&set(recipe['train_tasks'])
atomic(ROOT/'recipe.json',recipe)
atomic(ROOT/'evidence/warm_weight_rehash.json',dict(passed=True,recipe_sha256=digest(ROOT/'recipe.json'),at=time.time(),shards=weights['shards']))
for gpu in range(4):
 stage='warm_start' if gpu<3 else 'orpo';i=models[stage]['identity']
 cfg=dict(name=f'query_hybrid_{stage}_g{gpu}',base=i['base'],adapter=i['adapter'],identities=[i],gpu=gpu,cpus=[32+gpu],port=21800+gpu,
 memory_gib=30,max_input_tokens=15360,max_new_tokens=1024,prompt_cache=dict(enabled=True,max_bytes=1610612736,min_tokens=128))
 atomic(ROOT/'services'/f'replica{gpu}.json',cfg)
for stage,port,gpus in [('warm_start',21810,range(3)),('orpo',21811,[3])]:
 atomic(ROOT/'services'/f'{stage}.broker.json',dict(name='query_hybrid_'+stage,port=port,cpus=[37],identities=[models[stage]['identity']],replicas=[f'http://127.0.0.1:{21800+g}' for g in gpus],journal=str(ROOT/'services'/f'{stage}.requests.sqlite'),task_affinity=True))
(ROOT/'services/token').write_text(secrets.token_hex(32));os.chmod(ROOT/'services/token',0o600)
atomic(ROOT/'services/vae_pool.json',dict(domain='ceb_query_runtime',token_file=str(ROOT/'services/token'),workers=[dict(url='http://127.0.0.1:21820')]))
atomic(ROOT/'status.json',dict(status='preflight',tasks=200,warm_measurements=0,orpo_attempts=0,rl_enabled=False))
print('Prepared',ROOT,flush=True)
