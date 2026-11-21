"""Live batch JSON probe on the training smoke task; no held-out oracle budget."""
import sys,time
from common import *
from batch import ensure_batch, parse_batch
from evaluate import connect,get,put,peptide_setup,query_setup

def main(domain):
    recipe=domain_recipe(verify(),domain);settings=recipe[domain];task=settings['smoke_task']
    directory=ROOT/domain/'batch_smoke'/task
    if domain=='query':directory=directory/recipe['batch_policy_sha256'][:12]
    guard=lock(directory/'task.lock')
    db=connect(directory)
    context,initial,validate,_=(peptide_setup(task,settings) if domain=='peptide' else query_setup(task,settings,0))
    from transformers import AutoTokenizer
    from qwen_inference_transport import SharedQwenClient
    import torch
    torch.manual_seed(recipe['seed'])
    config=ROOT/domain/'services/client.json'
    if os.environ.get('SEMANTICOPT_BATCH_PROBE_URL'):
        cfg=load(config);cfg['url']=os.environ['SEMANTICOPT_BATCH_PROBE_URL']
        config=directory/'client.json';atomic(config,cfg)
    client=SharedQwenClient(config,settings['base'],settings['adapter'])
    os.environ['ML_QWEN_REQUEST_NAMESPACE']=str(directory);os.environ['ML_RUN_ID']=ROOT.name
    client.import_state(get(db,'inference_rng'))
    if not get(db,'batch_request_0_0',{}).get('response'):client.ensure_loaded()
    tokenizer=AutoTokenizer.from_pretrained(settings['adapter'],local_files_only=True,trust_remote_code=True)
    size=recipe['batch_policy']['size']
    # One response suffices to validate the live batch wire format; production
    # keeps its full retry allowance. This probe never spends a test oracle slot.
    recipe=dict(recipe,batch_policy=dict(recipe['batch_policy'],max_requests=1))
    batch=ensure_batch(db,get,put,recipe=recipe,domain=domain,context=context,initial=initial,rows=[],total=size,
        validate=validate,seen={canonical(r['configuration']) for r in initial},tokenizer=tokenizer,client=client)
    raw=get(db,'batch_request_0_0');raw_count=0
    try:raw_count=len(parse_batch(raw['response']))
    except Exception:pass
    llm=sum(i['provenance']['source']=='llm' for i in batch['items'])
    value=dict(passed=len(batch['items'])==size and raw_count==size,
        at=time.time(),task=task,batch_policy_sha256=recipe['batch_policy_sha256'],
        requested=size,probe_max_requests=1,first_response_items=raw_count,llm_valid_items=llm,
        random_fallback_items=size-llm,llm_requests=batch['llm_requests'],
        batch_code_sha256=digest(ROOT/'code/batch.py'),adapter_sha256=settings['adapter_sha256'])
    atomic(ROOT/domain/'batch_preflight.json',value);print(canonical(value),flush=True)
    if not value['passed']:raise RuntimeError(f'Expected a complete JSON response with {size} items')

if __name__=='__main__':main(sys.argv[1])
