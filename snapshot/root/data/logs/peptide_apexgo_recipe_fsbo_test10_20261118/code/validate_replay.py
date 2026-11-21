"""Real training, source storage, resume and new online/FSBO integration checks."""
from common import *


def main():
    import numpy as np
    import torch
    from Levenshtein import distance
    from evaluate import FSBO
    setup();cfg=prepare();checks={}
    checks.update(read(ROOT/'smoke_training.json')['checks'])
    ds=read(ROOT/'sequence_data/manifest.json')
    checks['all800_source_tasks_in_sequence_corpus']=ds['source_tasks']==cfg['train_tasks'] and ds['rows']==4053113
    arrays={k:np.load(ROOT/'sequence_data'/(k+'.npy'),mmap_mode='r') for k in ['train','val','test']}
    train_set=set(arrays['train']);val_set=set(arrays['val']);test_set=set(arrays['test'])
    checks['actual_sequence_splits_disjoint']=not train_set&val_set and not train_set&test_set and not val_set&test_set
    checks['actual_sequence_splits_unique']=len(train_set)+len(val_set)+len(test_set)==ds['unique_sequences']
    del train_set,val_set,test_set,arrays
    store=read(ROOT/'source_snapshot/storage_manifest.json');db=corpus_db()
    for task in cfg['train_tasks']:
        digest,payload=db.execute('SELECT payload_sha256,payload FROM tasks WHERE id=?',(task,)).fetchone()
        assert digest==store['payload_hashes'][task]==hashlib.sha256(payload).hexdigest()
    checks['all800_private_source_payloads_match']=True;db.close()
    initial=read(ROOT/'validation_initial_fsbo/meta_status.json')
    checks['initial_scratch_FSBO_uses_all_component_tasks']=initial['all_tasks_sampled'] and initial['tasks_sampled']==4 and initial['reload_verified'] and initial['current_task'] is None
    path=ROOT/'validation_fullscale/adapted_online/pep_219';result=read(path/'result.json')
    checks['online_update_and_following_acquisition']=result['e2e_updates']==1 and result['rounds']>=3
    event=result['adaptation_events'][0]
    checks['online_VAE_reconstruction_KL_only']=event['objective']=='reconstruction+0.0001*KL'
    checks['online_VAE_accepted_real_weight_change']=event['accepted'] and event['vae_change_l2']>0
    meta=next((path/'source_refresh').glob('round_*'));manifest=read(meta/'manifest.json');status=read(meta/'meta_status.json')
    checks['all_component_sources_plus_current_trained']=status['status']=='complete' and status['tasks_sampled']==5 and status['all_tasks_sampled']
    checks['source_episodes_remain_separate']=set(manifest['offsets'])==set(manifest['source_tasks']+['pep_219'])
    checks['scratch_training_and_selected_reload']=status['initialization']=='scratch' and status['reload_verified']
    start=torch.load(meta/'initial_weights.pt',map_location='cpu',weights_only=False)
    selected=torch.load(meta/'weights',map_location='cpu',weights_only=False)
    checks['fresh_FSBO_network_trained']=any(not torch.equal(v,selected['net'][k]) for k,v in start['net'].items())
    saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
    checks['refreshed_bounds_persisted']=np.allclose(saved['fsbo_low'],manifest['lower']) and np.allclose(saved['fsbo_scale'],manifest['scale'])
    fsbo=FSBO(OLD/'fsbo/weights',read(OLD/'corpus/manifest.json'),max_updates=2)
    fsbo.gp.load_state_dict(selected['full'],strict=True);checks['new_prior_loads_strictly']=True
    ledger=[json.loads(line) for line in (path/'oracle_requests.jsonl').read_text().splitlines()];events=[x['event'] for x in ledger]
    checks['saved_scores_not_requested_twice']=len(events)==len(set(events)) and events.count(1)==1
    checks['fresh_oracle_accounting']=sum(x['physical_requested'] for x in ledger)==result['fresh_evaluations']
    seen=set();fresh=0
    for row in result['observations']:
        if row['phase']!='initialization':
            assert row['cached']==(row['sequence'] in seen)
            assert row['feasible']==(1-distance(row['sequence'],result['reference'])/len(result['reference'])>=.75)
            fresh+=not row['cached']
        seen.add(row['sequence'])
    checks['similarity_and_cache_accounting']=fresh==result['fresh_evaluations']
    checks['feasible_new_candidates_after_update']=any(r['phase']!='initialization' and r['feasible'] for r in result['observations'])
    vae,data=load_vae(cfg['author_vae']);vae.load_state_dict(saved['vae'])
    actual=encode(vae,data,[r['sequence'] for r in saved['state']['rows']])
    checks['target_coordinates_current_after_resume']=np.allclose(actual,saved['state']['features'],rtol=1e-5,atol=1e-5)
    # Inject a failed decoder audit after a real optimizer step, and verify
    # that neither the changed VAE nor an unnecessary FSBO refresh is adopted.
    import reconstruction
    from evaluate import joint_update
    original_audit=reconstruction.audit;calls=[];before=cpu_state(vae)
    def rejected_audit(*args,**kwargs):
        value=original_audit(*args,**kwargs);calls.append(True)
        if len(calls)==2:value=dict(value,feasible=0)
        return value
    reconstruction.audit=rejected_audit
    try:
        tiny=dict(rows=saved['state']['rows'][:100],constraints=saved['state']['constraints'][:100])
        rejection=joint_update(vae,data,None,None,None,tiny)
    finally:reconstruction.audit=original_audit
    checks['bad_online_decoder_update_rolled_back']=(not rejection['accepted'] and
        all(torch.equal(v,vae.state_dict()[k].cpu()) for k,v in before.items()))
    value=dict(passed=all(checks.values()),checks=checks,
        component_note='Four real source tasks plus current task exercise online scratch retraining; production uses800 and100000steps.',
        code_sha256={str(p):sha(p) for p in (ROOT/'code').glob('*.py')},config_sha256=sha(ROOT/'config.json'))
    write(ROOT/'validation.json',value);print(json.dumps(value),flush=True);assert value['passed']


if __name__=='__main__':main()
