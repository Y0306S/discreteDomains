"""Audit real replay training, saved-score recovery, and resumed coordinates."""
from common import *


def main():
    import numpy as np
    import torch
    from Levenshtein import distance
    from evaluate import FSBO
    setup();cfg=prepare();checks={}
    checks['source800_disjoint_from_heldout']=len(cfg['train_tasks'])==800 and not set(cfg['train_tasks'])&set(cfg['tasks'])
    smoke=read(ROOT/'smoke_reconstruction.json')
    checks['stable_smoke_decoder_retention']=(smoke['stable_two_task_smoke']['feasible']>=.8*smoke['author']['feasible'] and
        smoke['stable_two_task_smoke']['reconstruction_loss']<=1.25*smoke['author']['reconstruction_loss'])
    path=ROOT/'validation_fullscale/adapted_online/pep_219'
    result=read(path/'result.json')
    checks['joint_update_retrain_and_following_acquisition']=result['e2e_updates']==1 and result['rounds']>=3
    meta=path/'source_refresh/round_001';manifest=read(meta/'manifest.json');status=read(meta/'meta_status.json')
    checks['all_component_sources_plus_current_trained']=status['status']=='complete' and status['tasks_sampled']==5 and status['all_tasks_sampled']
    checks['source_episodes_remain_separate']=set(manifest['offsets'])==set(manifest['source_tasks']+['pep_219'])
    checks['scratch_training_and_selected_reload']=status['initialization']=='scratch' and status['reload_verified']
    initial=torch.load(meta/'initial_weights.pt',map_location='cpu',weights_only=False)
    selected=torch.load(meta/'weights',map_location='cpu',weights_only=False)
    checks['fresh_network_actually_trained']=any(not torch.equal(v,selected['net'][k]) for k,v in initial['net'].items())
    saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
    checks['refreshed_bounds_persisted']=np.allclose(saved['fsbo_low'],manifest['lower']) and np.allclose(saved['fsbo_scale'],manifest['scale'])
    fsbo=FSBO(OLD/'fsbo/weights',read(OLD/'corpus/manifest.json'),max_updates=2)
    fsbo.gp.load_state_dict(selected['full'],strict=True)
    checks['new_prior_loads_strictly']=True
    ledger=[json.loads(line) for line in (path/'oracle_requests.jsonl').read_text().splitlines()]
    events=[x['event'] for x in ledger]
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
    checks['all_target_coordinates_current_after_resume']=np.allclose(actual,saved['state']['features'],rtol=1e-5,atol=1e-5)
    value=dict(passed=all(checks.values()),checks=checks,
        component_note='Four real source tasks and current task exercise the complete path; production uses all 800 sources and 100000 steps.',
        code_sha256={str(p):sha(p) for p in (ROOT/'code').glob('*.py')})
    write(ROOT/'validation.json',value);print(json.dumps(value),flush=True)
    assert value['passed']


if __name__=='__main__':main()
