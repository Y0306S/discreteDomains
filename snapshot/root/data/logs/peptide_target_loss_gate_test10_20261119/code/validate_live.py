"""Audit real GPU integration and degraded-checkpoint rollback before launch."""
import copy
import json
from pathlib import Path
import time

from common import *


def main():
    import torch
    import numpy as np
    from Levenshtein import distance
    from target_gate import guarded_retrain
    from reconstruction import audit
    setup()
    directory=ROOT/'validation_fullscale/adapted_online/pep_219'
    result=read(directory/'result.json');event=result['adaptation_events'][0]
    meta=event['source_meta_retraining']
    saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    checks=dict(live_gpu_update_accepted=event['accepted'] and event['vae_change_l2']>0,
        target_loss_did_not_worsen=event['target_loss_after']['mean_target_loss']<=event['target_loss_before']['mean_target_loss']+1e-12,
        two_real_optimizer_steps=event['optimizer_steps']==2,
        scratch_FSBO_rebuilt=meta['initialization']=='scratch' and meta['reload_verified'],
        separate_four_sources_plus_current=meta['source_tasks']==4 and meta['tasks_sampled']==5 and meta['all_tasks_sampled'],
        post_update_decoding_and_scoring=result['events']==3 and result['recenter_fresh']>0)
    requests=[json.loads(s) for s in (directory/'oracle_requests.jsonl').read_text().splitlines()]
    checks['oracle_ledger_matches_fresh']=sum(v['physical_requested'] for v in requests)==result['fresh_evaluations']
    checks['request_event_ids_unique']=len({r['event'] for r in requests})==len(requests)
    seen=set();fresh=0
    for row in result['observations']:
        if row['phase']!='initialization':
            assert row['cached']==(row['sequence'] in seen)
            assert row['feasible']==(1-distance(row['sequence'],result['reference'])/len(result['reference'])>=.75)
            fresh+=not row['cached']
        seen.add(row['sequence'])
    checks['cache_and_similarity_accounting']=fresh==result['fresh_evaluations']
    vae,data=load_vae(ROOT/'vae/weights.pt')
    vae.load_state_dict(saved['vae'])
    encoded=encode(vae,data,[r['sequence'] for r in saved['state']['rows']])
    checks['saved_coordinates_match_retained_VAE']=np.allclose(encoded,saved['state']['features'],rtol=1e-5,atol=1e-5)
    # A known degraded real checkpoint exercises the new numerical gate on GPU.
    parent=Path(prepare()['parent_experiment'])
    initial=torch.load(ROOT/'vae/weights.pt',map_location='cpu',weights_only=False)
    candidate=torch.load(parent/'evaluation/adapted_online/pep_999/checkpoint.pt',map_location='cpu',weights_only=False)['vae']
    vae.load_state_dict(initial)
    old_result=read(parent/'evaluation/adapted_online/pep_999/result.json')
    state=dict(rows=[r for r in old_result['observations'] if r['phase']=='initialization'])
    def swap(records):vae.load_state_dict(candidate)
    rejection=guarded_retrain(vae,data,state,swap,audit,read(ROOT/'smoke_training.json')['before'])
    checks['real_GPU_target_degradation_rejected']=not rejection['accepted'] and 'target_loss_worse_than_pre_update' in rejection['rejection_reasons']
    checks['GPU_rollback_restores_every_tensor']=all(torch.equal(v,initial[k]) for k,v in cpu_state(vae).items())
    value=dict(passed=all(checks.values()),checks=checks,host=__import__('socket').gethostname(),
        gpu=torch.cuda.get_device_name(),validation_oracle_calls=result['fresh_evaluations'],
        validation_directory=str(directory),production_oracle_calls=0,
        component_budget='GPU VAE update + four real source tasks/current task,50 FSBO steps; production800/100000 unchanged',
        target_degradation_rejection=rejection,
        code_sha256={str(p):sha(p) for p in (ROOT/'code').glob('*.py')},validated_at=time.time())
    write(ROOT/'validation/live_gpu.json',value)
    print(json.dumps(value,indent=2));assert value['passed'],checks


if __name__=='__main__':main()
