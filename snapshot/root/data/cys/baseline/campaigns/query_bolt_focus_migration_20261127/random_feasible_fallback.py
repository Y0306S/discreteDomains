"""Explicitly scoped random connected-plan fallback, with durable provenance."""
import copy
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

POLICY_NAME='random_feasible_fallback_20261128.json'


def checkpoint_write(path,value):
    from bolt_proposal_recovery import write_checkpoint
    write_checkpoint(path,value)


def state_digest(state):
    return hashlib.sha256(pickle.dumps(dict(rows=state['rows'],turbo=state['turbo']),protocol=4)).hexdigest()


def draw_random(runner,task,seen,rng):
    from random_plans import query_proposal
    from workload.workloads import WorkloadSpec,OracleCodec
    from oracle.oracle import plan_has_crossjoin
    definition,codec,_,_=runner.definition(task)
    raw=query_proposal(definition,codec,rng)['plan']
    plan=runner.canonicalize(task,raw)
    plan=runner.canonicalize(task,plan,strict=True)
    if runner.plan_key(plan) in seen:
        raise ValueError('duplicate canonical plan')
    spec=WorkloadSpec.from_definition(definition,OracleCodec.Aliases)
    if plan_has_crossjoin(spec,list(plan)):
        raise ValueError('random plan has a cross join')
    return plan


def fill_random(runner,task,state,vae,number,checkpoint,path,policy):
    import numpy as np
    seed=runner.seed_for(task,'user_approved_random_feasible_fallback',number,runner.recipe()['seed'])
    rng=random.Random(seed)
    if checkpoint.get('random_rng_state') is not None:rng.setstate(checkpoint['random_rng_state'])
    seen={runner.plan_key(row['plan']) for row in state['rows']}
    seen.update(runner.plan_key(row['plan']) for row in checkpoint['accepted'])
    checkpoint.setdefault('random_draws',0)
    for _ in range(checkpoint['random_draws'],policy['max_random_draws']):
        if len(checkpoint['accepted'])==10:break
        checkpoint['random_draws']+=1
        try:plan=draw_random(runner,task,seen,rng)
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            checkpoint['rejected'].append(dict(source='random_feasible_fallback',error=str(exc),
                draw=checkpoint['random_draws']))
        else:
            checkpoint['accepted'].append(dict(plan=plan,proposed_z=None,
                proposal_source='random_feasible_fallback',fallback_type='random_connected_feasible',
                random_seed=seed,random_draw=checkpoint['random_draws'],
                decode_retries=None,raw_vae_tokens=None,native_canonicalized=True))
            seen.add(runner.plan_key(plan))
        checkpoint['random_rng_state']=rng.getstate()
        checkpoint_write(path,checkpoint)
    if len(checkpoint['accepted'])!=10:
        raise RuntimeError(f'{task}: cannot produce ten distinct feasible plans after bounded random fallback')
    pending=[row for row in checkpoint['accepted'] if row.get('proposed_z') is None]
    if pending:
        z=np.asarray(vae.encode([row['plan'] for row in pending]),dtype=np.float32)
        if z.shape!=(len(pending),64) or not np.isfinite(z).all():
            raise ValueError('Invalid random-plan posterior means')
        for row,latent in zip(pending,z.tolist()):row['proposed_z']=latent
    checkpoint['fit'].update(random_fallback_count=sum(
        row.get('proposal_source')=='random_feasible_fallback' for row in checkpoint['accepted']),
        random_fallback_seed=seed,random_draws=checkpoint['random_draws'],
        proposal_policy_amendment=policy['policy_path'],proposal_policy_sha256=policy['policy_sha256'])
    checkpoint_write(path,checkpoint)


def save_provenance(runner,task,number,checkpoint,policy):
    path=runner.ROOT/'runs'/task/'random_feasible_provenance.json'
    data=json.loads(path.read_text()) if path.exists() else dict(task=task,
        policy_path=policy['policy_path'],policy_sha256=policy['policy_sha256'],rounds={})
    if data['policy_sha256']!=policy['policy_sha256']:raise ValueError('Fallback provenance policy changed')
    random_items=[row for row in checkpoint['accepted'] if row.get('proposal_source')=='random_feasible_fallback']
    data['rounds'][str(number)]=dict(random_plans=[row['plan'] for row in random_items],
        random_count=len(random_items),latent_count=10-len(random_items),
        random_seed=checkpoint['fit'].get('random_fallback_seed'),
        random_draws=checkpoint.get('random_draws',0),
        prior_failed_proposal=checkpoint['fit'].get('prior_failed_proposal'),at=time.time())
    runner.atomic(path,data)


def batch_result(runner,task,number,checkpoint,policy):
    save_provenance(runner,task,number,checkpoint,policy)
    runner.atomic(runner.ROOT/'runs'/task/'proposal_progress.json',dict(task=task,round=number,
        latent_batches=checkpoint['next_retry'],latent_attempts=checkpoint['next_retry']*10,
        accepted=len(checkpoint['accepted']),rejected=len(checkpoint['rejected']),
        random_fallback_count=checkpoint['fit'].get('random_fallback_count',0),
        random_draws=checkpoint.get('random_draws',0),policy_amendment=policy['policy_path'],at=time.time()))
    return dict(items=checkpoint['accepted'],rejected=checkpoint['rejected'],fit=checkpoint['fit'])


def next_batch(runner,task,state,vae,number,policy):
    import torch
    path=runner.ROOT/'runs'/task/f'random_fallback_proposal_{number:03}.pickle'
    identity=dict(task=task,round=number,state_sha256=state_digest(state),
        recipe_sha256=runner.digest(runner.ROOT/'recipe.json'),policy_sha256=policy['policy_sha256'])
    checkpoint=None
    if path.exists():
        with path.open('rb') as stream:checkpoint=pickle.load(stream)
        if checkpoint['identity']!=identity:raise ValueError('Random fallback proposal checkpoint mismatch')
        if len(checkpoint['accepted'])==10 and all(row.get('proposed_z') is not None for row in checkpoint['accepted']):
            return batch_result(runner,task,number,checkpoint,policy)
    observed=[];observed_keys=set()
    for row in state['rows']:
        key=runner.plan_key(row['plan'])
        if not row['censored'] and key not in observed_keys:
            observed.append(row);observed_keys.add(key)
    if not observed:raise ValueError('No uncensored incumbent')
    seed=runner.seed_for(task,'lolbo',number,runner.recipe()['seed'])
    limit=runner.recipe()['proposal_policy']['max_latent_batches']
    if checkpoint is None:
        checkpoint=dict(identity=identity,accepted=[],rejected=[],next_retry=0,
            fit=dict(train_rows=len(observed),epochs=20 if number==1 else 2,seed=seed))
        if number==policy['failed_round']:
            # The exhausted original batch never persisted its accepted candidates.
            # None were measured. Replace that unfinished batch without replaying SQL.
            checkpoint['next_retry']=limit
            checkpoint['fit'].update(epochs=0,prior_failed_proposal=policy['failed_proposal'],
                original_partial_proposals_persisted=False,
                continuation='replace previously exhausted unmeasured batch with random feasible plans')
        checkpoint_write(path,checkpoint)
    if checkpoint['next_retry']<limit:
        x=torch.tensor([row['z'] for row in observed],dtype=torch.float32)
        y=torch.tensor([-row['runtime'] for row in observed],dtype=torch.float32).view(-1,1)
        device=torch.device('cpu')
        model,mll=runner.lolbo.make_surrogate(x,device)
        if checkpoint.get('model_state') is None:
            torch.manual_seed(seed % (2**63-1))
            model=runner.fit_lolbo(model,mll,x,y,20 if number==1 else 2)
            checkpoint['model_state']=model.state_dict();checkpoint_write(path,checkpoint)
        else:model.load_state_dict(checkpoint['model_state']);model.eval()
        turbo=runner.lolbo.TurboState(**state['turbo'])
        seen={runner.plan_key(row['plan']) for row in state['rows']}
        seen.update(runner.plan_key(row['plan']) for row in checkpoint['accepted'])
        for retry in range(checkpoint['next_retry'],limit):
            torch.manual_seed((seed+retry) % (2**63-1))
            with torch.no_grad():
                z=runner.lolbo.generate_batch(turbo,model,x,y,batch_size=10,n_candidates=2000,
                    device=device,unit_cube=False)
            if z.shape!=(10,64) or not torch.isfinite(z).all():raise ValueError('Invalid latent batch')
            latents=z.tolist();decoded=vae.decode(latents,seed=(seed+retry) % (2**31-1))
            if len(decoded)!=10:raise ValueError('Decoder response length changed')
            for latent,item in zip(latents,decoded):
                try:
                    plan=runner.canonicalize(task,item['plan']);plan=runner.canonicalize(task,plan,strict=True)
                    key=runner.plan_key(plan)
                    if item.get('fallback') or key in seen:raise ValueError('decoder fallback or duplicate')
                except (ValueError,TypeError,KeyError,IndexError) as exc:
                    checkpoint['rejected'].append(dict(retry=retry,error=str(exc),decoded=item));continue
                checkpoint['accepted'].append(dict(plan=plan,proposed_z=latent,raw_vae_tokens=item['plan'],
                    decode_retries=item.get('decode_retries'),native_canonicalized=item['plan']!=plan,
                    proposal_source='lolbo'))
                seen.add(key)
                if len(checkpoint['accepted'])==10:break
            checkpoint['next_retry']=retry+1;checkpoint_write(path,checkpoint)
            runner.atomic(runner.ROOT/'runs'/task/'proposal_progress.json',dict(task=task,round=number,
                latent_batches=retry+1,accepted=len(checkpoint['accepted']),
                rejected=len(checkpoint['rejected']),policy_amendment=policy['policy_path'],at=time.time()))
            if len(checkpoint['accepted'])==10:return batch_result(runner,task,number,checkpoint,policy)
    fill_random(runner,task,state,vae,number,checkpoint,path,policy)
    return batch_result(runner,task,number,checkpoint,policy)


def install(runner,task,operations):
    path=Path(operations)/POLICY_NAME
    if not path.exists():return
    config=json.loads(path.read_text())
    task_policy=config.get('tasks',{}).get(task)
    if task_policy is None:return
    policy=dict(task_policy,policy_path=str(path),policy_sha256=runner.digest(path),
        max_random_draws=config['max_random_draws'])
    original=runner.next_batch
    def amended(current,state,vae,number):
        if current!=task or number<policy['failed_round']:return original(current,state,vae,number)
        return next_batch(runner,current,state,vae,number,policy)
    runner.next_batch=amended
    original_observed=runner.observed
    def observed(plan,result,index,source):
        row=original_observed(plan,result,index,source)
        ledger=runner.ROOT/'runs'/task/'random_feasible_provenance.json'
        if ledger.exists():
            provenance=json.loads(ledger.read_text())
            key=runner.plan_key(plan)
            for number,data in provenance['rounds'].items():
                if any(runner.plan_key(candidate)==key for candidate in data['random_plans']):
                    row.update(source_phase='random_feasible_fallback',proposal_source='random_feasible_fallback',
                        proposal_policy_sha256=policy['policy_sha256'],proposal_round=int(number))
                    break
        return row
    runner.observed=observed
