"""Matched FSBO runs with optional task-specific VAE/kernel joint adaptation."""
import argparse
import copy
import json
import random
import time
from common import *

import numpy as np
import torch
import gpytorch

sys.path.insert(0,'/root/data/cys/baseline/repos/FSBO')
from fsbo_modules import DeepKernelGP
from fsbo_target_refit import fit_target
import peptide_local_policy as policy
from lolbo.utils.bo_utils.turbo import TurboState,update_state
from lolbo.utils.bo_utils.constrained_max_posterior_sampling import MaxPosteriorSampling


class FSBO:
    def __init__(self,weights,manifest,max_updates=1000):
        self.gp=DeepKernelGP(256,os.devnull,1,hidden_size=[128,128],kernel='rbf',ard=False,
            checkpoint=str(Path(weights).parent),epochs=max_updates)
        w=torch.load(weights,map_location='cpu',weights_only=False)
        self.gp.model.load_state_dict(w['gp'],strict=True)
        self.gp.likelihood.load_state_dict(w['likelihood'],strict=True)
        self.gp.feature_extractor.load_state_dict(w['net'],strict=True)
        self.prior=cpu_state(self.gp)
        self.low=torch.tensor(manifest['lower'],device='cuda',dtype=torch.float32)
        high=torch.tensor(manifest['upper'],device='cuda',dtype=torch.float32)
        self.scale=torch.where(high>self.low,high-self.low,1.)
        self.max_updates=max_updates

    def normalize(self,x):
        return (x-self.low)/self.scale

    def fit(self,x,y,seed):
        self.gp.load_state_dict(self.prior,strict=True)
        self.offset=float(np.min(y))
        self.yscale=float(np.max(y)-self.offset) or 1.
        x=self.normalize(torch.as_tensor(x,device='cuda'))
        y=torch.as_tensor((np.asarray(y)-self.offset)/self.yscale,device='cuda',dtype=torch.float32)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            result=fit_target(self.gp,x,y,self.max_updates,patience=16,min_delta=.0001,lr=.001,seed=seed)
        return result

    def posterior(self,x,observation_noise=False):
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        dist=self.gp.model(self.gp.feature_extractor(self.normalize(x)))
        return GPyTorchPosterior(dist)


def joint_update(vae,data,fsbo,constraint,cmll,state):
    """Train both VAE and exact FSBO; then the caller refreshes all coordinates."""
    from uniref_vae.load_vae import vae_forward
    d=dict(y=np.asarray([r['score'] for r in state['rows']]),
        constraints=state['constraints'].reshape(-1))
    ids=training_indices(d)
    sequences=[state['rows'][i]['sequence'] for i in ids]
    y=torch.as_tensor((d['y'][ids]-fsbo.offset)/fsbo.yscale,device='cuda',dtype=torch.float32)
    c=torch.as_tensor(d['constraints'][ids],device='cuda',dtype=torch.float32)
    model=fsbo.gp
    initial={k:p.detach().cpu().clone() for k,p in vae.named_parameters()}
    initial_gp={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
    vae.train();model.model.train();model.likelihood.train();model.feature_extractor.train();constraint.train()
    optimizer=torch.optim.Adam([
        {'params':vae.parameters()}, {'params':model.model.parameters()},
        {'params':model.feature_extractor.parameters()}, {'params':constraint.parameters()}],lr=.001)
    batch=max(1,2560//max(map(len,sequences)))
    losses=[]
    for epoch in range(2):
        for start in range(0,len(sequences),batch):
            optimizer.zero_grad(set_to_none=True)
            z,recon=vae_forward(sequences[start:start+batch],data,vae)
            features=model.feature_extractor(fsbo.normalize(z))
            model.model.set_train_data(features,y[start:start+batch],strict=False)
            objective_loss=-model.mll(model.model(features),y[start:start+batch])
            constraint_loss=-cmll(constraint(z),c[start:start+batch])
            loss=recon+objective_loss+constraint_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite VAE/FSBO joint update')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(),1.)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params']):
                raise ValueError('Nonfinite joint VAE/FSBO gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
    vae.eval();constraint.eval();constraint.likelihood.eval()
    delta=lambda a,b:sum(float((b[k]-v).square().sum()) for k,v in a.items() if v.is_floating_point())**.5
    dv=delta(initial,{k:p.detach().cpu() for k,p in vae.named_parameters()})
    dg=delta(initial_gp,{k:p.detach().cpu() for k,p in model.named_parameters()})
    if not np.isfinite([dv,dg]).all() or dv<=0 or dg<=0:
        raise ValueError('VAE and FSBO must both change at adaptation')
    fsbo.prior=cpu_state(model)
    return dict(optimizer_steps=len(losses),selected_observations=len(ids),first_loss=losses[0],
        last_loss=losses[-1],vae_change_l2=dv,fsbo_change_l2=dg)


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())


def rng_restore(rng):
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
    random.setstate(rng['python']);np.random.set_state(rng['numpy'])


def run(task,arm,max_events=None,force_update=False,stop_after_score=False):
    clock_start=time.monotonic()
    cfg=prepare()
    assert task in cfg['tasks'] and arm in cfg['arms']
    setup(seed=cfg['seed'])
    if (force_update or stop_after_score) and not max_events:
        raise ValueError('Forced updates and staged stops are validation controls only')
    directory=ROOT/('validation_fullscale' if max_events else 'evaluation')/arm/task
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists() and not max_events:
        return read(directory/'result.json')
    source=read(SOURCE/'recipe.json')
    original=arm=='author_frozen'
    manifest=read((OLD if original or max_events else ROOT)/'corpus/manifest.json')
    vae_path=Path(cfg['author_vae']) if original else ROOT/'vae/weights.pt'
    weights=(OLD if original else ROOT)/'fsbo/weights'
    # Smoke tests can exercise the full online path before offline training finishes.
    if max_events:
        vae_path=Path(cfg['author_vae'])
        weights=OLD/'fsbo/weights'
        manifest=read(OLD/'corpus/manifest.json')
    from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
    previous=os.getcwd()
    os.chdir(UPSTREAM)
    try:
        objective=ApexConstrainedDiverseObjective(task_id='apex',task_specific_args=['bacteria_0'],
            path_to_vae_statedict=str(vae_path),max_string_length=30)
    finally:
        os.chdir(previous)
    vae,data=objective.vae,objective.dataobj
    objective.objective_function.cache.clear()
    fsbo=FSBO(weights,manifest,max_updates=1000)
    pool=read('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json')[task]
    reference=source['references'][task]
    assert pool['reference']==reference
    recipe=read(BASE/'recipe.json')
    recipe.update(proposals=cfg['proposals'],fresh_budget=cfg['fresh_budget'])
    def constraints(seqs):
        from Levenshtein import distance
        return np.asarray([[.75-(1-distance(s,reference)/len(reference))] for s in seqs],dtype=np.float32)
    identity=dict(config_sha256=sha(ROOT/'config.json'),task=task,arm=arm,
        vae_sha256=sha(vae_path),fsbo_sha256=sha(weights),code_sha256=sha(__file__),
        smoke=bool(max_events),force_update=force_update)
    import subprocess
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    write(directory/'runtime_environment.json',dict(torch=torch.__version__,gpytorch=gpytorch.__version__,
        visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),hardware=hardware,
        cpu_threads=torch.get_num_threads(),tf32=False,identity=identity,
        timing_definition='In-process wall from run entry, includes model load and file I/O; excludes interpreter/import startup.'))
    checkpoint=directory/'checkpoint.pt'
    pending_path=directory/'pending.pt'
    recovered_event=None
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint.exists() else None
    if pending_path.exists():
        pending_saved=torch.load(pending_path,map_location='cpu',weights_only=False)
        if saved is None or saved['state']['progress']['events']<pending_saved['pending_event']['event']:
            saved=pending_saved
            recovered_event=saved['pending_event']
        else:
            pending_path.unlink()
    if saved is not None:
        assert saved['identity']==identity
        state=saved['state'];vae.load_state_dict(saved['vae']);fsbo.prior=saved['fsbo_prior']
        cm,cmll=make_gp(state['features'][:saved['inducing_count']])
        cm.load_state_dict(saved['constraint'])
        cmll.num_data=len(state['rows'])
        rng_restore(saved['rng'])
    else:
        xs,ys=pool['sequences'],pool['scores']
        features=encode(vae,data,xs)
        rows=[dict(sequence=s,score=float(y),mic=-float(y),phase='initialization',round=0,
            feasible=True,cached=True,elapsed=0.) for s,y in zip(xs,ys)]
        state=dict(rows=rows,features=features,constraint_z=features.copy(),constraints=constraints(xs),
            stagnation=0,recenter_plan=None,last_event_size=len(rows),last_phase=None,
            progress=dict(rounds=0,proposals=0,fresh_evaluations=0,cache_hits=0,
                acquisition_fresh=0,recenter_fresh=0,recenter_proposals=0,events=0,seconds=0.),
            timing=dict(initialization=0.,target_refit=0.,constraint_fit=0.,acquisition=0.,
                decode=0.,oracle=0.,encoding=0.,joint_update=0.,reencoding=0.,checkpoint=0.),
            adaptation_events=[],fit_events=[],wall_seconds=0.)
        if not (state['constraints']<=0).all():
            raise ValueError('Invalid starting peptide feasibility')
        state['trust_region']=policy.new_region(rows,state['constraints'],50,recipe['local'],TurboState)
        cm,cmll=make_gp(features)
        state['timing']['initialization']=time.monotonic()-clock_start
    accumulated_wall=state['wall_seconds']
    active_start=clock_start
    def snapshot_value():
        state['wall_seconds']=accumulated_wall+time.monotonic()-active_start
        return dict(identity=identity,state=state,vae=cpu_state(vae),fsbo_prior=fsbo.prior,
            constraint=cpu_state(cm),inducing_count=cm.variational_strategy.inducing_points.shape[0],rng=rng_state())
    def persist():
        before=time.monotonic()
        torch.cuda.synchronize()
        state['wall_seconds']=accumulated_wall+time.monotonic()-active_start
        save(checkpoint,snapshot_value())
        state['timing']['checkpoint']+=time.monotonic()-before
        result=dict(identity=identity,status='running',**state['progress'],timing=state['timing'],
            wall_seconds=state['wall_seconds'],e2e_updates=len(state['adaptation_events']),
            best_feasible_mic=state['rows'][policy.incumbent(state['rows'])]['mic'])
        write(directory/'status.json',result)
    while (state['progress']['proposals']<recipe['proposals'] and
           state['progress']['fresh_evaluations']<recipe['fresh_budget']):
        started=time.monotonic()
        if max_events and state['progress']['events']>=max_events:
            break
        if recovered_event is None:
            if force_update and not state['adaptation_events'] and state['progress']['events']==1:
                state['stagnation']=10
            previous_plan=state['recenter_plan']
            phase=policy.next_phase(state,recipe)
            new_recenter=(phase=='recenter' and state['recenter_plan'] is not previous_plan)
            if new_recenter and arm=='adapted_online':
                before=time.monotonic()
                y=np.asarray([r['score'] for r in state['rows']])
                fit=fsbo.fit(state['features'],y,1+state['progress']['events'])
                state['timing']['target_refit']+=time.monotonic()-before
                before=time.monotonic()
                cmll.num_data=len(state['rows'])
                event=joint_update(vae,data,fsbo,cm,cmll,state)
                torch.cuda.synchronize()
                event['joint_seconds']=time.monotonic()-before
                state['timing']['joint_update']+=event['joint_seconds']
                before=time.monotonic()
                state['features']=encode(vae,data,[r['sequence'] for r in state['rows']])
                state['constraint_z']=state['features'].copy()
                # Changed VAE coordinates require a fresh constraint model as well.
                cm,cmll=make_gp(state['features'][:100])
                cmll.num_data=len(state['rows'])
                cm=fit_gp(cm,cmll,state['constraint_z'],state['constraints'],20)
                event['reencoding_constraint_refit_seconds']=time.monotonic()-before
                state['timing']['reencoding']+=event['reencoding_constraint_refit_seconds']
                event.update(round=state['progress']['rounds'],observations=len(state['rows']),
                    fresh_evaluations=state['progress']['fresh_evaluations'])
                state['adaptation_events'].append(event)
            extra={}
            if phase=='recenter':
                plan=state['recenter_plan']
                end=min(len(plan['inputs']),plan['cursor']+plan['batch'])
                z=encode(vae,data,plan['inputs'][plan['cursor']:end],mean=False)
                extra['next_cursor']=end
            else:
                before=time.monotonic()
                full=state['progress']['events']==0 or state.get('last_phase')=='recenter'
                n=len(state['rows']) if full else state['last_event_size']
                cmll.num_data=len(state['rows'])
                cm=fit_gp(cm,cmll,state['constraint_z'][-n:],state['constraints'][-n:],20 if state['progress']['events']==0 else 2,total=len(state['rows']))
                state['timing']['constraint_fit']+=time.monotonic()-before
                before=time.monotonic()
                fit=fsbo.fit(state['features'],[r['score'] for r in state['rows']],1+state['progress']['events'])
                state['timing']['target_refit']+=time.monotonic()-before
                state['fit_events'].append(dict(round=state['progress']['rounds'],**fit))
                before=time.monotonic()
                center=torch.as_tensor(state['constraint_z'][policy.incumbent(state['rows'])])
                candidates=policy.candidate_pool(center,state['trust_region'],recipe['n_candidates'],20.).cuda()
                count=min(50,recipe['proposals']-state['progress']['proposals'],recipe['fresh_budget']-state['progress']['fresh_evaluations'])
                with torch.no_grad():
                    z=MaxPosteriorSampling(model=fsbo,constraint_models=[cm],constrained=True,replacement=False)(candidates,num_samples=count).cpu().numpy()
                torch.cuda.synchronize()
                state['timing']['acquisition']+=time.monotonic()-before
            before=time.monotonic()
            with torch.no_grad():
                sequences=objective.vae_decode(torch.as_tensor(z))
            state['timing']['decode']+=time.monotonic()-before
            cache={r['sequence']:r['score'] for r in state['rows']}
            remaining=recipe['fresh_budget']-state['progress']['fresh_evaluations']
            fresh=set()
            end=0
            for sequence in sequences:
                if sequence not in cache and sequence not in fresh:
                    if len(fresh)==remaining:
                        break
                    fresh.add(sequence)
                end+=1
            sequences=sequences[:end];z=z[:end]
            if not sequences:
                raise ValueError('Empty event before budget exhaustion')
            if any(not s or len(s)>30 or s.count('C')>1 or set(s)-set('ACDEFGHIKLMNPQRSTVWY') for s in sequences):
                raise ValueError('Invalid decoded peptide')
            unknown=list(dict.fromkeys(s for s in sequences if s not in cache))
            before=time.monotonic()
            known={r['sequence']:v for r,v in zip(state['rows'],state['features'])}
            if unknown:
                known.update(zip(unknown,encode(vae,data,unknown)))
            canonical=np.stack([known[s] for s in sequences])
            state['timing']['encoding']+=time.monotonic()-before
            recovered_event=dict(extra,event=state['progress']['events']+1,phase=phase,z=z,sequences=sequences,
                constraints=constraints(sequences),canonical_z=canonical,unknown=unknown,
                proposal_seconds=time.monotonic()-started)
            value=snapshot_value();value['pending_event']=recovered_event
            save(pending_path,value)
        sequences=recovered_event['sequences']
        unknown=recovered_event['unknown']
        phase=recovered_event['phase']
        cache={r['sequence']:r['score'] for r in state['rows']}
        if 'scores' not in recovered_event:
            before=time.monotonic()
            with (directory/'oracle_requests.jsonl').open('a') as ledger:
                ledger.write(json.dumps(dict(event=recovered_event['event'],sequences=unknown,physical_requested=len(unknown)))+'\n')
                ledger.flush();os.fsync(ledger.fileno())
            scores=objective.query_oracle(unknown) if unknown else []
            if len(scores)!=len(unknown) or not np.isfinite(scores).all():
                raise ValueError('Invalid APEX score')
            state['timing']['oracle']+=time.monotonic()-before
            recovered_event['scores']=list(map(float,scores))
            recovered_event['oracle_seconds']=time.monotonic()-before
            value=snapshot_value();value['pending_event']=recovered_event
            save(pending_path,value)
        if stop_after_score:
            return dict(status='validation_scored_pending',event=recovered_event['event'])
        scores=recovered_event['scores']
        new=dict(zip(unknown,map(float,scores)))
        seen=set(cache);hits=[]
        for s in sequences:
            hits.append(s in seen);seen.add(s)
        cache.update(new)
        pending=dict(recovered_event,seconds=recovered_event['proposal_seconds']+recovered_event['oracle_seconds'],
            scored=dict(scores=[cache[s] for s in sequences],fresh_evaluations=len(new),cache_hits=hits))
        policy.commit(state,pending,recipe,update_state,TurboState)
        state['last_phase']=phase
        wall_elapsed=accumulated_wall+time.monotonic()-active_start
        for row in state['rows'][-len(sequences):]:
            row['wall_elapsed']=wall_elapsed
        persist()
        pending_path.unlink()
        recovered_event=None
        print(json.dumps(dict(task=task,arm=arm,**state['progress'],updates=len(state['adaptation_events']))),flush=True)
    persist()
    finished=(state['progress']['proposals']>=recipe['proposals'] or state['progress']['fresh_evaluations']>=recipe['fresh_budget'])
    result=dict(read(directory/'status.json'),status='complete' if finished else 'smoke_complete',
        observations=state['rows'],adaptation_events=state['adaptation_events'],fit_events=state['fit_events'],
        phase_metrics=policy.phase_metrics(state['rows']),reference=reference,
        initialization_best_mic=min(-float(y) for y in pool['scores']))
    write(directory/'result.json',result)
    return result


if __name__=='__main__':
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--max-events',type=int)
    parser.add_argument('--force-update',action='store_true')
    parser.add_argument('--stop-after-score',action='store_true')
    args=parser.parse_args()
    (ROOT/'locks').mkdir(exist_ok=True)
    with (ROOT/'locks'/(args.arm+'_'+args.task+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.task,args.arm,args.max_events,args.force_update,args.stop_after_score)
