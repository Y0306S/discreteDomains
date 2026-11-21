#!/usr/bin/env python3
"""Versioned frozen-VAE local/recenter wrapper for peptide BO methods.

Existing pinned implementations/recipes are read-only dependencies. This is a
new method variant; preparation never starts a campaign.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import peptide_local_policy as policy
import peptide_transfer_eval as transfer
from peptide_transfer_eval import bo, protocol, storage, torch

ROOT = Path(__file__).resolve().parent
FORMAT = 'peptide-frozen-local-recenter-v1'
METHODS = ('fsbo', 'rgpe', 'tpe', 'gp', 'hebo', 'smac', 'random')
BASELINES = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
TRANSFERS = Path('/root/data/logs/peptide_transfer_test200_20261028_v1')
TRAIN800 = storage.SOURCE
SHARED_SETTINGS = ('initialization', 'proposals', 'fresh_budget', 'similarity',
                   'seed', 'gp_initial_epochs', 'gp_update_epochs', 'learning_rate',
                   'center_policy', 'init_policy', 'budget_definition', 'oracle',
                   'higher_is_better', 'vae', 'upstream', 'cpu_upstream')


def train800_settings():
    return storage.native.verify(TRAIN800)


def bolt_settings_check():
    """Read configuration/AST only. No LOLBO instantiation, tuning or execution."""
    campaign = TRAIN800
    recipe = train800_settings()
    upstream = Path(recipe['upstream'])
    tree = ast.parse((upstream/'lolbo_scripts/optimize.py').read_text())
    init = next(n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
                if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    defaults = {arg.arg:ast.literal_eval(value) for arg,value in
                zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}
    checks = dict(batch_size=recipe['batch_size']==50,
        initial_epochs=recipe['gp_initial_epochs']==20,
        update_epochs=recipe['gp_update_epochs']==defaults['num_update_epochs']==2,
        learning_rate=recipe['learning_rate']==defaults['learning_rte']==0.001,
        stagnation_rounds=recipe['e2e_every']==defaults['e2e_freq']==10,
        similarity=recipe['similarity']==0.75,
        native_ts=defaults['acq_func']=='ts', native_gp_dkl=defaults['surrogate_type']=='gp_dkl',
        top_k=defaults['k']==1000, native_e2e=defaults['update_e2e'] is True,
        wrapper_native_dispatch='native.run_task(args)' in (ROOT/'peptide_test200.py').read_text(),
        per_reference_feasible_center=recipe['center_policy']=='best_observed_feasible')
    script = (upstream/'lolbo_scripts/run_single_seed_0.sh').read_text()
    checks['author_launch_batch'] = '--bsz 50' in script
    checks['author_launch_objective'] = '[bacteria_0]' in script
    checks['author_launch_max_length'] = '--max_string_length 30' in script
    turbo = ast.parse((Path(recipe['cpu_upstream'])/'lolbo/utils/bo_utils/turbo.py').read_text())
    cls = next(n for n in turbo.body if isinstance(n, ast.ClassDef) and n.name=='TurboState')
    fields = {n.target.id:eval(compile(ast.Expression(n.value), '<field>', 'eval'),
        {'__builtins__':{}, 'float':float, 'torch':torch}) for n in cls.body
        if isinstance(n, ast.AnnAssign) and n.value is not None}
    for local_key, native_key in [('tr_length','length'),('tr_length_min','length_min'),
        ('tr_length_max','length_max'),('failure_tolerance','failure_tolerance'),
        ('success_tolerance','success_tolerance')]:
        checks[local_key] = policy.DEFAULTS[local_key]==fields[native_key]
    return dict(passed=all(checks.values()), checks=checks, executed_lolbo=False,
        campaign=str(campaign),
        recipe_sha256=bo.digest(campaign/'recipe.json'),
        benchmark_overrides=dict(initialization=[100,1000],
            fresh_budget=[5000,200000], seed=[recipe['seed'],0],
            explanation='Existing common benchmark protocol versus author seed-0 launch; preserved'),
        source=str(upstream/'lolbo_scripts/run_single_seed_0.sh'))


def parent_recipe(method):
    path = (TRANSFERS/method if method in ('fsbo','rgpe') else BASELINES/'tpe')/'recipe.json'
    recipe = transfer.verify(path.parent) if method in ('fsbo','rgpe') else storage.verify(path.parent)
    return path, recipe


def prepare(args):
    check = bolt_settings_check()
    if not check['passed']:
        raise ValueError('Existing LOLBO configuration does not match the checked BOLT settings')
    settings = dict(policy.DEFAULTS, tr_length=args.tr_length,
        failure_tolerance=args.failure_tolerance, perturb_dimensions=args.perturb_dimensions)
    source = train800_settings()
    if not (settings['tr_length_min'] <= args.tr_length <= settings['tr_length_max']
            and args.failure_tolerance > 0 and 0 < args.perturb_dimensions <= 256):
        raise ValueError('Invalid existing locality settings')
    for method in args.methods.split(','):
        if method not in METHODS:
            raise ValueError('Unknown peptide method: '+method)
        parent, base = parent_recipe(method)
        directory = args.campaign/method
        if (directory/'recipe.json').exists():
            raise ValueError('Use a new campaign directory; recipe already exists: '+str(directory))
        recipe = copy.deepcopy(base)
        recipe.update({key:source[key] for key in SHARED_SETTINGS})
        if args.scope == 'train':
            recipe['prepared_pool'] = source['prepared_pool']
        recipe.update(format=FORMAT, method=method, adapter=str(Path(__file__).resolve()),
            parent_recipe=str(parent), parent_sha256=bo.digest(parent),
            local=settings, vae_policy='frozen_author', e2e_every=0,
            e2e_policy='disabled_user_requested', recenter_policy='native_posterior_reconstruction_frozen_vae',
            coordinate_policy='canonical_objective_and_actual_decoder_input_constraint',
            evaluation_scope=args.scope+'800' if args.scope=='train' else 'test200',
            allowed_tasks=recipe['train'] if args.scope=='train' else recipe['test'],
            batch_size=source['batch_size'] if method in ('fsbo','rgpe','tpe','gp') else 1,
            proposals=args.proposals, fresh_budget=args.fresh_budget,
            n_candidates=args.candidates, local_variant=method+'-local-recenter',
            settings_source=str(TRAIN800/'recipe.json'),
            settings_source_sha256=bo.digest(TRAIN800/'recipe.json'),
            source_task_overlap=(args.scope=='train' and method in ('fsbo','rgpe')),
            bolt_check=check, prepared_at=time.time())
        recipe['code'] = dict(base['code'])
        for name in ('peptide_local_bo.py','peptide_local_policy.py'):
            recipe['code'][str(ROOT/name)] = bo.digest(ROOT/name)
        recipe['assets'][recipe['prepared_pool']] = bo.digest(Path(recipe['prepared_pool']))
        recipe['packages'] = {n:importlib.metadata.version(n) for n in
            ('numpy','torch','gpytorch','botorch','optuna')}
        if method == 'hebo':
            recipe['packages']['HEBO'] = importlib.metadata.version('HEBO')
        if method == 'smac':
            recipe['packages']['openbox'] = importlib.metadata.version('openbox')
            recipe['packages']['scikit-learn'] = importlib.metadata.version('scikit-learn')
            # Same sklearn PRF selected by installed OpenBox when pyrfr is absent.
            recipe['smac_surrogate'] = 'sk_prf'
        bo.atomic_json(directory/'recipe.json', recipe)
    bo.atomic_json(args.campaign/'bolt_settings_check.json', check)
    print(json.dumps(dict(prepared=str(args.campaign), methods=args.methods,
                          vae_updates=False, launched=False)), flush=True)


def verify(directory):
    recipe = json.loads((directory/'recipe.json').read_text())
    if (recipe['format']!=FORMAT or recipe['method'] not in METHODS
            or recipe['vae_policy']!='frozen_author' or recipe['e2e_every']!=0
            or recipe['adapter']!=str(Path(__file__).resolve())):
        raise ValueError('Invalid frozen local recipe')
    if bo.digest(Path(recipe['parent_recipe']))!=recipe['parent_sha256']:
        raise ValueError('Parent recipe changed')
    if bo.digest(Path(recipe['settings_source']))!=recipe['settings_source_sha256']:
        raise ValueError('Train800 LOLBO settings recipe changed')
    for section in ('code','assets'):
        for name, expected in recipe[section].items():
            if bo.digest(Path(name)) != expected:
                raise ValueError('Pinned dependency changed: '+name)
    for name, version in recipe['packages'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('Package version changed: '+name)
    if recipe['method'] in ('fsbo','rgpe'):
        base = transfer.verify(Path(recipe['parent_recipe']).parent)
        if (recipe['fsbo_weights_sha256']!=base['fsbo_weights_sha256']
                or recipe['corpus_sha256']!=base['corpus_sha256']
                or recipe['rgpe_model_hashes']!=base['rgpe_model_hashes']):
            raise ValueError('Transfer source identity changed')
        if recipe.get('target_fit_amendment_sha256')!=base.get('target_fit_amendment_sha256'):
            raise ValueError('Transfer target-fit amendment changed')
    return recipe


def make_gp(x, weights=None):
    import gpytorch
    from gpytorch.mlls import PredictiveLogLikelihood
    from lolbo.utils.bo_utils.ppgpr import GPModelDKL
    inducing = min(len(x),1024) if weights is None else weights['variational_strategy.inducing_points'].shape[0]
    model = GPModelDKL(torch.as_tensor(x[:inducing]),
                       gpytorch.likelihoods.GaussianLikelihood())
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    return model, PredictiveLogLikelihood(model.likelihood, model, num_data=len(x))


def make_study(state, recipe):
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.trial import create_trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = state.get('sampler')
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=recipe['seed'], constraints_func=policy.tpe_constraints)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    for record in state['tpe_trials']:
        study.add_trial(create_trial(params=record['params'],
            distributions={k:FloatDistribution(*v) for k,v in record['bounds'].items()},
            value=record['score'], user_attrs={'constraints':record['constraints']},
            system_attrs={'constraints':record['constraints']}))
    return study


def tpe_record(z, score, constraint, ranges=None):
    values = np.asarray(z).reshape(-1)
    return dict(params={f'z{i:03d}':float(v) for i,v in enumerate(values)},
        bounds={f'z{i:03d}':([float(v),float(v)] if ranges is None else
                [float(ranges[0][i]),float(ranges[1][i])]) for i,v in enumerate(values)},
        score=float(score), constraints=[float(constraint)])


def propose_tpe(state, recipe):
    from optuna.distributions import FloatDistribution
    center = torch.as_tensor(state['constraint_z'][policy.incumbent(state['rows'])])
    low, high = policy.bounds(center, state['trust_region'])
    study = make_study(state, recipe)
    count = min(recipe['batch_size'],recipe['proposals']-state['progress']['proposals'],
                recipe['fresh_budget']-state['progress']['fresh_evaluations'])
    records = []
    for active in policy.sparse_mask(count,256,recipe['local']['perturb_dimensions']):
        lo,hi = low.clone(),high.clone()
        lo[~active],hi[~active] = center[~active],center[~active]
        dists = {f'z{i:03d}':FloatDistribution(float(lo[i]),float(hi[i])) for i in range(256)}
        trial = study.ask(fixed_distributions=dists)
        records.append(dict(params=trial.params,bounds={k:[d.low,d.high] for k,d in dists.items()},
                            number=trial.number))
    # Fixed coordinates are part of ask; the whole batch is told after scoring.
    z = np.asarray([[record['params'][f'z{i:03d}'] for i in range(256)]
                    for record in records],dtype=np.float32)
    return z,dict(tpe_pending=records,sampler=copy.deepcopy(study.sampler))


def tell_tpe(state, pending, recipe):
    if pending['phase']=='acquisition':
        from optuna.distributions import FloatDistribution
        from optuna.trial import create_trial, TrialState, Trial
        state['sampler'] = pending['sampler']
        study = make_study(state, recipe)
        trials = []
        for record in pending['tpe_pending']:
            study.add_trial(create_trial(params=record['params'],
                distributions={k:FloatDistribution(*v) for k,v in record['bounds'].items()},
                state=TrialState.RUNNING))
            trials.append(Trial(study,study.trials[-1]._trial_id))
        n = len(pending['sequences'])
        for i,(trial,record) in enumerate(zip(trials,pending['tpe_pending'])):
            if i>=n:
                # A fresh-budget prefix may leave unscored suffix trials.
                study.tell(trial,state=TrialState.FAIL)
                continue
            c = float(pending['constraints'][i][0])
            trial.set_user_attr('constraints',[c])
            study.tell(trial,pending['scored']['scores'][i])
            state['tpe_trials'].append(dict(params=record['params'],bounds=record['bounds'],
                score=pending['scored']['scores'][i],constraints=[c]))
        state['sampler'] = copy.deepcopy(study.sampler)
    else:
        for z,y,c in zip(pending['z'],pending['scored']['scores'],pending['constraints']):
            state['tpe_trials'].append(tpe_record(z,y,c[0]))


def fit_gp(model, mll, state, recipe, target):
    from lolbo.utils.utils import update_surr_model
    # After recenter, refresh on all observations, as native V5 does after re-encoding.
    full = state['progress']['events']==0 or state.get('last_phase')=='recenter'
    size = len(state['rows']) if full else state['last_event_size']
    epochs = recipe['gp_initial_epochs'] if state['progress']['events']==0 else recipe['gp_update_epochs']
    mll.num_data = len(state['rows'])
    return update_surr_model(model,mll,recipe['learning_rate'],
        torch.as_tensor(state['constraint_z'][-size:]),torch.as_tensor(target[-size:]).flatten(),epochs)


def strict_hebo_predict(model, x, xe):
    """HEBO posterior/scaling, failing explicitly instead of random fallback."""
    import gpytorch
    tx,te = model.xtrans(x,xe)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
        pred = model.gp(tx,te)
        if model.pred_likeli:
            pred = model.lik(pred)
        mean = model.yscaler.inverse_transform(pred.mean.reshape(-1,model.num_out))
        variance = pred.variance.reshape(-1,model.num_out)*model.yscaler.std**2
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise RuntimeError('HEBO posterior is not finite')
        return mean,variance.clamp(min=torch.finfo(variance.dtype).eps)


def acquisition(state, recipe, constraint, objective_gp, model_client):
    method = recipe['method']
    if method=='tpe':
        return propose_tpe(state,recipe)
    center = torch.as_tensor(state['constraint_z'][policy.incumbent(state['rows'])])
    candidates = policy.candidate_pool(center,state['trust_region'],recipe['n_candidates'],
                                       recipe['local']['perturb_dimensions'])
    count = min(recipe['batch_size'],recipe['proposals']-state['progress']['proposals'],
                recipe['fresh_budget']-state['progress']['fresh_evaluations'])
    if method in ('fsbo','rgpe','gp'):
        from lolbo.utils.bo_utils.constrained_max_posterior_sampling import MaxPosteriorSampling
        if method in ('fsbo','rgpe'):
            fit = model_client.request('fit',x=transfer.normalized(state['features'],recipe),
                score=np.asarray([r['score'] for r in state['rows']]),
                seed=recipe['seed']+state['progress']['events'])
            model = transfer.BridgeModel(model_client,recipe)
        else:
            model, fit = objective_gp, dict(method='gp',observations=len(state['rows']))
        with torch.no_grad():
            z = MaxPosteriorSampling(model=model,constraint_models=[constraint],
                constrained=True,replacement=False)(candidates,num_samples=count)
        return z.numpy(),dict(fit=fit)
    if method=='random':
        return candidates[:count].numpy(),{}
    valid = policy.constrain_indices(constraint,candidates)
    pool = candidates[valid]
    x = np.asarray(state['constraint_z'])
    mic = -np.asarray([r['score'] for r in state['rows']]).reshape(-1,1)
    if method=='smac':
        from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
        from openbox.acquisition_function.acquisition import EI
        # RF/EI are retained; optimize on BOLT's constrained sparse local pool.
        low, high = x.min(0)-1., x.max(0)+1.
        model = skRandomForestWithInstances(np.zeros(256,dtype=np.uint),
            list(zip(low,high)),seed=recipe['seed']+state['progress']['events'],n_jobs=1)
        model.train(x,mic)
        acq = EI(model)
        acq.update(eta=-state['rows'][policy.incumbent(state['rows'])]['score'])
        index = int(np.argmax(acq(pool.numpy(),convert=False)))
    elif method=='hebo':
        from hebo.models.model_factory import get_model
        from hebo.acquisitions.acq import MACE
        from hebo.models.gp.gp_util import default_kern
        from gpytorch.constraints import GreaterThan
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        from sklearn.preprocessing import power_transform
        # Retain HEBO's response transform, including its raw-response fallback.
        try:
            transformed = power_transform(mic/mic.std(),
                method='yeo-johnson' if mic.min()<=0 else 'box-cox')
            if transformed.std()<.5:
                transformed = power_transform(mic/mic.std(),method='yeo-johnson')
            if transformed.std()<.5:
                raise ValueError('Degenerate power transform')
        except (ValueError,FloatingPointError):
            transformed = mic
        model = get_model('gp',256,0,1,lr=0.01,num_epochs=100,
                          verbose=False,noise_lb=8e-4,pred_likeli=False)
        train_x = torch.as_tensor(x)
        train_e = torch.zeros((len(x),0),dtype=torch.long)
        train_y = torch.as_tensor(transformed,dtype=torch.float32)
        # The installed HEBO SGLD can underflow a positive lengthscale to zero
        # in 256 dimensions. Keep its kernel/optimizer, with a numerical floor.
        model.fit_scaler(train_x,train_e,train_y)
        sx,se,sy = model.xtrans(train_x,train_e,train_y)
        kernel = default_kern(sx,se,sy,256,True)
        for child in kernel.modules():
            if hasattr(child,'raw_lengthscale') and child.raw_lengthscale is not None:
                lengthscale = child.lengthscale.detach().clone()
                child.register_constraint('raw_lengthscale',GreaterThan(1e-6))
                child.lengthscale = lengthscale.clamp_min(2e-6)
        model.conf['kern'] = kernel
        model.fit(train_x,train_e,train_y)
        if not all(torch.isfinite(p).all() for p in model.gp.parameters()):
            raise RuntimeError('HEBO GP fitting failed; refusing its random-prediction fallback')
        bx = torch.as_tensor(x[policy.incumbent(state['rows'])]).reshape(1,-1)
        # Apply the same strict prediction to the incumbent and every MACE input.
        model.predict = lambda cx,ex: strict_hebo_predict(model,cx,ex)
        best,_ = model.predict(bx,torch.zeros((1,0),dtype=torch.long))
        iteration = max(1,len(x))
        kappa = np.sqrt(.5*2*((2+256/2)*np.log(iteration)+np.log(3*np.pi**2/(3*.01))))
        acq = MACE(model,best_y=float(best.item()),kappa=kappa)
        with torch.no_grad():
            values = acq(pool,torch.zeros((len(pool),0),dtype=torch.long)).numpy()
        if not np.isfinite(values).all():
            raise RuntimeError('HEBO acquisition is not finite')
        front = NonDominatedSorting().do(values,only_non_dominated_front=True)
        index = int(np.random.choice(front))
    else:
        raise ValueError('No acquisition adapter for '+method)
    return pool[index:index+1].numpy(),dict(acquisition=method+'-native-acquisition-local-pool',
                                           sampled_feasible_pool=len(pool))


def history(state,recipe,complete=False):
    best = state['rows'][policy.incumbent(state['rows'])]
    return dict(state['progress'],identity=state['identity'],
        status='complete' if complete else 'running',method=recipe['local_variant'],
        reference=recipe['references'][state['identity']['task']],
        initialization=recipe['initialization'],proposal_budget=recipe['proposals'],
        fresh_budget=recipe['fresh_budget'],observations=state['rows'],
        best_feasible_mic=best['mic'],best_feasible_sequence=best['sequence'],
        e2e_updates=0,vae_policy='frozen_author',phase_metrics=policy.phase_metrics(state['rows']))


def persist(db,directory,state,recipe):
    state['rng'] = bo.capture_rng()
    with db:
        storage.put(db,'state',state)
    bo.atomic_json(directory/'progress.json',dict(state['progress'],identity=state['identity']))
    bo.atomic_json(directory/'history.json',history(state,recipe))


def initialize(recipe,task,client,identity,turbo_class):
    reference,sequences,scores = bo.load_pool(recipe,task)
    if reference!=recipe['references'][task]:
        raise ValueError('Initialization/reference mismatch')
    features = transfer.encode_mean(client,sequences)
    rows = [dict(sequence=s,score=float(y),mic=-float(y),feasible=True,cached=True,
        phase='initialization',round=0,elapsed=0.) for s,y in zip(sequences,scores)]
    constraints = np.asarray([[recipe['similarity']-bo.similarity(s,reference)] for s in sequences],dtype=np.float32)
    state = dict(identity=identity,rows=rows,features=features,constraint_z=features.copy(),
        constraints=constraints,constraint_weights=None,objective_weights=None,
        stagnation=0,recenter_plan=None,last_event_size=len(rows),last_phase=None,
        sampler=None,tpe_trials=[],progress=dict(rounds=0,proposals=0,fresh_evaluations=0,
            cache_hits=0,acquisition_fresh=0,recenter_fresh=0,recenter_proposals=0,events=0,seconds=0.))
    state['trust_region'] = policy.new_region(rows,constraints,recipe['batch_size'],recipe['local'],turbo_class)
    if recipe['method']=='tpe':
        state['tpe_trials'] = [tpe_record(z,y,c[0],(np.minimum(z,-4.),np.maximum(z,4.)))
                              for z,y,c in zip(features,scores,constraints)]
    return state


def run(args):
    recipe = verify(args.campaign)
    if args.task not in recipe['allowed_tasks']:
        raise ValueError('Task outside the recipe scope')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Coordinator requires CUDA_VISIBLE_DEVICES=""; inference stays on the service')
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{args.cpu})
    identity = dict(format=FORMAT,recipe_sha256=bo.digest(args.campaign/'recipe.json'),task=args.task)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    with transfer.transfer.lock(args.campaign/'locks'/(args.task+'.lock')):
        directory = args.campaign/'runs'/args.task
        archive = directory.with_suffix('.tar.gz')
        if archive.exists():
            storage.cleanup_tpe(directory,archive,identity)
            return storage.verify_tpe_archive(archive,identity)
        directory.mkdir(parents=True,exist_ok=True)
        db = storage.connect(directory/'runtime.sqlite')
        client = model_client = None
        try:
            random.seed(recipe['seed']); np.random.seed(recipe['seed']); torch.manual_seed(recipe['seed'])
            objective,client = transfer.create_objective(recipe,args.task,args.server,identity)
            objective.vae.eval()
            for parameter in objective.vae.parameters():
                parameter.requires_grad_(False)
            from lolbo.utils.bo_utils.turbo import TurboState,update_state
            state = storage.get(db,'state')
            if state is None:
                state = initialize(recipe,args.task,client,identity,TurboState)
                persist(db,directory,state,recipe)
            if state['identity']!=identity:
                raise ValueError('Recovery identity changed')
            protocol.check_accounting(recipe,state['progress'],state['rows'])
            if any(np.asarray(state[k]).shape!=(len(state['rows']),256) for k in ('features','constraint_z')):
                raise ValueError('Recovered coordinates are misaligned')
            objective.xs_to_scores_dict = {r['sequence']:r['score'] for r in state['rows']}
            objective.num_calls = state['progress']['fresh_evaluations']
            constraint,mll = make_gp(state['constraint_z'],state['constraint_weights'])
            objective_gp = objective_mll = None
            if recipe['method']=='gp':
                objective_gp,objective_mll = make_gp(state['constraint_z'],state['objective_weights'])
            if recipe['method'] in ('fsbo','rgpe'):
                base = transfer.verify(Path(recipe['parent_recipe']).parent)
                model_client = transfer.ModelClient(Path(recipe['parent_recipe']),base)
            bo.restore_rng(state['rng'])
            while (state['progress']['proposals']<recipe['proposals']
                   and state['progress']['fresh_evaluations']<recipe['fresh_budget']):
                pending = storage.get(db,'pending')
                if pending is None:
                    if stopped:
                        break
                    started = time.monotonic()
                    phase = policy.next_phase(state,recipe)
                    extra = {}
                    if phase=='recenter':
                        plan = state['recenter_plan']
                        end = min(len(plan['inputs']),plan['cursor']+plan['batch'])
                        with torch.no_grad():
                            z,_ = objective.vae_forward(plan['inputs'][plan['cursor']:end])
                        z = z.detach().cpu().numpy().reshape(-1,256)
                        extra['next_cursor'] = end
                    else:
                        if recipe['method'] not in ('tpe','random'):
                            constraint = fit_gp(constraint,mll,state,recipe,state['constraints'])
                            state['constraint_weights'] = bo.cpu_copy(constraint.state_dict())
                        if objective_gp is not None:
                            objective_gp = fit_gp(objective_gp,objective_mll,state,recipe,
                                np.asarray([r['score'] for r in state['rows']]))
                            state['objective_weights'] = bo.cpu_copy(objective_gp.state_dict())
                        z,extra = acquisition(state,recipe,constraint,objective_gp,model_client)
                    with torch.no_grad():
                        sequences = list(objective.vae_decode(torch.as_tensor(z)))
                    for sequence in sequences:
                        bo.validate_sequence(sequence)
                    remaining = recipe['fresh_budget']-state['progress']['fresh_evaluations']
                    sequences = protocol.bounded_prefix(sequences,objective.xs_to_scores_dict,remaining)
                    if not sequences:
                        raise ValueError('Empty event before budget exhaustion')
                    z = z[:len(sequences)]
                    pending = dict(extra,event=state['progress']['events']+1,phase=phase,z=z,
                        sequences=sequences,seconds=time.monotonic()-started,
                        constraints=objective.compute_constraints(sequences).numpy(),
                        oracle_request=dict(op='oracle',sequences=list(dict.fromkeys(
                            s for s in sequences if s not in objective.xs_to_scores_dict)),
                            nonce=int(torch.randint(0,2**62,(1,)).item())))
                    # Save proposal and its post-proposal RNG before requesting scores.
                    state['rng'] = bo.capture_rng()
                    with db:
                        storage.put(db,'state',state); storage.put(db,'pending',pending)
                if args.stop_after_stage=='proposal' and 'scored' not in pending:
                    return history(state,recipe)
                if 'scored' not in pending:
                    requested = pending['oracle_request']['sequences']
                    started = time.monotonic()
                    reply = client.request(pending['oracle_request']) if requested else {'scores':[]}
                    values = np.asarray(reply['scores'],dtype=float)
                    if values.shape!=(len(requested),) or not np.isfinite(values).all():
                        raise ValueError('Invalid oracle result')
                    new = dict(zip(requested,values.tolist()))
                    cache = dict(objective.xs_to_scores_dict,**new)
                    seen,hits = set(objective.xs_to_scores_dict),[]
                    for s in pending['sequences']:
                        hits.append(s in seen); seen.add(s)
                    pending['scored'] = dict(scores=[cache[s] for s in pending['sequences']],
                        new_scores=new,cache_hits=hits,fresh_evaluations=len(new))
                    pending['seconds'] += time.monotonic()-started
                    with db:
                        storage.put(db,'pending',pending)
                if args.stop_after_stage=='score':
                    return history(state,recipe)
                if 'canonical_z' not in pending:
                    known = {}
                    for row,z in zip(state['rows'],state['features']):
                        known.setdefault(row['sequence'],z)
                    unknown = list(dict.fromkeys(s for s in pending['sequences'] if s not in known))
                    if unknown:
                        known.update(zip(unknown,transfer.encode_mean(client,unknown)))
                    pending['canonical_z'] = np.stack([known[s] for s in pending['sequences']])
                    pending['rng'] = bo.capture_rng()
                    with db:
                        storage.put(db,'pending',pending)
                bo.restore_rng(pending['rng'])
                if recipe['method']=='tpe':
                    tell_tpe(state,pending,recipe)
                policy.commit(state,pending,recipe,update_state,TurboState)
                state['last_phase'] = pending['phase']
                objective.xs_to_scores_dict.update(pending['scored']['new_scores'])
                objective.num_calls += pending['scored']['fresh_evaluations']
                protocol.check_accounting(recipe,state['progress'],state['rows'],
                                          type('State',(),{'objective':objective})())
                state['rng'] = bo.capture_rng()
                with db:
                    storage.put(db,'state',state); storage.put(db,'pending',None)
                persist(db,directory,state,recipe)
                print(json.dumps(dict(method=recipe['local_variant'],task=args.task,
                    phase=pending['phase'],**state['progress'])),flush=True)
                if stopped or (args.stop_after_events and state['progress']['events']>=args.stop_after_events):
                    break
            complete = (state['progress']['proposals']>=recipe['proposals']
                        or state['progress']['fresh_evaluations']>=recipe['fresh_budget'])
            result = history(state,recipe,complete)
            bo.atomic_json(directory/'history.json',result)
        finally:
            if model_client is not None:
                model_client.close()
            if client is not None:
                client.close()
            db.close()
        if complete:
            storage.tpe_archive(directory,identity)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    check = commands.add_parser('check-bolt')
    check.add_argument('--output',type=Path)
    prep = commands.add_parser('prepare')
    prep.add_argument('--campaign',type=Path,required=True)
    prep.add_argument('--methods',default='fsbo,rgpe,tpe,gp,hebo,smac')
    prep.add_argument('--scope',choices=('train','test'),default='train')
    prep.add_argument('--proposals',type=int,default=5000)
    prep.add_argument('--fresh-budget',type=int,default=5000)
    prep.add_argument('--candidates',type=int,default=5000)
    prep.add_argument('--tr-length',type=float,default=policy.DEFAULTS['tr_length'])
    prep.add_argument('--failure-tolerance',type=int,default=policy.DEFAULTS['failure_tolerance'])
    prep.add_argument('--perturb-dimensions',type=float,default=policy.DEFAULTS['perturb_dimensions'])
    task = commands.add_parser('run-task')
    task.add_argument('--campaign',type=Path,required=True)
    task.add_argument('--task',required=True)
    task.add_argument('--cpu',type=int,required=True)
    task.add_argument('--server',default='http://127.0.0.1:18871')
    task.add_argument('--stop-after-events',type=int,default=0)
    task.add_argument('--stop-after-stage',choices=('proposal','score'))
    args = parser.parse_args()
    if args.command=='check-bolt':
        report = bolt_settings_check()
        if args.output:
            bo.atomic_json(args.output,report)
        print(json.dumps(report,indent=2))
        if not report['passed']:
            raise SystemExit(1)
    elif args.command=='prepare':
        if min(args.proposals,args.fresh_budget,args.candidates)<=0 or args.candidates<50:
            parser.error('Budgets must be positive and candidate pool at least 50')
        prepare(args)
    else:
        run(args)


if __name__=='__main__':
    main()
