#!/usr/bin/env python3
"""Isolated train-only query VAE fine-tuning, re-embedding, and FSBO training."""
import argparse, collections, hashlib, io, json, math, os, random, sqlite3, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
OLD = Path('/root/data/cys/baseline/campaigns/query_our_fsbo_test200_20261116')
TRANSFER = Path('/root/data/logs/query_transfer_train800_20261106_v1')
REPO = Path('/root/data/cys/llmoptimizer_benchmark')
sys.path[:0] = [str(REPO/'tools'), str(OLD/'batch_code')]

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)); temp.replace(path)
def load(path): return json.loads(Path(path).read_text())
def selection(count):
    m = load(TRANSFER/'corpus/manifest.json')
    train = (SOURCE/'train800.txt').read_text().split()
    test = (SOURCE/'test200.txt').read_text().split()
    assert len(train)==800 and len(test)==200 and not set(train)&set(test)
    groups = collections.defaultdict(list)
    for task in m['source_tasks']: groups[task.split('_')[1].rstrip('0123456789')].append(task)
    # rstrip loses the instance suffix while retaining template letters, e.g. 11A.
    rng = random.Random(123)
    for group in sorted(groups): rng.shuffle(groups[group])
    eligible = sum(map(len, groups.values())); assert count<=eligible
    quota = {g:int(count*len(ts)/eligible) for g,ts in groups.items()}
    for g in sorted(groups, key=lambda g:(-(count*len(groups[g])/eligible-quota[g]),g))[:count-sum(quota.values())]: quota[g]+=1
    selected = sorted(t for g in groups for t in groups[g][:quota[g]])
    valid = set(random.Random(124).sample(selected, max(1,len(selected)//5)))
    return dict(requested_tasks=count, seed=123, selection='proportional_query_template_largest_remainder_seed123_no_test_results',
        source_tasks=selected, fsbo_train_tasks=[t for t in selected if t not in valid],
        fsbo_validation_tasks=[t for t in selected if t in valid],
        task_template_counts={g:quota[g] for g in sorted(quota)},
        train_split_sha256=digest(SOURCE/'train800.txt'), test_split_sha256=digest(SOURCE/'test200.txt'),
        source_corpus_sha256=digest(TRANSFER/'corpus/manifest.json'),
        excluded_test_tasks=test, heldout_overlap=0)

def prepare(args):
    import query_transfer as qt
    from query_data import canonicalize, plan_key
    import numpy as np
    selected = selection(args.task_count)
    config = dict(selected, epochs=30, patience=8, batch_size=64, learning_rate=1e-4,
        betas=[.9,.99], weight_decay=.01, lr_warmup='10_percent_of_maximum_updates_capped_at2048',
        vae_initial_checkpoint=str(SOURCE/'native/vae/CEB_64_author.ckpt'),
        vae_initial_sha256=digest(SOURCE/'native/vae/CEB_64_author.ckpt'),
        vae_objective='native_reconstruction_plus_KL_unchanged', vae_validation='disjoint_training_split_tasks_only',
        fsbo_steps=100000, fsbo_settings='unchanged_train_taskset_transfer_fsbo',
        test_evaluation_tuning=False, old_campaign=str(OLD))
    target=ROOT/'training/config.json'
    if target.exists(): assert load(target)==config, 'Training config changed'
    atomic(target,config)
    source_plan=load(SOURCE/'plan.json'); old_manifest=load(TRANSFER/'corpus/manifest.json')
    tasks={}; total=0
    for index,task in enumerate(selected['source_tasks']):
        history_path=SOURCE/'outputs'/task/'lolbo/seed0/history.json'
        assert digest(history_path)==old_manifest['source_hashes'][task]
        history=load(history_path)
        distinct, counts=qt.source_rows(history,task,source_plan,lambda p:canonicalize(task,p))
        assert len(distinct)>=2
        plans=[]; runtimes=[]; source_rows=[]; seen=set()
        for plan,(row,runtime) in distinct.items():
            p=canonicalize(task,list(plan),strict=True); key=plan_key(p)
            if key in seen: continue
            seen.add(key); plans.append(p); runtimes.append(runtime); source_rows.append(row)
        assert len(plans)>=2 and np.isfinite(runtimes).all()
        tasks[task]=dict(plans=plans,y=runtimes,source_rows=source_rows,source_sha256=digest(history_path),counts=counts)
        total+=len(plans)
        if (index+1)%10==0: print(json.dumps(dict(stage='prepare',tasks=index+1,plans=total)),flush=True)
    result=dict(config_sha256=digest(target),tasks=tasks,observations=total,oracle_calls=0)
    atomic(ROOT/'training/plans.json',result)
    atomic(ROOT/'training/status.json',dict(stage='prepared',tasks=len(tasks),plans=total,at=time.time()))
    print('PREPARED',len(tasks),total,flush=True)

def vae(args):
    import numpy as np, torch, lightning
    cfg=load(ROOT/'training/config.json'); corpus=load(ROOT/'training/plans.json')
    assert corpus['config_sha256']==digest(ROOT/'training/config.json')
    os.sched_setaffinity(0,{24,25}); torch.set_num_threads(2)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    torch.cuda.set_per_process_memory_fraction(.12)
    sys.path.insert(0,str(SOURCE/'native'))
    from vae.model import VAEModule, encode_batch
    model=VAEModule.load_from_checkpoint(cfg['vae_initial_checkpoint'],map_location='cpu').cuda()
    assert model.hparams.bn_size*model.hparams.d_neck==64
    train=[p for t in cfg['fsbo_train_tasks'] for p in corpus['tasks'][t]['plans']]
    valid=[p for t in cfg['fsbo_validation_tasks'] for p in corpus['tasks'][t]['plans']]
    optim=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']))
    max_steps=cfg['epochs']*math.ceil(len(train)/cfg['batch_size']); warmup=max(1,min(2048,max_steps//10))
    out=ROOT/'training/vae'; out.mkdir(exist_ok=True)
    identity=dict(config_sha256=digest(ROOT/'training/config.json'),plans_sha256=digest(ROOT/'training/plans.json'),
        trainer_sha256=digest(__file__),initial_sha256=cfg['vae_initial_sha256'],native_model_sha256=digest(SOURCE/'native/vae/model.py'))
    def evaluate():
        model.eval(); vals=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(99123)
            for start in range(0,len(valid),cfg['batch_size']):
                plans=valid[start:start+cfg['batch_size']]
                stats=model.model(encode_batch(plans,model.vocab).cuda())
                vals.append((len(plans),{k:float(stats[k].mean()) for k in ['loss','recon_loss','kldiv','recon_token_acc','recon_string_acc']}))
        return {k:sum(n*v[k] for n,v in vals)/len(valid) for k in vals[0][1]}
    def save_checkpoint(path,epoch,step,metrics):
        state=dict(state_dict=model.state_dict(),hyper_parameters=dict(model.hparams),
            **{'pytorch-lightning_version':lightning.__version__},epoch=epoch,global_step=step,
            retraining_identity=identity,validation=metrics)
        temp=path.with_suffix('.tmp');torch.save(state,str(temp));temp.replace(path)
    baseline=evaluate(); best=baseline['loss']; best_epoch=0; step=0; start_epoch=1
    state_path=out/'training_state.pt'
    if state_path.exists():
        state=torch.load(str(state_path),map_location='cpu',weights_only=False);assert state['identity']==identity
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
        for value in optim.state.values():
            for k,v in value.items():
                if torch.is_tensor(v): value[k]=v.cuda()
        best=state['best'];best_epoch=state['best_epoch'];step=state['step'];start_epoch=state['epoch']+1
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    else: save_checkpoint(out/'best.ckpt',0,0,baseline)
    started=time.monotonic()
    for epoch in range(start_epoch,cfg['epochs']+1):
        model.train(); order=np.random.RandomState(cfg['seed']+epoch).permutation(len(train)); losses=[]
        for start in range(0,len(order),cfg['batch_size']):
            step+=1
            for group in optim.param_groups:group['lr']=cfg['learning_rate']*min(1.,step/warmup)
            tokens=encode_batch([train[i] for i in order[start:start+cfg['batch_size']]],model.vocab).cuda()
            optim.zero_grad(set_to_none=True);stats=model.model(tokens);loss=stats['loss']
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optim.step();losses.append(float(loss.detach()))
        metrics=evaluate()
        if metrics['loss']<best:
            best=metrics['loss'];best_epoch=epoch;save_checkpoint(out/'best.ckpt',epoch,step,metrics)
        state=dict(identity=identity,model=model.state_dict(),optimizer=optim.state_dict(),epoch=epoch,step=step,
            best=best,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        temp=state_path.with_suffix('.tmp');torch.save(state,str(temp));temp.replace(state_path)
        status=dict(status='training',epoch=epoch,max_epochs=cfg['epochs'],step=step,best_epoch=best_epoch,
            best_validation_loss=best,validation=metrics,baseline_validation=baseline,train_loss=float(np.mean(losses)),
            train_tasks=len(cfg['fsbo_train_tasks']),validation_tasks=len(cfg['fsbo_validation_tasks']),
            train_plans=len(train),validation_plans=len(valid),elapsed_seconds=time.monotonic()-started,identity=identity)
        atomic(out/'status.json',status);print(json.dumps({k:v for k,v in status.items() if k!='identity'}),flush=True)
        if epoch-best_epoch>=cfg['patience']:break
    restored=VAEModule.load_from_checkpoint(str(out/'best.ckpt'),map_location='cpu')
    selected=torch.load(str(out/'best.ckpt'),map_location='cpu',weights_only=False)
    assert restored.vocab==model.vocab and restored.hparams.bn_size*restored.hparams.d_neck==64
    assert selected['retraining_identity']==identity
    if best_epoch==0:raise RuntimeError('Fine-tuning did not improve train-only validation; do not silently promote the old model')
    status.update(status='complete',checkpoint_sha256=digest(out/'best.ckpt'),reload_verified=True,
        selected_validation=selected['validation'],weights_changed=any(not torch.equal(v.cpu(),restored.state_dict()[k]) for k,v in VAEModule.load_from_checkpoint(cfg['vae_initial_checkpoint'],map_location='cpu').state_dict().items()))
    assert status['weights_changed'];atomic(out/'status.json',status)

def export(args):
    import numpy as np, torch
    import peptide_transfer as shared
    cfg=load(ROOT/'training/config.json');plans=load(ROOT/'training/plans.json');status=load(ROOT/'training/vae/status.json')
    assert status['status']=='complete' and status['reload_verified']
    torch.set_num_threads(2);os.sched_setaffinity(0,{24,25})
    sys.path.insert(0,str(SOURCE/'native'));from vae.model import VAEModule,encode_batch
    checkpoint=ROOT/'training/vae/best.ckpt';assert digest(checkpoint)==status['checkpoint_sha256']
    model=VAEModule.load_from_checkpoint(str(checkpoint),map_location='cpu').eval().requires_grad_(False)
    folder=ROOT/'training/corpus';folder.mkdir(exist_ok=True);db=shared.connect(folder/'corpus.sqlite')
    lower=np.full(64,np.inf);upper=np.full(64,-np.inf);hashes={};source_hashes={};observations=0
    for index,task in enumerate(cfg['source_tasks']):
        rows=plans['tasks'][task];features=[]
        with torch.inference_mode():
            for start in range(0,len(rows['plans']),128):
                mean,_=model.model.encode(encode_batch(rows['plans'][start:start+128],model.vocab));features.append(mean.reshape(-1,64).numpy().astype(np.float32))
        x=np.concatenate(features);assert x.shape==(len(rows['y']),64) and np.isfinite(x).all()
        payload=shared.pack_arrays(X=x,y=np.asarray(rows['y'],np.float64),source_rows=np.asarray(rows['source_rows'],np.int32))
        sha=hashlib.sha256(payload).hexdigest()
        with db:db.execute('INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?)',(task,len(x),rows['source_sha256'],sha,sqlite3.Binary(payload)))
        lower=np.minimum(lower,x.min(0));upper=np.maximum(upper,x.max(0));hashes[task]=sha;source_hashes[task]=rows['source_sha256'];observations+=len(x)
    manifest=dict(domain='ceb_query_runtime',policy='uncensored_first_distinct_plan_min_two',status='complete',purpose='production_training',
        representation='finetuned_train80_VAE_posterior_mean_64',normalization='selected_training_tasks_coordinate_minmax',
        dimension=64,source=str(SOURCE),source_tasks=cfg['source_tasks'],fsbo_train_tasks=cfg['fsbo_train_tasks'],fsbo_validation_tasks=cfg['fsbo_validation_tasks'],
        train_split_sha256=cfg['train_split_sha256'],test_split_sha256=cfg['test_split_sha256'],
        config_sha256=digest(ROOT/'training/config.json'),plans_sha256=digest(ROOT/'training/plans.json'),
        vae_checkpoint=str(checkpoint),vae_sha256=digest(checkpoint),lower=lower.tolist(),upper=upper.tolist(),
        task_hashes=hashes,source_hashes=source_hashes,observations=observations,oracle_calls=0,acquisition_wrapper=None)
    assert not set(manifest['source_tasks'])&set(cfg['excluded_test_tasks']) and np.all(upper>lower)
    with db:shared.put_meta(db,'manifest',manifest)
    db.close();atomic(folder/'manifest.json',manifest);print('REEMBEDDED',len(hashes),observations,flush=True)

def fsbo(args):
    import numpy as np
    import query_transfer as qt, peptide_transfer as shared
    sys.path.insert(0,str(REPO/'examples/benchmark/domains'))
    import train_taskset_transfer as trainer
    cfg=load(ROOT/'training/config.json');folder=ROOT/'training/corpus';manifest=load(folder/'manifest.json')
    assert digest(manifest['vae_checkpoint'])==manifest['vae_sha256'] and set(manifest['source_tasks'])==set(cfg['source_tasks'])
    db=sqlite3.connect('file:'+str(folder/'corpus.sqlite')+'?mode=ro',uri=True);tasks={}
    assert shared.get_meta(db,'manifest')==manifest
    for task in cfg['source_tasks']:
        data=qt.read_task(db,task);assert data['payload_sha256']==manifest['task_hashes'][task]
        tasks[task]=dict(X=shared.normalize(data['X'],manifest),y_maximize=-data['y'],y_minimize=data['y'],source_sha256=data['source_sha256'])
    db.close()
    def identity(settings,_):
        return dict(domain='ceb_query_runtime',method='fsbo',purpose='production_training',corpus_sha256=digest(folder/'manifest.json'),
            train_split_sha256=cfg['train_split_sha256'],test_split_sha256=cfg['test_split_sha256'],representation=manifest['representation'],seed=123,
            trainer_sha256=digest(__file__),shared_training_loop_sha256=digest(trainer.__file__),device='cpu',threads=2,acquisition_wrapper=None)
    trainer.identity=identity
    output=ROOT/'training/fsbo';output.mkdir(exist_ok=True);os.sched_setaffinity(0,{24,25})
    settings=argparse.Namespace(corpus=folder,output=output,seed=123,threads=2,steps=cfg['fsbo_steps'],purpose='production_training',method='fsbo',resume=True)
    trainer.fsbo(settings,manifest,tasks)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','vae','export','fsbo']);parser.add_argument('--task-count',type=int,default=80);args=parser.parse_args()
    try:globals()[args.stage](args)
    except BaseException as exc:
        atomic(ROOT/'training/failure.json',dict(stage=args.stage,error=repr(exc),at=time.time()));raise
if __name__=='__main__':main()
