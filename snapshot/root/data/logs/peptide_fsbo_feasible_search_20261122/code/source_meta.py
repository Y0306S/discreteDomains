"""Refresh every source observation, then train a new FSBO prior on CPU.

The source tasks remain independent GP episodes. They are never pooled into
one GP with labels from different peptide references.
"""
import argparse
import shutil
import subprocess
from common import *


def refresh_and_retrain(vae,data,state,task,directory,component=False):
    import numpy as np
    import torch
    cfg=prepare()
    assert task in cfg['tasks'] and task not in cfg['train_tasks']
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    seed=cfg['fsbo_meta_seed']
    identity=dict(task=task,round=state['progress']['rounds'],observations=len(state['rows']),
        config_sha256=sha(ROOT/'config.json'),component=component)
    # A trigger may be repeated after interruption before its optimizer
    # checkpoint commits. Verify the exact VAE tensor contents, not filenames.
    h=hashlib.sha256()
    for name,value in sorted(vae.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    identity['vae_tensor_sha256']=h.hexdigest()
    idpath=directory/'identity.json'
    if idpath.exists():assert read(idpath)==identity,'Refresh identity changed'
    else:write(idpath,identity)
    status=directory/'meta_status.json'
    if not status.exists() or read(status)['status']!='complete':
        manifest_path=directory/'manifest.json'
        if not manifest_path.exists():
            source=corpus_db()
            ids=cfg['train_tasks'][:4] if component else cfg['train_tasks']
            manifest_source=read(OLD/'corpus/manifest.json')
            lengths={}
            for index,t in enumerate(ids):
                d=task_data(source,t)
                lengths[t]=min(120,len(d['y'])) if component else len(d['y'])
            total=sum(lengths.values())+len(state['rows'])
            x=np.lib.format.open_memmap(directory/'features.npy',mode='w+',dtype='float32',shape=(total,256))
            y=np.lib.format.open_memmap(directory/'labels.npy',mode='w+',dtype='float32',shape=(total,))
            low,high=np.full(256,np.inf),np.full(256,-np.inf)
            offsets={};cursor=0;started=time.monotonic()
            for index,t in enumerate(ids):
                d=task_data(source,t);n=lengths[t]
                z=encode(vae,data,d['sequences'][:n].tolist())
                x[cursor:cursor+n]=z;y[cursor:cursor+n]=d['y'][:n]
                offsets[t]=[cursor,cursor+n];cursor+=n
                low=np.minimum(low,z.min(0));high=np.maximum(high,z.max(0))
                write(directory/'refresh_status.json',dict(status='encoding',source_tasks=index+1,
                    required_source_tasks=len(ids),observations=cursor,elapsed_seconds=time.monotonic()-started))
            target=encode(vae,data,[r['sequence'] for r in state['rows']])
            x[cursor:]=target;y[cursor:]=[r['score'] for r in state['rows']]
            offsets[task]=[cursor,total];x.flush();y.flush();del x,y
            source.close()
            scale=np.where(high>low,high-low,1.)
            write(manifest_path,dict(identity=identity,source_tasks=ids,current_task=task,offsets=offsets,
                lower=low.tolist(),scale=scale.tolist(),normalization='source800_only_refreshed',
                steps=50 if component else cfg['online_meta_steps'],seed=seed,
                validate_every=25 if component else cfg['online_meta_validate_every'],
                checkpoint_every=25 if component else cfg['online_meta_checkpoint_every'],
                validation_tasks=ids if component else manifest_source['fsbo_validation_tasks'],
                features_sha256=sha(directory/'features.npy'),labels_sha256=sha(directory/'labels.npy'),
                encoding_seconds=time.monotonic()-started,total_observations=total))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (directory/'meta.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',__file__,'--train',str(directory)],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=read(directory/'manifest.json');info=read(status)
    assert info['initialization']=='scratch' and info['all_tasks_sampled']
    assert info['source_tasks']==(4 if component else 800)
    weights=torch.load(directory/'weights',map_location='cpu',weights_only=False)
    # DeepKernelGP's full state_dict has shared likelihood aliases.
    prior=weights['full']
    # Keep model/checkpoint/provenance, discard only reproducible 4GB matrices.
    for name in ['features.npy','labels.npy']:
        (directory/name).unlink(missing_ok=True)
    return prior,np.asarray(manifest['lower'],dtype=np.float32),np.asarray(manifest['scale'],dtype=np.float32),info


def train(directory):
    import numpy as np
    import torch
    import gpytorch
    if sys.version_info < (3,10) and str(ROOT/'deps') in sys.path:
        sys.path.remove(str(ROOT/'deps'))
    sys.path.insert(0,'/root/data/cys/baseline/repos/FSBO')
    from fsbo_modules import DeepKernelGP
    directory=Path(directory);m=read(directory/'manifest.json')
    assert not torch.cuda.is_available(),'Source meta-training must use CPU'
    torch.set_num_threads(2)
    assert sha(directory/'features.npy')==m['features_sha256']
    assert sha(directory/'labels.npy')==m['labels_sha256']
    x=np.load(directory/'features.npy',mmap_mode='r');y=np.load(directory/'labels.npy',mmap_mode='r')
    lo=np.asarray(m['lower'],dtype=np.float32);scale=np.asarray(m['scale'],dtype=np.float32)
    ids=m['source_tasks']+([m['current_task']] if m['current_task'] is not None else [])
    assert len(ids)==len(set(ids)) and m['current_task'] not in m['source_tasks']
    rng=np.random.RandomState(m['seed']);torch.manual_seed(m['seed'])
    model=DeepKernelGP(256,os.devnull,m['seed'],hidden_size=[128,128],kernel='rbf',ard=False,checkpoint=str(directory))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    valid_rng=np.random.RandomState(m['seed']+1)
    train_indices={};validation=[];ranges={}
    for task in ids:
        a,b=m['offsets'][task];values=np.asarray(y[a:b]);ranges[task]=(float(values.min()),float(values.max()-values.min()) or 1.)
        indices=np.arange(a,b)
        if task in m['validation_tasks']:
            order=valid_rng.permutation(indices)
            size=min(50,max(1,len(order)//4))
            validation.append((task,order[:size],order[size:2*size]))
            indices=order[2*size:]
        train_indices[task]=indices
        assert len(indices)>0
    def tensors(task,index):
        offset,width=ranges[task]
        return torch.as_tensor((x[index]-lo)/scale,dtype=torch.float32),torch.as_tensor((y[index]-offset)/width,dtype=torch.float32)
    def weights():
        return dict(gp=cpu_state(model.model),likelihood=cpu_state(model.likelihood),net=cpu_state(model.feature_extractor),full=cpu_state(model))
    def validate():
        model.model.eval();model.feature_extractor.eval();model.likelihood.eval()
        losses=[]
        with torch.no_grad():
            for task,support,query in validation:
                sx,sy=tensors(task,support);qx,qy=tensors(task,query)
                model.model.set_train_data(model.feature_extractor(sx),sy,strict=False)
                posterior=model.likelihood(model.model(model.feature_extractor(qx)))
                losses.append(float(-posterior.log_prob(qy)/len(qy)))
        value=float(np.mean(losses));assert np.isfinite(value)
        return value
    checkpoint=directory/'training_state.pt'
    start=0;best=float('inf');best_step=0;seen=set();elapsed=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['manifest_sha256']==sha(directory/'manifest.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng'])
        start,best,best_step,seen,elapsed=saved['step'],saved['best'],saved['best_step'],set(saved['seen']),saved['elapsed_seconds']
    else:
        save(directory/'initial_weights.pt',weights())
    started=time.monotonic();order=ids.copy()
    for step in range(start+1,m['steps']+1):
        # An initial full pass guarantees that all 800 source tasks and the
        # current task participate; later episodes are sampled uniformly.
        task=order[step-1] if step<=len(order) else ids[rng.randint(len(ids))]
        indices=train_indices[task];indices=rng.choice(indices,min(50,len(indices)),replace=False)
        tx,ty=tensors(task,indices)
        model.model.train();model.likelihood.train();model.feature_extractor.train()
        optimizer.zero_grad(set_to_none=True)
        features=model.feature_extractor(tx);model.model.set_train_data(features,ty,strict=False)
        loss=-model.mll(model.model(features),ty)
        if not torch.isfinite(loss):raise ValueError('Nonfinite scratch FSBO training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite scratch FSBO gradient')
        optimizer.step();seen.add(task)
        if step%m['validate_every']==0 or step==m['steps']:
            value=validate()
            if value<best and (step>=len(ids) or m['identity']['component']):
                best,best_step=value,step;save(directory/'weights',weights())
        if step%m['checkpoint_every']==0 or step==m['steps']:
            seconds=elapsed+time.monotonic()-started
            save(checkpoint,dict(manifest_sha256=sha(directory/'manifest.json'),model=cpu_state(model),
                optimizer=optimizer.state_dict(),rng=rng.get_state(),torch_rng=torch.get_rng_state(),
                step=step,best=best,best_step=best_step,seen=sorted(seen),elapsed_seconds=seconds))
            info=dict(status='training',step=step,requested_steps=m['steps'],best_step=best_step,
                best_validation_loss=best,initialization='scratch',source_tasks=len(m['source_tasks']),
                current_task=m['current_task'],tasks_sampled=len(seen),all_tasks_sampled=len(seen)==len(ids),
                elapsed_seconds=seconds,identity=m['identity'])
            write(directory/'meta_status.json',info);print(json.dumps(info),flush=True)
    selected=torch.load(directory/'weights',map_location='cpu',weights_only=False)
    model.model.load_state_dict(selected['gp']);model.likelihood.load_state_dict(selected['likelihood']);model.feature_extractor.load_state_dict(selected['net'])
    assert np.isclose(validate(),best,rtol=1e-5,atol=1e-5),'Selected FSBO reload differs'
    info.update(status='complete',weights_sha256=sha(directory/'weights'),reload_verified=True)
    write(directory/'meta_status.json',info)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',required=True)
    train(parser.parse_args().train)
