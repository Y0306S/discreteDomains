"""Joint LOLBO replay, then refresh all 4M train800 canonical embeddings."""
import argparse
import io
import json
import os
import time
from types import SimpleNamespace
from common import *


def train(limit=None,attempt=0):
    import numpy as np
    import torch
    from uniref_vae.load_vae import vae_forward
    from lolbo.utils.utils import update_models_end_to_end_with_constraints
    cfg=prepare()
    learning_rate=cfg['offline_vae_learning_rate']*(.1**attempt)
    setup()
    model,data=load_vae(cfg['author_vae'])
    directory=ROOT/('smoke_offline' if limit else 'vae')
    directory.mkdir(exist_ok=True)
    path=directory/'state.pt'
    order=np.random.RandomState(123).permutation(cfg['train_tasks']).tolist()
    if limit:
        order=order[:limit]
    start,elapsed=0,0.
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['config_sha256']==sha(ROOT/'config.json')
        assert state['attempt']==attempt
        model.load_state_dict(state['vae'])
        start,elapsed=state['index'],state['elapsed_seconds']
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
    objective=SimpleNamespace(vae=model,vae_forward=lambda xs:vae_forward(xs,data,model))
    db=corpus_db()
    started=time.monotonic()
    for index,task in enumerate(order[start:],start):
        before=time.monotonic()
        d=task_data(db,task)
        assert str(d['reference'])==read(SOURCE/'recipe.json')['references'][task]
        ids=training_indices(d)
        sequences=d['sequences'][ids].tolist()
        y=d['y'][ids].astype(np.float32)
        # Raw MIC magnitudes overwhelmed reconstruction in the previous pilot.
        # Match FSBO's per-task response scaling for representation adaptation.
        y=(y-y.min())/(float(y.max()-y.min()) or 1.)
        c=d['constraints'][ids].astype(np.float32)
        x=encode(model,data,sequences,mean=False)
        gp,mll=make_gp(x)
        cm,cmll=make_gp(x)
        fit_gp(gp,mll,x,y,20)
        fit_gp(cm,cmll,x,c,20)
        initial=cpu_state(model)
        update_models_end_to_end_with_constraints(sequences,y,objective,gp,mll,learning_rate,2,
            train_c_scores=torch.as_tensor(c[:,None]),c_models=[cm],c_mlls=[cmll])
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite adapted VAE weights')
        delta=sum(float((v-initial[k]).square().sum()) for k,v in cpu_state(model).items() if v.is_floating_point())**.5
        if delta<=0:
            raise ValueError('Joint update did not change VAE weights')
        item=dict(task=task,index=index+1,total=len(order),selected_observations=len(ids),
            attempt=attempt,vae_learning_rate=learning_rate,
            task_seconds=time.monotonic()-before,elapsed_seconds=elapsed+time.monotonic()-started,
            vae_parameter_change_l2=delta)
        with (directory/'events.jsonl').open('a') as f:
            f.write(json.dumps(item)+'\n')
        save(path,dict(config_sha256=sha(ROOT/'config.json'),attempt=attempt,vae=cpu_state(model),index=index+1,
            elapsed_seconds=item['elapsed_seconds'],rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        write(directory/'status.json',dict(item,status='complete' if index+1==len(order) else 'training'))
        print(json.dumps(item),flush=True)
        del gp,mll,cm,cmll,initial,x,d
    save(directory/'weights.pt',cpu_state(model))
    summary=read(directory/'status.json')
    summary.update(status='complete',weights_sha256=sha(directory/'weights.pt'),new_oracle_calls=0)
    write(directory/'status.json',summary)
    db.close()


def export():
    import numpy as np
    import torch
    cfg=prepare()
    setup()
    assert read(ROOT/'vae/status.json')['status']=='complete'
    weights=ROOT/'vae/weights.pt'
    model,data=load_vae(weights)
    directory=ROOT/'corpus'
    directory.mkdir(exist_ok=True)
    source=corpus_db()
    target=sqlite3.connect(str(directory/'corpus.sqlite'))
    target.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,sha TEXT NOT NULL,payload BLOB NOT NULL)')
    low,high=np.full(256,np.inf),np.full(256,-np.inf)
    started=time.monotonic()
    count=0
    for index,task in enumerate(cfg['train_tasks']):
        cached=target.execute('SELECT sha,payload FROM tasks WHERE id=?',(task,)).fetchone()
        if cached:
            assert hashlib.sha256(cached[1]).hexdigest()==cached[0]
            with np.load(io.BytesIO(cached[1]),allow_pickle=False) as a:
                x=a['X'];n=len(x)
        else:
            d=task_data(source,task)
            x=encode(model,data,d['sequences'].tolist())
            n=len(x)
            stream=io.BytesIO()
            np.savez_compressed(stream,X=x,y=d['y'],constraints=d['constraints'],sequences=d['sequences'])
            payload=stream.getvalue()
            with target:
                target.execute('INSERT INTO tasks VALUES(?,?,?)',(task,hashlib.sha256(payload).hexdigest(),payload))
        low=np.minimum(low,x.min(0));high=np.maximum(high,x.max(0));count+=n
        write(directory/'status.json',dict(status='encoding',tasks=index+1,total=800,observations=count,
            elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(encoding_tasks=index+1,observations=count)),flush=True)
    manifest=read(OLD/'corpus/manifest.json')
    manifest.update(representation='adapted_train800_VAE_posterior_mean_256',vae_sha256=sha(weights),
        database=str(directory/'corpus.sqlite'),lower=low.tolist(),upper=high.tolist(),
        original_manifest_sha256=sha(OLD/'corpus/manifest.json'),observations=count,
        task_hashes=dict(target.execute('SELECT id,sha FROM tasks')),elapsed_seconds=time.monotonic()-started)
    assert count==4053113
    write(directory/'manifest.json',manifest)
    write(directory/'status.json',dict(status='complete',tasks=800,observations=count,elapsed_seconds=manifest['elapsed_seconds']))
    source.close();target.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','export'])
    parser.add_argument('--limit',type=int)
    parser.add_argument('--attempt',type=int,default=0)
    args=parser.parse_args()
    train(args.limit,args.attempt) if args.stage=='train' else export()
