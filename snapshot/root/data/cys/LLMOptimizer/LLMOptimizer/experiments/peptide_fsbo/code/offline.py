"""APEXGo sequence reconstruction+KL training, continued on source800.

The actual author train_vae.py uses Adam(2e-4), batch512, bf16 and GradScaler.
Its unused VAEModule scheduler is not part of that training path.
"""
import argparse
import importlib.util
import math
from common import *


def dataset():
    import numpy as np
    cfg=prepare();directory=ROOT/'sequence_data';directory.mkdir(exist_ok=True)
    manifest=directory/'manifest.json'
    if manifest.exists():
        value=read(manifest);assert value['config_sha256']==sha(ROOT/'config.json')
        for name,digest in value['files'].items():assert sha(directory/name)==digest
        return value
    db=corpus_db();unique=set();rows=0;hashes={};started=time.monotonic()
    for index,task in enumerate(cfg['train_tasks']):
        digest,payload=db.execute('SELECT payload_sha256,payload FROM tasks WHERE id=?',(task,)).fetchone()
        assert hashlib.sha256(payload).hexdigest()==digest
        with np.load(io.BytesIO(payload),allow_pickle=False) as archive:sequences=archive['sequences'].tolist()
        if any(not s or len(s)>30 or set(s)-set('ACDEFGHIKLMNPQRSTVWY') for s in sequences):
            raise ValueError('Invalid source training sequence: '+task)
        unique.update(sequences);rows+=len(sequences);hashes[task]=digest
        if (index+1)%20==0:
            write(directory/'status.json',dict(status='collecting',source_tasks=index+1,total=800,
                rows=rows,unique_sequences=len(unique),elapsed_seconds=time.monotonic()-started))
    db.close();sequences=np.asarray(sorted(unique),dtype='U30');del unique
    order=np.random.RandomState(cfg['fsbo_meta_seed']).permutation(len(sequences))
    n=len(sequences);boundaries=[0,n-int(n/10),n-int(n/20),n];files={};counts={}
    for i,name in enumerate(('train','val','test')):
        path=directory/(name+'.npy');np.save(path,sequences[order[boundaries[i]:boundaries[i+1]]],allow_pickle=False)
        files[path.name]=sha(path);counts[name]=boundaries[i+1]-boundaries[i]
    value=dict(config_sha256=sha(ROOT/'config.json'),source_tasks=cfg['train_tasks'],
        source_payload_sha256=hashes,rows=rows,unique_sequences=n,counts=counts,files=files,
        split='90/5/5 after deduplication and seeded shuffling; no sequence overlap',
        heldout_tasks_used=False,scores_used=False,elapsed_seconds=time.monotonic()-started)
    write(manifest,value);write(directory/'status.json',dict(value,status='complete'))
    print(json.dumps(dict(dataset=counts,unique_sequences=n)),flush=True);return value


def author_model():
    """Use the pinned, unmodified author model with our compatible tokenizer."""
    import torch
    cfg=prepare();native,data=load_vae(cfg['author_vae']);del native;torch.cuda.empty_cache()
    directory=ROOT/'reference/APEXGo/generation';sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('apexgo_author_vae',directory/'vae.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.path.remove(str(directory))
    model=module.InfoTransformerVAE(dataset=data,d_model=128,kl_factor=.0001,
        encoder_dropout=.05,decoder_dropout=.05,encoder_dim_feedforward=256,
        decoder_dim_feedforward=256,encoder_num_layers=6,decoder_num_layers=6).cuda()
    model.load_state_dict(torch.load(cfg['author_vae'],map_location='cpu',weights_only=False),strict=True)
    model.max_string_length=30;return model,data


def validation(model,data,sequences):
    import numpy as np
    import torch
    model.eval();losses=[]
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        torch.manual_seed(8123);torch.cuda.manual_seed_all(8123)
        for start in range(0,len(sequences),512):
            with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
                out=model(tokens(list(sequences[start:start+512]),data))
            losses.append(float(out['loss']))
    value=float(np.mean(losses))
    if not np.isfinite(value):raise ValueError('Nonfinite validation loss')
    return value


def update_batch(model,data,optimizer,scaler,sequences):
    import torch
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):out=model(tokens(list(sequences),data))
    loss=out['loss']
    if not torch.isfinite(loss):raise ValueError('Nonfinite VAE loss')
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
    return float(loss.detach())


def smoke():
    import numpy as np
    import torch
    from reconstruction import audit
    cfg=prepare();setup();torch.set_float32_matmul_precision('medium')
    model,data=author_model();before=audit(model,data);db=corpus_db()
    with np.load(io.BytesIO(db.execute('SELECT payload FROM tasks WHERE id=?',(cfg['train_tasks'][0],)).fetchone()[0]),allow_pickle=False) as a:
        sequences=a['sequences'][:512].tolist()
    db.close();native,ndata=load_vae(cfg['author_vae']);model.eval();native.eval();x=tokens(sequences[:32],data)
    with torch.no_grad():
        torch.manual_seed(91);a=model(x);torch.manual_seed(91);b=native(x)
    exact=all(torch.equal(a[k],b[k]) for k in ('loss','recon_loss','kldiv')) and torch.equal(a['z'].flatten(1),b['z'].flatten(1))
    objective_ok=torch.allclose(a['loss'],a['recon_loss']+.0001*a['kldiv'])
    del native;torch.cuda.empty_cache();initial=cpu_state(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0002);scaler=torch.amp.GradScaler('cuda');model.train()
    loss=update_batch(model,data,optimizer,scaler,sequences);after=audit(model,data)
    path=ROOT/'smoke_training/state.pt'
    save(path,dict(vae=cpu_state(model),optimizer=optimizer.state_dict(),scaler=scaler.state_dict()))
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['vae']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
    checks=dict(author_and_local_forward_identical=exact,reconstruction_plus_kl_only=bool(objective_ok),
        weights_changed=any(not torch.equal(v,saved['vae'][k]) for k,v in initial.items()),
        finite_training_loss=bool(np.isfinite(loss)),optimizer_resumed=len(optimizer.state)>0,
        source800_disjoint_from_test=not set(cfg['train_tasks'])&set(cfg['tasks']))
    value=dict(passed=all(checks.values()),checks=checks,before=before,after=after,train_loss=loss,
        model_source_sha256=sha(ROOT/'reference/APEXGo/generation/vae.py'))
    write(ROOT/'smoke_training.json',value);print(json.dumps(value),flush=True);assert value['passed']


def train():
    import numpy as np
    import torch
    from reconstruction import audit
    cfg=prepare();setup();torch.set_float32_matmul_precision('medium');metadata=dataset()
    directory=ROOT/'vae';directory.mkdir(exist_ok=True)
    train_rows=np.load(ROOT/'sequence_data/train.npy',mmap_mode='r');val_rows=np.load(ROOT/'sequence_data/val.npy',mmap_mode='r')
    model,data=author_model();optimizer=torch.optim.Adam(model.parameters(),lr=.0002);scaler=torch.amp.GradScaler('cuda')
    checkpoint=directory/'state.pt';start_epoch=0;start_batch=0;elapsed=0.;step=0;baseline=audit(model,data)
    epoch_sums=0.;epoch_steps=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['config_sha256']==sha(ROOT/'config.json')
        model.load_state_dict(saved['vae']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start_epoch,start_batch,step=saved['epoch'],saved['batch'],saved['step'];elapsed=saved['elapsed_seconds']
        best=saved['best'];baseline=saved['baseline'];epoch_sums,epoch_steps=saved['epoch_sums'],saved['epoch_steps']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        best=dict(validation_loss=validation(model,data,val_rows),epoch=0,initialization=True,quality=baseline)
        save(directory/'weights.pt',cpu_state(model));write(directory/'reconstruction_baseline.json',baseline);write(directory/'selected.json',best)
    started=time.monotonic();epochs=cfg['offline_vae_epochs']
    def snapshot(epoch,next_batch):
        save(checkpoint,dict(config_sha256=sha(ROOT/'config.json'),vae=cpu_state(model),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch,batch=next_batch,step=step,elapsed_seconds=elapsed+time.monotonic()-started,
            best=best,baseline=baseline,epoch_sums=epoch_sums,epoch_steps=epoch_steps,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    for epoch in range(start_epoch,epochs):
        order=np.random.RandomState(cfg['fsbo_meta_seed']+epoch).permutation(len(train_rows));total=math.ceil(len(order)/512);model.train()
        for number in range(start_batch if epoch==start_epoch else 0,total):
            loss=update_batch(model,data,optimizer,scaler,train_rows[order[number*512:(number+1)*512]])
            step+=1;epoch_steps+=1;epoch_sums+=loss
            if step%100==0 or number+1==total:
                if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite VAE weights')
                snapshot(epoch,number+1)
                value=dict(status='training',epoch=epoch+1,epochs=epochs,batch=number+1,total_batches=total,step=step,
                    mean_loss=epoch_sums/epoch_steps,elapsed_seconds=elapsed+time.monotonic()-started,
                    learning_rate=.0002,batch_size=512,objective='reconstruction+0.0001*KL',best=best)
                write(directory/'status.json',value);print(json.dumps(value),flush=True)
        if epoch%cfg['offline_vae_validation_frequency']==0 or epoch+1==epochs:
            value_loss=validation(model,data,val_rows);quality=audit(model,data)
            passed=quality['feasible']>=.8*baseline['feasible'] and quality['reconstruction_loss']<=1.25*baseline['reconstruction_loss']
            record=dict(epoch=epoch+1,validation_loss=value_loss,quality=quality,decoder_gate_passed=passed)
            with (directory/'validation_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if passed and value_loss<best['validation_loss']:
                best=dict(validation_loss=value_loss,epoch=epoch+1,initialization=False,quality=quality)
                save(directory/'weights.pt',cpu_state(model));write(directory/'selected.json',best)
            print(json.dumps(dict(validation=record,selected=best)),flush=True)
        epoch_sums=0.;epoch_steps=0;snapshot(epoch+1,0)
    write(directory/'status.json',dict(status='complete',epochs=epochs,step=step,
        elapsed_seconds=elapsed+time.monotonic()-started,weights_sha256=sha(directory/'weights.pt'),selected=best,
        baseline=baseline,new_oracle_calls=0,source_sequences=metadata['unique_sequences']))


def export():
    import numpy as np
    cfg=prepare();setup();assert read(ROOT/'vae/status.json')['status']=='complete'
    weights=ROOT/'vae/weights.pt';model,data=load_vae(weights);directory=ROOT/'corpus';directory.mkdir(exist_ok=True)
    identity=dict(vae_sha256=sha(weights),config_sha256=sha(ROOT/'config.json'))
    if (directory/'identity.json').exists():assert read(directory/'identity.json')==identity
    else:write(directory/'identity.json',identity)
    source=corpus_db();low,high=np.full(256,np.inf),np.full(256,-np.inf);count=0;hashes={};started=time.monotonic()
    for index,task in enumerate(cfg['train_tasks']):
        path=directory/(task+'.npz')
        if path.exists():
            payload=path.read_bytes()
            with np.load(io.BytesIO(payload),allow_pickle=False) as a:x=a['X'];n=len(x)
        else:
            d=task_data(source,task);x=encode(model,data,d['sequences'].tolist());n=len(x)
            stream=io.BytesIO();np.savez_compressed(stream,X=x,y=d['y'],constraints=d['constraints'],sequences=d['sequences'])
            payload=stream.getvalue();temporary=path.with_suffix('.npz.tmp');temporary.write_bytes(payload);temporary.replace(path)
        hashes[task]=hashlib.sha256(payload).hexdigest();assert sha(path)==hashes[task]
        low=np.minimum(low,x.min(0));high=np.maximum(high,x.max(0));count+=n
        write(directory/'status.json',dict(status='encoding',tasks=index+1,total=800,observations=count,elapsed_seconds=time.monotonic()-started))
        if (index+1)%20==0:print(json.dumps(dict(encoded_tasks=index+1,observations=count)),flush=True)
    source.close();manifest=read(OLD/'corpus/manifest.json')
    manifest.update(representation='APEXGo_recipe_train800_VAE_posterior_mean_256',vae_sha256=sha(weights),
        storage='hash_checked_npz_shards',database=str(directory),lower=low.tolist(),upper=high.tolist(),
        original_manifest_sha256=sha(OLD/'corpus/manifest.json'),observations=count,task_hashes=hashes,elapsed_seconds=time.monotonic()-started)
    assert count==4053113;write(directory/'manifest.json',manifest)
    write(directory/'status.json',dict(status='complete',tasks=800,observations=count,elapsed_seconds=manifest['elapsed_seconds']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['dataset','smoke','train','export'])
    args=parser.parse_args();globals()[args.stage]()
