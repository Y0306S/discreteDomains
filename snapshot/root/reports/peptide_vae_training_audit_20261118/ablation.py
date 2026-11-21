"""Controlled source-only learning-rate/label-scale diagnosis, no oracle calls."""
import sys
from pathlib import Path
sys.path.insert(0,'/root/data/logs/peptide_vae_fsbo_source_replay_test10_20261118/code')
from common import *
from reconstruction import audit
import numpy as np
import torch
from uniref_vae.load_vae import vae_forward

OUT=Path(__file__).resolve().parent


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--lr',type=float,required=True);p.add_argument('--scale',choices=['raw','minmax'],required=True)
    args=p.parse_args();setup();cfg=prepare()
    model,data=load_vae(cfg['author_vae']);db=corpus_db()
    order=np.random.RandomState(123).permutation(cfg['train_tasks']).tolist()[:8]
    events=[];started=time.monotonic()
    for index,task in enumerate(order):
        d=task_data(db,task);ids=training_indices(d);sequences=d['sequences'][ids].tolist()
        y=d['y'][ids].astype(np.float32);c=d['constraints'][ids].astype(np.float32)
        raw_range=[float(y.min()),float(y.max())]
        if args.scale=='minmax':y=(y-y.min())/(float(y.max()-y.min()) or 1.)
        x=encode(model,data,sequences,mean=False)
        gp,mll=make_gp(x);cm,cmll=make_gp(x)
        fit_gp(gp,mll,x,y,20);fit_gp(cm,cmll,x,c,20)
        model.train();gp.train();cm.train()
        optimizer=torch.optim.Adam([{'params':model.parameters()},{'params':gp.parameters()},{'params':cm.parameters()}],lr=args.lr)
        batch=max(1,2560//max(map(len,sequences)))
        metrics=[];first=None
        for epoch in range(2):
            for start in range(0,len(sequences),batch):
                optimizer.zero_grad(set_to_none=True)
                z,recon=vae_forward(sequences[start:start+batch],data,model)
                objective=-mll(gp(z),torch.as_tensor(y[start:start+batch],device='cuda'))
                constraint=-cmll(cm(z),torch.as_tensor(c[start:start+batch],device='cuda'))
                if first is None:
                    parameters=list(model.encoder.parameters())
                    def norm(term):
                        grads=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
                        return sum(float(g.square().sum()) for g in grads if g is not None)**.5
                    first=dict(reconstruction_encoder_grad=norm(recon),objective_encoder_grad=norm(objective),constraint_encoder_grad=norm(constraint))
                loss=recon+objective+constraint
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                metrics.append([float(recon.detach()),float(objective.detach()),float(constraint.detach())])
        model.eval();gp.eval();cm.eval()
        event=dict(index=index+1,task=task,lr=args.lr,scale=args.scale,raw_y_range=raw_range,
            mean_losses=dict(zip(['reconstruction','objective','constraint'],np.mean(metrics,axis=0).tolist())),
            first_batch_gradients=first)
        if index in [0,3,7]:event['reconstruction_audit']=audit(model,data)
        events.append(event);print(json.dumps(event),flush=True)
        del gp,mll,cm,cmll,x,d
    db.close()
    write(OUT/f'ablation_{args.scale}_{args.lr:g}.json',dict(events=events,elapsed_seconds=time.monotonic()-started,
        source_only=True,source_tasks=order,author_vae_sha256=sha(cfg['author_vae'])))


if __name__=='__main__':main()
