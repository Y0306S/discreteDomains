"""Decoder retention check using source tasks only, never held-out outcomes."""
from common import *


def audit(model,data,seed=7123):
    import numpy as np
    import torch
    from Levenshtein import distance
    from uniref_vae.load_vae import vae_decode,vae_forward
    cfg=prepare();db=corpus_db()
    chosen=np.random.RandomState(seed).choice(cfg['train_tasks'],32,replace=False)
    sequences=[];references=[]
    for task in chosen:
        d=task_data(db,task)
        # Source initialization observations have known task feasibility.
        for s in d['sequences'][:4].tolist():
            sequences.append(s);references.append(str(d['reference']))
    db.close();model.eval()
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        z=encode(model,data,sequences)
        decoded=vae_decode(torch.as_tensor(z),model,data)
        losses=[]
        for i in range(0,len(sequences),32):
            _,loss=vae_forward(sequences[i:i+32],data,model);losses.append(float(loss))
    similarities=[1-distance(s,r)/len(r) for s,r in zip(decoded,references)]
    return dict(tasks=32,sequences=len(sequences),feasible=sum(s>=.75 for s in similarities),
        mean_similarity=float(np.mean(similarities)),reconstruction_loss=float(np.mean(losses)),
        source_only=True)


def main():
    import torch
    setup();cfg=prepare()
    original,data=load_vae(cfg['author_vae']);baseline=audit(original,data)
    del original;torch.cuda.empty_cache()
    adapted,data=load_vae(ROOT/'vae/weights.pt');result=audit(adapted,data)
    passed=(result['feasible']>=.8*baseline['feasible'] and
        result['reconstruction_loss']<=1.25*baseline['reconstruction_loss'])
    value=dict(passed=passed,baseline=baseline,adapted=result,vae_sha256=sha(ROOT/'vae/weights.pt'))
    write(ROOT/'reconstruction_gate.json',value)
    print(json.dumps(value),flush=True)
    if not passed:raise ValueError('Adapted VAE failed source-only decoder retention gate')


if __name__=='__main__':main()
