"""Retrospective CPU-only target retention/decoder probe. No oracle calls or training."""
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
UP=Path('/root/data/cys/BOLT/optimization/peptides')
OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(ROOT/'deps'))
sys.path.insert(0,str(UP))
import torch
import numpy as np
from Levenshtein import distance
from uniref_vae import data as native_data
sys.modules['data']=native_data
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
spec=importlib.util.spec_from_file_location('author_cpu_vae',ROOT/'reference/APEXGo/generation/vae.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
vocab=__import__('pandas').read_csv(UP/'uniref_vae/1mer_vocab.csv',header=None).values.squeeze().tolist()
data=native_data.DatasetKmers(k=1,load_data=False,vocab=vocab)
model=module.InfoTransformerVAE(dataset=data,d_model=128,kl_factor=.0001,
    encoder_dim_feedforward=256,decoder_dim_feedforward=256,encoder_num_layers=6,decoder_num_layers=6)
model.max_string_length=30
assert not torch.cuda.is_available() and model.device.type=='cpu'
cfg=json.loads((ROOT/'config.json').read_text())
result=json.loads((ROOT/'evaluation/adapted_online/pep_999/result.json').read_text())
init=[r['sequence'] for r in result['observations'] if r['phase']=='initialization']
ref=result['reference']
inc='KLYWRSFAITALAL'
anchors=list(dict.fromkeys([ref,inc]+init[:14]))
def tokens(seq):
    return native_data.collate_fn([data.encode(s).unsqueeze(0) for s in data.tokenize_sequence(seq)])
def decode(z,seed):
    torch.manual_seed(seed)
    sample=model.sample(z=z.reshape(-1,2,128))
    return [data.decode(sample[i]) for i in range(sample.shape[0])]
def finite_weights():
    return all(torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point())
models=[('Author frozen',Path(cfg['author_vae'])),('Before online tuning',ROOT/'vae/weights.pt'),
        ('After online tuning',ROOT/'evaluation/adapted_online/pep_999/checkpoint.pt')]
summaries=[];rows=[];start=time.monotonic()
for label,path in models:
    weights=torch.load(path,map_location='cpu',weights_only=False)
    if label=='After online tuning':weights=weights['vae']
    model.load_state_dict(weights,strict=True);del weights
    model.eval();assert finite_weights()
    with torch.inference_mode():
        x=tokens(init)
        losses=[]
        for seed in [8131,8132,8133]:
            torch.manual_seed(seed)
            out=model(x)
            losses.append({k:float(out[k]) for k in ['loss','recon_loss','kldiv','recon_token_acc','recon_string_acc']})
        mu,_=model.encode(x)
        recon=[]
        for seed in [7131,7132,7133]:
            decoded=decode(mu,seed)
            sims=[1-distance(s,ref)/len(ref) for s in decoded]
            rec=dict(seed=seed,n=len(decoded),exact_reconstructions=sum(a==b for a,b in zip(decoded,init)),
                     feasible=sum(s>=.75 for s in sims),mean_similarity=float(np.mean(sims)))
            recon.append(rec)
            rows.extend(dict(model=label,kind='initial_reconstruction',seed=seed,input=a,decoded=b,
                             feasible=distance(b,ref)<=3,edits_from_reference=distance(b,ref)) for a,b in zip(init,decoded))
        centers,_=model.encode(tokens(anchors))
        centers=centers.reshape(-1,256)
        local=[]
        # Fixed sparse perturbations about the same observed/reference anchors;
        # not GP-selected candidates and not a replay of the historical run.
        for length in [.1,.2,.8]:
            generator=torch.Generator().manual_seed(19001)
            indexes=torch.arange(128)%len(anchors)
            center=centers[indexes]
            mask=torch.rand((128,256),generator=generator)<20/256
            delta=(torch.rand((128,256),generator=generator)*2-1)*4*length
            z=center+mask*delta
            decoded=decode(z,19002)
            sims=[1-distance(s,ref)/len(ref) for s in decoded]
            rec=dict(tr_length=length,n=128,feasible=sum(s>=.75 for s in sims),
                     unique_feasible=len({s for s,sim in zip(decoded,sims) if sim>=.75}),
                     mean_similarity=float(np.mean(sims)),median_edits=float(np.median([distance(s,ref) for s in decoded])))
            local.append(rec)
            rows.extend(dict(model=label,kind='local_decode',seed=19002,tr_length=length,input=anchors[int(i)],decoded=s,
                             feasible=distance(s,ref)<=3,edits_from_reference=distance(s,ref)) for i,s in zip(indexes,decoded))
    summary=dict(model=label,path=str(path),weights_finite=True,eval_losses=losses,
                 initial_reconstruction_probes=recon,local_sparse_decode_probes=local)
    summaries.append(summary);print(json.dumps(summary),flush=True)
    (OUT/'vae_cpu_probe_partial.json').write_text(json.dumps(summaries,indent=2)+'\n')
report=dict(task='pep_999',device='CPU',threads=2,elapsed_seconds=time.monotonic()-start,
            training_performed=False,oracle_calls=0,models=summaries,
            caveats=['Retrospective fp32 CPU comparison; not the original bf16 GPU training loss.',
                     'Initial100 sequences are a retention probe, not held-out target validation: they were observed during optimization.',
                     'Seeds are paired and constant across models; autoregressive decoding is stochastic.',
                     'Sparse local decoding uses fixed anchors and masks, not historical GP-selected candidates or a causal ablation.',
                     'Intermediate pre/post-update VAE tensors were not retained separately, so exact per-update target rollback decisions cannot be reconstructed.'])
(OUT/'vae_cpu_probe.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'vae_cpu_probe_sequences.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader();writer.writerows(rows)
print('COMPLETE',report['elapsed_seconds'],flush=True)
