"""Summarize saved loss audit and read-only CPU checkpoint probes."""
import csv
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
probe=json.loads((OUT/'vae_cpu_probe.json').read_text())
audit=list(csv.DictReader((OUT/'vae_loss_gate_audit.csv').open()))
focus=[r for r in audit if r['task']=='pep_999']
before=probe['models'][1];after=probe['models'][2]
loss_before=statistics.mean(r['loss'] for r in before['eval_losses'])
loss_after=statistics.mean(r['loss'] for r in after['eval_losses'])
metrics=[]
for model in probe['models']:
    recon=model['initial_reconstruction_probes']
    metrics.append(dict(model=model['model'],total_eval_loss_mean=statistics.mean(r['loss'] for r in model['eval_losses']),
                        reconstruction_cross_entropy_mean=statistics.mean(r['recon_loss'] for r in model['eval_losses']),
                        kl_mean=statistics.mean(r['kldiv'] for r in model['eval_losses']),
                        exact_reconstructions=sum(r['exact_reconstructions'] for r in recon),
                        feasible_reconstructions=sum(r['feasible'] for r in recon),sampled_reconstructions=sum(r['n'] for r in recon),
                        local_01_feasible=next(r['feasible'] for r in model['local_sparse_decode_probes'] if r['tr_length']==.1),
                        local_02_feasible=next(r['feasible'] for r in model['local_sparse_decode_probes'] if r['tr_length']==.2),
                        local_08_feasible=next(r['feasible'] for r in model['local_sparse_decode_probes'] if r['tr_length']==.8)))
with (OUT/'vae_checkpoint_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
labels=['Author\nfrozen','Before online\ntuning','After online\ntuning']
colors=['#1f77b4','#888888','#d62728']
ax=axes[0]
for i,model in enumerate(probe['models']):
    vals=[r['loss']*1000 for r in model['eval_losses']]
    ax.bar(i,np.mean(vals),color=colors[i],alpha=.8)
    ax.scatter([i]*len(vals),vals,color='black',s=12,zorder=3)
ax.set(xticks=range(3),xticklabels=labels,ylabel='Target evaluation loss × 1,000',title='pep_999 retention loss increased')
ax.grid(axis='y',alpha=.2)
ax=axes[1]
for i,m in enumerate(probe['models']):
    local=m['local_sparse_decode_probes']
    ax.plot([r['tr_length'] for r in local],[100*r['feasible']/r['n'] for r in local],'-o',label=labels[i].replace('\n',' '),color=colors[i],lw=2)
ax.set(title='Fixed local decoder probes',xlabel='Latent trust-region length',ylabel='Decoded candidates feasible (%)',xticks=[.1,.2,.8],ylim=(-3,105))
ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[2]
rounds=[int(r['round']) for r in focus]
source=[float(r['source_probe_change_percent']) for r in focus]
ax.plot(rounds,source,'o-',color='#d62728')
ax.axhline(0,color='gray',lw=1)
ax.set(title='Source-only checks missed target drift',xlabel='Online update after round',ylabel='Source check loss change per update (%)',xticks=rounds,ylim=(-1,1))
ax.grid(alpha=.2)
fig.text(.5,.005,'CPU fp32, no training or MIC calls. Target retention uses 100 previously observed sequences and three paired seeds; local probes are not BO-selected candidates.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1))
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'vae_loss_and_decoder_probe.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

text=f'''# pep_999 VAE loss and rollback investigation

The saved evidence supports target-specific VAE degradation, not a demonstrated numerical collapse. A new CPU-only comparison of actual saved checkpoints found that target evaluation loss increased {loss_after/loss_before:.2f}×, despite stable source retention metrics. Reconstruction still works for nearly all known target sequences. This can contribute to poor search, but does not establish how much of the MIC gap was caused by VAE updates versus the changed FSBO prior and search trajectory.

## Direct checkpoint comparison

The same 100 initialization sequences from pep_999 were evaluated before and after online tuning, with three paired seeds, no dropout, no new training and no oracle calls. These sequences were already observed and used during optimization: this is a retention diagnostic, not an independent held-out generalization test. Computation uses CPU fp32, so these losses should not be numerically equated with the original bf16 training losses.

| Saved VAE | Mean target evaluation loss | Exact autoregressive reconstruction | Feasible reconstruction | Feasible fixed local probes, length0.1 | Length0.2 | Length0.8 |
|---|---:|---:|---:|---:|---:|---:|
'''
for m in metrics:
    text+=f"| {m['model']} | {m['total_eval_loss_mean']:.7f} | {m['exact_reconstructions']}/300 | {m['feasible_reconstructions']}/300 | {m['local_01_feasible']}/128 | {m['local_02_feasible']}/128 | {m['local_08_feasible']}/128 |\n"
text+='''
Both before- and after-online checkpoint weights are finite. The final target loss increase comes from reconstruction cross entropy; mean KL falls slightly. Teacher-forced token/string argmax accuracy stays at100%, illustrating that argmax accuracy alone misses deteriorating probability calibration and autoregressive failures. The final VAE is not completely broken, but preserves the target distribution less well.

The local decoding test fixes the reference/incumbent/initial anchors, random seeds, sparse masks and perturbation scale across models. At the native starting trust-region length0.8, all three VAEs produced0/128 feasible samples. Smaller perturbations dramatically improve decoder validity in this diagnostic. These samples were not selected by the GP, do not reproduce the historical candidate trajectory, and have no MIC measurements. The figures support testing a tighter neighborhood; they do not establish an optimal trust-region setting or better MIC.

## What the seven actual update logs show

Each pep_999 update trained for two epochs on128–154 unique selected sequences, below the batch size512: only two attempted optimizer steps per update. Training uses dropout and random posterior samples, and bf16 with GradScaler. The saved first/last loss values are stochastic training-mode observations, not evaluation of the same checkpoint under paired randomness. The log calls them optimizer_steps, but does not record actual GradScaler-skipped steps or gradient norms. There is no complete within-step history of reconstruction/KL components.

| Update round | First training loss | Last training loss | Source retention check change | Accepted |
|---|---:|---:|---:|---|
'''
for r in focus:
    text+=f"| {r['round']} | {float(r['first_loss']):.7f} | {float(r['last_loss']):.7f} | {float(r['source_probe_change_percent']):+.3f}% | {r['accepted']} |\n"
text+='''
All logged pep_999 losses are finite. Source probe validity remained128/128 at every update, with source loss changes below0.4% in magnitude per update. This is inconsistent with an obvious numerical blow-up. It does not rule out cumulative target forgetting, which the new checkpoint test detects.

## Rollback already implemented, and its limits

The online update copies the VAE state before training. It restores those weights when the source-only gate fails. Acceptance requires at least80% of the larger pre-update/author source validity count and at most1.25× the smaller pre-update/author source loss. With128 feasible source reconstructions before an update, the count gate can allow a fall to103, so it is not a strict retention guarantee.

There is a real rollback example in this same campaign: pep_673 round69. Its training loss fell from0.0003538 to0.0003469, but the source retention loss rose56.0%; the gate rejected it and the logged post-rollback weight change was0. Thus a rule based only on falling training loss would have accepted an update that the existing evaluation check correctly rejected.

The source metric is named reconstruction_loss in the audit, but the called vae_forward returns total reconstruction-plus-KL loss. The components should be logged separately. The source gate covers32 source tasks and128 sequences, not pep_999, so it cannot certify retention of the current target or feasibility in its search neighborhood.

Nonfinite VAE loss currently raises an exception. It does not take the normal rejected-update rollback branch. GradScaler can skip nonfinite-gradient steps, but skipped-step counts are not logged. The code should distinguish numerical rejection from infrastructure failures rather than treat every exception as model degradation.

## Recommended transactional acceptance policy

1. Before attempting an update, retain a durable pre-update checkpoint and baseline measurements. Preserve the current known-good VAE, FSBO prior, normalization, embeddings, constraint model and trust-region state as one consistent generation. Keep probe RNG independent of search RNG.
2. Log reconstruction and KL separately, finite loss/gradients/weights, gradient norm, scaler value and actual optimizer steps. On nonfinite values or a confirmed numerical failure, reject the attempted update and restore the pre-update generation. Propagate storage/device faults distinctly.
3. Evaluate the old and candidate VAE on exactly the same fixed source set and target-known anchor set, using paired random seeds and evaluation mode. Include total loss, reconstruction loss, exact/feasible autoregressive reconstruction and a fixed local decoder probe. A loss increase on a fixed set is much more meaningful than last-training-loss/first-training-loss.
4. Compare with both the immediately preceding accepted checkpoint and a fixed target-retention baseline from before online tuning, to catch gradual cumulative drift. Use numerical floors and measured repeat variability for near-zero reconstruction losses. Calibrate acceptance margins on development tasks; do not tune a threshold after seeing held-out pep_999 MIC outcomes. Target diagnostic anchors must come only from already observed sequences, never future proposals or the frozen run's later solution. A target subset withheld from VAE gradient updates can provide a stronger generalization check.
5. Accept only after those checks pass. Then re-encode and rebuild the matching downstream models. If a numerical failure occurs during rebuilding, retain the old consistent generation. On rejection, continue BO from the old checkpoint and record the reason, attempt and counters. Do not roll back measured observations, erase oracle costs or change the historical incumbent.

Target-retention checking would flag the final-versus-initial loss drift shown here under a reasonable calibrated criterion. Exact intermediate rejection decisions cannot be reconstructed: individual pre/post-update VAE tensor snapshots and target probe results were not retained. Therefore it is not established that the proposed policy would have rejected a specific historical update, prevented the plateau or achieved the frozen MIC.

For this task, rollback is a useful guardrail but not the entire fix. Feasibility was already poor before the first update; the native wide-neighborhood probe fails even for the author VAE. The next controlled experiment should separate initial adaptation, online updates and latent search radius, while retaining identical evaluation accounting. No training policy or scientific result was modified in this investigation.

[Loss and decoder comparison](vae_loss_and_decoder_probe.png) · [Raw checkpoint probe](vae_cpu_probe.json) · [Loss gate audit](vae_loss_gate_audit.csv) · [Metric CSV](vae_checkpoint_metrics.csv)
'''
(OUT/'VAE_ROLLBACK_INVESTIGATION.md').write_text(text)
print(json.dumps(dict(target_loss_before=loss_before,target_loss_after=loss_after,ratio=loss_after/loss_before,
                     metrics=metrics),indent=2),flush=True)
