import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path('/root/reports/peptide_train800_trust_region_20261121')
s=json.loads((OUT/'summary.json').read_text())
cells=s['decoder_cells'];miccells=s['mic_cells'];moves=s['historical']['improving_moves_by_distance']
# Publication-readable diagnostic plots, no smoothing of categorical/radius results.
colors={2:'#0072B2',3:'#D55E00',100:'#009E73'};labels={2:'Local ≤2 edits',3:'Local ≤3 edits',100:'Global constraint only'}
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for k in [2,3,100]:
 c=[x for x in cells if x['model']=='adapted' and x['local_edits']==k and x['split']=='all800' and x['radius']>0]
 x=[v['radius'] for v in c]
 axs[0,0].plot(x,[v['mean_valid_percent'] for v in c],'-o',color=colors[k],label=labels[k],ms=4)
 axs[0,1].plot(x,[v['mean_unique_novel'] for v in c],'-o',color=colors[k],label=labels[k],ms=4)
 m=[x for x in miccells if x['local_edits']==k and x['split']=='all800' and x['budget']==20 and x['radius']>0]
 axs[1,0].plot([v['radius'] for v in m],[v['geomean_best_over_initial'] for v in m],'-o',color=colors[k],label=labels[k],ms=4)
 axs[1,1].plot([v['radius'] for v in m],[v['filled_budget_percent'] for v in m],'-o',color=colors[k],label=labels[k],ms=4)
for ax in axs.flat:
 ax.set_xscale('log');ax.set_xticks([.1,.125,.15,.175,.2,.3,.8]);ax.set_xticklabels(['0.10','0.125','0.15','0.175','0.20','0.30','0.80'],rotation=45);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.set_xlabel('Latent trust-region length');ax.grid(alpha=.2);ax.legend(fontsize=8)
axs[0,0].set_ylabel('Feasible draws (%)');axs[0,1].set_ylabel('Mean unique unseen feasible peptides / 96 draws')
axs[1,0].set_ylabel('Geometric mean best MIC / initial best (↓)');axs[1,1].set_ylabel('Tasks filling 20-candidate budget (%)')
fig.suptitle('800 training peptides: fixed-pool search diagnostic, not BO convergence')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'trust_region_800.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for i,k in enumerate([2,3]):
 for model,ls in [('adapted','-'),('author','--')]:
  c=[x for x in cells if x['model']==model and x['local_edits']==k and x['split']=='all800' and x['radius'] in [.15,.2,.8]]
  axs[0].plot([str(v['radius']) for v in c],[v['mean_unique_novel'] for v in c],ls+'o',color=colors[k],label=f'{model}, ≤{k}edits')
 distances=sorted(int(k) for k in moves)
axs[1].bar(distances,[moves[str(d)] for d in distances],color=['#0072B2' if d<=2 else '#D55E00' if d==3 else '#999999' for d in distances])
axs[0].set(xlabel='Latent length',ylabel='Mean unique unseen feasible /96draws');axs[0].legend(fontsize=8)
axs[1].set(xlabel='Edits from previous best peptide',ylabel='Observed historical incumbent improvements')
for ax in axs:ax.grid(axis='y',alpha=.2)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'frozen_and_history.{ext}',dpi=180)
plt.close(fig)
