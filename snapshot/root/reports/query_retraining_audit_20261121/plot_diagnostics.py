from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
x=json.loads((O/'decoder_probe.json').read_text());s=json.loads((O/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12.5,4.6))
regimes=['encoded_mean','local_005_span','local_020_span','native_initial_candidate'];labels=['Encoded mean','Small noise\n5% span','Larger noise\n20% span','Native initial\nsearch geometry']
a=axes[0];idx=np.arange(4)
for delta,name,label,color in [(-.19,'original','Original VAE','#777777'),(.19,'retrained','Retrained VAE','#CC79A7')]:
 vals=[100*x['models'][name]['regimes'][r]['production_valid_before_retries']/x['models'][name]['regimes'][r]['draws'] for r in regimes]
 bars=a.bar(idx+delta,vals,width=.38,label=label,color=color)
 a.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],fontsize=8,padding=2)
a.set_xticks(idx,labels,fontsize=8);a.set_ylim(0,110);a.set_ylabel('Valid non-crossjoin plans (%)');a.set_title('Decoder probe: 240 draws per setting');a.legend(frameon=False,fontsize=8)
a=axes[1]
keys=['versus_frozen_FSBO','versus_SemanticOPT','versus_previous_hybrid'];labels=['Frozen FSBO\n123 queries','SemanticOPT\n123 queries','Previous hybrid\n21 queries'];vals=[s['fixed_200_proposals'][k]['geomean_new_over_comparator'] for k in keys]
low=[s['fixed_200_proposals'][k]['descriptive_task_bootstrap_95pct_ratio_interval'][0] for k in keys];high=[s['fixed_200_proposals'][k]['descriptive_task_bootstrap_95pct_ratio_interval'][1] for k in keys]
a.errorbar(np.arange(3),vals,yerr=[np.subtract(vals,low),np.subtract(high,vals)],fmt='o',color='#CC79A7',capsize=5);a.axhline(1,color='#666666',linestyle='--',lw=1);a.set_xticks(np.arange(3),labels);a.set_ylim(.55,1.1);a.set_ylabel('Retrained hybrid / comparator runtime');a.set_title('At 200 new proposals; lower is better')
fig.text(.5,.01,'Historical runtime comparison: initializers, storage and source training corpus differ. Decoder draws are clustered within15queries.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1))
for ext in ['png','pdf','svg']:fig.savefig(O/('diagnostics.'+ext),dpi=180)
