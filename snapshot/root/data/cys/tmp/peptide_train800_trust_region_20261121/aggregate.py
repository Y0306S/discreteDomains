"""Validate and summarize all800 paired source diagnostics; no production mutation."""
import csv,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=Path('/root/reports/peptide_train800_trust_region_20261121');OUT.mkdir(exist_ok=True)
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def table(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def boot_mean(a,seed=1201):
 a=np.asarray(a,dtype=float);rng=np.random.default_rng(seed)
 means=a[rng.integers(len(a),size=(3000,len(a)))].mean(1)
 return [float(x) for x in np.quantile(means,[.025,.975])]
def geo(a):return float(np.exp(np.mean(np.log(a))))
config=json.loads(Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/config.json').read_text())
tasks=set(config['train_tasks']);assert len(tasks)==800
allmodels={};decoder=[];physical_draws=0
for model,radii in [('adapted',[0.,.1,.125,.15,.175,.2,.3,.8]),('author',[.15,.2,.8])]:
 a=[json.loads(p.read_text()) for p in sorted((ROOT/'tasks'/model).glob('*.json'))]
 assert len(a)==800 and {r['task'] for r in a}==tasks,(model,len(a))
 assert Counter(r['development_split'] for r in a)=={'calibration640':640,'confirmation160':160}
 allmodels[model]=a
 for x in a:
  assert x['status']=='complete' and x['identity']['radii']==radii and x['identity']['seeds']==[31001,31002,31003]
  assert len(x['summary'])==len(radii) and all(s['draws']==96 for s in x['summary'])
  physical_draws+=sum(s['draws'] for s in x['summary'])
  for s in x['summary']:
   for k in [1,2,3,100]:
    c=s['counts'][str(k)]
    assert 0<=c['unique_novel_valid']<=c['novel_valid_draws']<=c['valid_draws']<=96
    decoder.append(dict(model=model,task=x['task'],split=x['development_split'],global_edits=x['global_edit_budget'],radius=s['radius'],local_edits=k,valid_fraction=c['valid_draws']/96,novel_valid_fraction=c['novel_valid_draws']/96,unique_novel=c['unique_novel_valid']))
assert physical_draws==844800
cells=[]
for model,radii in [('adapted',[0.,.1,.125,.15,.175,.2,.3,.8]),('author',[.15,.2,.8])]:
 for radius in radii:
  for k in [1,2,3,100]:
   for split in ['all800','calibration640','confirmation160']:
    group=[r for r in decoder if r['model']==model and r['radius']==radius and r['local_edits']==k and (split=='all800' or r['split']==split)]
    u=[r['unique_novel'] for r in group];v=[r['valid_fraction'] for r in group]
    cells.append(dict(model=model,radius=radius,local_edits=k,split=split,tasks=len(group),mean_valid_percent=100*float(np.mean(v)),mean_unique_novel=float(np.mean(u)),median_unique_novel=float(np.median(u)),unique_novel_q10=float(np.quantile(u,.1)),unique_novel_q90=float(np.quantile(u,.9)),zero_novel_percent=100*float(np.mean(np.array(u)==0)),at_least20_percent=100*float(np.mean(np.array(u)>=20))))
scored=[json.loads(p.read_text()) for p in sorted((ROOT/'scored').glob('*.json'))]
assert len(scored)==800 and {r['task'] for r in scored}==tasks,len(scored)
microws=[]
for x in scored:
 assert x['baseline']=='fresh_rescored100' and len(x['initial_predictions_values'])==100
 p=ROOT/'tasks/adapted'/(x['task']+'.json');assert hashlib.sha256(p.read_bytes()).hexdigest()==x['decoder_result_sha256']
 assert len(x['measurements'])==8*4*4
 for m in x['measurements']:
  microws.append(dict(task=x['task'],split=x['development_split'],global_edits=x['global_edit_budget'],**m))
miccells=[]
for radius in [0.,.1,.125,.15,.175,.2,.3,.8]:
 for k in [1,2,3,100]:
  for budget in [5,10,20,100000]:
   for split in ['all800','calibration640','confirmation160']:
    g=[r for r in microws if r['radius']==radius and r['local_edits']==k and r['nominal_candidate_budget']==budget and (split=='all800' or r['split']==split)]
    a=[r['best_over_initial'] for r in g];n=[r['actual_candidates_scored'] for r in g]
    miccells.append(dict(radius=radius,local_edits=k,budget=budget,split=split,tasks=len(g),geomean_best_over_initial=geo(a),mean_best_over_initial=float(np.mean(a)),median_best_over_initial=float(np.median(a)),improved_tasks=int(sum(x<1-1e-6 for x in a)),improved_at_least1percent_tasks=int(sum(x<.99 for x in a)),filled_budget_percent=100*float(np.mean([r['filled_budget'] for r in g])),mean_candidates=float(np.mean(n))))
# Report length/reference-edit-budget strata, rather than only cohort means.
strata=[]
for budget_group in [2,3,4,5,6]:
 for radius in [.15,.175,.2,.3,.8]:
  for k in [2,3,100]:
   d=[r for r in decoder if r['model']=='adapted' and r['radius']==radius and r['local_edits']==k and r['global_edits']==budget_group]
   m=[r for r in microws if r['radius']==radius and r['local_edits']==k and r['global_edits']==budget_group and r['nominal_candidate_budget']==20]
   assert len(d)==len(m)>0
   strata.append(dict(global_edit_budget=budget_group,radius=radius,local_edits=k,tasks=len(m),mean_valid_percent=100*float(np.mean([r['valid_fraction'] for r in d])),mean_unique_novel=float(np.mean([r['unique_novel'] for r in d])),zero_novel_percent=100*float(np.mean([r['unique_novel']==0 for r in d])),geomean_best_over_initial=geo([r['best_over_initial'] for r in m]),improved_tasks=sum(r['best_over_initial']<1-1e-6 for r in m),mean_actual_candidates=float(np.mean([r['actual_candidates_scored'] for r in m])),filled20_percent=100*float(np.mean([r['filled_budget'] for r in m]))))
table(OUT/'strata_summary.csv',strata)
# Compare paired task outcomes, preserving tasks as the sampling unit (not draws).
paired=[]
for radius in [.1,.125,.15,.175,.2,.3,.8]:
 for split in ['all800','calibration640','confirmation160']:
  g={k:{r['task']:r for r in microws if r['radius']==radius and r['local_edits']==k and r['nominal_candidate_budget']==20 and (split=='all800' or r['split']==split)} for k in [2,3,100]}
  for ka,kb in [(3,2),(100,3)]:
   for matched in [False,True]:
    ids=sorted(t for t in g[ka] if not matched or g[ka][t]['filled_budget'] and g[kb][t]['filled_budget'])
    if not ids:continue
    dif=np.array([math.log(g[ka][t]['best_over_initial']/g[kb][t]['best_over_initial']) for t in ids])
    ci=boot_mean(dif)
    paired.append(dict(radius=radius,split=split,method_a_local_edits=ka,method_b_local_edits=kb,require_both20=matched,tasks=len(ids),geomean_a_over_b=float(np.exp(np.mean(dif))),ci95_low=float(np.exp(ci[0])),ci95_high=float(np.exp(ci[1])),a_better=int(sum(dif<-1e-6)),b_better=int(sum(dif>1e-6)),tied=int(sum(np.abs(dif)<=1e-6))))
# Select radius for each local mode using only640development tasks, then report160.
choices=[]
for k in [2,3,100]:
 candidates=[c for c in miccells if c['local_edits']==k and c['budget']==20 and c['split']=='calibration640' and c['radius'] in [.125,.15,.175,.2,.3]]
 best=min(candidates,key=lambda c:c['geomean_best_over_initial'])
 confirm=next(c for c in miccells if c['local_edits']==k and c['budget']==20 and c['split']=='confirmation160' and c['radius']==best['radius'])
 choices.append(dict(local_edits=k,selected_radius=best['radius'],selection='minimum640task geometric MIC ratio at capped20available candidates; unequal fill rates reported',calibration=best,confirmation=confirm))
audit=json.loads((ROOT/'source_audit.json').read_text());assert len(audit['rows'])==800
moves=Counter(str(m['edits_from_previous_best']) for r in audit['rows'] for m in r['improving_moves'])
hist=dict(distinct_novel=sum(r['distinct_novel'] for r in audit['rows']),distinct_novel_feasible=sum(r['distinct_novel_feasible'] for r in audit['rows']),improving_moves_by_distance=dict(sorted(moves.items(),key=lambda x:int(x[0]))),total_improving_moves=sum(moves.values()),improving_moves_over3=sum(v for k,v in moves.items() if int(k)>3),global_edit_budget_counts=dict(Counter(str(r['allowed_reference_edits']) for r in audit['rows'])))
failed=json.loads((ROOT/'failed_attempt_predictions.json').read_text())
parity=json.loads((ROOT/'native_scorer_parity.json').read_text());assert parity['status']=='passed'
physical=dict(decoder_draws=physical_draws,source_tasks=800,adapted_model_tasks=800,author_model_tasks=800,candidate_sequence_predictions=sum(r['physical_candidate_predictions'] for r in scored),fresh_initial_predictions=80000,parity_predictions=64,failed_attempt_predictions=failed['candidate_predictions']+failed['initial_predictions'])
physical['decoder_draws_accounting']='Retained complete paired draws. Uncommitted partial work during two-to-four-shard transition is not quantified.'
physical['total_apex_sequence_predictions']=sum(physical[k] for k in ['candidate_sequence_predictions','fresh_initial_predictions','parity_predictions','failed_attempt_predictions'])
summary=dict(status='complete',physical=physical,historical=hist,radius_choices=choices,decoder_cells=cells,mic_cells=miccells,paired_comparisons=paired,max_historical_initial_relative_error=max(r['max_calibration_relative_error'] for r in scored),native_parity=parity,limitations=['Not sequential BO or a4000call optimization campaign. Fixed unranked decoded pools, no GP selection or feedback.', '96draws per task/radius, paired3seeds. Seeds pooled; bootstrap samples tasks.', '640/160 source development split; both already representation-trained, progress aggregates inspected, not a blind held-out benchmark. Heldout200 excluded.', 'Fixed centers are reference+first3initial peptides, selected without candidate outcomes; not an adaptive incumbent replay.', 'Capped20comparison uses available candidates when pools underfill; mean actual counts, fill rates, and both-filled paired subsets are reported.', 'APEX MICs are model predictions in micromolar, not laboratory measurements.', 'Source movement audit is historical policy-dependent; large moves do not prove smaller-step paths cannot find the same peptide.'])
dump(OUT/'summary.json',summary)
table(OUT/'decoder_per_task.csv',decoder);table(OUT/'decoder_summary.csv',cells);table(OUT/'mic_per_task.csv',microws);table(OUT/'mic_summary.csv',miccells);table(OUT/'paired_2_vs_3.csv',paired)
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
print(json.dumps(dict(physical=physical,choices=[(c['local_edits'],c['selected_radius'],c['calibration']['geomean_best_over_initial'],c['confirmation']['geomean_best_over_initial']) for c in choices],max_historical_relative_error=summary['max_historical_initial_relative_error']),indent=2))
