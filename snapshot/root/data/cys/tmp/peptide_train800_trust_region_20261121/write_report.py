import hashlib,json,shutil
from pathlib import Path
OUT=Path('/root/reports/peptide_train800_trust_region_20261121');ROOT=Path(__file__).resolve().parent
s=json.loads((OUT/'summary.json').read_text());choices={c['local_edits']:c for c in s['radius_choices']}
def dec(radius,k,model='adapted',split='all800'):
 return next(r for r in s['decoder_cells'] if r['model']==model and r['radius']==radius and r['local_edits']==k and r['split']==split)
def mic(radius,k,split='all800'):
 return next(r for r in s['mic_cells'] if r['radius']==radius and r['local_edits']==k and r['split']==split and r['budget']==20)
rows=[]
for radius in [.15,.175,.2,.3,.8]:
 for k in [2,3,100]:
  d=dec(radius,k);m=mic(radius,k)
  rows.append(f"|{radius:g}|{'global only' if k==100 else str(k)}|{d['mean_valid_percent']:.3f}%|{d['mean_unique_novel']:.2f}|{d['zero_novel_percent']:.1f}%|{m['geomean_best_over_initial']:.4f}|{m['improved_tasks']}/800|{m['mean_candidates']:.1f}|{m['filled_budget_percent']:.1f}%|")
choice_rows=[]
for k in [2,3,100]:
 c=choices[k];a=c['calibration'];b=c['confirmation']
 choice_rows.append(f"|{'global only' if k==100 else str(k)}|{c['selected_radius']:g}|{a['geomean_best_over_initial']:.4f}|{b['geomean_best_over_initial']:.4f}|{b['improved_tasks']}/160|{b['filled_budget_percent']:.1f}%|")
pair=next(r for r in s['paired_comparisons'] if r['radius']==choices[3]['selected_radius'] and r['split']=='all800' and r['method_a_local_edits']==3 and r['method_b_local_edits']==2 and not r['require_both20'])
matched=next(r for r in s['paired_comparisons'] if r['radius']==choices[3]['selected_radius'] and r['split']=='all800' and r['method_a_local_edits']==3 and r['method_b_local_edits']==2 and r['require_both20'])
frozen=[]
for radius in [.15,.2,.8]:
 for k in [2,3]:
  a=dec(radius,k,'adapted');b=dec(radius,k,'author')
  frozen.append(f"|{radius:g}|{k}|{a['mean_unique_novel']:.2f}|{b['mean_unique_novel']:.2f}|{a['mean_valid_percent']:.3f}%|{b['mean_valid_percent']:.3f}%|")
physical=s['physical'];hist=s['historical']
text=f'''# Peptide trust regions: all800 training-task diagnostic —21November2026

## Findings

The completed diagnostic supports testing a three-edit local mode alongside a two-edit conservative control. It does not support a universal two-edit cap, and it does not establish that online training caused the pep_999 gap. Smaller latent neighborhoods produce much more useful sequence-space coverage than the original starting length0.8. A local cap alone is insufficient: use several feasible centers and retain an explicitly global-feasible exploration arm.

These are fixed-pool decoder and predicted-MIC diagnostics, **not800 completed sequential BO runs**. No production model, running optimization policy or historical result was changed. The held-out200 tasks were excluded. New isolated code is available for review, not deployed.

## Why frozen FSBO could succeed with the same wide region

Frozen and online used the same nominal recipe: length0.8, minimum0.0078125, maximum1.6; approximately20/256 perturbed coordinates;5000Sobol candidates;batch50;seed1. Online evaluate.py reads the frozen recipe directly. Their100 initialization sequences, measured scores and order match. Both online runs fit the target GP before the first acquisition, so missing initial target data is not the explanation.

The models differ. Frozen uses the author VAE, whereas online uses an adapted VAE and updates its models. The same latent length therefore need not imply the same sequence-edit distances, centers or acquisition ranking. A read-only replay of saved acquisition batches through the native update rule gives a more precise answer: frozen and latest rollback both used length0.8in rounds1–32,0.4in rounds33–64,and0.2in rounds65–80. Frozen found its strong candidates at rounds69and72 **after shrinking to0.2**, and the latest online improvement at round67also used0.2. The previous online arm improved at round52at length0.4, resetting its failure counter, and stayed at0.4through round80. These are reconstructed states, not original radius telemetry; radius_replay.json records the trace. The rule matches the frozen CPU source snapshot. Thus frozen did not find its winner while still using the initial0.8length. The latent bound is center±4×length in each perturbed coordinate, not a fixed number of amino-acid edits. Length0.2does not mean a two-edit sequence constraint. Starting smaller could avoid waiting64rounds (3200proposals) for these two trajectories to contract, but improvement must still be tested sequentially.

Frozen also explored inefficiently on pep_999:17/3561fresh acquisitions were feasible within the4000proposal comparison, versus25/3729for the latest rollback arm. It found KLYWRSFAITALAL at round69, then KLYWRSFAITKAL at round72, reaching43.3079µM. The latest online run reached101.0068µM and never generated the latter sequence. More feasible candidates alone did not ensure a better optimum. This is one trajectory per method; neither luck nor a specific training failure is established as the sole cause.

## What was tested

- All800 configured training tasks, using the exact initialization manifest, task IDs and pinned model weights. IDs are read from the campaign, not assumed to be000–799.
- Adapted VAE at lengths0,0.1,0.125,0.15,0.175,0.2,0.3,0.8; author VAE at0.15,0.2,0.8.
- Three paired seeds,32decodes per seed,96draws per task/radius; **{physical['decoder_draws']:,} saved complete decodes**. Uncommitted partial decoding during the two-to-four-shard transition was not quantified and is additional compute.
- Fixed centers: reference and first three initialization sequences, chosen without generated-candidate outcomes. These are not adaptive incumbent trajectories.
- Exact original reference constraint: Levenshtein distance≤floor(reference length/4), plus peptide syntax, length≤30 and at most one C. Local1,2,3and global-only gates were compared. Local edits are measured from the generating center; global edits always from the task reference.
- Deduplicated, initially unseen globally feasible candidates were scored with the original eight-model APEX ensemble, bacteria0, native MIC transformation and averaging. Initial100were rescored on the same device and precision, preserving the stored historical scores separately.
- Predicted-MIC summaries take the first5,10or20available accepted candidates in generation order, and separately the entire finite pool. There is no GP ranking, sequential feedback or offline selection of the best candidates before scoring.

## Full800 results

Each decoder column uses the same96raw draws per task/radius. The MIC column uses at most20accepted unique candidates and includes the initial incumbent. A ratio below1means improved predicted MIC; lower is better. Underfilled pools consume fewer actual candidate predictions and must not be interpreted as matched20call runs.

|Latent length|Local edits|Feasible draws|Mean unique new feasible /96|Tasks with zero new|Geometric best MIC / initial|Tasks improved|Mean candidates used ≤20|Tasks filling20|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

At length{choices[3]['selected_radius']:g}, the paired three-edit/two-edit MIC ratio is **{pair['geomean_a_over_b']:.4f}**, with a95% task-bootstrap interval[{pair['ci95_low']:.4f},{pair['ci95_high']:.4f}]. Three edits performs better on{pair['a_better']}tasks, two edits on{pair['b_better']}, and{pair['tied']}tie. This averages over available pools, including underfill. Restricting to the{matched['tasks']}tasks where both gates fill20, the ratio is{matched['geomean_a_over_b']:.4f}, interval[{matched['ci95_low']:.4f},{matched['ci95_high']:.4f}]. The both-filled subset is selected for candidate availability and is not the whole population.

### Radius selection and internal check

For each mode, radius selection minimized the640-task geometric MIC ratio at capped20available candidates over0.125,0.15,0.175,0.2,0.3. The remaining160form an internal development check. Both groups were previously used in representation/source training; this is **not an unseen-task benchmark**, and aggregate progress previews were inspected during development. The bootstrap intervals describe task variability within this source cohort, not post-selection generalization guarantees.

|Local edits|Selected length|640selection MIC ratio|160internal-check ratio|160tasks improved|160tasks filling20|
|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(choice_rows)+'''

### Author versus adapted decoder

|Length|Local edits|Adapted mean unique new /96|Author mean unique new /96|Adapted feasible draws|Author feasible draws|
|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(frozen)+f'''

This compares decoder coverage, not frozen-versus-online BO convergence. No author-pool MIC experiment was run. Training effects cannot be inferred from this table alone.

## Historical evidence across all800

The pinned source corpus contains{hist['distinct_novel']:,}distinct initially novel sequences (counted within tasks), of which{hist['distinct_novel_feasible']:,}({100*hist['distinct_novel_feasible']/hist['distinct_novel']:.2f}%)meet the reference constraint. Across{hist['total_improving_moves']}recorded incumbent improvements,{hist['improving_moves_over3']}({100*hist['improving_moves_over3']/hist['total_improving_moves']:.2f}%)were more than three edits from the preceding best peptide. Source global-edit-budget task counts are{hist['global_edit_budget_counts']}.

This argues for diversity, multiple centers and bounded expansion. It does not prove smaller-step routes cannot reach the same sequences. These moves follow stored source-row order and reflect the historical sampling policy. A cap around the original reference is different from a cap around an adaptive center; the original global reference constraint must never be relaxed.

## Proposed method changes

1. **Decode, exact-filter, deduplicate, then rank actual sequences.** Apply the cheap original global constraint and optional local2/3gate before costly MIC prediction. Record raw decoded counts, reasons for rejection, cached candidates, requested batch size, actual fill and physical MIC calls separately. Never rewrite previous4000proposal results or silently pad batches.
2. **Use canonical sequence features for acquisition.** Re-encode each accepted decoded peptide with the current VAE, then apply the matching FSBO feature transform/acquisition. The existing policy ranks raw proposed latent coordinates although objective observations are stored at re-encoded canonical coordinates. That mismatch is a plausible issue, not yet a proven cause. After accepted VAE updates, re-encode observations consistently before GP fitting, as the current code already does.
3. **Compare a conservative2-edit arm and a broader3-edit arm**, using the same length0.175in the main two-versus-three-edit ablation to isolate the edit cap. Separately test the source-selected per-mode radii in policy_plan.json. Use incumbent, reference and diverse promising already measured feasible centers; retain a separately logged global-feasible exploration stream. Explore an initial80/20local/exploration allocation only as a preregistered pilot setting, not as a measured optimum. Global-feasible exploration still obeys the original reference constraint.
4. **Replace unproductive recenter replay with new candidate generation.** The latest pep_999run replayed2194cached recenter proposals with zero new sequences. Stop repeated decode/rescore loops when they add no novel candidates; diversify centers or expand generation within a bounded allowance. Prefer a short explicitly recorded batch over unbounded resampling. The current fixed-pool test does not calibrate a full adaptive expansion schedule.
5. **Keep the existing loss rollback, add paired decoder-health diagnostics.** Loss on known initial sequences checks retention, not useful exploration. Measure local feasible yield, novel yield and displacement on fixed anchors before/after each VAE retrain, without using held-out target outcomes. Calibrate a rejection threshold on source-task online updates before turning it into an automatic veto. Do not claim the present loss gate would have rejected pep_999: both latest updates passed and no rollback occurred.

The isolated policy_prototype.py implements center selection, exact sequence gates, canonical-feature acquisition callbacks, deduplication and index-preserving ranking. check_policy.py passed the boundary/alignment checks in policy_checks.log. This prototype does not integrate the live controller, train a GP or launch optimization.

## Controlled next optimization experiment

Use the existing training cohort for policy development, then lock the policy before a held-out-task benchmark. The previously examined10peptides (including pep_999) are exploratory targets, not pristine test cases; report them separately from the remaining190within the200-task set, and audit other prior target exposure before describing any subset as untouched. Start with source-stratified matched-seed sequential arms: unchanged geometry; sequence-first canonical ranking with global-only gate; the same pipeline with local2; and with local3. Hold VAE/source-FSBO weights and online-update policy fixed initially to isolate search changes. Track identical initial pools, feasible best MIC versus actual new MIC calls, all decoded/rejected/cached counts, and wall-time components. Then separately ablate multiple centers/global-exploration and re-enable online updates with the loss gate. A full800×4000BOcampaign was not launched by this diagnostic. Sequential optimization is still required before claiming convergence or runtime improvements at that budget; the final held-out benchmark need not itself contain800tasks.

## Oracle validation and compute accounting

The new scorer matched the unmodified native predict_APEX function exactly on32sequences on the same device ({physical['parity_predictions']}sequence predictions across both implementations). Native source/model SHA256hashes are retained. Historical initialization predictions differed by at most{100*s['max_historical_initial_relative_error']:.4f}%with this hardware/precision/batching; this check alone does not identify the source of that variation. Freshly rescored100-sequence baselines avoid mixing those values with newly scored candidates. Historical scores are untouched.

Physical APEX sequence predictions: {physical['candidate_sequence_predictions']:,}candidate predictions +{physical['fresh_initial_predictions']:,}initialization predictions +{physical['parity_predictions']}parity predictions +{physical['failed_attempt_predictions']}predictions in the preserved first attempt that failed its strict historical-score tolerance check =**{physical['total_apex_sequence_predictions']:,}**. Each sequence prediction uses all eight ensemble members. Scores are shared across overlapping radius/local-gate analyses within each task; summing table rows would overcount actual compute. Oracle calls here are predictions, not laboratory experiments.

Artifacts: radius_replay.json; trust_region_800.png/pdf/svg; frozen_and_history.png/pdf/svg; summary.json; decoder_summary.csv; mic_summary.csv; paired_2_vs_3.csv; per-task CSVs; strata_summary.csv; policy_plan.json. Shared raw draws, scored outputs, ledger and scripts are in{ROOT}.
'''
# Keep prose units and counts readable; code identifiers and artifact paths stay literal.
import re
text=re.sub(r'(?<=\d)(?=(?:tasks|task|rounds|round|proposals|proposal|sequences|sequence|predictions|prediction|draws|draw|decodes|Sobol|coordinates|initialization|initial|available|accepted|candidate|edits|local|global|observed|source|model|GP|BO|sequence|parity|total|fresh|recorded|incumbent|with|for|to|run|were|are|form|mean|reconstruction|feasible|ensemble|peptide|gate|were|and|or|but|with|in|of)\b)', ' ',text)
for a,b in [('all800','all 800'),('All800','All 800'),('length0','length 0'),('length1','length 1'),('at0','at 0'),('at1','at 1'),('at round','at round'),('batch50','batch 50'),('seed1','seed 1'),('by at most','by at most'),('floor(reference length/4)','floor(reference length / 4)'),('initial100','initial 100'),('initialization100','initialization 100'),('local1','local 1'),('local2','local 2'),('local3','local 3'),('local2/3gate','local 2/3 gate'),('all eight','all eight'),('predict_APEX','predict_APEX'),('SHA256hashes','SHA256 hashes'),('were more than','were more than'),('original starting length0.8','original starting length 0.8'),('both gates fill20','both gates fill 20'),('filling20','filling 20'),('from0','from 0'),('over0','over 0'),('three edits on','three edits on'),('maximum1.6','maximum 1.6'),('minimum0.0078125','minimum 0.0078125'),('reference≤','reference ≤'),('counted within tasks','counted within tasks')]:text=text.replace(a,b)
lines=[]
for line in text.splitlines():
 if not line.startswith('Artifacts:'):
  line=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', line)
  line=re.sub(r'(?<=\d)(?=[A-Za-zµ])', ' ', line)
  line=re.sub(r'(?<=[%)])(?=[A-Za-z])', ' ', line)
  line=line.replace('SHA 256','SHA256').replace('BOcampaign','BO campaign')
 lines.append(line)
text='\n'.join(lines)+'\n'
(OUT/'REPORT.md').write_text(text)
plan=dict(status='proposed_not_deployed',cohort='source800 development; heldout200 excluded',production_policy_changed=False,local_modes={str(k):dict(local_edits=k,selected_start_length=choices[k]['selected_radius'],controlled_edit_cap_ablation_length=choices[3]['selected_radius']) for k in [2,3]},reference_rule='Levenshtein<=floor(reference_length/4), plus original sequence rules',center_policy=dict(maximum=4,include=['best measured feasible','reference','diverse among top32 measured feasible']),candidate_pipeline=['decode bounded pool','exact reference+syntax check','optional local edit check','deduplicate against observed and current pool','encode actual accepted sequences with current VAE','FSBO canonical feature transform','objective acquisition ranking','score selected candidates with MIC oracle','account actual calls and short batches'],future_multicenter_allocation=dict(local=.8,global_feasible_exploration=.2,status='pilot starting choice, not calibrated optimum'),training='hold VAE/source-FSBO frozen first to isolate search; separate later online ablation',required_next_validation='matched-seed sequential BO on source development, then locked target evaluation with the10already examined targets reported separately; audit remaining190for prior policy-development exposure; not completed here')
(OUT/'policy_plan.json').write_text(json.dumps(plan,indent=2))
# Keep reviewable code with the human report; raw datasets remain shared.
for name in ['policy_prototype.py','check_policy.py','probe800.py','score_candidates.py','audit_sources.py','aggregate.py','write_report.py','validate_raw.py','replay_radius.py']:
 p=ROOT/name
 if p.exists():shutil.copyfile(p,OUT/name)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.log'] and p.name!='artifact_hashes.json'}
(OUT/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2))
print('Wrote',OUT/'REPORT.md')
