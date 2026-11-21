# Peptide ORPO preference preparation

This is **our discrete peptide method**, initialized from the clean Qwen3.5-9B base, with no SFT/SemanticOPT adapter parent. Training has not been launched. Query training and the subsequent four-phase RL cycle are separate follow-up work; neither is implemented by this dataset builder.

## RL and evaluation prompt policy, updated November 11

`rl_prompt_policy.json` records the user's peptide/query-specific policy. All RL phases omit inferred Good/Bad Region summaries and textual latent-region bounds. Concrete measured good/bad examples from the current task and Recent50 remain valid observation history; they are not region summaries. The audited ORPO dataset is unchanged.

During RS and exploitation **after round 50 (round 51 onward)**, similar-task performance evidence is restricted to measured top/bottom discrete candidates. Each source card must also describe its actual task, objective and constraints, and show explicit source-versus-current differences. Peptide cards show both reference sequences; query cards show the actual query text/descriptions and configuration spaces. Source scores are not target scores. The prompt instructs the model to analyze differences and adapt the evidence, validate current-task constraints and novelty, and then return only the single-candidate JSON. There is no extra visible rationale field and no claim that a prompt can prove the model performed the requested internal analysis.

`code/rl_prompt_policy.py` implements this restricted card schema for both domains and a peptide RL prompt builder. It rejects extra region fields, missing metadata, nonfinite measurements, self-retrieval and tasks outside the caller's allowed training set; it requires real source evidence for RS and post-50 exploitation. Source ranking and measurement verification remain the caller's responsibility. The query rollout adapter and full RL pipeline are not yet connected to this helper. Integration must use the correct training-only split and rerun full token checks including retrieval cards before RL starts. No change to the phase schedule is implied.

This prompt restriction does not remove FSBO's internal trust region or change its measured success/failure updates. The LLM continues to recommend a discrete candidate rather than issuing region-control commands.

## Prompt and action contract

Each call requests one amino-acid sequence, not a 3D protein structure, latent vector, or batch. The answer is exactly `{"sequence":"..."}`. The prompt contains the task reference, measured current best MIC, five good and five bad feasible observations, and the most recent 50 distinct valid measured peptides in chronological order. Each hard rule has an illustrative example.

Hard constraints: uppercase `ACDEFGHIKLMNPQRSTVWY`; length 1–30; at most one C; `1 - Levenshtein(candidate, reference)/len(reference) >= 0.75`; no previously proposed/evaluated sequence. Lower APEX bacteria_0 MIC is better. Format examples are explicitly hypothetical and do not waive task constraints.

Recent50 counts first measured valid sequences. Cached repeats, similarity failures, malformed sequences, and failed decodes do not count. Imported initialization measurements are admitted and explicitly distinguished from VAE-decoded proposals. `parse_action` validates the entire seen set; retry proposals belong in a separate rejection list, not Recent50. Good/bad can overlap Recent50 because they summarize the same causal prefix.

## Preference rule and causality

All 800 audited training tasks are scanned, with all 200 test IDs excluded. Source: `peptide_bo_train800_20261027_v5b`, a **LOLBO** archive, not an FSBO rollout. Source history hashes are checked against the pre-existing training audit. Acquisition/recenter rows were scored after VAE decode in the archived implementation. There is no per-row decode-success field and no claim of roundtrip reconstruction; sequence validity, constraints, finite measured scores, and source-code provenance are the available evidence.

Each prompt is frozen before a contiguous `(phase, round)` block. This conservatively treats all recenter sub-batches sharing that phase/round as one block. No same-block observations appear in its prompt. The best newly seen feasible improving sequence in a block is selected as chosen. It must strictly improve the best feasible MIC in the full prefix. The rejected peptide is unseen in that prefix, feasible, measured, and has MIC greater than or equal to the prefix best. Both targets must also be literally absent from the complete prompt, including reference and example text.

Negative labels may use later offline measurements from the same training task. Their sequences/results never enter the earlier causal prompt. Among eligible negatives, use the ML Pareto rule: maximize reference-independent peptide edit similarity to chosen and normalized worse MIC gap, then select the maximum 0.5/0.5 score. Peptide edit similarity uses maximum pair length; the hard reference similarity constraint separately uses reference length.

There are 1,263 pairs from 651 tasks. Of the 149 exclusions, 146 had no improvement. Two improved tasks (`pep_141`, `pep_783`) have improving targets appearing as substrings in the prompt, so the strict literal leakage check excludes them. `pep_411` has no eligible unseen nonimproving negative after its causal prefix. No synthetic positives or fabricated MIC labels are added. `data/tasks` also records causal context indices for all scanned tasks, but these inventories are not runnable RL rollouts.

## Validation and training preparation

`reports/independent_audit.json` validates the original full histories, chronological Recent50, source hashes, full-prefix novelty, baseline labels, selected negatives, and exact assembled training file. `reports/sample_prompt.txt` shows a real prompt. Five targeted unit tests cover the rules, filtering, duplicate checks, batch leakage, and Pareto selection.

The ORPO config copies the ML LLaMA-Factory recipe: clean base, ORPO beta 0.3, LoRA rank32/alpha64/dropout0.05/all-linear, one epoch, LR1e-5, accumulation8, BF16, SDPA flash enabled, gradient checkpointing, seed42. `prepare.sh` verifies exact answer/EOS tokens and assistant-only labels through actual LLaMA-Factory preprocessing, selecting a cutoff that preserves all constraints and Recent50. `train_orpo.sh` is a single-GPU entry requiring an explicit GPU and at least54 GiB free, with a48 GiB allocator cap; it is not an automatic queue and does not implement RL.

The completed token preflight passed for all 1,263 pairs: 5,153–6,693 tokens, cutoff 7,168, zero truncated examples, exact answer/EOS tokens and assistant-only labels verified. The one-epoch recipe expects 158 optimizer steps. Its CPU-only preparation entry hides GPUs because the initial check detected dependency-driven CUDA initialization; the successful rerun passed the CPU guard.

The shared `/root/data` filesystem had zero free inodes during preparation despite free bytes. Outputs were written under `/root` without deleting other experiments. This directory is a local preparation copy; shared storage needs inode capacity before a durable shared copy or new shared-output jobs can be created.

## Concurrent training setup proposal, November 11

`training_setup_proposal.json` and `reports/peptide_task_selection_audit.json` contain the offline setup proposal; **no new queue or training process was launched**. `assets/rl_task_inventory.json` lists all 800 training references, length/sequence descriptors, three nearest training-only references, actual causal-context indices, and a proposed full-coverage ordering interleaving length groups. This ordering covers each task once; it does not impose equal sampling weights or change the ORPO dataset.

All tasks share the APEX bacteria_0 MIC objective. The archive has no biological subtype labels. Reference lengths are 10–25: 391 tasks at 10–15 residues, 239 at 16–20, and 170 at 21–25. This is reference length, not a new candidate-length constraint; candidates retain the existing 1–30 and reference-similarity rules. Charge/composition descriptors are sequence-derived proxies. Nearest-reference ranking is a retrieval candidate list, not a calibrated transfer predictor: 131 references have no other training reference with edit similarity >= 0.5, so actual retrieval integration must handle weak evidence explicitly.

All 800 tasks have early causal contexts, including tasks without a recorded BO improvement. Only 754 have archived pre-acquisition contexts for proposal round >= 51; the other 46 must not be represented as late-stage examples by relabeling earlier rounds. Proposal round r uses the prefix before that acquisition, after r-1 completed acquisition rounds. Exact ML prefix counts and peptide proposal round numbers must not be interchanged.

The proposed resource plan waits for the current ML evaluation to release its twelve replicas and for SemanticOPT's four verified replicas (one per A800) to load. Use one A800 for ORPO and then RL, sharing it with one capped21 GiB SemanticOPT replica. Admission requires at least54 GiB actually free; the ORPO allocator cap is48 GiB. The existing ML GRPO code computes fixed-reference probabilities by swapping adapters on a single base model, so a second GPU-resident reference base is unnecessary. These are proposed limits, not a successful peptide backward-memory measurement. The peptide collector/reward adapter, retrieval-prompt token check, live oracle/backward checks, and checkpoint/resume validation remain launch gates.

The November 11 storage check showed shared inodes had become available again (roughly182,000 free at the check). Durable training checkpoints should use shared storage after verifying its byte/inode headroom, rather than depending on the local root filesystem. The existing ORPO configuration and its successful preflight were left intact during discussion; moving its output/config requires a new config hash/token gate before launch.
