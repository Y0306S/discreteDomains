# APEXGo VAE recipe + FSBO: ten-task peptide pilot

This run replaces the paused sequential GP-guided source800 VAE replay. Original baselines and rejected experiments remain untouched.

Official source: https://github.com/Yimeng-Zeng/APEXGo . The paper is *A generative artificial intelligence approach for antibiotic optimization*: https://doi.org/10.1101/2024.11.27.625757 . `reference/provenance.json` pins the upstream tree and every copied source file. Our original frozen checkpoint is byte-for-byte identical to the author's epoch118 checkpoint.

The executed author training path is `generation/train_vae.py` with `generation/train.sh`: Adam lr0.0002; batch512; bf16 autocast and GradScaler; reconstruction cross-entropy +0.0001*KL; k1 amino-acid tokens; d_model128, bottleneck2 (latent256), six encoder and decoder layers, FF256, dropout0.05. It does not call the schedulers in the separate unused VAEModule.

## Offline continued training

Start from the released epoch118 weights, not random initialization. Use all4,053,113 historical measured observations from the800 source tasks, deduplicate to3,853,427 sequences, then seeded shuffle and split90/5/5. Train3,468,085; validation192,671; test192,671. The ten heldout optimization tasks supply no training or checkpoint-selection data. Source labels do not enter the VAE loss.

Run four continuation epochs. Validate after epochs1 and4, matching the released epoch%3 validation cadence. Include the initial author checkpoint as a selection candidate. Select minimum source validation loss subject to retained decoder quality. Log whether a trained checkpoint beats initialization. The original APEXGo model was trained118 epochs on its own UniRef-derived data; this is a smaller continuation experiment on our source800 corpus, not a fresh reproduction of that pretraining dataset or budget.

Source SQLite produced intermittent malformed-image errors on shared storage despite repeat successful reads. Training instead reads independently hashed immutable NPZ payload files under `source_snapshot/payloads`; every task retains its original corpus payload SHA256. Both original and copied SQLite files are preserved for audit.

## FSBO and online training

Re-encode every source observation using the selected VAE. Train a new FSBO model FROM SCRATCH for100,000 episodes on CPU using all800 source tasks. Keep tasks separate. Reserve validation observations within160 source tasks; their remaining observations can train. Model selection uses source validation predictive NLL. This all800 population differs from the original640/160 task-level FSBO split, so final performance differences evaluate the entire revised method and do not isolate VAE effects.

For each target, use the same initialization100, batch50, 5,000 acquisition proposal/fresh-call caps, and local acquisition/recentering policy as the archived baseline. Every10 stagnant rounds triggers two VAE epochs over the deduplicated currently observed recent50 + feasible top1000 sequences, using the SAME reconstruction/KL, Adam0.0002, batch512, bf16/GradScaler process. No objective-GP or constraint-GP loss backpropagates into the VAE. Compare a fixed source-only128-sequence probe before/after. Roll back if feasible retention<80% or reconstruction loss>125% of the pre-update value and the fixed original-author baseline (both must pass). After an accepted update re-encode all800 source tasks plus this task's observed prefix; train FSBO from scratch100,000 episodes, refresh coordinate scaling, re-encode the target, and refit its constraint GP. No other test-task observations enter these updates.

## Evaluation and operation

Same prespecified ten tasks, seed1, four GPU evaluator processes across GPUs0–3. The historical completed frozen VAE+FSBO baseline is reused. Four real source tasks plus the current target exercise the full update/retrain/resume path before production. Interrupted saved oracle responses are reused.

`pipeline.py` runs VAE training → decoder retention gate → full embeddings → initial FSBO → heldout evaluation → paired comparison. Checkpoints include optimizer/scaler/RNG states. An independently scheduled ten-minute read-only monitor writes `monitor/latest.json` and history. It cannot send future chat messages. On completion it writes `monitor/completion.json`, `RESULTS.md`, `paired_results.csv`, `summary.json`, `convergence.csv`, and `comparison.png/pdf/svg`.

Do not call this improved until the actual heldout results support it. Single-step reconstruction checks are implementation checks, not final optimization evidence.
