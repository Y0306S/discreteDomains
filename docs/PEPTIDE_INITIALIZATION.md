# Measured peptide initialization: 100 feasible samples per task

The repository includes the existing random-mutation initialization pools,
unchanged from the experiment files. These are saved starting samples with APEX
predicted MIC outcomes, not newly generated candidates or wet-lab measurements.
No samples were selected by MIC rank. The budget is 100 samples per task.

| Pool | Tasks | Entries | File |
| --- | ---: | ---: | --- |
| Original pool, training and held-out | 1,000 | 100,000 | [initialization.json](../snapshot/root/data/logs/peptide_bo_train800_20261027_v5b/initialization.json) |
| Revalidated held-out pool | 200 | 20,000 | [test200_revalidated_initialization.json](../snapshot/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json) |

The second file contains the same held-out sequences, with revalidated scores,
original scores and revalidation provenance. It does not add another 100 samples
per task. Use the original file for the 800 training tasks and the revalidated
file for the 200 held-out tasks. The exact task split and file hashes are in
[peptide_initialization_manifest.json](peptide_initialization_manifest.json).

Each task maps to `reference`, `sequences`, `scores` and `sources`. Arrays are
aligned by index. Scores are **negative predicted MIC**: `mic = -score`, so a
higher score is better. The revalidated file additionally preserves
`original_scores` and `score_revalidation`. The original source/provenance fields
are retained without rewriting the measured values.

All 100 candidates within each task are distinct and differ from its reference.
Each candidate uses the 20 standard amino-acid letters, has length 1–30,
contains at most one cysteine, and satisfies
`Levenshtein(candidate, reference) <= floor(len(reference) / 4)`
(similarity at least 0.75). A reference is task context and need not itself pass
candidate constraints; for example, `pep_235` has two cysteines, but all its
initialization candidates pass.

Verify hashes, split membership, all candidate constraints and original versus
revalidated sequence alignment without loading models or making oracle calls:

```bash
python3 migration/verify_peptide_initialization.py
python3 migration/verify_snapshot.py
```

The snapshot paths restore to the original absolute locations with
`migration/restore_layout.py`. This addition supplies the initialization files;
model weights, source histories and other large runtime artifacts still follow
the migration guide. Historical experiments and running campaigns are unchanged.
