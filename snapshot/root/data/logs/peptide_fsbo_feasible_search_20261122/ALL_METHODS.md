# New peptide FSBO versus saved methods

Same ten peptides, shared initialization100, and4000proposal/fresh ceilings. MIC in µM; lower is better.

| Method | Mean final MIC | New FSBO wins/losses |
| --- | ---: | ---: |
| New feasible-search FSBO | 10.4592 | — |
| Original frozen FSBO | 28.4054 | 10/0 |
| Online VAE + FSBO | 28.9512 | 10/0 |
| SemanticOPT with fallback | 14.2838 | 8/2 |
| Online VAE + FSBO + rollback (20 rounds, max2) | 27.1539 | 10/0 |

| Task | New FSBO | SemanticOPT with fallback | Winner |
| --- | ---: | ---: | --- |
| pep_117 | 2.0311 | 2.6607 | New FSBO |
| pep_219 | 2.1105 | 6.2636 | New FSBO |
| pep_221 | 4.9715 | 4.0452 | SemanticOPT |
| pep_575 | 3.9817 | 15.1235 | New FSBO |
| pep_673 | 11.2652 | 21.6679 | New FSBO |
| pep_921 | 9.0620 | 11.0047 | New FSBO |
| pep_925 | 11.0927 | 16.0202 | New FSBO |
| pep_967 | 37.4630 | 33.9040 | SemanticOPT |
| pep_980 | 1.6270 | 2.5731 | New FSBO |
| pep_999 | 20.9871 | 29.5751 | New FSBO |

Same ten tasks, shared initial100 and4000proposal/fresh ceilings; actual fresh counts differ

FSBO seed1; SemanticOPT seed42; one trajectory per task/method

SemanticOPT includes random fallback: prior audit reports78.3% of scored proposals

MIC is APEX predicted micromolar; lower is better

Previously examined ten-task cohort; no claim of generalization to remaining tasks

Sources:

- /root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/capped_convergence.csv
- /root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/semanticopt_comparison.csv
- /root/data/logs/peptide_fsbo_feasible_search_20261122/summary.json
