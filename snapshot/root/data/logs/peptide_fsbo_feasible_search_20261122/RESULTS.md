# Same-ten peptide FSBO rerun

Completed: 10/10 tasks. Same initial100, seed1, batch50,
4000 acquisition proposals and 4000 shared fresh-evaluation cap. Lower MIC is better.
Positive reduction means the new search has a lower MIC at the same proposal count.

| Task | Status | Proposals | New MIC | Previous MIC at same budget | MIC reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| pep_117 | complete | 4000 | 2.0311 | 2.5742 | +0.5431 |
| pep_219 | complete | 4000 | 2.1105 | 3.7835 | +1.6730 |
| pep_221 | complete | 4000 | 4.9715 | 16.6053 | +11.6337 |
| pep_575 | complete | 4000 | 3.9817 | 17.5888 | +13.6071 |
| pep_673 | complete | 4000 | 11.2652 | 18.8727 | +7.6076 |
| pep_921 | complete | 4000 | 9.0620 | 13.4354 | +4.3733 |
| pep_925 | complete | 4000 | 11.0927 | 13.3715 | +2.2789 |
| pep_967 | complete | 4000 | 37.4630 | 81.3230 | +43.8601 |
| pep_980 | complete | 4000 | 1.6270 | 2.9779 | +1.3509 |
| pep_999 | complete | 4000 | 20.9871 | 101.0068 | +80.0197 |

Baseline: /root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119

`summary.json` and `task_results.csv` include fresh acquisition feasibility,
runtime, and retraining counts. Completed-task averages cover only the tasks
that have finished. The controller refreshes this report while workers run.
Runtime comparisons use shared A16 GPUs with different concurrent workloads.
This is one seed on ten previously examined tasks.

![Matched-budget comparison](comparison.png)
