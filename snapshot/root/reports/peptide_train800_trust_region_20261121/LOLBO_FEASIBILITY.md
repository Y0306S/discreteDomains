# LOLBO feasibility on the same ten peptides

Fresh acquisition and recenter predictions are counted separately. Initialization and cache hits are excluded. The prefix includes at most the first 80 acquisition rounds (4000 proposals); tasks can terminate earlier because acquisition and recenter share a 5000 fresh-call cap. These rates do not constitute an equal-cost optimization-quality comparison.

|Peptide|Acquisition proposals represented|Acquisition feasible/fresh|Recenter feasible/fresh|Combined feasible rate|
|---|---:|---:|---:|---:|
|pep_117|3418|0/3052 (0.00%)|325/1948 (16.68%)|6.50%|
|pep_219|2597|0/2547 (0.00%)|823/2453 (33.55%)|16.46%|
|pep_221|4000|20/3212 (0.62%)|340/1740 (19.54%)|7.27%|
|pep_575|3661|7/3002 (0.23%)|770/1998 (38.54%)|15.54%|
|pep_673|3000|0/2845 (0.00%)|518/2155 (24.04%)|10.36%|
|pep_921|4000|53/3180 (1.67%)|470/1754 (26.80%)|10.60%|
|pep_925|4000|25/2985 (0.84%)|539/1272 (42.37%)|13.25%|
|pep_967|4000|84/2811 (2.99%)|489/1738 (28.14%)|12.60%|
|pep_980|3000|0/2621 (0.00%)|279/2379 (11.73%)|5.58%|
|pep_999|3371|8/2988 (0.27%)|891/2012 (44.28%)|17.98%|

Aggregated counts and complete-run results: lolbo_feasibility_audit.json. All recorded sequence constraints were recomputed exactly, initialization equivalence checked, and history SHA 256 verified against manifests and the prior 200-task archive audit. No new inference, oracle predictions or training were performed.

## Completed-run totals

All ten completed LOLBO runs each used 5,000 fresh predictions, including recentering. Acquisition: 221/30372 (0.728%); recentering: 5529/19628 (28.169%); combined: 5750/50000 (11.500%).

For pep_999, acquisition gave 8/2,988 feasible (0.268%), recentering 891/2,012 (44.284%), combined 899/5,000 (17.980%). The run stopped at 3,371 acquisition proposals because the shared fresh-call budget was exhausted. Best feasible MIC stayed at its initialization value of 112.6502 µM despite many new feasible recenter candidates. These phase rates establish feasibility/novelty, not useful objective improvement.
