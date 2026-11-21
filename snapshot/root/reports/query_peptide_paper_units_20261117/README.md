# Convergence in paper objective units

Metric reference: BOLT paper Figure 2, PDF page 8: summed query runtime and average predicted MIC. Units: seconds and micromol/L (paper pages 3/6). Figure 1 on PDF page 7 instead uses normalized summed objectives; these additional figures intentionally show raw objective units.

Query curve = sum across 200 tasks of best uncensored query-plan runtime in seconds; it is not cumulative optimization wall time.

Peptide curve = arithmetic mean across 200 tasks of best feasible APEX bacteria_0 predicted MIC, in micromol/L (equivalently micromolar, µM). These are predictions, not laboratory MIC measurements.

APEX_predict.py explicitly applies 10 ** (6 - model_prediction) before averaging its eight models; saved objective values already have MIC units. No extra logarithm or back-transform is applied.

Peptide summed-MIC alternative is also exported, following the aggregate used by Figure 1 / Table 11 without normalization.

X preserves the requested new-acquisition-proposal accounting: query 0–1000, peptide 0–5000. This differs from the paper Figure 1 oracle-call x-axis when cache hits/recentering occur. Initialization: query49, peptide100, excluded from x.

SemanticOPT measured endpoints are held flat after query200 / peptide1000. Other completed runs retain final incumbents after exhausting their proposal or fresh-oracle budget. No new observations are fabricated.

Shared palette and PCHIP helper are reused exactly, with checkpoints every 50 proposals. Raw integer-budget values and aggregate sums/means are exported; smoothing is display interpolation.

No per-task min–max normalization or geometric mean is used in these new physical-unit figures. Aggregation differs from the previous normalized-score plots and can change method ordering.

All200 tasks are included with identical starting pools. FSBO is the frozen-VAE local/recenter run; TPE is the completed global variant. Historical campaign device/date/seed/timeout differences remain.

SemanticOPT includes constrained random fallback: query32690/40000; peptide136255/200000. Refer to source audits for full method provenance.

No uncertainty bands added: one optimizer seed per method/task. The paper Figure 1 includes standard-error bands, but this request changes objective units only.

Outputs: query_physical_units, peptide_physical_units, query_peptide_physical_units, peptide_summed_mic (PNG/PDF/SVG); convergence.csv; endpoints.csv; audit.json.
Reproduce: `python3 build_plots.py`. Existing normalized-score figures are retained.
