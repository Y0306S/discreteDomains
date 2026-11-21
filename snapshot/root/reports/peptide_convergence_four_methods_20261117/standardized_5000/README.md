# Peptide convergence: standardized 5,000-proposal view

SemanticOPT is held at its proposal-1,000 result through 5,000; carried forward values are not new measurements.

Original SemanticOPT run: 136,255/200,000 random fallback proposals.
Baselines retain measured incumbents after their proposal or fresh-call limit; recentering costs extra oracle calls.

Both domains use the exact shared STYLE entries and interpolate() PCHIP helper.

Checkpoints every 50 proposals through 5,000 use the same PCHIP smoothing. Interpolation preserves these checkpoint values and monotonicity, and never overshoots adjacent endpoints. Raw per-proposal values remain in the CSV; the smoothed line between checkpoints is a visual interpolation, not new measurements.

Per-task min–max bounds retain the full-source normalization audit: pool feasible/uncensored full measurements, drop slowest floor(20%) for bounds only. Bounds are retrospective.

Peptide x now counts acquisition proposals, including cached proposals; extra recentering oracle calls do not increment this axis. This differs from the earlier fresh-evaluation plots.

Both normalized convergence figures display the requested 0.8–1.0 score-axis range; CSV scores retain the full [0,1] range. Source data are read-only.

Files: normalized convergence, relative objective, summed score (PNG/PDF/SVG), CSV, per-task arrays and audit.
