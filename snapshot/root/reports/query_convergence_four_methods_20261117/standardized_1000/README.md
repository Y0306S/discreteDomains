# Query convergence: standardized 1,000-proposal view

SemanticOPT is held at its proposal-200 result through 1,000; carried forward values are not new measurements.

Original SemanticOPT run: 32,690/40,000 random fallback proposals. SQL caps, seeds, hosts and run dates differ.

Both domains use the exact shared STYLE entries and interpolate() PCHIP helper.

Checkpoints every 50 proposals through 1,000 use the same PCHIP smoothing. Interpolation preserves these checkpoint values and monotonicity, and never overshoots adjacent endpoints. Raw per-proposal values remain in the CSV; the smoothed line between checkpoints is a visual interpolation, not new measurements.

Per-task min–max bounds retain the full-source normalization audit: pool feasible/uncensored full measurements, drop slowest floor(20%) for bounds only. Bounds are retrospective.

Query x counts charged acquisition proposals, including cache hits.

Both normalized convergence figures display the requested 0.8–1.0 score-axis range; CSV scores retain the full [0,1] range. Source data are read-only.

Files: normalized convergence, relative objective, summed score (PNG/PDF/SVG), CSV, per-task arrays and audit.
