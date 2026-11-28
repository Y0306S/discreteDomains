# SemanticOPT improvement after 400 evaluations

Only pilot queries with a completed, saved 800-proposal trajectory are included. Queries completed at their 400-evaluation target cannot answer whether there is improvement after 400 and are excluded. Running pilots are also excluded, so the cohort is fixed over the entire plot.

The horizontal axis counts proposal evaluations, excluding 49 saved Bao initialization measurements; 400 proposals are 40 batches of ten, and 800 are 80 batches. Evaluation zero is the best uncensored successful Bao initialization result. The incumbent can improve only on uncensored successful measurements. Timeouts remain charged evaluations. No smoothing, extrapolation, padding, or statistical uncertainty bands are used.

The primary chart normalizes each query's best runtime by its own value at evaluation 400; a value below one shows improvement beyond that budget. Its black line shows the sum of the included queries' incumbent runtimes, divided by that sum at evaluation 400. This normalization is specific to the plateau question and differs from the earlier multi-method min–max score and normalized summed-runtime figures. The second chart exports the individual trajectories in seconds.

This snapshot contains 9 completed 800-proposal pilots. 4 improved between 400 and 800 evaluations. Their summed best runtime fell from 23.843045 to 22.307855 seconds, a 6.44% reduction. The trajectories are descriptive evidence for this cohort, not an estimate of how all ten pilots or 200 held-out queries behave. The completed cohort may favor faster/easier queries.

Outputs: semanticopt_completed800_plateau.png/.pdf/.svg, semanticopt_completed800_runtime_seconds.png/.pdf/.svg, convergence.csv, summary.json, observations_snapshot.json.

Reproduce: python3 build_plot.py. All campaign inputs are read-only.
