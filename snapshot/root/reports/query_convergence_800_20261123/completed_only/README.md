# Completed query methods:800evaluations

Only LLM + FSBO, FSBO, LOLBO and TPE are included. All200queries in each method have a saved trajectory through800total evaluation slots. SemanticOPT and ongoingBOLTcontinuation are excluded.

The x-axis includes initialization:30BOLTmeasurements for the hybrid,49Baomeasurements for standalone baselines. Curves start after initialization and end at800. Baseline cache hits remain charged slots, so this is not the paper's exact fresh-oracle-call axis.

Mean normalized score: for each query pool uncensored runtimes from these four methods' first800total slots, discard the slowest floor(20%) only to choose the upper bound h, set l to the pooled minimum, then average clip((h-best_runtime)/(h-l),0,1) equally across200queries. The bounds are retrospective and fixed over each curve. Display score axis is0.8–1.0; exported values preserve[0,1].

BOLT paper Scaling Multi-Task Bayesian Optimization, Figure1, PDFpage7: curves sum best query runtimes across held-out test tasks and label that summed objective normalized. The caption/text does not specify the normalization denominator. We follow the summed aggregation and explicitly choose a single shared reference, sum of best runtimes from the49Bao starts. We do not claim that this unspecified denominator matches the paper's original plotting code.

Normalized summed runtime R(n)=sum_q best_runtime_q(n) / 617.204758197 seconds. Sum first, then divide by the same constant for every method. This is neither a geometric mean nor a mean per-query runtime ratio, and is not1-minus-runtime.

Uncensored successful runtimes alone improve incumbents; timeouts remain evaluation slots. Each curve uses all200queries at every displayed point. No extrapolation, smoothing, ongoing tasks or seed uncertainty bands.

Raw histories were verified against the prior saved curve snapshot; all600baseline source file hashes were verified. Bounds use only the four displayed methods and the requested800-slot horizon, so scores differ from the previous four-method report that includedSemanticOPT.

Initialization pools, optimization seeds, hosts, dates, timeout policies and storage differ across method families. These are descriptive historical comparisons. Production files and running tasks were not modified.

Outputs:query_completed_800(two panels),query_normalized_score_800,query_normalized_summed_runtime_800, each PNG/PDF/SVG; convergence.csv;normalization_bounds.csv;audit.json.

Reproduce:python3 build_plot.py.
