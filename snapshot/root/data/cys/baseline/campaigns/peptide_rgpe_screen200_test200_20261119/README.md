Frozen-author-VAE peptide RGPE screening experiment, 19 November 2026.

The finite queue covers exactly the existing 200 held-out references, disjoint
from the 800 immutable source GPs. It uses the same revalidated 100-point
initialization and the original author VAE checkpoint. No source GP, source
representation, FSBO model, or VAE is retrained. The current task GP is refitted
and RGPE weights are recomputed each acquisition round, as in RGPE.

Each acquisition generates 1,500 native sparse local latent candidates. Expected
improvement from the fitted target GP shortlists 200, minimizing MIC after
conversion to the GP's standardized output units. Stable original-index ties
make screening deterministic. The original constrained joint Thompson sampler
then chooses 50 without replacement from the full weighted independent-GP
posterior over those 200 points. The full covariance and squared source weights
are preserved. Exact decoded-sequence similarity checks and cached duplicate
accounting remain in the existing audited task state machine.

Each task stops at 4,000 acquisition proposals or 4,000 total fresh logical APEX
evaluations, whichever comes first. Initialization is excluded. Recenter oracle
calls count toward the fresh cap; recenter proposals have their own counter.
Thus acquisition is capped at 80 batches of 50; duplicates can leave fewer than
4,000 fresh calls. Native frozen-VAE recentering remains enabled.

Worker admission targets 20 CPU task workers on LLMOptimizer2, 16 on
LLMOptimizer3, and 12 on LLMOptimizer4 (48 total). Workers use one BLAS/PyTorch
thread. Each host reserves 16 GiB memory headroom and at least 4 GiB growth
capacity per task. Admission pauses above 90% measured CPU quota, when memory
growth cannot be reserved, when inference services are unavailable, or when
shared storage has less than 60 GiB free. Fixed disjoint host partitions cover
88, 64, and 48 tasks respectively. One interrupted worker retry reuses durable
pending state. Failures remain visible in host status JSON rather than becoming
completed results.

Four existing validated FP64 CUDA math services use A800 GPUs 0–3 on
LLMOptimizer2 (ports 22440–22443), bounded to 25% allocator capacity each with
12 GiB source caches. Eight frozen VAE/APEX services use A16 GPUs
1,2,4,6,8,10,12,13 on LLMOptimizer3 (ports 22460–22467), bounded to 20% each.
They reject checkpoint weights that differ from the author checkpoint. All
existing Query and BORA services remain allocated. Six owned idle TaskSet LLM
services were retired after verifying all 200 tasks / 60 workers finished and
that their ports had no established clients; identities are saved in evidence.
The extra CPU host on SSH port 38783 rejected the available keys, so it is not
included in these counts.

The queue is started on all three hosts but waits for gate.py. A real canary
uses all 800 sources, commits one 50-proposal event, and closes normally. The
gate checks the bank, frozen representation, 200-point full covariance,
nonzero oracle progress, pending-state commit, split, and focused numerical
validations. Production reopens that same checkpoint and continues. Timing
telemetry lives outside runtime directories so final archival can use the
unchanged checksum/identity/accounting verification and cleanup.

Validation includes an independent EI calculation, output direction and units,
zero variance, ties, RNG preservation and shortlist mapping. Three real source
GPs plus a target fixture with four nonzero weights validate CPU/GPU agreement
for means, full 200-point covariance and 50 correlated draws. This is a component
check; the full-bank canary is the production admission check.

After the canary, benchmark.py compares full 1,500-point acquisition with
screened 200-point acquisition using the same fitted target snapshot, 800-bank
weights, candidate pool and sampling seed. Two alternating-order repeats report
screening, merge compute, queue, and total acquisition time. No oracle calls are
made by the benchmark. It is an initial-checkpoint acquisition comparison, not
an end-to-end or mature-GP speed estimate. Target fitting, source ranking/LOOCV,
VAE/oracle time and contention remain separate costs. Historical full RGPE used
5,000 candidates and 5,000 budgets; retrospective partial historical results
must be compared only at matched measured support and labeled as a different
recipe. There is no completed 200-task historical RGPE baseline here.

Run `python status.py` for read-only progress. Host dispatchers, gate, canary,
inference services and SSH forwards are detached owned processes; process JSON
files record their PIDs, command lines and process start ticks. `pins.json`
records code and recipe identities. Live and per-fit timing files are under
telemetry. No measured speedup is claimed before matched_speed.json completes.
