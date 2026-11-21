# Query VAE pool and CPU capacity assessment

Ten **query** VAE inference services are running on LLMOptimizer2. Physical
`nvidia-smi` GPUs 0, 1 and 2 have two each; GPU 3 (the fourth GPU) has four.
They use the frozen author query VAE and its 64-dimensional posterior mean,
matching the newly trained query FSBO/RGPE features. Peptide services and
other existing experiments were preserved.

## Using the pool

`pool.json` lists localhost ports 19140–19149 and the private token-file path.
A dedicated SSH tunnel exposes the same localhost endpoints on LLMOptimizer4.
Process records and individual logs are in this directory. Services have one
CPU thread each; the CUDA allocator is capped at 2% per process. CUDA context
memory is additional to that cap. Requests are serialized within each service.

```python
import sys
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
from query_vae_client import QueryVAEBackend

backend = QueryVAEBackend(
    '/root/data/logs/query_vae_cpu_capacity_20261106_v1/pool.json',
    task=query_task_id,
    fallback=known_valid_query_plan,
    seed=123,
)
latent_means = backend.encode(query_plans)
decoded_rows = backend.decode(latent_means, seed=456)
```

The adapter canonicalizes plans before encoding. Task-aware decoding uses the
native cross-join rejection and fallback policy. Encode requests round-robin;
decode requests use a stable task-to-service mapping. Explicit seeds are
reproducible despite interleaved requests. Returned means are raw latent
coordinates: transfer methods still need their saved corpus normalization.

This connects query inference for future runners. The FSBO/RGPE acquisition
runner and target-query Bao warmup are not implemented here. No new optimization
campaign or Thompson sampling wrapper was started.

## Validation

- `pool_validation.json`: all ten services passed concurrent real-query encode,
  decode, canonical re-encode, and seed reproducibility checks, with no errors.
  Concurrent validation took 8.6 seconds; this is a small functional check,
  not a sustained throughput measurement.
- `corpus_encoding_validation.json`: inference agrees with saved query training
  features within 9.54e-7 absolute error.
- `remote_health.json` and `remote_inference_validation.json`: all ten endpoints
  are reachable from LLMOptimizer4, with actual query inference verified.
- VAE services perform no database runtime measurements.

## CPU findings and proposed allocation

An isolated clean PostgreSQL copy on LLMOptimizer4 evaluated one measured Bao
plan for each of six training queries. Each CPU condition had one warmup and
two measured runs per query. The historical campaign's 1 GiB shared buffers,
64 MiB work memory and two parallel workers were retained.

| PostgreSQL cores | Measured runs | Five-second timeouts | Median wall time |
| --- | --- | --- | --- |
| 1 | 12 | 10 | 5.03 s (mostly censored) |
| 2 | 12 | 0 | 3.58 s |
| 3 | 12 | 0 | 2.40 s |

The median per-query slowdown at two cores versus three was 44%. Queries
actually launched up to two parallel PostgreSQL workers. Reducing database
cores changes measured runtimes and can change censoring.

The fourth dedicated optimizer core is an allocation choice. Proposed new
workers use **three cores total**, with a single-thread optimizer sharing one
of its database's three cores. This requires synchronous model fitting and
runtime evaluation: the optimizer must wait while PostgreSQL executes. The
database retains three cores. Validate this sharing in the eventual runner;
this probe did not execute a full FSBO/RGPE optimizer loop.

Proposed initial stage: **20 query workers**, 4 on LLMOptimizer2 and 16 on
LLMOptimizer4. Proposed target: **24**, 6 plus 18, reached via a 22-worker stage.
These are proposals, not running processes. CPU quotas are 40 and 64 cores,
regardless of the larger host CPU counts.

Before the initial stage, enlarge LLMOptimizer4's `/dev/shm` from 128 to
192 GiB or validate another local database storage layout. Eighteen copies
need about 157 GiB of database storage alone. No mount was changed. Reserve
another 1 GiB shared buffers and provisionally 2 GiB optimizer/query working
memory per query worker; validate real peak usage before scaling.

Require at least 30 GiB local and 32 GiB remote cgroup memory headroom, four
measured local CPU quota cores of spare capacity without sustained throttling,
and ten spare remote quota cores at the target. Existing local memory usage
fluctuates, so six local workers are conditional on admission checks.

`proposed_plan.json` contains the machine-readable proposal.
`cpu_benchmark.json` and `cpu_benchmark_summary.json` contain raw results and
limitations: six plans, fixed 3/1/2 condition order, and no concurrency pilot.
The benchmark database was stopped and removed after measurement.
