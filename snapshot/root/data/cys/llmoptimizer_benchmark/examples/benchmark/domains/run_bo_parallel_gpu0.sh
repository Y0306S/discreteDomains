#!/usr/bin/env bash
# Protein BO on GPU 0 only — all 1000 BOLT start sequences (pep_000..pep_999).
# Query BO deferred until RL config is finalized.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
exec bash "$REPO/examples/benchmark/domains/run_protein_bo_all1000_gpu0.sh"
#!/usr/bin/env bash
# Protein BO on GPU 0 only — all 1000 BOLT start sequences (pep_000..pep_999).
# Query BO deferred until RL config is finalized.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
exec bash "$REPO/examples/benchmark/domains/run_protein_bo_all1000_gpu0.sh"
