#!/usr/bin/env bash
set -euo pipefail

# The prior release cannot resume checkpoints in a fresh process.
exec bash /root/data/logs/peptide_bo_verified_20261003_v4/launch_on_llmoptimizer3.sh "$@"
