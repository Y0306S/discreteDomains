# Frozen-VAE peptide local/recenter variants

`tools/peptide_local_bo.py` wraps FSBO, RGPE, TPE, GP, HEBO and SMAC with the
existing BOLT local search and finite recenter proposal policy. Existing pinned
implementations, recipes, completed results and transfer source models remain
read-only dependencies. Results use the method name `<method>-local-recenter`.

FSBO, RGPE and TPE use the peptide **train800 LOLBO recipe settings** from
`/root/data/logs/peptide_bo_train800_20261027_v5b/recipe.json`:

| Setting | Value |
| --- | --- |
| Initial measured sequences | 100 per reference |
| Acquisition batch size | 50, including TPE |
| Acquisition proposal limit | 5,000 |
| Shared fresh acquisition/recenter limit | 5,000 |
| Similarity threshold | 0.75 |
| Random seed | 1 |
| Target constraint GP initial/update epochs | 20 / 2 |
| Target constraint GP learning rate | 0.001 |
| Recenter trigger | 10 consecutive acquisition rounds without improvement |
| Recenter inputs | Recent batch plus unique feasible top 1,000 |
| Recenter passes | 2, with native length-dependent minibatching |
| Initial/minimum/maximum region length | 0.8 / 0.5**7 / 1.6 |
| Success/failure tolerance | 10 / 32 |
| Sparse perturbation probability | 20 / 256 |
| Candidate pool for posterior/acquisition methods | 5,000 |

The author VAE stays in evaluation mode with gradients disabled. There are no
VAE optimizer steps, joint end-to-end updates or representation changes.
Recenter uses native posterior-sampled encoding and native decoding with these
fixed weights. This does not reproduce LOLBO's jointly updated representation,
and no feasibility improvement has yet been established for these variants.

FSBO and RGPE retain their existing objective backends and frozen canonical
posterior-mean source representation. Target constraint models learn from the
actual latent vectors sent to the decoder. Transfer target refitting retains
its method-specific settings; LOLBO's GP epochs do not replace FSBO training or
RGPE ranking. GP uses native constrained Thompson sampling. TPE retains
independent constrained Optuna TPE: sparse fixed coordinates are declared in
each ask distribution, all 50 proposals are asked before batch feedback, and
every scored proposal is told with its measured constraint. No constant-liar
or multivariate-TPE option is introduced.

HEBO and SMAC retain batch-one MACE and RF/EI respectively, restricted to a
native sparse local pool with sampled constraint feasibility (or native least
violation fallback). HEBO retains its GP/optimizer, with a numerical lengthscale
floor of 1e-6 and explicit posterior failures in place of the installed library's
silent random-prediction fallback. SMAC uses the installed OpenBox sklearn PRF
fallback because pyrfr is absent. These are separately named variants.

Recenter events consume the same fresh-evaluation budget as acquisition and
have separate proposal/feasibility counters. They do not increment acquisition
rounds or update the acquisition trust region. SQLite checkpoints persist
actual constraint coordinates, canonical transfer features, models, TPE
sampler/trials, pending scores, RNG and the finite recenter cursor. A scored
pending event resumes without another oracle call.

Run from the repository using the existing environment:

```bash
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/root/data/cys/envs/bolt_clean_venv/bin/python

# Configuration/AST inspection only; does not execute LOLBO.
$PY tools/peptide_local_bo.py check-bolt

# Prepare new recipes only; no workers or optimization jobs start.
$PY tools/peptide_local_bo.py prepare \
  --campaign /root/data/logs/peptide_frozen_local_recenter_20261101_v1 \
  --scope train --methods fsbo,rgpe,tpe,gp,hebo,smac

# Execute one explicitly selected task; choose an available CPU/service.
$PY tools/peptide_local_bo.py run-task \
  --campaign /root/data/logs/peptide_frozen_local_recenter_20261101_v1/tpe \
  --task pep_000 --cpu 18 --server http://127.0.0.1:18871

# Component/recovery checks; no APEX calls or LOLBO optimization.
$PY tests/test_peptide_local_bo.py -v
```

`--scope train` selects the original train800 starting pool. Transfer models
were trained on those 800 tasks, so train-scope FSBO/RGPE recipes explicitly
record `source_task_overlap=true`; results are training diagnostics. For held-out
evaluation, prepare a separate campaign with `--scope test`, retaining these
same train800-derived optimizer settings and the validated test200 pool.
Preparation never changes the source bank or retrains the VAE.

Use the standalone test command above: the frozen release's namespace package
can conflict with the repository package under unittest discovery. Completed
campaigns have not been rerun, and LOLBO has only undergone a settings check.
