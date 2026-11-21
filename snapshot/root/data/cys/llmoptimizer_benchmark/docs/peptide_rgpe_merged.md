RGPE acquisition uses the posterior of a sum of independent GP components:

```
mean(X) = sum_t w_t mean_t(X)
cov(X,X) = sum_t w_t**2 cov_t(X,X)
```

Weights are normalized once. Components predict in standardized output units;
the target offset and scale are applied once after merging. The covariance
includes correlations between different candidates. Diagonal variances alone
would not preserve batch Thompson sampling.

The previous peptide acquisition implementation already used these formulas
and sampled one merged posterior. `tools/peptide_rgpe_merged.py` reduces its
memory further: it evaluates each source's covariance in 256-candidate blocks
and adds those blocks directly to one merged covariance. It uses the pinned
skopt GP's fitted kernel and training Cholesky factor. A single merged Cholesky
factor generates all 50 batch samples. Numerical retries add the same bounded
jitter as the prior implementation. Zero-weight sources are not loaded.

Source predictions still require computation. The method does not merge the
training sets or replace source GPs with one newly trained GP. Target fitting,
source ranking, joint target LOOCV ranking, dilution, and learned weights are
inherited unchanged. LOOCV remains a separate potentially expensive stage.
Memory still grows with the candidate covariance and the size of a component's
training data; this does not guarantee freedom from OOM at arbitrary sizes.

Deployment preserves the original recipe bytes, checkpoint identities, pinned
code, source bank, and stored scores. An acquisition amendment pins four new
modules and a validation report. `peptide_rgpe_merged_launch.py` runs the existing
frozen local wrapper with the versioned worker entry. The worker inherits the
original fit/snapshot/restore methods and replaces acquisition sampling only.
It records its implementation/hash identity in each task's `rgpe_worker.json`
and its latest sample diagnostics in `rgpe_acquisition.json`.

`peptide_rgpe_merged_dispatch.py` retains the prior CPU, quota, memory and task
claim guards and launches the amended entry. For deployment, stop admissions,
drain existing workers, back up SQLite checkpoints, archive the old dispatch
plan/status, bind the validated amendment in the new plan, and restart both
host dispatchers. An old live worker must never be adopted as though it were
running the new implementation. Resume existing pending requests and scored
events using their existing identities and nonces.

The VAE remains frozen. Train800 search settings and the held-out 200-task
queue remain unchanged. No LOLBO optimization is executed for this change.

Validation artifacts live in `reports/rgpe_merged_20261102/`. Tests cover the
formulas, full covariance, negative correlations, zero-weight loading, source
lifetime, 100/800 contributors, numerical jitter, score sign, seeded samples,
one merged factorization, source model equivalence, production candidate count,
compact snapshots and subprocess restart. Existing wrapper recovery and budget
tests are also run. Diagnostic stress weights are never written to production
checkpoints or the source bank.
