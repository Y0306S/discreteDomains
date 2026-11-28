# Query continuation launch

Verified 2026-11-23T22:34:35.596063+08:00. Both controllers are running, both have committed new SQL measurements, and neither reports worker failures.

| Method | Active task workers / private DBs | Model services | Endpoint |
|---|---:|---|---|
| SemanticOPT | 20 | 20 LLM replicas | 400 proposals per query; fixed first ten to 800 |
| Saved BOLT30 + frozen-VAE LOLBO | 7 initially | 8 VAE services | 30 retained + 770 new = 800 total observations per query |

At verification, SemanticOPT had committed 31 new evaluations and BOLT 243. These counters are a launch snapshot; current status is in each campaign.

SemanticOPT campaign: /root/data/cys/baseline/campaigns/semanticopt_query_400_pilot800_20261123

BOLT campaign: /root/data/cys/baseline/campaigns/query_bolt30_lolbo_total800_20261123

Read current consolidated progress with `python3 /root/data/cys/baseline/campaigns/semanticopt_query_400_pilot800_20261123/control/status.py`.

The old hybrid's recovery watchdog is disabled, its completion evidence is retained, and its private databases were handed off after idle checks. All 200 SemanticOPT record prefixes were verified unchanged. Both models and generation policies remain pinned. The full 15,360 input-token envelope works on the A16 pairs without prompt truncation; three A800 replicas per card are admitted following measured generation peaks.

BOLT's prior proposal blockage was corrected by applying native execution canonicalization before strict plan validation. The canary audit verifies the resulting ten plans and rejects decoder fallback. The canary's ten SQL evaluations count toward its task budget. BOLT is the frozen-VAE LOLBO variant described in its recipe, not an end-to-end VAE retraining experiment.

The eighth BOLT SQL worker was subsequently authorized and launched on2026-11-24; see the admission update below. Automatic transfer of finished SemanticOPT slots into additional BOLT workers is not armed in this initial launch; the finite controllers continue their present worker pools to the specified task budgets.

## Eighth-worker admission,2026-11-24

User authorized launching the reserved worker on Optimizer3. It uses CPUs16–18 for a private PostgreSQL database and CPU19 for the optimizer (port20437). The eight existing VAE services are shared. A separate supervised slot uses the original task leases and unchanged recipe/checkpoint identity; the original seven-worker controller keeps running. Consolidated status includes all eight workers.

Database evidence: /root/data/cys/baseline/campaigns/query_bolt30_lolbo_total800_20261123/evidence/worker07/database.ready.json. A live pg_basebackup attempt failed because a source tablespace symlink exceeded PostgreSQL's tar-header path limit; no active task was interrupted. The final database is an offline copy of retired, cleanly shut-down worker22, with every copied file verified bySHA256, followed by exactIMDbtable-count, hint-extension and resource-setting checks. Startup memory headroom was about15GiB.

Supervisor records: /root/data/cys/baseline/campaigns/query_bolt30_lolbo_total800_20261123/control/extra_controller.status.json. New slot log: /root/data/cys/baseline/campaigns/query_bolt30_lolbo_total800_20261123/logs/slot07.log.
