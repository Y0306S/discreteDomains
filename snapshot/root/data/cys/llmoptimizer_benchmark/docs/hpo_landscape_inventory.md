# HPO landscape inventory (beyond query, protein, DL, tabular ML)

Reference note for future planning. Last updated: 2026-08-25.

## Current domains (planned or in flight)

| Domain | Formulation | Oracle | Status |
|--------|-------------|--------|--------|
| **Tabular ML (CASH)** | Named HPs | Train sklearn → bal_acc / mse | Primary suite (`real_hpo_v2_mltest20_algo40`) |
| **Query optimization** | Join-order VAE → `z` | EXPLAIN cost (Postgres / dummy) | In flight — CEB VAE, IMDb stack |
| **Protein / peptides** | Sequence VAE → `z` | APEX MIC; ProteinGym GFP/AAV/GB1 + surrogate | AMP live; ProteinGym planned |
| **DL HPO** | Named HPs (no domain VAE) | zaxie `tune_report.json` lookup | Bootstrapped from real Optuna TPE logs |

See also: [`examples/benchmark/domains/README.md`](../examples/benchmark/domains/README.md), [domain transfer plan](/root/.cursor/plans/domain_transfer_plan_886395b6.plan.md), [ProteinGym VAE HPO plan](/root/.cursor/plans/proteingym_vae_hpo_8c94bf46.plan.md).

## What makes a landscape worth adding

A candidate domain is useful for **our** loop (LLM+TPE, experience bank, TUNE/LOCK, optional latent `z`) only if it has:

1. **Many related tasks** — TUNE/LOCK splits and retrieval are meaningful, not a single surface.
2. **Cheap, deterministic oracle** — lookup bench or fast simulator; not multi-day training per trial.
3. **Encoder or ConfigSpace** — either VAE/graph encode → `z` (protein/query style) or named HPs (ML/DL style).
4. **Literature alignment** — papers already do BO / AC / latent optimization on it.

**Poor fits:** synthetic BBOB alone (ELA diagnostics only), live CIFAR from scratch (too slow), full ProteinGym 217-assay leaderboard (fitness *prediction*, not latent HPO), prompt/LLM-config tuning (noisy oracle).

## Two formulation buckets

```text
Named HPs (like tabular ML + DL)
  tabular CASH     ← have
  DL (zaxie)       ← have
  NAS-Bench-201    ← natural 5th
  SAT/MIP AC       ← strongest new transfer story for experience bank

Encode → z → decode (like protein + query)
  peptides / GFP-AAV  ← have / planned
  join-order VAE      ← have
  molecules (SMILES)  ← strongest new latent story
  CompilerGym / graphs← optional
```

## Recommended next landscapes (ranked)

| Rank | Landscape | Same as | Why it fits | Skip unless |
|------|-----------|---------|-------------|-------------|
| **1** | **Molecules** (SMILES/SELFIES VAE → QED / penalized logP / TDC oracles) | Protein (`z`) | Other LOL-BO / JT-VAE / CDDD workhorse. Same decode→oracle stack as peptides. Many tasks = property or target. ZINC / GuacaMol / TDC are standard. | You want zero new VAEs |
| **2** | **NAS-Bench-201** (cell → accuracy lookup) | DL (named / discrete) | Already in domain-transfer plan as cheap DL public suite. Three datasets = three tasks, zero training. Graph encoding (Arch2Vec) optional later. | zaxie already saturates the DL story |
| **3** | **SAT / MIP algorithm configuration** (AClib, SATzilla features) | Tabular ML | Best analogue of CASH for *combinatorial* AC. Instance features = task vector; ISAC / SATzilla / SMAC literature. Hundreds of instances. | You only care about latent `z` |
| **4** | **CompilerGym** (LLVM flags / passes → runtime or code size) | Query (structured + named) | Programs = tasks; oracle is local and cheap. Closer to query-opt than classic HPO. | IMDb query env still unfinished |
| **5** | **Materials / crystals** (CDVAE-style `z` → formation energy / bandgap surrogate) | Protein (`z`) | Same encode→BO papers; DFT expensive — need surrogate like ProteinGym. | After GFP/AAV surrogate is proven |

## Explicitly not primary

| Landscape | Reason |
|-----------|--------|
| BBOB / COCO | Good for ELA / landscape-shape analysis (`task_similarity_research.md` §4), not a product domain |
| LCBench | Overlaps zaxie role; named in domain plan as optional later |
| Live RL training | Too expensive per trial |
| Docking from scratch | Needs cached surrogate oracle |
| Full ProteinGym 217 | Wrong task type (zero-shot ranking, not encode→HPO) |

## Practical recommendation (2026-08-25)

**Finish the current four first:** GFP/AAV surrogate + query Postgres oracle before opening a fifth adapter. Extra landscapes only help the paper if LOCK eval is real search, not fake-library bootstrap.

**If adding one more:**

- **Latent path:** **Molecules** — reuse VAE+LOLBO stack, new alphabet, cheap oracles.
- **Named-HP path:** **NAS-Bench-201** (public comparable DL suite, already in transfer plan) or **AClib SAT** (strongest experience-transfer story, less code reuse).

## Related reading in-repo

- [`docs/task_similarity_research.md`](task_similarity_research.md) — §7 “Other search spaces” (NAS, protein, query, SAT, BBOB)
- [`domain_transfer_plan`](/root/.cursor/plans/domain_transfer_plan_886395b6.plan.md) — LCBench / NAS-Bench-201 as DL cheap env
- [`proteingym_vae_hpo`](/root/.cursor/plans/proteingym_vae_hpo_8c94bf46.plan.md) — GFP/AAV/GB1 as protein sub-landscapes, not full ProteinGym leaderboard
