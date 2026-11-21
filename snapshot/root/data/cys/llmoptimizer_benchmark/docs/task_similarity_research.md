# Task similarity for experience retrieval

Research notes on how papers decide that two optimization tasks are “the same
kind of problem,” mapped onto our Experience-v2 bank and the RC/RS bake-off.

**What we actually need.** Given a new HPO slot `(dataset, algorithm)`, rank
past experiences so the injected prefer/avoid advice *transfers*. Two tasks are
similar iff ranking configs on A predicts ranking them on B. That is not a
property of an English blurb.

**What we already run.**

| Ranker | Code | Signal |
| --- | --- | --- |
| `embed_text` cosine (v1) | unused in live P5+ | wording of a summary string |
| **cascade** (`P5plusA5T50X50RC`) | `retrieve.py` | hard filters → cold mix (algo match, profile, role Jaccard, optional semantic) → Spearman reweight after local prefix |
| **surrogate** (`P5plusA5T50X50RS`) | `surrogate_landscape.py` | same cold pool, then per-algo GP landscape Spearman on the target prefix |

LODO on the 733-run bank: cascade top-1 held-out ρ ≈ 0.33 vs random same-algo 0.20
vs oracle 0.68. Similarity exists; text cosine is not what finds it.

Never overwrite `ptllm_tpe__P5plusA5T50X50__H2`. New similarity ideas get a new
prompt id (as RC/RS already do).

---

# How papers identify similar tasks

Papers do **not** use one definition. They pick a proxy for “will transfer
help,” then either (i) retrieve neighbors, (ii) weight source surrogates, or
(iii) learn a shared prior so similarity is implicit. The survey
[Bai et al., arXiv:2302.05927](https://arxiv.org/abs/2302.05927) splits transfer
BO into initial-design / search-space / surrogate / acquisition — every bucket
needs a relatedness rule.

Below, methods are grouped by **what they compare**, not by year.

---

## 1. Dataset / instance features (cold start, no HPO trials yet)

Use when the target has no local scores. This is Auto-sklearn’s original move
and the algorithm-selection literature’s default.

| Method | Similarity rule | Papers | Fit for us |
| --- | --- | --- | --- |
| **Handcrafted meta-features** | L1/L2/cosine on `(n, p, class imbalance, skew, missingness, landmarkers)` | [Feurer et al. AAAI 2015](https://ojs.aaai.org/index.php/AAAI/article/view/9354) (init SMBO from kNN datasets); Auto-sklearn; [Pfahringer landmarkers 2000](https://link.springer.com/chapter/10.1007/3-540-45164-1_38); Wistuba MF1/MF2 | We already have `problem.profile` and `profile_cos` in COLD_WEIGHTS. Weak if only n/p. |
| **Landmarkers** | Run cheap default models (1-NN, Naive Bayes, Decision Stump); vector of their scores *is* the task fingerprint | Pfahringer; Auto-sklearn; Reif et al. | Cheap, HPO-relevant. Not in the bank today. |
| **Dataset2Vec** | DeepSet encoder over `(x, y)` pairs; trained on a dataset-similarity auxiliary task | [Jomaa et al. DMKD 2021](https://arxiv.org/abs/1905.12930) | Schema-agnostic. Beats Feurer MFs for HPO transfer in their paper. Needs a trained encoder. |
| **End-to-end landmark net (DKLM)** | NN embeds the *set of evaluated (config, score) pairs*; deep-kernel GP | [Jomaa, OpenReview](https://openreview.net/pdf?id=ceb4a7449da04e42e4ff2cbbead8095229b9e54d) | Closer to Layer C than to raw data. Needs many source runs. |
| **TabPFN / tabular FM pooling** | Frozen foundation model; pool last-layer tokens / OOF embeddings into one vector per table | [Hollmann et al., Nature 2025](https://www.nature.com/articles/s41586-024-08328-6); TabPFN-2.5 | Most novel cheap encoder we do not use. One forward pass on a CSV subsample. |
| **Learned landmarkers** | Ask whether a NN encoder recovers landmarkers useful for HPO warm-start | [arXiv:2507.12604](https://arxiv.org/abs/2507.12604) | Confirms learned table reps can replace hand landmarkers. |
| **OTDD / s-OTDD** | Optimal-transport distance between feature–label measures (not between summaries) | [Alvarez-Melis & Fusi, ICML 2020](https://arxiv.org/abs/2002.02923); [s-OTDD, ICML 2025](https://arxiv.org/abs/2501.18901) | Geometry of the table. Heterogeneous schemas need a ground metric. s-OTDD is near-linear. |
| **Task2Vec** | Diagonal Fisher of a frozen probe net = fixed-dim task embedding | [Achille et al. ICCV 2019](https://arxiv.org/abs/1902.03545) | Built for vision. Analogue for us: Fisher of a small MLP / TabPFN on the table. |
| **CKA / RSA / MMD** | Representation or distribution distance between two tables (or two probe nets) | Kornblith CKA; Kriegeskorte RSA; Gretton MMD | Domain-adaptation toolkit. No HPO label. Use as a cold gate, not as the HPO score. |
| **ISAC clustering** | g-means on instance-feature vectors; one config per cluster | [Kadioglu et al. ECAI 2010](https://ai.dmi.unibas.ch/research/reading_group/kadioglu-et-al-ecai2010.pdf) | Algorithm *configuration*, not HPO transfer, but the similarity is the same object: nearest cluster in feature space. |
| **SATzilla / AutoFolio** | Empirical hardness model: predict per-solver runtime from instance features, then pick | [Xu, Hutter, Hoos, Leyton-Brown JAIR](https://jair.org/index.php/jair/article/view/10556); SAT’24 revisit of SATzilla features | For CASH / solver portfolios. Our analogue: predict “will this experience’s incumbent be good here.” |

**Takeaway.** Every successful AutoML warm-start paper from 2013–2018 used
*dataset* kNN, not text kNN. Feurer 2015 literally: “start from configs that
worked on similar datasets,” similarity = meta-feature distance.

---

## 2. Shared-config ranking (the definition of HPO similarity)

Once even a few target evaluations exist, papers stop asking “are the tables
alike?” and ask “do the *response surfaces* agree?”

| Method | Similarity rule | Papers | Fit for us |
| --- | --- | --- | --- |
| **Pairwise ranking kernel (TST-R)** | Fraction of config pairs whose order agrees on source vs target; Nadaraya–Watson mix of source GPs | [Wistuba, Schilling, Schmidt-Thieme, ECML 2016](https://www.ismll.uni-hildesheim.de/pub/pdfs/wistuba_et_al_ECML_2016.pdf); MLJ 2018 scalable GPs | This **is** our Spearman rerank, with a GP wrapper. Bandwidth ρ is their only knob. |
| **RGPE ranking loss** | Weight source GP `i` by how well it *ranks* the target’s observed points (not MSE) | [Feurer, Letham, Bakshy, AutoML 2018 / arXiv:1802.02219](https://arxiv.org/abs/1802.02219) | Gold standard. Scale-invariant. We mix Spearman into the cascade score (`spearman_mix=0.4`) instead of fitting per-source GPs. |
| **TAF / TAF-R / TAF-ME** | Same ranking weights, applied to the *acquisition* (source improvement + target EI), and faded once the source has nothing better left | [Wistuba et al. MLJ 2018](https://doi.org/10.1007/s10994-017-5684-y) | Directly justifies `experience_retrieve_fade`. Similarity × remaining value. |
| **SCoT ranking GP** | One GP on `(meta-features, config)` that predicts *ranks*, so scores on different datasets become comparable | [Bardenet, Brendel, Kégl, Sebag, ICML 2013](https://proceedings.mlr.press/v28/bardenet13.html) | First paper that said “practitioners generalize across similar problems.” Needs meta-features. |
| **Active Testing** | Iteratively pick the source whose predicted ranking is most informative given current disagreements | [Leite, Brazdil, 2012] | Retrieval as an active query, not a one-shot kNN. |
| **Quantile / Gaussian copula (GCP)** | Map each task’s scores to quantiles; transfer if the config→quantile map agrees | [Salinas, Shen, Perrone, ICML 2020](https://arxiv.org/abs/2003.13370) | Same-space only. Robust to RMSE vs R² vs bal_acc — exactly our mix of cls/rgs. |
| **Shared-design Spearman** | Evaluate (or look up) a common probe set; ρ of scores | Standard label in transfer papers; our LODO oracle | Free if histories overlap; else an 8–16 point probe. Our cascade after trial 24. |
| **Surrogate-landscape Spearman** | Fit one GP per `(algo, dataset)` offline; ρ(μ_source(x_i), y_target_i)` on the target prefix | Our RS arm; cousin of RGPE without online weight fitting | Already in flight vs cascade. |
| **Importance-vector cosine** | fANOVA / ablation importance over named roles. Same knobs matter → same advice | [van Rijn & Hutter 2018](https://arxiv.org/abs/1710.04725) “HP importance across datasets”; [Hutter et al. ICML 2014](https://proceedings.mlr.press/v32/hutter14.pdf) fANOVA; Probst tunability | Free in `knowledge.factors`. Cascade uses it after the prefix (`importance_cos=0.45`). Cold start zeros it. |
| **TransBO two-phase** | Phase 1: weight source surrogates *without* meta-features; Phase 2: fine-tune on target. Explicitly attacks negative transfer from treating all sources equally | [KDD 2022](https://doi.org/10.1145/3534678.3539255) | Their complaint about SMFO/SGPR (“use all sources equally”) is why we drop ρ<0. |

**Negative transfer is the failure mode.** SMFO (Wistuba 2015, sequential
model-*free*) and SGPR average every source. TransBO, RGPE, TST-R, and TAF all
exist to *stop* transferring from dissimilar tasks. Our `drop_negative_spearman`
is the cheap version of that.

---

## 3. Learned task kernels and shared priors (similarity is implicit)

These methods rarely emit a neighbor list. They learn `k(task, task')` or a
prior that makes related tasks close in function space.

| Method | How relatedness is modeled | Papers | Fit for us |
| --- | --- | --- | --- |
| **Multi-task BO** | Kronecker kernel `k_x(x,x') · k_task(t,t')`; `k_task` is learned from all observations | [Swersky, Snoek, Adams, NeurIPS 2013](https://arxiv.org/abs/1304.5231); Krause & Ong contextual BO | Cubic in tasks×points. Needs a shared config grid. Not a retriever. |
| **ABLR** | Shared NN basis; one Bayesian linear head per task | [Perrone et al. 2018](https://arxiv.org/abs/1802.02219) (also in Feurer’s comparison) | Similarity = cosine in the learned basis. Needs a big multi-task matrix. |
| **FSBO / DKT** | Deep kernel GP pre-trained on source tasks, few-shot adapted | [Wistuba & Grabocka 2021](https://arxiv.org/abs/2101.07667); Patacchiola DKT | Same-space HPO. Overkill for prompt retrieval; right for a BO hybrid. |
| **HyperBO** | Pre-train GP mean+kernel prior by minimizing KL to source functions; tasks conditionally independent given the prior | [Wang et al. JMLR 2024](https://www.jmlr.org/papers/v25/23-0269.html) | They say explicitly: *modeling k(task,task') is hard*; a shared prior scales linearly. Includes a protein-sequence tuning set. |
| **MetaBO** | Meta-learn the acquisition as a neural net over source functions | [Volpp et al. arXiv:1904.02642](https://arxiv.org/abs/1904.02642) | Similarity baked into AF weights. No neighbor list to inject into a prompt. |
| **FMLP / DNGO / BOHAMIANN** | NN (or BNN) maps `(config, dataset-id)` → score | Springenberg; Snoek DNGO | Dataset id is a learned embedding — that embedding *is* the task vector. |
| **OptFormer** | Autoregressive transformer over HPO traces; transfer by conditioning on a text task card | [Chen et al.](https://arxiv.org/abs/2205.13320) | Learned similarity via attention over traces, not cosine on blurbs. |
| **PFNs4BO** | Prior-data fitted net approximates the BO posterior in one forward pass | [Müller, Feurer, Hollmann, Hutter, ICML 2023](https://proceedings.mlr.press/v202/muller23a.html) | In-context: the “similar tasks” are the prior samples the net was trained on. |
| **BOLT** | LLM encoder of the *task description* + large-scale multi-task BO | [arXiv:2503.08131](https://arxiv.org/abs/2503.08131) | Closest LLM×MTBO cousin. They still beat “text embedding alone” with a GP head. |

**Takeaway.** From Swersky 2013 through HyperBO 2024 the field moved *away*
from declaring similarity and *toward* learning a prior. For prompt retrieval we
still need an explicit top-k, so RGPE/TST-R-style scores remain the right
interface even if a HyperBO prior sits behind them.

---

## 4. Search-space / landscape geometry (when knobs do not line up)

Advice only transfers if knobs can be aligned. `space_hash` is all-or-nothing.
Papers that leave the shared-grid assumption use geometry.

| Method | Idea | Papers | When it wins |
| --- | --- | --- | --- |
| **Exact space hash** | Same ConfigSpace → copy configs and importance as-is | Our filter | Same algorithm, same bounds. Highest precision, zero novelty. |
| **Role-set overlap** | Jaccard of `{learning_rate, tree_depth, n_estimators, …}` | Designed into Experience v2 `factor.role`; van Rijn & Hutter roles | xgboost vs lightgbm vs catboost. Cascade already scores `role_jaccard`. |
| **OTMANN** | Optimal transport on NAS cell graphs so architectures in different search spaces can be compared | [Kandasamy et al. NeurIPS 2018](https://arxiv.org/abs/1802.07191) | Heterogeneous *spaces*, not datasets. Analogue: OT on HP graphs. |
| **Gromov–Wasserstein / fused OT** | Align two `(config, score)` clouds when coordinates differ: match pairwise score geometry | GW literature; used in heterogeneous transfer | AdaBoost 4-d vs XGB 9-d. Heavy. Do not ship before Spearman-on-aligned-roles works. |
| **ELA landscape fingerprint** | Exploratory landscape features (dispersion, information content, meta-model R², …) of the HP loss surface; cluster tasks in ELA space | [Mersmann et al. GECCO 2011](https://www.coseal.net/ela/); [HPO×ELA, PPSN 2022 / arXiv:2208.00220](https://arxiv.org/abs/2208.00220); flacco / pflacco | Needs a filled design. HPO×ELA showed XGB HPO problems cluster with *low-multimodality* BBOB functions — i.e. landscape shape *is* a task descriptor. |
| **AKT-MTBO kernelized autoencoder** | Learn a kernel-induced task similarity, then pick auxiliary tasks with a heuristic to avoid negative transfer | IEEE SMC 2025 | Heavier; only if we jointly BO multiple tasks. |
| **Search-space pruning / bounding box** | Relatedness → restrict the target box to the convex hull of source incumbents | [Perrone & Shen 2019](https://arxiv.org/abs/1902.06642) search-space learning; listed as a baseline in Feurer 1802.02219 | Transfer of *region*, which is exactly our prefer interval. |

---

## 5. Trajectory and incumbent-curve shape

Cheap, weak alone (many runs stagnate), useful as a tie-break.

| Method | Signal | Papers / lineage |
| --- | --- | --- |
| **Best-so-far DTW** | Align incumbent curves | Time-series kNN; used as a heuristic in some AutoML dashboards |
| **Dynamics tags** | STAGNANT / JUMPY / incumbent_updates | Our `result.dynamics`; currently only tokens in `embed_text` |
| **Learning-curve similarity** | Freeze-thaw / LCNet: tasks with similar early-epoch curves share hyperparameters | [Swersky freeze-thaw](https://arxiv.org/abs/1406.2541); Klein LCNet |
| **WAC (warmstart AC)** | Replay high-performing configs from similar *instances* as the SMAC initial design | [Lindauer & Hutter AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11532) | Algorithm configuration, but the retrieval is “configs that worked on nearby instances.” |

---

## 6. LLM-native retrieval (not cosine-on-a-blurb)

Text embedding works when *descriptions differ a lot*. Our bank is homogeneous
(“classification with xgboost; prefer lower learning_rate”), so cosine collapses.

| Method | How they retrieve | Papers | Lesson |
| --- | --- | --- | --- |
| **ExpeL** | all-mpnet + FAISS over successful trajectories; plus a voted insight list | [Zhao, Lee, Hsu, AAAI 2024](https://arxiv.org/abs/2308.10144) | Split memory: insights global, trajectories from *true* neighbors. We already have `prompt_block` + takeaways. |
| **MLCopilot** | Embed the *user’s task description* with GPT-3; retrieve experience + knowledge pools; one-shot prompt | [Zhang et al. EACL 2024](https://aclanthology.org/2024.eacl-long.179.pdf) | They can do this because users write free-form task cards. Our slots are `(dataset, algo)` with near-identical cards. |
| **LLAMBO** | Zero-shot warm-start from the LLM’s pretraining, not from a private bank | [Liu et al. 2024](https://arxiv.org/abs/2402.03921) | Different problem: parametric memory vs our explicit bank. |
| **AgentHPO** | Iterates on *this* task’s history; no cross-task retrieval | [Liu et al. PMLR 2025](https://proceedings.mlr.press/v280/liu25c.html) | Confirms most LLM-HPO papers still ignore the bank. |
| **OptFormer / BOLT** | Condition a sequence model on a text task card *and* traces | Chen et al.; arXiv:2503.08131 | The card is a *feature*, not the distance. |
| **Structured transfer judge** | Show two experience cards with roles aligned; ask for a 0–1 transfer score and which factors to keep/drop | (ours, unimplemented) | Can say “same algorithm, opposite LR direction → do not inject,” which cosine cannot. |

---

## 7. Other search spaces we said we care about

Same three-layer test (task / representation / knowledge), different fingerprints.

| Domain | What papers use as “similar task” | Notes |
| --- | --- | --- |
| **NAS** | Path encoding, adjacency encoding, Arch2Vec, OTMANN transport on cell graphs, zero-cost proxies (Synflow, Grasp, Jacobian) | Space similarity dominates dataset similarity. |
| **Protein / sequence design** | BLAST / sequence identity; ESM / protein LM embeddings; HyperBO’s protein-sequence tuning split | HyperBO treats each protein as a task and learns a *shared GP prior*, not kNN on FASTA. |
| **Query / workload optimization** | SQL sketch / template fingerprint; predicate histogram; plan-tree edit distance; workload embedding | Representation layer is the query graph, not the HP vector. |
| **SAT / combinatorial AC** | SATzilla features (variable-clause graph, Horn proximity, probing); ISAC clusters | Instance features *are* the task vector. Direct analogue of meta-features. |
| **Continuous BBOB** | ELA feature vector; nearest BBOB neighbor predicts optimizer rank | HPO×ELA: this is how you ask “is this XGB landscape like f15?” |

**Rule we already wrote down:** retrieve with domain ∩ representation first;
use factor *roles* to transfer advice when knobs do not line up.

---

# What we have vs the literature

| Paper idea | In our stack today | Gap |
| --- | --- | --- |
| Meta-feature kNN (Feurer 2015) | `profile_cos` if `problem.profile` is filled | Profile is often empty; no landmarkers |
| Role Jaccard | yes, cold + warm | — |
| Importance cosine (van Rijn) | yes, after prefix | Zeroed at cold start (correct) |
| Spearman / TST-R ranking | cascade after `rerank_after` (24) | Mix 0.4, not a fitted RGPE weight |
| Per-source GP landscape | RS arm | Bake-off in flight |
| TAF-style fade | `experience_retrieve_fade` exists; **disabled** on RC/RS | Re-enable as a separate arm once RC/RS finish |
| Drop negative transfer | `drop_negative_spearman` | — |
| Dataset2Vec / TabPFN / OTDD | no | Cold-start gap vs n/p |
| Copula / quantile map (Salinas) | no | Would fix cls vs rgs scale before Spearman |
| RGPE loss (not Spearman mix) | no | Next upgrade of cascade Stage 3 |
| ELA fingerprint | no | Offline analysis, heavy online |
| HyperBO / FSBO / ABLR / PFN prior | no | Wrong interface for a prompt top-k |
| LLM judge / ExpeL split | no | Only after Stage 3 is in the loop |
| ISAC / SATzilla | no | Relevant if we add CASH / solver spaces |

---

# What is actually worth testing next

Do **not** replace `embed_text` with a larger text embedder. The bottleneck is
the signal, not the encoder. Do **not** mix a new ranker with a new T-gate or a
new hybrid selector in one arm.

One-factor arms, all extend `P5plusA5T50X50`, hybrid **H2**, seed 1, 80 trials,
40 unique slots. New prompt ids only.

| Priority | New id (suggested) | Change vs RC/RS | Question the paper literature asks |
| --- | --- | --- | --- |
| 0 | (running) `P5plusA5T50X50RC` / `RS` | cascade vs surrogate GP | Is TST-R Spearman enough, or do we need a landscape GP? |
| 1 | `…X50RFade` | RC + `experience_retrieve_fade: true` | TAF: fade source once it has nothing better |
| 1 | `…X50RCopula` | quantile-transform scores before Spearman | Salinas GCP: does scale mismatch kill ρ on rgs slots? |
| 2 | `…X50RRgpe` | replace mix-0.4 with ranking-loss weights | Feurer RGPE vs our linear mix |
| 2 | `…X50RLand` | add landmarkers / fill `problem.profile` | Feurer 2015 cold start |
| 3 | `…X50RTabpfn` | TabPFN (or Dataset2Vec) vector in the cold mix | Is the table geometry missing from n/p? |
| skip until 3 | OTDD, ELA, GW, HyperBO, LLM judge | research projects | Only if Stage 3 still leaves ρ@1 ≪ oracle 0.68 |

**Pre-registered decision rule (same as X50R):** unique-slot mean rank **and**
head-to-head vs `P5plusA5T50X50__H2`; unique slots must filter traj by
algorithm (`report_x50r_unique_slots.py`). Adopt only if both win.

**Offline first (zero API):** replay every ranker on finished X50 trajectories
and on the holdout bank the way `eval_lodo_similarity.py` already does. Live
arms only for methods that change *which* `prompt_block` the LLM sees.

---

# Reading list

**Surveys / framing**
- Bai, Li, Wang, Chen — Transfer learning for Bayesian optimization, [arXiv:2302.05927](https://arxiv.org/abs/2302.05927)

**Dataset / instance similarity**
- Feurer, Springenberg, Hutter — Initializing BO via meta-learning, [AAAI 2015](https://ojs.aaai.org/index.php/AAAI/article/view/9354)
- Jomaa et al. — Dataset2Vec, [arXiv:1905.12930](https://arxiv.org/abs/1905.12930)
- Alvarez-Melis & Fusi — OTDD, [arXiv:2002.02923](https://arxiv.org/abs/2002.02923)
- s-OTDD — [arXiv:2501.18901](https://arxiv.org/abs/2501.18901)
- Achille et al. — Task2Vec, [arXiv:1902.03545](https://arxiv.org/abs/1902.03545)
- Hollmann et al. — TabPFN, Nature 2025
- Kadioglu et al. — ISAC, ECAI 2010
- Xu, Hutter, Hoos, Leyton-Brown — SATzilla, JAIR

**Ranking / surrogate transfer (the HPO definition)**
- Bardenet et al. — SCoT, [ICML 2013](https://proceedings.mlr.press/v28/bardenet13.html)
- Wistuba et al. — TST-R, [ECML 2016](https://www.ismll.uni-hildesheim.de/pub/pdfs/wistuba_et_al_ECML_2016.pdf)
- Wistuba et al. — TAF, [MLJ 2018](https://doi.org/10.1007/s10994-017-5684-y)
- Feurer, Letham, Bakshy — RGPE, [arXiv:1802.02219](https://arxiv.org/abs/1802.02219)
- Salinas, Shen, Perrone — GCP / copula, [ICML 2020](https://arxiv.org/abs/2003.13370)
- TransBO — [KDD 2022](https://doi.org/10.1145/3534678.3539255)
- van Rijn & Hutter — HP importance across datasets, [arXiv:1710.04725](https://arxiv.org/abs/1710.04725)

**Learned priors (implicit similarity)**
- Swersky, Snoek, Adams — Multi-task BO, NeurIPS 2013
- Perrone et al. — ABLR, 2018
- Wistuba & Grabocka — FSBO, 2021
- Wang et al. — HyperBO, [JMLR 2024](https://www.jmlr.org/papers/v25/23-0269.html)
- Müller et al. — PFNs4BO, [ICML 2023](https://proceedings.mlr.press/v202/muller23a.html)
- Chen et al. — OptFormer
- BOLT — [arXiv:2503.08131](https://arxiv.org/abs/2503.08131)

**Landscape / heterogeneous spaces**
- HPO×ELA — [arXiv:2208.00220](https://arxiv.org/abs/2208.00220)
- Kandasamy et al. — OTMANN, NeurIPS 2018
- Perrone & Shen — search-space learning, 2019
- Lindauer & Hutter — WAC, [AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11532)

**LLM retrieval**
- Zhao et al. — ExpeL, AAAI 2024
- Zhang et al. — MLCopilot, EACL 2024
- Liu et al. — LLAMBO, [arXiv:2402.03921](https://arxiv.org/abs/2402.03921)
