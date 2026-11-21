import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys, json, statistics
sys.path.insert(0, '/root/data/cys/BOLT/optimization/peptides')
from apex_oracle.refseqs import REFERENCE_SEQUENCE
from Levenshtein import distance as lev

RUN_DIR = "/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10"
INDEX = f"{RUN_DIR}/index.jsonl"
INIT_SEQ_FILE = "/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated/refs_0_1000_init.txt"
INIT_SCORE_FILE = "/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated/refs_0_1000_scores.csv"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ---- Load rows ----
rows = []
with open(INDEX) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

print(f"Total rows read: {len(rows)}")

task_ids = [r["task_id"] for r in rows]
unique_ids = set(task_ids)
print(f"Unique task_ids: {len(unique_ids)}")

from collections import Counter, defaultdict
cnt = Counter(task_ids)
dups = {k: v for k, v in cnt.items() if v > 1}
print(f"Duplicate task_ids count: {len(dups)}")
if dups:
    print("Duplicates:", dups)

# Deduplicate keeping LAST occurrence
dedup = {}
order = []
for r in rows:
    tid = r["task_id"]
    if tid not in dedup:
        order.append(tid)
    dedup[tid] = r
final_rows = [dedup[tid] for tid in order]
print(f"Rows after dedup (keep last): {len(final_rows)}")

# ---- Load init pool lazily per-index ----
print("\nLoading init pool files (this may take a moment)...")
needed_indices = set()
for r in final_rows:
    needed_indices.add(r["constraint_index"])
max_idx = max(needed_indices)
min_idx = min(needed_indices)
print(f"constraint_index range needed: {min_idx}..{max_idx}, count={len(needed_indices)}")

# We'll read only needed line ranges: for index i, rows [i*1000, i*1000+100)
# To avoid loading full 1M-line file into memory as list of strings unnecessarily large,
# but 1M short lines is fine memory-wise (~20MB). Let's just load fully for simplicity & speed.
with open(INIT_SEQ_FILE) as f:
    init_seqs_all = f.read().splitlines()
with open(INIT_SCORE_FILE) as f:
    init_scores_all = f.read().splitlines()

print(f"init_seqs_all lines: {len(init_seqs_all)}, init_scores_all lines: {len(init_scores_all)}")

def init_pool_best_mic(i):
    start = i * 1000
    end = start + 100
    scores = [float(x) for x in init_scores_all[start:end]]
    best_score = max(scores)  # score = -MIC, so best (max score) => min MIC
    best_mic = -best_score
    return best_mic, scores

# ================= Check 2: constraint compliance =================
print("\n=== Check 2: CONSTRAINT COMPLIANCE (similarity >= 0.75) ===")
violations_below_075 = []
mismatch_similarity = []
TOL = 1e-6
for r in final_rows:
    i = r["constraint_index"]
    ref = REFERENCE_SEQUENCE[i]
    seq = r["best_feasible_seq"]
    if seq is None:
        continue
    d = lev(seq, ref)
    sim = (len(ref) - d) / len(ref)
    if sim < 0.75 - TOL:
        violations_below_075.append((r["task_id"], i, sim))
    reported = r.get("best_feasible_similarity")
    if reported is not None and abs(sim - reported) > TOL:
        mismatch_similarity.append((r["task_id"], i, sim, reported))

print(f"Rows with best_feasible_seq present: {sum(1 for r in final_rows if r.get('best_feasible_seq') is not None)}")
print(f"Rows with best_feasible_seq is None: {sum(1 for r in final_rows if r.get('best_feasible_seq') is None)}")
print(f"Violations (recomputed similarity < 0.75): {len(violations_below_075)}")
for v in violations_below_075:
    print("  VIOLATION:", v)
print(f"Mismatches between recomputed and reported similarity (tol 1e-6): {len(mismatch_similarity)}")
for m in mismatch_similarity:
    print("  MISMATCH:", m)

# ================= Check 3: seq validity =================
print("\n=== Check 3: best_feasible_seq validity (len<=30, <=1 'C', standard AAs) ===")
len_violations = []
cys_violations = []
alphabet_violations = []
for r in final_rows:
    seq = r.get("best_feasible_seq")
    if seq is None:
        continue
    if len(seq) > 30:
        len_violations.append((r["task_id"], len(seq)))
    if seq.count("C") > 1:
        cys_violations.append((r["task_id"], seq.count("C")))
    bad_chars = set(seq) - VALID_AA
    if bad_chars:
        alphabet_violations.append((r["task_id"], seq, bad_chars))

print(f"Length > 30 violations: {len(len_violations)}: {len_violations}")
print(f"More than one 'C' violations: {len(cys_violations)}: {cys_violations}")
print(f"Non-standard AA violations: {len(alphabet_violations)}: {alphabet_violations}")

# ================= Check 4: reference & constraint_index =================
print("\n=== Check 4: reference field & constraint_index vs pep_<i> ===")
ref_mismatches = []
idx_mismatches = []
for r in final_rows:
    tid = r["task_id"]
    # tid format pep_<i>
    try:
        i_from_tid = int(tid.split("_")[1])
    except Exception:
        i_from_tid = None
    i = r["constraint_index"]
    if i_from_tid is not None and i != i_from_tid:
        idx_mismatches.append((tid, i, i_from_tid))
    ref_expected = REFERENCE_SEQUENCE[i] if 0 <= i < len(REFERENCE_SEQUENCE) else None
    if r.get("reference") != ref_expected:
        ref_mismatches.append((tid, i, r.get("reference"), ref_expected))

print(f"constraint_index vs task_id index mismatches: {len(idx_mismatches)}: {idx_mismatches}")
print(f"reference field mismatches vs REFERENCE_SEQUENCE[i]: {len(ref_mismatches)}: {ref_mismatches}")

# ================= Check 5: n_init_feasible == n_init_points == 100 =================
print("\n=== Check 5: n_init_feasible == n_init_points == 100 ===")
init_exceptions = []
for r in final_rows:
    nif = r.get("n_init_feasible")
    nip = r.get("n_init_points")
    if nif != 100 or nip != 100:
        init_exceptions.append((r["task_id"], nif, nip))
print(f"Exceptions: {len(init_exceptions)}")
for e in init_exceptions:
    print("  ", e)

# ================= Check 6: distributions =================
print("\n=== Check 6: distributions (min/median/max) ===")
fields = ["n_rounds", "e2e_updates", "n_acquisition_calls", "n_e2e_calls", "n_oracle_calls", "seconds", "n_obs"]
for fld in fields:
    vals = [r[fld] for r in final_rows if r.get(fld) is not None]
    if not vals:
        print(f"{fld}: no values")
        continue
    print(f"{fld}: n={len(vals)} min={min(vals)} median={statistics.median(vals)} max={max(vals)}")

# ================= Check 7: budget_respected / truncated =================
print("\n=== Check 7: budget_respected & truncated ===")
n_budget_false = sum(1 for r in final_rows if r.get("budget_respected") is False)
print(f"Rows with budget_respected == false: {n_budget_false}")
if n_budget_false:
    print("  task_ids:", [r["task_id"] for r in final_rows if r.get("budget_respected") is False])

trunc_counter = Counter()
for r in final_rows:
    t = r.get("truncated")
    if t is not None:
        trunc_counter[t] += 1
print(f"Rows with non-null truncated: {sum(trunc_counter.values())}")
for k, v in trunc_counter.items():
    print(f"  truncated={k!r}: {v}")

# ================= Check 8: sanity best_feasible MIC <= best init MIC =================
print("\n=== Check 8: SANITY best_feasible MIC <= best_init_MIC + 1e-6 ===")
sanity_violations = []
missing_init_data = []
for r in final_rows:
    i = r["constraint_index"]
    if i < 0 or (i + 1) * 1000 > len(init_scores_all):
        missing_init_data.append(r["task_id"])
        continue
    best_init_mic, scores = init_pool_best_mic(i)
    bfo = r.get("best_feasible_objective")
    if bfo is None:
        continue
    best_feasible_mic = -bfo
    if best_feasible_mic > best_init_mic + 1e-6:
        sanity_violations.append((r["task_id"], i, best_feasible_mic, best_init_mic))

print(f"Missing init data (out of range): {len(missing_init_data)}: {missing_init_data}")
print(f"Sanity violations (best_feasible MIC > best_init MIC): {len(sanity_violations)}")
for v in sanity_violations:
    print("  VIOLATION:", v)

# Also cross-check reported init_max_similarity field isn't part of MIC but let's also
# double check n_init_feasible corresponds to init pool feasibility (bonus, using recompute)
print("\n=== Bonus: recompute init pool feasibility count (similarity>=0.75) for cross-check ===")
init_feas_mismatches = []
for r in final_rows:
    i = r["constraint_index"]
    if i < 0 or (i + 1) * 1000 > len(init_seqs_all):
        continue
    start = i * 1000
    end = start + 100
    seqs = init_seqs_all[start:end]
    ref = REFERENCE_SEQUENCE[i]
    feas_count = 0
    for s in seqs:
        d = lev(s, ref)
        sim = (len(ref) - d) / len(ref)
        if sim >= 0.75 - 1e-9:
            feas_count += 1
    if feas_count != r.get("n_init_feasible"):
        init_feas_mismatches.append((r["task_id"], i, feas_count, r.get("n_init_feasible")))
print(f"Init feasibility recompute mismatches: {len(init_feas_mismatches)}")
for m in init_feas_mismatches[:20]:
    print("  ", m)
if len(init_feas_mismatches) > 20:
    print(f"  ... and {len(init_feas_mismatches)-20} more")

print("\nDONE")
