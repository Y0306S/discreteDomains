import json, sys, warnings, math
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

sys.path.insert(0, '/root/data/cys/BOLT/optimization/peptides')
from apex_oracle.refseqs import REFERENCE_SEQUENCE

RUN_DIR = "/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10"
INIT_TXT = "/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated/refs_0_1000_init.txt"
SCORES_CSV = "/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated/refs_0_1000_scores.csv"

# --- Load index.jsonl, dedup by task_id keeping LAST occurrence ---
rows = {}
n_lines = 0
with open(f"{RUN_DIR}/index.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        r = json.loads(line)
        rows[r["task_id"]] = r

print(f"# Loaded {n_lines} lines, {len(rows)} unique task_ids after dedup (kept last occurrence)")

# --- Load init scores (flat, 1000 refs x 1000 rows each) ---
with open(SCORES_CSV) as f:
    scores = [float(x) for x in f if x.strip() != ""]
print(f"# scores.csv rows: {len(scores)}")

with open(INIT_TXT) as f:
    init_seqs = [x.strip() for x in f if x.strip() != ""]
print(f"# init_seqs rows: {len(init_seqs)}")

assert len(scores) == 1000000
assert len(init_seqs) == 1000000

def task_index(task_id):
    return int(task_id.split("_")[1])

# Build per-task records
records = []
for task_id, r in rows.items():
    i = task_index(task_id)
    lo, hi = i * 1000, i * 1000 + 100
    block_scores = scores[lo:hi]
    block_seqs = init_seqs[lo:hi]
    best_idx = max(range(100), key=lambda k: block_scores[k])
    init_best_score = block_scores[best_idx]
    init_best_seq = block_seqs[best_idx]
    init_mic = -init_best_score

    bo_seq = r.get("best_feasible_seq")
    bo_mic = -r["best_feasible_objective"]

    improved_seq = (bo_seq != init_best_seq)
    improved_score_tol = (bo_mic < init_mic - 1e-6)  # lower mic better; improvement means bo_mic < init_mic
    tied_score_tol = (abs(bo_mic - init_mic) <= 1e-6)

    ref_len = len(REFERENCE_SEQUENCE[i])
    ref_str = REFERENCE_SEQUENCE[i]
    consistent_ref = (ref_str == r.get("reference"))

    records.append(dict(
        task_id=task_id, idx=i, init_mic=init_mic, bo_mic=bo_mic,
        init_best_seq=init_best_seq, bo_seq=bo_seq,
        improved_seq=improved_seq, improved_score_tol=improved_score_tol,
        tied_score_tol=tied_score_tol, ref_len=ref_len, consistent_ref=consistent_ref,
    ))

n_tasks = len(records)
print(f"# Reference-string consistency check (row['reference'] == REFERENCE_SEQUENCE[idx]): "
      f"{sum(rr['consistent_ref'] for rr in records)}/{n_tasks} consistent")

# ============================================================
print("\n" + "="*70)
print("SECTION 1: Improvement counts (sequence-based vs score-based)")
print("="*70)

n_improved_seq = sum(rr['improved_seq'] for rr in records)
n_improved_score = sum(rr['improved_score_tol'] for rr in records)
n_tied_score = sum(rr['tied_score_tol'] for rr in records)
n_worse_or_tied_seq = n_tasks - n_improved_seq

print(f"Total tasks present: {n_tasks}")
print(f"Improved by SEQUENCE-difference method: {n_improved_seq} ({100*n_improved_seq/n_tasks:.2f}%)")
print(f"Not improved (seq identical to init best) by SEQUENCE method: {n_worse_or_tied_seq} ({100*n_worse_or_tied_seq/n_tasks:.2f}%)")
print()
print(f"Improved by SCORE method (bo_mic < init_mic - 1e-6): {n_improved_score} ({100*n_improved_score/n_tasks:.2f}%)")
print(f"Tied by SCORE method (|bo_mic - init_mic| <= 1e-6): {n_tied_score} ({100*n_tied_score/n_tasks:.2f}%)")

# agreement check
mismatches = [rr for rr in records if rr['improved_seq'] != rr['improved_score_tol']]
print(f"\nAgreement between the two methods: {n_tasks - len(mismatches)}/{n_tasks} tasks agree")
if mismatches:
    print(f"  {len(mismatches)} MISMATCHES found. Examples (task_id, improved_seq, improved_score_tol, tied_score_tol, init_mic, bo_mic):")
    for rr in mismatches[:15]:
        print(f"    {rr['task_id']}: seq={rr['improved_seq']} score={rr['improved_score_tol']} tied={rr['tied_score_tol']} init={rr['init_mic']:.6f} bo={rr['bo_mic']:.6f}")
else:
    print("  No mismatches -- the two methods fully agree.")

# ============================================================
print("\n" + "="*70)
print("SECTION 2: Distribution of %MIC reduction for IMPROVED tasks (seq-based)")
print("="*70)

improved = [rr for rr in records if rr['improved_seq']]
pct_reductions = [(rr['init_mic'] - rr['bo_mic']) / rr['init_mic'] * 100 for rr in improved]

if pct_reductions:
    srt = sorted(pct_reductions)
    n = len(srt)
    med = srt[n//2] if n % 2 == 1 else (srt[n//2-1]+srt[n//2])/2
    print(f"n improved = {n}")
    print(f"min    = {min(srt):.4f}%")
    print(f"median = {med:.4f}%")
    print(f"mean   = {sum(srt)/n:.4f}%")
    print(f"max    = {max(srt):.4f}%")
    for thresh in [1,5,10,25,50]:
        c = sum(1 for x in pct_reductions if x >= thresh)
        print(f"  improved by >= {thresh:>2}%: {c} tasks ({100*c/n_tasks:.2f}% of all present tasks, {100*c/n:.2f}% of improved tasks)")
else:
    print("No improved tasks found.")

# ============================================================
print("\n" + "="*70)
print("SECTION 3: Overall summed / mean / median MIC, init vs post-BO")
print("="*70)

def summarize(recs, label):
    inits = [rr['init_mic'] for rr in recs]
    bos = [rr['bo_mic'] for rr in recs]
    n = len(recs)
    if n == 0:
        print(f"{label}: no tasks")
        return
    sum_init, sum_bo = sum(inits), sum(bos)
    pct_change = (sum_bo - sum_init) / sum_init * 100
    mean_init, mean_bo = sum_init/n, sum_bo/n
    si, sb = sorted(inits), sorted(bos)
    def med(a):
        m = len(a)
        return a[m//2] if m%2==1 else (a[m//2-1]+a[m//2])/2
    med_init, med_bo = med(si), med(sb)
    print(f"\n-- {label} (n={n}) --")
    print(f"  Summed init MIC    = {sum_init:.4f}")
    print(f"  Summed post-BO MIC = {sum_bo:.4f}")
    print(f"  Percent change (sum) = {pct_change:.4f}%")
    print(f"  Mean init MIC    = {mean_init:.4f}   |  Mean post-BO MIC    = {mean_bo:.4f}")
    print(f"  Median init MIC  = {med_init:.4f}   |  Median post-BO MIC  = {med_bo:.4f}")
    return dict(n=n, sum_init=sum_init, sum_bo=sum_bo, pct_change=pct_change,
                mean_init=mean_init, mean_bo=mean_bo, med_init=med_init, med_bo=med_bo)

overall_stats = summarize(records, "OVERALL (all present tasks)")

# ============================================================
print("\n" + "="*70)
print("SECTION 4: TEST split (pep_900-pep_999) vs TRAIN split (pep_000-pep_899)")
print("="*70)

test_recs = [rr for rr in records if 900 <= rr['idx'] <= 999]
train_recs = [rr for rr in records if 0 <= rr['idx'] <= 899]
other_recs = [rr for rr in records if not (0 <= rr['idx'] <= 999)]
if other_recs:
    print(f"NOTE: {len(other_recs)} tasks fall outside idx 0..999: {[r['task_id'] for r in other_recs]}")

test_stats = summarize(test_recs, "TEST split pep_900..pep_999")
train_stats = summarize(train_recs, "TRAIN split pep_000..pep_899")

# ============================================================
print("\n" + "="*70)
print("SECTION 5: pep_900..pep_919 (20 tasks) vs published reference 625.95")
print("="*70)

sub20 = [rr for rr in records if 900 <= rr['idx'] <= 919]
sub20_stats = summarize(sub20, "pep_900..pep_919")
if sub20_stats:
    ratio = sub20_stats['sum_bo'] / 625.95
    print(f"\n  Our summed post-BO MIC over these tasks = {sub20_stats['sum_bo']:.4f}")
    print(f"  Published reference summed MIC          = 625.95")
    print(f"  Ratio (ours / published)                 = {ratio:.4f}")
    print(f"  n tasks present in this subset out of 20 = {sub20_stats['n']}")
    missing20 = [f"pep_{i:03d}" for i in range(900,920) if f"pep_{i:03d}" not in rows]
    if missing20:
        print(f"  MISSING task_ids in this subset: {missing20}")

# ============================================================
print("\n" + "="*70)
print("SECTION 6: Breakdown by reference-length band")
print("="*70)

bands = [(10,12), (13,14), (15,17), (18,20), (21,25)]
for lo, hi in bands:
    band_recs = [rr for rr in records if lo <= rr['ref_len'] <= hi]
    n = len(band_recs)
    if n == 0:
        print(f"\nBand {lo}-{hi}: n=0 (no tasks present)")
        continue
    n_imp = sum(rr['improved_seq'] for rr in band_recs)
    pct_reds = [(rr['init_mic']-rr['bo_mic'])/rr['init_mic']*100 for rr in band_recs if rr['improved_seq']]
    mean_pct_red_improved_only = sum(pct_reds)/len(pct_reds) if pct_reds else float('nan')
    # also mean pct reduction over ALL tasks in band (treating non-improved as their actual, possibly 0 or negative, value)
    pct_reds_all = [(rr['init_mic']-rr['bo_mic'])/rr['init_mic']*100 for rr in band_recs]
    mean_pct_red_all = sum(pct_reds_all)/len(pct_reds_all)
    print(f"\nBand {lo}-{hi} aa: n={n}, improved={n_imp} ({100*n_imp/n:.2f}%)")
    print(f"  mean %reduction (over improved tasks only, n={len(pct_reds)}) = {mean_pct_red_improved_only:.4f}%")
    print(f"  mean %reduction (over ALL tasks in band, incl. non-improved) = {mean_pct_red_all:.4f}%")

# Spearman correlation: reference length vs improved (0/1), and vs pct reduction (all tasks)
def spearman(x, y):
    n = len(x)
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0]*len(v)
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[idx[j+1]] == v[idx[i]]:
                j += 1
            avg_rank = (i+j)/2 + 1
            for k in range(i, j+1):
                ranks[idx[k]] = avg_rank
            i = j+1
        return ranks
    rx, ry = rank(x), rank(y)
    mean_rx, mean_ry = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mean_rx)*(ry[i]-mean_ry) for i in range(n))
    denx = math.sqrt(sum((rx[i]-mean_rx)**2 for i in range(n)))
    deny = math.sqrt(sum((ry[i]-mean_ry)**2 for i in range(n)))
    if denx == 0 or deny == 0:
        return float('nan')
    return num/(denx*deny)

lengths = [rr['ref_len'] for rr in records]
improved_flags = [1 if rr['improved_seq'] else 0 for rr in records]
pct_reds_all_tasks = [(rr['init_mic']-rr['bo_mic'])/rr['init_mic']*100 for rr in records]

rho_improved = spearman(lengths, improved_flags)
rho_pctred = spearman(lengths, pct_reds_all_tasks)
print(f"\nSpearman correlation (ref length, improved 0/1)        = {rho_improved:.4f}   (n={len(records)})")
print(f"Spearman correlation (ref length, % MIC reduction)     = {rho_pctred:.4f}   (n={len(records)})")

try:
    from scipy.stats import spearmanr
    rho1, p1 = spearmanr(lengths, improved_flags)
    rho2, p2 = spearmanr(lengths, pct_reds_all_tasks)
    print(f"[scipy check] rho(length, improved) = {rho1:.4f}, p={p1:.4g}")
    print(f"[scipy check] rho(length, pct_red)   = {rho2:.4f}, p={p2:.4g}")
except ImportError:
    print("[scipy not available; using manual implementation above only]")

# ============================================================
print("\n" + "="*70)
print("SECTION 7: Tasks reaching MIC < 8")
print("="*70)

n_bo_lt8 = sum(1 for rr in records if rr['bo_mic'] < 8)
n_init_lt8 = sum(1 for rr in records if rr['init_mic'] < 8)
print(f"Tasks with post-BO MIC < 8:  {n_bo_lt8} / {n_tasks} ({100*n_bo_lt8/n_tasks:.2f}%)")
print(f"Tasks with init-pool best MIC < 8: {n_init_lt8} / {n_tasks} ({100*n_init_lt8/n_tasks:.2f}%)")
newly_crossed = sum(1 for rr in records if rr['bo_mic'] < 8 and rr['init_mic'] >= 8)
lost = sum(1 for rr in records if rr['bo_mic'] >= 8 and rr['init_mic'] < 8)
print(f"  (of which newly crossed under 8 by BO: {newly_crossed}; regressed above 8 after BO: {lost})")

# ============================================================
print("\n" + "="*70)
print("SECTION 8: Median MIC and mean log10(MIC), init vs post-BO")
print("="*70)

def med(a):
    s = sorted(a)
    m = len(s)
    return s[m//2] if m%2==1 else (s[m//2-1]+s[m//2])/2

def log_stats(recs, label):
    inits = [rr['init_mic'] for rr in recs]
    bos = [rr['bo_mic'] for rr in recs]
    n = len(recs)
    if n == 0:
        print(f"{label}: no tasks")
        return
    med_init, med_bo = med(inits), med(bos)
    # guard against non-positive values for log10
    bad_init = [x for x in inits if x <= 0]
    bad_bo = [x for x in bos if x <= 0]
    log_init = [math.log10(x) for x in inits if x > 0]
    log_bo = [math.log10(x) for x in bos if x > 0]
    mean_log_init = sum(log_init)/len(log_init) if log_init else float('nan')
    mean_log_bo = sum(log_bo)/len(log_bo) if log_bo else float('nan')
    print(f"\n-- {label} (n={n}) --")
    print(f"  Median MIC:        init={med_init:.4f}   post-BO={med_bo:.4f}")
    print(f"  Mean log10(MIC):   init={mean_log_init:.4f} (n={len(log_init)})   post-BO={mean_log_bo:.4f} (n={len(log_bo)})")
    if bad_init or bad_bo:
        print(f"  NOTE: excluded non-positive MIC values from log10: init={len(bad_init)}, post-BO={len(bad_bo)}")

log_stats(records, "OVERALL")
log_stats(test_recs, "TEST split pep_900..pep_999")

print("\n" + "="*70)
print("DONE")
print("="*70)
