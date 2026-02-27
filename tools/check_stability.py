#!/usr/bin/env python
"""
Cross-validate our instability index implementation vs Biopython.
Also checks the generated_stable.fasta distribution against an independent library.
"""
import sys, os
# Allow running from tools/ subdir or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from peptidegen.evaluation.stability import calculate_instability_index

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    biopython_ok = True
except ImportError:
    biopython_ok = False
    print("WARNING: Biopython not installed. Run: pip install biopython")

# ───────────────────────────────────────────────
# 1. Known reference sequences with literature II
# ───────────────────────────────────────────────
test_seqs = [
    # (name, sequence, approx_expected_II)
    ("ACTH(1-13)",          "SYSMEHFRWGKPV",           None),
    ("Insulin A-chain",     "GIVEQCCTSICSLYQLENYCN",    None),
    ("All-Ala (very stable)","AAAAAAAAAA",               None),
    ("Gen sample 1",        "GKKHKKKWEKDIKCNNKE",        None),
    ("Gen sample 2",        "KKKLEV",                    None),
    ("Gen sample 3",        "YVLKKKKDFDQWD",             None),
    ("Gen sample 4",        "SCGWGLEKWFYDLYKCNVMMKCAFWIVVTKK", None),
]

print("=" * 72)
print("INSTABILITY INDEX CROSS-VALIDATION")
print("=" * 72)
header = f"{'Name':<28} {'Sequence':<34} {'Our II':>8}"
if biopython_ok:
    header += f" {'Bio II':>8} {'Delta':>7} {'Match':>6}"
print(header)
print("-" * 72)

mismatches = 0
for name, seq, _ in test_seqs:
    our_ii = calculate_instability_index(seq)
    row = f"{name:<28} {seq:<34} {our_ii:>8.2f}"
    if biopython_ok:
        try:
            bio_ii = ProteinAnalysis(seq).instability_index()
            delta  = abs(our_ii - bio_ii)
            match  = delta < 1.0   # allow rounding tolerance
            if not match:
                mismatches += 1
            row += f" {bio_ii:>8.2f} {delta:>7.2f} {'OK' if match else 'FAIL':>6}"
        except Exception as e:
            row += f"  (bio error: {e})"
    print(row)

if biopython_ok:
    print()
    if mismatches == 0:
        print(f"PASS: all {len(test_seqs)} sequences match Biopython (delta < 1.0)")
    else:
        print(f"FAIL: {mismatches}/{len(test_seqs)} sequences disagree with Biopython")

# ───────────────────────────────────────────────
# 2. Distribution check on generated_stable.fasta
# ───────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
fasta = os.path.join(PROJECT_ROOT, "results/generated_stable.fasta")
if not os.path.exists(fasta):
    print(f"\nSkipping distribution check: {fasta} not found.")
    sys.exit(0)

sequences = []
with open(fasta) as f:
    for line in f:
        line = line.strip()
        if not line.startswith(">") and line:
            sequences.append(line.upper())

ii_values = [calculate_instability_index(s) for s in sequences]
stable_ours = sum(1 for v in ii_values if v < 40)

print()
print("=" * 72)
print(f"DISTRIBUTION CHECK on {fasta} (n={len(sequences)})")
print("=" * 72)
print(f"  Our II < 40    : {stable_ours}/{len(sequences)} = {stable_ours/len(sequences)*100:.1f}%")

if biopython_ok:
    # Sample first 200 seqs for speed
    sample = sequences[:200]
    bio_ii_vals = []
    for seq in sample:
        try:
            bio_ii_vals.append(ProteinAnalysis(seq).instability_index())
        except Exception:
            pass
    bio_stable = sum(1 for v in bio_ii_vals if v < 40)
    print(f"  Bio II < 40    : {bio_stable}/{len(sample)} = {bio_stable/len(sample)*100:.1f}%  (sampled first {len(sample)})")
    import statistics
    print(f"  Our  mean II   : {statistics.mean(ii_values):.2f}  std: {statistics.stdev(ii_values):.2f}")
    print(f"  Bio  mean II   : {statistics.mean(bio_ii_vals):.2f}  std: {statistics.stdev(bio_ii_vals):.2f}")
else:
    import statistics
    print(f"  Mean II        : {statistics.mean(ii_values):.2f}  std: {statistics.stdev(ii_values):.2f}")
    print(f"  Min/Max II     : {min(ii_values):.2f} / {max(ii_values):.2f}")

print()
# Histogram of II distribution
buckets = [0, 10, 20, 30, 40, 50, 60, 80, float("inf")]
labels  = ["0-10","10-20","20-30","30-40","40-50","50-60","60-80","80+"]
counts  = [0]*len(labels)
for v in ii_values:
    for i in range(len(buckets)-1):
        if buckets[i] <= v < buckets[i+1]:
            counts[i] += 1
            break
print("  II Distribution:")
for label, count in zip(labels, counts):
    bar = "#" * (count * 40 // max(counts, default=1))
    print(f"    {label:>6}: {bar:<40} {count}")
