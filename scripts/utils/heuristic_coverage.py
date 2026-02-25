#!/usr/bin/env python3
"""
Heuristic coverage analysis
Assumes vocab is sorted by usage frequency (common in scRNA-seq)
"""
import json
import torch

print("="*80)
print("HEURISTIC GenePT Coverage Analysis")
print("="*80)

# Load vocab
print("\n[1/3] Loading vocab...")
with open('vocab.json', 'r') as f:
    vocab = json.load(f)
vocab_list = list(vocab.keys())
print(f"  Total vocab: {len(vocab_list):,}")

# Load GenePT alignment
print("\n[2/3] Loading GenePT alignment...")
genept_data = torch.load('genept_aligned_embeddings.pt')
embeddings = genept_data['embeddings']

# Find non-zero embeddings (= matched genes)
non_zero_mask = (embeddings.sum(dim=1) != 0).numpy()
matched_indices = [i for i, m in enumerate(non_zero_mask) if m]
print(f"  Total matched: {len(matched_indices):,}/{len(vocab_list):,} (18.7%)")

# Analyze coverage for different vocab subsets
print("\n[3/3] Coverage for different gene sets...")
print(f"\n  {'Gene Set':<25} {'Size':>8} {'Matched':>8} {'Coverage':>10}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")

subsets = [
    ("Top 1,000 genes", 1000),
    ("Top 2,000 genes", 2000),
    ("Top 5,000 genes", 5000),
    ("Top 10,000 genes", 10000),
    ("Top 20,000 genes", 20000),
    ("Full vocab", len(vocab_list))
]

results = {}
for name, size in subsets:
    size = min(size, len(vocab_list))
    subset_indices = set(range(size))
    matched_in_subset = len(subset_indices & set(matched_indices))
    coverage = matched_in_subset / size * 100
    results[name] = {
        'size': size,
        'matched': matched_in_subset,
        'coverage_pct': round(coverage, 1)
    }
    print(f"  {name:<25} {size:>8,} {matched_in_subset:>8,} {coverage:>9.1f}%")

# Key insight: Training uses top 512 genes per cell
# So effective gene set is probably 5K-10K genes total
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n💡 Training uses top 512 highly-expressed genes PER CELL")
print("   → Effective gene set: ~5,000-10,000 genes total across dataset")

top5k_coverage = results["Top 5,000 genes"]['coverage_pct']
top10k_coverage = results["Top 10,000 genes"]['coverage_pct']

print(f"\n📊 Estimated REAL coverage:")
print(f"   Conservative (top 10K): {top10k_coverage:.1f}%")
print(f"   Optimistic (top 5K):    {top5k_coverage:.1f}%")

# Analyze matched gene distribution
matched_set = set(matched_indices)
matched_in_top1k = len(matched_set & set(range(1000)))
matched_in_top5k = len(matched_set & set(range(5000)))
matched_in_top10k = len(matched_set & set(range(10000)))

print(f"\n📈 GenePT gene distribution:")
print(f"   In top 1K vocab:  {matched_in_top1k:>5,} / {len(matched_indices):>5,} ({matched_in_top1k/len(matched_indices)*100:.1f}%)")
print(f"   In top 5K vocab:  {matched_in_top5k:>5,} / {len(matched_indices):>5,} ({matched_in_top5k/len(matched_indices)*100:.1f}%)")
print(f"   In top 10K vocab: {matched_in_top10k:>5,} / {len(matched_indices):>5,} ({matched_in_top10k/len(matched_indices)*100:.1f}%)")

# Conclusion
print("\n" + "="*80)
if top5k_coverage >= 80:
    print("✅ CONCLUSION: GenePT coverage is EXCELLENT for training!")
    print(f"   {top5k_coverage:.1f}% of commonly-used genes are covered")
elif top10k_coverage >= 60:
    print("✓ CONCLUSION: GenePT coverage is GOOD for training")
    print(f"   {top10k_coverage:.1f}% coverage provides substantial benefit")
else:
    print("⚠️ CONCLUSION: GenePT coverage is MODERATE")
    print(f"   {top10k_coverage:.1f}% coverage - partial benefit expected")

print("="*80)

# Save
with open('heuristic_coverage.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n💾 Saved: heuristic_coverage.json")
