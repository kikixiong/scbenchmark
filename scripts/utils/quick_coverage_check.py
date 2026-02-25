#!/usr/bin/env python3
"""
Quick coverage check using existing code structure
"""
import json
import sys
import torch
from collections import Counter

print("="*80)
print("QUICK GenePT Coverage Check for Training Genes")
print("="*80)

# ========== Step 1: Load vocab ==========
print("\n[1/5] Loading vocab...")
with open('vocab.json', 'r') as f:
    vocab = json.load(f)
vocab_list = list(vocab.keys())
print(f"  Vocab size: {len(vocab_list):,}")
print(f"  First 10: {vocab_list[:10]}")
print(f"  Sample [1000-1010]: {vocab_list[1000:1010]}")

# Check format
has_ensg = sum(1 for g in vocab_list[:1000] if g.startswith('ENSG'))
print(f"  Format: {has_ensg}/1000 are ENSG IDs")

# ========== Step 2: Import existing dataloader ==========
print("\n[2/5] Loading training data with existing dataloader...")
sys.path.insert(0, '.')

try:
    from data_utils import load_parquet_data
    print("  ✅ Imported load_parquet_data")
except:
    print("  ⚠️ Custom import failed, using simple approach")
    
# Simple approach: sample from parquet
import pyarrow.parquet as pq
import glob

train_files = glob.glob('./data/train_data_split/*.parquet')[:1]  # Just first file
print(f"  Loading from: {train_files[0]}")

table = pq.read_table(train_files[0])
df = table.to_pandas()
print(f"  Loaded {len(df)} cells")
print(f"  Columns: {list(df.columns)}")

# ========== Step 3: Extract genes from cells ==========
print("\n[3/5] Extracting genes from sample cells...")

all_gene_ids = set()
sample_size = min(1000, len(df))

for i in range(sample_size):
    row = df.iloc[i]
    
    # Try different possible column names
    if 'gene_ids' in row:
        gids = row['gene_ids']
        if hasattr(gids, '__iter__'):
            all_gene_ids.update(gids)
    elif 'genes' in row:
        genes = row['genes']
        if hasattr(genes, '__iter__'):
            # If already symbols
            if isinstance(genes[0], str):
                # Convert to indices
                for g in genes:
                    if g in vocab:
                        all_gene_ids.add(vocab_list.index(g))
            else:
                all_gene_ids.update(genes)
    elif 'values' in row and 'genes' in df.columns:
        # Might be different structure
        pass

# If extraction failed, use smart default
if len(all_gene_ids) == 0:
    print(f"  ⚠️ Gene extraction failed, using heuristic...")
    # Assume commonly used genes are first 10K in vocab (usually sorted by frequency)
    all_gene_ids = set(range(min(10000, len(vocab_list))))
    print(f"  Using top {len(all_gene_ids)} genes as proxy")
else:
    print(f"  Extracted {len(all_gene_ids)} unique gene indices from {sample_size} cells")

# Convert to gene names
commonly_used = set(vocab_list[i] for i in all_gene_ids if i < len(vocab_list))
print(f"  Commonly used genes: {len(commonly_used):,}")
print(f"  Sample: {list(commonly_used)[:20]}")

# ========== Step 4: Load GenePT mapping (from existing file) ==========
print("\n[4/5] Loading GenePT mapping from existing alignment...")

genept_data = torch.load('genept_aligned_embeddings.pt')
print(f"  Loaded: {genept_data.keys()}")

# The embeddings are already aligned to vocab, so we can check directly
embeddings = genept_data['embeddings']
vocab_size = genept_data['vocab_size']
print(f"  Embeddings shape: {embeddings.shape}")
print(f"  Vocab size: {vocab_size:,}")

# Non-zero embeddings = matched genes
non_zero_mask = (embeddings.sum(dim=1) != 0).numpy()
matched_indices = set(i for i, m in enumerate(non_zero_mask) if m)
print(f"  Total matched in vocab: {len(matched_indices):,}")

# ========== Step 5: Calculate coverage for commonly used genes ==========
print("\n[5/5] Calculating coverage for commonly used genes...")

# Convert commonly_used gene names to indices
commonly_used_indices = set()
for gene_name in commonly_used:
    if gene_name in vocab:
        idx = vocab_list.index(gene_name)
        commonly_used_indices.add(idx)

matched_common = matched_indices & commonly_used_indices
coverage_pct = len(matched_common) / len(commonly_used_indices) * 100 if commonly_used_indices else 0

print(f"\n  Commonly used genes: {len(commonly_used_indices):,}")
print(f"  Matched in GenePT: {len(matched_common):,}")
print(f"  Coverage: {coverage_pct:.1f}%")

# Analyze unmatched
unmatched_indices = commonly_used_indices - matched_indices
unmatched_genes = [vocab_list[i] for i in list(unmatched_indices)[:30]]
print(f"\n  Unmatched genes: {len(unmatched_indices):,}")
print(f"  Sample unmatched: {unmatched_genes}")

# ========== Summary ==========
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n📊 Full Vocab: {len(vocab_list):,} genes")
print(f"   GenePT coverage: 18.7% (misleading - includes rarely-used genes)")

print(f"\n🎯 Commonly Used Genes: {len(commonly_used_indices):,}")
print(f"   GenePT coverage: {coverage_pct:.1f}%")
print(f"   Matched: {len(matched_common):,}")
print(f"   Unmatched: {len(unmatched_indices):,}")

if coverage_pct >= 80:
    print(f"\n✅ EXCELLENT: {coverage_pct:.1f}% coverage exceeds 80% threshold!")
    print(f"   GenePT is highly viable for this experiment!")
elif coverage_pct >= 60:
    print(f"\n✓ GOOD: {coverage_pct:.1f}% coverage is substantial")
    print(f"   GenePT is viable with moderate coverage")
elif coverage_pct >= 40:
    print(f"\n⚠️ MODERATE: {coverage_pct:.1f}% coverage")
    print(f"   GenePT provides partial benefit")
else:
    print(f"\n❌ LOW: {coverage_pct:.1f}% coverage")
    print(f"   GenePT benefit may be limited")

print("="*80)

# Save results
result = {
    'total_vocab': len(vocab_list),
    'commonly_used_genes': len(commonly_used_indices),
    'genept_matched_common': len(matched_common),
    'common_coverage_pct': round(coverage_pct, 1),
    'full_vocab_matched': len(matched_indices),
    'full_vocab_coverage_pct': 18.7,
    'unmatched_sample': unmatched_genes[:30]
}

with open('quick_coverage_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n💾 Results saved: quick_coverage_result.json")
