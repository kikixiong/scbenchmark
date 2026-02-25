#!/usr/bin/env python3
# Step 4: 验证对齐质量

import torch

datasets = ["Myeloid", "pancread", "lupus", "scfoundation", "Multiple_Sclerosis"]

print("="*60)
print("Step 4: 验证对齐质量")
print("="*60)

for dataset in datasets:
    emb_path = f"/root/autodl-tmp/genept_official_experiment/genept_mapped_{dataset}.pt"
    
    try:
        emb = torch.load(emb_path, map_location='cpu')
        
        zero_count = (emb.sum(dim=1) == 0).sum().item()
        zero_rate = 100 * zero_count / emb.shape[0]
        
        print(f"\n{dataset}:")
        print(f"  Shape: {emb.shape}")
        print(f"  Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
        print(f"  Zero vectors: {zero_count}/{emb.shape[0]} ({zero_rate:.1f}%)")
        
    except Exception as e:
        print(f"\n{dataset}: ❌ Error - {e}")

print("\n" + "="*60)
print("✅ 验证完成！")
print("="*60)
