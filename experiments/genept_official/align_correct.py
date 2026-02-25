#!/usr/bin/env python3
# 正确版本：使用完整映射链对齐GenePT embeddings

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path

print("="*60)
print("GenePT → scBench 正确对齐")
print("映射链: scBench ID → Symbol → Entrez ID → Embedding")
print("="*60)

# 1. 加载vocab.json（基因符号→scBench ID）
print("\n1. 加载vocab.json...")
with open("/root/autodl-tmp/scbenchmark/vocab.json", "r") as f:
    symbol_to_scbench_id = json.load(f)

# 创建反向映射：scBench ID → 基因符号
scbench_id_to_symbol = {v: k for k, v in symbol_to_scbench_id.items()}
print(f"   Vocab size: {len(symbol_to_scbench_id)}")
print(f"   scBench ID范围: {min(symbol_to_scbench_id.values())} ~ {max(symbol_to_scbench_id.values())}")
print(f"   示例: ID 3 → {scbench_id_to_symbol.get(3, 'Unknown')}")
print(f"   示例: ID 57 → {scbench_id_to_symbol.get(57, 'Unknown')}")

# 2. 加载mygene映射（Entrez ID → 基因符号）
print("\n2. 加载mygene映射...")
with open("/root/autodl-tmp/genept_official_experiment/gene_id_mapping.json", "r") as f:
    mygene_data = json.load(f)

entrez_to_symbol = mygene_data["entrez_to_symbol"]

# 创建反向映射：基因符号(大写) → Entrez ID
symbol_to_entrez = {}
for entrez_id, symbol in entrez_to_symbol.items():
    symbol_upper = symbol.upper()
    # 处理一对多情况（保留第一个）
    if symbol_upper not in symbol_to_entrez:
        symbol_to_entrez[symbol_upper] = int(entrez_id)

print(f"   Entrez→Symbol映射: {len(entrez_to_symbol)}")
print(f"   Symbol→Entrez映射: {len(symbol_to_entrez)}")

# 3. 加载GenePT embeddings
print("\n3. 加载GenePT embeddings...")
genept_dir = Path("/root/autodl-tmp/genept_official_experiment/intersect/GENEPT-ADA")
gene_list_df = pd.read_csv(genept_dir / "GENEPT-ADA_genelist.txt", header=None)
genept_entrez_ids = gene_list_df[0].astype(int).tolist()
embeddings = pd.read_csv(genept_dir / "GENEPT-ADA_emb.csv", header=None).values

entrez_to_emb = {entrez_id: embeddings[i] for i, entrez_id in enumerate(genept_entrez_ids)}
print(f"   GenePT genes: {len(genept_entrez_ids)}")
print(f"   Embedding dim: {embeddings.shape[1]}")

# 4. 构建完整映射链：scBench ID → Entrez ID → Embedding
print("\n4. 构建映射链...")
scbench_id_to_emb = {}
chain_success = 0
chain_fail_at_symbol = 0
chain_fail_at_entrez = 0
chain_fail_at_embedding = 0

for scbench_id, symbol in scbench_id_to_symbol.items():
    symbol_upper = symbol.upper()
    
    # 步骤1: Symbol → Entrez
    if symbol_upper not in symbol_to_entrez:
        chain_fail_at_symbol += 1
        continue
    
    entrez_id = symbol_to_entrez[symbol_upper]
    
    # 步骤2: Entrez → Embedding
    if entrez_id not in entrez_to_emb:
        chain_fail_at_embedding += 1
        continue
    
    scbench_id_to_emb[scbench_id] = entrez_to_emb[entrez_id]
    chain_success += 1

print(f"   ✅ 成功映射: {chain_success}/{len(scbench_id_to_symbol)}")
print(f"   ❌ Symbol未找到: {chain_fail_at_symbol}")
print(f"   ❌ Embedding未找到: {chain_fail_at_embedding}")
print(f"   总成功率: {100*chain_success/len(scbench_id_to_symbol):.1f}%")

# 5. 为scBench数据集创建对齐embeddings
print("\n5. 对齐到scBench数据集...")
datasets = ["Myeloid", "pancread", "lupus", "scfoundation", "Multiple_Sclerosis"]
data_base = Path("/root/autodl-tmp/scbenchmark/data/downstreams/classification/processed_data")

alignment_stats = {}
embedding_dim = embeddings.shape[1]

for dataset_name in datasets:
    print(f"\n  {dataset_name}:")
    
    pt_file = data_base / f"{dataset_name}_data.pt"
    if not pt_file.exists():
        print(f"    ⚠️ 文件不存在")
        continue
    
    # 加载数据
    data = torch.load(pt_file, map_location='cpu', weights_only=False)
    genes_list = data['genes']
    
    # 获取唯一基因ID
    all_gene_ids = set()
    for gene_arr in genes_list:
        all_gene_ids.update(gene_arr.tolist())
    all_gene_ids = sorted(list(all_gene_ids))
    
    # 对齐
    aligned_embeddings = []
    matched_count = 0
    missing_ids = []
    
    for gene_id in all_gene_ids:
        if gene_id in scbench_id_to_emb:
            aligned_embeddings.append(scbench_id_to_emb[gene_id])
            matched_count += 1
        else:
            aligned_embeddings.append(np.zeros(embedding_dim))
            if len(missing_ids) < 10:
                symbol = scbench_id_to_symbol.get(gene_id, f"Unknown(ID={gene_id})")
                missing_ids.append((gene_id, symbol))
    
    aligned_tensor = torch.tensor(np.array(aligned_embeddings), dtype=torch.float32)
    match_rate = 100 * matched_count / len(all_gene_ids)
    
    print(f"    唯一基因: {len(all_gene_ids)}")
    print(f"    ✅ 匹配: {matched_count}/{len(all_gene_ids)} ({match_rate:.1f}%)")
    if missing_ids:
        print(f"    未匹配示例:")
        for gid, sym in missing_ids[:5]:
            print(f"      ID {gid} ({sym})")
    
    # 保存
    save_path = f"/root/autodl-tmp/genept_official_experiment/genept_aligned_{dataset_name}_FINAL.pt"
    save_data = {
        'embeddings': aligned_tensor,
        'gene_ids': all_gene_ids,
        'matched_count': matched_count,
        'total_genes': len(all_gene_ids)
    }
    torch.save(save_data, save_path)
    print(f"    💾 已保存: {save_path.split('/')[-1]}")
    
    alignment_stats[dataset_name] = {
        "total_genes": len(all_gene_ids),
        "matched": matched_count,
        "match_rate": round(match_rate, 2),
        "file": save_path
    }

# 保存统计
with open("/root/autodl-tmp/genept_official_experiment/alignment_stats_FINAL.json", "w") as f:
    json.dump(alignment_stats, f, indent=2)

print(f"\n{'='*60}")
print("✅ 最终对齐完成！")
print(f"{'='*60}")
print(f"\n{'Dataset':<20} {'基因数':<10} {'匹配':<10} {'匹配率':<10}")
print("-" * 60)
for dataset, stats in alignment_stats.items():
    print(f"{dataset:<20} {stats['total_genes']:<10} {stats['matched']:<10} {stats['match_rate']:.1f}%")
