#!/usr/bin/env python3
# Step 2: 使用mygene进行基因ID映射

import pandas as pd
import json
import mygene

print("="*60)
print("Step 2: 使用mygene进行基因ID映射")
print("="*60)

# 1. 加载GenePT的Entrez Gene IDs
genept_dir = "/root/autodl-tmp/genept_official_experiment/intersect/GENEPT-ADA"
genept_genes_df = pd.read_csv(f"{genept_dir}/GENEPT-ADA_genelist.txt", header=None)
genept_entrez_ids = genept_genes_df[0].astype(str).tolist()

print(f"✅ GenePT Entrez IDs: {len(genept_entrez_ids)} genes")
print(f"   示例: {genept_entrez_ids[:10]}")

# 2. 使用mygene查询基因符号
mg = mygene.MyGeneInfo()

print("\n🔍 查询基因符号（这可能需要几分钟）...")
results = mg.querymany(
    genept_entrez_ids, 
    scopes='entrezgene', 
    fields='symbol,ensembl.gene',
    species='human',
    returnall=True
)

# 3. 构建映射字典
entrez_to_symbol = {}
entrez_to_ensembl = {}
missing_count = 0

for item in results['out']:
    entrez = str(item.get('query', ''))
    symbol = item.get('symbol', None)
    ensembl = None
    
    if 'ensembl' in item:
        if isinstance(item['ensembl'], list):
            ensembl = item['ensembl'][0].get('gene', None) if item['ensembl'] else None
        else:
            ensembl = item['ensembl'].get('gene', None)
    
    if symbol:
        entrez_to_symbol[entrez] = symbol
    if ensembl:
        entrez_to_ensembl[entrez] = ensembl
    
    if not symbol and not ensembl:
        missing_count += 1

print(f"\n✅ 映射完成:")
print(f"   Entrez → Symbol: {len(entrez_to_symbol)} genes")
print(f"   Entrez → Ensembl: {len(entrez_to_ensembl)} genes")
print(f"   未匹配: {missing_count} genes")

# 4. 保存映射表
mapping_data = {
    "entrez_to_symbol": entrez_to_symbol,
    "entrez_to_ensembl": entrez_to_ensembl,
    "total_genept_genes": len(genept_entrez_ids),
    "mapped_to_symbol": len(entrez_to_symbol),
    "mapped_to_ensembl": len(entrez_to_ensembl),
    "missing": missing_count
}

with open("/root/autodl-tmp/genept_official_experiment/gene_id_mapping.json", "w") as f:
    json.dump(mapping_data, f, indent=2)

print(f"\n✅ 映射表已保存到: gene_id_mapping.json")
