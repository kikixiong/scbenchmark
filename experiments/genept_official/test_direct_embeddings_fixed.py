#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import json

print("="*60)
print("实验1: 直接用GenePT对齐Embeddings（不fusion）")
print("="*60)

datasets = ["Myeloid", "pancread", "lupus", "scfoundation", "Multiple_Sclerosis"]
results = []

for dataset_name in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # 1. 加载GenePT对齐的embeddings
    emb_path = f"/root/autodl-tmp/genept_official_experiment/genept_aligned_{dataset_name}_FINAL.pt"
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    gene_embeddings = emb_data['embeddings']  # [n_genes, 1536]
    gene_ids = emb_data['gene_ids']  # 基因ID列表
    
    # 创建gene_id到embedding索引的映射
    gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
    
    print(f"✅ GenePT embeddings loaded: {gene_embeddings.shape}")
    print(f"   Matched: {emb_data['matched_count']}/{emb_data['total_genes']}")
    
    # 检查零向量
    zero_mask = (gene_embeddings.sum(dim=1) == 0)
    zero_count = zero_mask.sum().item()
    zero_rate = 100*zero_count/gene_embeddings.shape[0]
    print(f"   Zero vectors: {zero_count}/{gene_embeddings.shape[0]} ({zero_rate:.1f}%)")
    
    # 2. 加载scBench数据（稀疏格式）
    data_path = f"/root/autodl-tmp/scbenchmark/data/downstreams/classification/processed_data/{dataset_name}_data.pt"
    data_dict = torch.load(data_path, map_location='cpu', weights_only=False)
    
    cell_genes_list = data_dict['genes']  # list of numpy arrays (gene IDs per cell)
    cell_expressions_list = data_dict['expressions']  # list of numpy arrays (expression values per cell)
    cell_labels = data_dict['cls_name']  # list of labels
    
    print(f"✅ Data loaded: {len(cell_labels)} cells")
    
    # 3. 转换标签为数值
    unique_labels = sorted(list(set(cell_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_numeric = torch.tensor([label_to_idx[label] for label in cell_labels])
    print(f"   Classes: {len(unique_labels)}")
    
    # 4. 对每个细胞：聚合gene_embeddings成cell embedding
    print("   Aggregating cell embeddings...")
    
    cell_embeddings_list = []
    for i in range(len(cell_labels)):
        if i % 2000 == 0:
            print(f"     Processing cell {i}/{len(cell_labels)}...")
        
        # 获取该细胞的基因ID和表达值
        cell_gene_ids = cell_genes_list[i]  # numpy array
        cell_expr = cell_expressions_list[i]  # numpy array
        
        # 构建该细胞的embedding向量（用零向量初始化）
        cell_emb = torch.zeros(1536, dtype=torch.float32)
        
        # 对每个表达的基因，累加加权embedding
        total_weight = 0.0
        for gene_id, expr_val in zip(cell_gene_ids, cell_expr):
            if gene_id in gene_id_to_idx:
                emb_idx = gene_id_to_idx[gene_id]
                cell_emb += float(expr_val) * gene_embeddings[emb_idx]
                total_weight += float(expr_val)
        
        # 归一化
        if total_weight > 0:
            cell_emb /= total_weight
        
        cell_embeddings_list.append(cell_emb.numpy())
    
    cell_embeddings = torch.tensor(np.array(cell_embeddings_list), dtype=torch.float32)
    print(f"✅ Cell embeddings: {cell_embeddings.shape}")
    
    # 5. 运行5次trials
    for trial in range(5):
        print(f"\n  Trial {trial+1}/5 (seed={1000+trial})")
        
        # 设置seed
        torch.manual_seed(1000 + trial)
        np.random.seed(1000 + trial)
        
        # 70/30 split
        n_train = int(0.7 * len(cell_embeddings))
        n_test = len(cell_embeddings) - n_train
        
        dataset = TensorDataset(cell_embeddings, labels_numeric)
        train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # 6. 训练Linear classifier
        n_classes = len(unique_labels)
        classifier = nn.Linear(1536, n_classes)
        optimizer = optim.Adam(classifier.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(50):
            # Train
            classifier.train()
            for batch_emb, batch_labels in train_loader:
                optimizer.zero_grad()
                logits = classifier(batch_emb)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
            
            # Eval every 5 epochs
            if (epoch + 1) % 5 == 0:
                classifier.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_emb, batch_labels in test_loader:
                        logits = classifier(batch_emb)
                        _, predicted = torch.max(logits, 1)
                        total += batch_labels.size(0)
                        correct += (predicted == batch_labels).sum().item()
                
                acc = 100 * correct / total
                if acc > best_acc:
                    best_acc = acc
        
        print(f"    Best accuracy: {best_acc:.2f}%")
        
        results.append({
            "dataset": dataset_name,
            "method": "direct_genept",
            "trial": trial,
            "accuracy": best_acc,
            "zero_vector_rate": zero_rate
        })

# 保存结果
output_path = "/root/autodl-tmp/genept_official_experiment/direct_embeddings_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print(f"✅ 实验1完成！结果保存到: {output_path}")
print("="*60)

# 打印汇总
for dataset in datasets:
    trials = [r for r in results if r['dataset'] == dataset]
    accs = [r['accuracy'] for r in trials]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"{dataset}: {mean_acc:.2f}±{std_acc:.2f}%")
