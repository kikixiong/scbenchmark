#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import json

print("="*60)
print("实验2: GenePT对齐Embeddings + 4种Fusion方法")
print("="*60)

datasets = ["Myeloid", "pancread", "lupus", "scfoundation", "Multiple_Sclerosis"]
fusion_methods = ["weighted_avg", "concatenation", "gating", "attention"]
results = []

for dataset_name in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载GenePT embeddings
    emb_path = f"/root/autodl-tmp/genept_official_experiment/genept_aligned_{dataset_name}_FINAL.pt"
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    gene_embeddings = emb_data['embeddings']
    gene_ids = emb_data['gene_ids']
    gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
    
    # 加载数据
    data_path = f"/root/autodl-tmp/scbenchmark/data/downstreams/classification/processed_data/{dataset_name}_data.pt"
    data_dict = torch.load(data_path, map_location='cpu', weights_only=False)
    cell_genes_list = data_dict['genes']
    cell_expressions_list = data_dict['expressions']
    cell_labels = data_dict['cls_name']
    
    # 转换标签
    unique_labels = sorted(list(set(cell_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_numeric = torch.tensor([label_to_idx[label] for label in cell_labels])
    
    for fusion_method in fusion_methods:
        print(f"\n--- {dataset_name} | {fusion_method} ---")
        
        # 聚合cell embeddings
        cell_embeddings_list = []
        for i in range(len(cell_labels)):
            cell_gene_ids = cell_genes_list[i]
            cell_expr = torch.tensor(cell_expressions_list[i], dtype=torch.float32)
            
            # 获取该细胞表达的基因的embeddings
            expressed_embeddings = []
            expressed_expr = []
            for idx, gene_id in enumerate(cell_gene_ids):
                if gene_id in gene_id_to_idx:
                    emb_idx = gene_id_to_idx[gene_id]
                    expressed_embeddings.append(gene_embeddings[emb_idx])
                    expressed_expr.append(cell_expr[idx])
            
            if len(expressed_embeddings) == 0:
                # 如果没有匹配的基因，用零向量
                cell_emb = torch.zeros(1536 if fusion_method != "concatenation" else 1540, dtype=torch.float32)
            else:
                expressed_embeddings = torch.stack(expressed_embeddings)  # [n_expressed, 1536]
                expressed_expr = torch.tensor(expressed_expr, dtype=torch.float32)
                
                # 应用fusion方法
                if fusion_method == "weighted_avg":
                    weights = expressed_expr / (expressed_expr.sum() + 1e-8)
                    cell_emb = (expressed_embeddings.T @ weights)
                
                elif fusion_method == "concatenation":
                    weights = expressed_expr / (expressed_expr.sum() + 1e-8)
                    avg_emb = expressed_embeddings.T @ weights
                    stats = torch.tensor([expressed_expr.mean(), expressed_expr.std(), expressed_expr.max(), float(len(expressed_expr))])
                    cell_emb = torch.cat([avg_emb, stats])
                
                elif fusion_method == "gating":
                    expr_norm = (expressed_expr - expressed_expr.mean()) / (expressed_expr.std() + 1e-8)
                    gates = torch.sigmoid(expr_norm)
                    gated_emb = expressed_embeddings * gates.unsqueeze(1)
                    cell_emb = gated_emb.mean(dim=0)
                
                elif fusion_method == "attention":
                    attn_weights = torch.softmax(expressed_expr, dim=0)
                    cell_emb = expressed_embeddings.T @ attn_weights
            
            cell_embeddings_list.append(cell_emb.numpy())
        
        cell_embeddings = torch.tensor(np.array(cell_embeddings_list), dtype=torch.float32)
        emb_dim = cell_embeddings.shape[1]
        print(f"  Embedding dim: {emb_dim}")
        
        # 5 trials
        trial_accs = []
        for trial in range(5):
            torch.manual_seed(1000 + trial)
            np.random.seed(1000 + trial)
            
            n_train = int(0.7 * len(cell_embeddings))
            dataset_obj = TensorDataset(cell_embeddings, labels_numeric)
            train_dataset, test_dataset = random_split(dataset_obj, [n_train, len(dataset_obj) - n_train])
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64)
            
            n_classes = len(unique_labels)
            classifier = nn.Linear(emb_dim, n_classes)
            optimizer = optim.Adam(classifier.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()
            
            best_acc = 0
            for epoch in range(50):
                classifier.train()
                for batch_emb, batch_labels in train_loader:
                    optimizer.zero_grad()
                    logits = classifier(batch_emb)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                
                if (epoch + 1) % 5 == 0:
                    classifier.eval()
                    correct = total = 0
                    with torch.no_grad():
                        for batch_emb, batch_labels in test_loader:
                            logits = classifier(batch_emb)
                            _, predicted = torch.max(logits, 1)
                            total += batch_labels.size(0)
                            correct += (predicted == batch_labels).sum().item()
                    
                    acc = 100 * correct / total
                    best_acc = max(best_acc, acc)
            
            trial_accs.append(best_acc)
            print(f"  Trial {trial+1}: {best_acc:.2f}%")
        
        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        print(f"  ✅ Result: {mean_acc:.2f}±{std_acc:.2f}%")
        
        results.append({
            "dataset": dataset_name,
            "fusion": fusion_method,
            "accuracies": trial_accs,
            "mean": mean_acc,
            "std": std_acc
        })

# 保存结果
output_path = "/root/autodl-tmp/genept_official_experiment/fusion_results_official.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print(f"✅ 实验2完成！结果保存到: {output_path}")
print("="*60)
