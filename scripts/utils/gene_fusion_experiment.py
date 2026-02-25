#!/usr/bin/env python
"""
Gene Embedding Fusion Experiment v2
Handles sparse gene expression data format
Optimized for 32-core CPU parallel processing
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Setup logging
OUTPUT_DIR = Path("/root/autodl-tmp/gene-embedding-fusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Baseline metrics for comparison
BASELINE_METRICS = {
    'Myeloid': {'KNN': 40.82, 'Cls': 38.62},
    'pancread': {'KNN': 74.22, 'Cls': 76.24},
    'lupus': {'KNN': 48.86, 'Cls': 59.77}
}

# Dataset paths
DATASETS = {
    'Myeloid': 'data/downstreams/classification/processed_data/Myeloid_data.pt',
    'pancread': 'data/downstreams/classification/processed_data/pancread_data.pt',
    'lupus': 'data/downstreams/classification/processed_data/lupus_data.pt'
}

MODEL_PATH = 'save_pretrain/difference_aligned_v3/best_model.pt'


class GeneEmbeddingFusion:
    """Gene embedding fusion strategies for sparse gene expression data"""
    
    def __init__(self, gene_embeddings):
        """
        Args:
            gene_embeddings: Tensor of shape (n_genes, embedding_dim)
        """
        self.gene_embeddings = gene_embeddings
        self.embedding_dim = gene_embeddings.shape[1]
        self.n_genes = gene_embeddings.shape[0]
        
    def weighted_avg(self, gene_indices, expression_values):
        """
        Weighted average: Σ(expr_i * emb_i) / Σ(expr_i)
        
        Args:
            gene_indices: list of numpy arrays, each containing gene indices for one cell
            expression_values: list of numpy arrays, each containing expression values
        Returns:
            cell_embeddings: Tensor of shape (n_cells, embedding_dim)
        """
        cell_embs = []
        
        for genes, exprs in zip(gene_indices, expression_values):
            if len(genes) == 0:
                # Empty cell - use zero vector
                cell_embs.append(torch.zeros(self.embedding_dim))
                continue
            
            # Get embeddings for expressed genes
            gene_embs = self.gene_embeddings[genes]  # (n_expressed_genes, emb_dim)
            
            # Weighted average
            expr_tensor = torch.from_numpy(exprs).float()
            weights = expr_tensor / (expr_tensor.sum() + 1e-8)
            
            cell_emb = (weights.unsqueeze(1) * gene_embs).sum(dim=0)
            cell_embs.append(cell_emb)
        
        return torch.stack(cell_embs)
    
    def concatenation(self, gene_indices, expression_values):
        """
        Concatenation: [weighted_avg; stats(mean, std, max, n_expressed)]
        """
        # Get weighted average
        weighted = self.weighted_avg(gene_indices, expression_values)
        
        # Compute statistics
        stats_list = []
        for exprs in expression_values:
            if len(exprs) == 0:
                stats = torch.tensor([0.0, 0.0, 0.0, 0.0])
            else:
                mean_expr = exprs.mean()
                std_expr = exprs.std() if len(exprs) > 1 else 0.0
                max_expr = exprs.max()
                n_expressed = float(len(exprs))
                stats = torch.tensor([mean_expr, std_expr, max_expr, n_expressed])
            stats_list.append(stats)
        
        stats_tensor = torch.stack(stats_list)
        
        # Concatenate
        cell_emb = torch.cat([weighted, stats_tensor], dim=1)
        return cell_emb
    
    def gating(self, gene_indices, expression_values):
        """
        Gating: mean(sigmoid(normalize(expr)) * emb)
        """
        cell_embs = []
        
        for genes, exprs in zip(gene_indices, expression_values):
            if len(genes) == 0:
                cell_embs.append(torch.zeros(self.embedding_dim))
                continue
            
            # Get embeddings
            gene_embs = self.gene_embeddings[genes]
            
            # Normalize expression
            expr_tensor = torch.from_numpy(exprs).float()
            if len(exprs) > 1:
                expr_norm = (expr_tensor - expr_tensor.mean()) / (expr_tensor.std() + 1e-8)
            else:
                expr_norm = expr_tensor
            
            # Apply sigmoid gating
            gates = torch.sigmoid(expr_norm)
            
            # Gated embeddings
            gated = gates.unsqueeze(1) * gene_embs
            cell_emb = gated.mean(dim=0)
            cell_embs.append(cell_emb)
        
        return torch.stack(cell_embs)
    
    def attention(self, gene_indices, expression_values):
        """
        Attention: Σ(softmax(expr) * emb)
        """
        cell_embs = []
        
        for genes, exprs in zip(gene_indices, expression_values):
            if len(genes) == 0:
                cell_embs.append(torch.zeros(self.embedding_dim))
                continue
            
            # Get embeddings
            gene_embs = self.gene_embeddings[genes]
            
            # Softmax attention over expression
            expr_tensor = torch.from_numpy(exprs).float()
            attention = torch.softmax(expr_tensor, dim=0)
            
            # Attention-weighted sum
            cell_emb = (attention.unsqueeze(1) * gene_embs).sum(dim=0)
            cell_embs.append(cell_emb)
        
        return torch.stack(cell_embs)


def extract_gene_embeddings(model_path):
    """Extract gene embedding matrix from pretrained model"""
    logger.info(f"Loading model from {model_path}")
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Direct access - state dict is the OrderedDict itself
    if 'module.embedding.weight' in state_dict:
        gene_emb = state_dict['module.embedding.weight']
        logger.info(f"Found gene embeddings: shape={gene_emb.shape}")
    else:
        logger.error(f"Could not find gene embeddings. Available keys: {list(state_dict.keys())[:20]}")
        raise ValueError("Gene embeddings not found")
    
    return gene_emb


def load_dataset(dataset_path):
    """Load dataset from .pt file"""
    logger.info(f"Loading dataset from {dataset_path}")
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    
    # Extract components
    gene_indices = [g.astype(np.int64) for g in data['genes']]
    expression_values = [e.astype(np.float32) for e in data['expressions']]
    labels = data['cls_name']
    
    logger.info(f"Dataset: {len(gene_indices)} cells")
    
    return gene_indices, expression_values, labels


def evaluate_classifier(X_train, y_train, X_test, y_test, clf_type='LR', n_jobs=8):
    """Train and evaluate a classifier"""
    if clf_type == 'LR':
        clf = LogisticRegression(max_iter=1000, n_jobs=n_jobs, random_state=42)
    elif clf_type == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {clf_type}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return acc


def run_single_experiment(dataset_name, dataset_path, gene_embeddings, fusion_method, clf_type, n_jobs=8):
    """Run a single experiment configuration"""
    start_time = time.time()
    
    try:
        # Load dataset
        gene_indices, expression_values, labels = load_dataset(dataset_path)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Apply fusion method
        fusion = GeneEmbeddingFusion(gene_embeddings)
        fusion_func = getattr(fusion, fusion_method)
        
        with torch.no_grad():
            cell_embeddings = fusion_func(gene_indices, expression_values)
        
        # Convert to numpy
        X = cell_embeddings.numpy()
        
        # 5-fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            acc = evaluate_classifier(X_train, y_train, X_test, y_test, clf_type, n_jobs=n_jobs)
            fold_accs.append(acc)
        
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        
        elapsed = time.time() - start_time
        
        result = {
            'dataset': dataset_name,
            'fusion': fusion_method,
            'classifier': clf_type,
            'mean_acc': mean_acc * 100,
            'std_acc': std_acc * 100,
            'elapsed_sec': elapsed,
            'embedding_dim': X.shape[1]
        }
        
        logger.info(f"✓ {dataset_name} | {fusion_method} | {clf_type} | "
                   f"Acc: {mean_acc*100:.2f}±{std_acc*100:.2f}% | {elapsed:.1f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ {dataset_name} | {fusion_method} | {clf_type} | Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'fusion': fusion_method,
            'classifier': clf_type,
            'mean_acc': 0,
            'std_acc': 0,
            'elapsed_sec': 0,
            'error': str(e)
        }


def main():
    """Main experiment runner"""
    logger.info("=" * 80)
    logger.info("Gene Embedding Fusion Experiment v2 - 32 Core Parallel Execution")
    logger.info("=" * 80)
    
    # Check CPU cores
    n_cores = mp.cpu_count()
    logger.info(f"Available CPU cores: {n_cores}")
    
    # Extract gene embeddings
    gene_embeddings = extract_gene_embeddings(MODEL_PATH)
    logger.info(f"Gene embeddings shape: {gene_embeddings.shape}")
    
    # Define experiment configurations
    fusion_methods = ['weighted_avg', 'concatenation', 'gating', 'attention']
    classifiers = ['LR', 'MLP']
    
    # Create all experiment configurations
    experiments = []
    for dataset_name, dataset_path in DATASETS.items():
        for fusion in fusion_methods:
            for clf in classifiers:
                experiments.append({
                    'dataset_name': dataset_name,
                    'dataset_path': dataset_path,
                    'fusion_method': fusion,
                    'clf_type': clf
                })
    
    logger.info(f"Total experiments: {len(experiments)} (3 datasets × 4 fusions × 2 classifiers)")
    
    # Determine parallelization strategy
    n_parallel_jobs = min(24, n_cores) 
    n_sklearn_jobs = max(1, n_cores // n_parallel_jobs)
    
    logger.info(f"Strategy: {n_parallel_jobs} parallel experiments, "
               f"{n_sklearn_jobs} cores per sklearn classifier")
    
    # Run experiments in parallel
    start_time = time.time()
    
    results = Parallel(n_jobs=n_parallel_jobs, verbose=10)(
        delayed(run_single_experiment)(
            exp['dataset_name'],
            exp['dataset_path'],
            gene_embeddings,
            exp['fusion_method'],
            exp['clf_type'],
            n_jobs=n_sklearn_jobs
        )
        for exp in experiments
    )
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time/60:.2f} minutes")
    
    # Save results
    df_results = pd.DataFrame(results)
    results_csv = OUTPUT_DIR / 'results.csv'
    df_results.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")
    
    # Generate comparison analysis
    generate_comparison_report(df_results, total_time, n_cores, n_parallel_jobs)
    
    logger.info("=" * 80)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 80)


def generate_comparison_report(df_results, total_time, n_cores, n_parallel_jobs):
    """Generate detailed comparison report"""
    
    report_path = OUTPUT_DIR / 'comparison.md'
    
    with open(report_path, 'w') as f:
        f.write("# Gene Embedding Fusion vs Transformer Baseline\n\n")
        f.write(f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Execution Time:** {total_time/60:.2f} minutes\n\n")
        f.write(f"**CPU Cores:** {n_cores} (used {n_parallel_jobs} parallel jobs)\n\n")
        
        f.write("## Performance Summary\n\n")
        
        # Best results per dataset
        for dataset in DATASETS.keys():
            f.write(f"### {dataset}\n\n")
            
            dataset_results = df_results[df_results['dataset'] == dataset].copy()
            dataset_results = dataset_results.sort_values('mean_acc', ascending=False)
            
            f.write("| Fusion Method | Classifier | Accuracy | Baseline (Cls) | Gap |\n")
            f.write("|---------------|------------|----------|----------------|-----|\n")
            
            baseline_acc = BASELINE_METRICS[dataset]['Cls']
            
            for _, row in dataset_results.iterrows():
                fusion = row['fusion']
                clf = row['classifier']
                acc = row['mean_acc']
                std = row['std_acc']
                gap = acc - baseline_acc
                pct = (acc / baseline_acc) * 100 if baseline_acc > 0 else 0
                
                f.write(f"| {fusion} | {clf} | {acc:.2f}±{std:.2f}% | "
                       f"{baseline_acc:.2f}% | {gap:+.2f}% ({pct:.1f}%) |\n")
            
            f.write("\n")
            
            # Best result for this dataset
            if len(dataset_results) > 0:
                best = dataset_results.iloc[0]
                best_pct = (best['mean_acc'] / baseline_acc) * 100 if baseline_acc > 0 else 0
                f.write(f"**Best:** {best['fusion']} + {best['classifier']} = "
                       f"{best['mean_acc']:.2f}±{best['std_acc']:.2f}% "
                       f"({best_pct:.1f}% of baseline)\n\n")
        
        f.write("## Overall Analysis\n\n")
        
        # Find best overall method
        avg_by_fusion = df_results.groupby(['fusion', 'classifier'])['mean_acc'].mean().reset_index()
        avg_by_fusion = avg_by_fusion.sort_values('mean_acc', ascending=False)
        
        f.write("### Average Performance Across All Datasets\n\n")
        f.write("| Fusion Method | Classifier | Avg Accuracy |\n")
        f.write("|---------------|------------|-------------|\n")
        for _, row in avg_by_fusion.iterrows():
            f.write(f"| {row['fusion']} | {row['classifier']} | {row['mean_acc']:.2f}% |\n")
        f.write("\n")
        
        best_overall = avg_by_fusion.iloc[0]
        f.write(f"**Best Overall:** {best_overall['fusion']} + {best_overall['classifier']} "
               f"({best_overall['mean_acc']:.2f}%)\n\n")
        
        # Baseline comparison
        f.write("### Comparison to Baseline\n\n")
        
        baseline_avg = np.mean([BASELINE_METRICS[ds]['Cls'] for ds in DATASETS.keys()])
        fusion_avg = df_results['mean_acc'].mean()
        
        f.write(f"- **Baseline Average (Transformer):** {baseline_avg:.2f}%\n")
        f.write(f"- **Fusion Average (All methods):** {fusion_avg:.2f}%\n")
        f.write(f"- **Gap:** {fusion_avg - baseline_avg:+.2f}%\n")
        f.write(f"- **Percentage of Baseline:** {(fusion_avg/baseline_avg)*100:.1f}%\n\n")
        
        # Parallelization efficiency
        f.write("### Parallelization Efficiency\n\n")
        
        total_experiments = len(df_results)
        avg_time_per_exp = df_results['elapsed_sec'].mean()
        sequential_time = total_experiments * avg_time_per_exp
        speedup = sequential_time / total_time if total_time > 0 else 0
        efficiency = (speedup / n_parallel_jobs) * 100 if n_parallel_jobs > 0 else 0
        
        f.write(f"- **Sequential Time (estimated):** {sequential_time/60:.2f} minutes\n")
        f.write(f"- **Parallel Time (actual):** {total_time/60:.2f} minutes\n")
        f.write(f"- **Speedup:** {speedup:.2f}x\n")
        f.write(f"- **Parallel Efficiency:** {efficiency:.1f}%\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        f.write("### 1. Can Gene Embedding Fusion Replace Transformer Forward Pass?\n\n")
        
        if fusion_avg >= baseline_avg * 0.9:
            f.write("✅ **YES** - Fusion methods achieve >90% of baseline performance, "
                   "making them viable alternatives for certain use cases.\n\n")
        elif fusion_avg >= baseline_avg * 0.7:
            f.write("⚠️ **PARTIAL** - Fusion methods achieve 70-90% of baseline performance. "
                   "Suitable for rapid prototyping but not production.\n\n")
        else:
            f.write("❌ **NO** - Fusion methods significantly underperform the baseline. "
                   "Transformer forward pass is necessary for competitive results.\n\n")
        
        f.write("### 2. Best Fusion Strategy\n\n")
        f.write(f"**{best_overall['fusion']}** consistently performs best across datasets.\n\n")
        
        f.write("### 3. Dataset-Specific Observations\n\n")
        for dataset in DATASETS.keys():
            dataset_best = df_results[df_results['dataset'] == dataset].nlargest(1, 'mean_acc').iloc[0]
            baseline = BASELINE_METRICS[dataset]['Cls']
            pct = (dataset_best['mean_acc'] / baseline) * 100 if baseline > 0 else 0
            f.write(f"- **{dataset}:** {pct:.1f}% of baseline "
                   f"(best: {dataset_best['fusion']} + {dataset_best['classifier']})\n")
        f.write("\n")
        
        f.write("### 4. Parallel Speedup\n\n")
        f.write(f"Using {n_parallel_jobs} parallel jobs on {n_cores} cores achieved "
               f"**{speedup:.2f}x speedup** with **{efficiency:.1f}% efficiency**.\n\n")
        
        if efficiency > 80:
            f.write("✅ Excellent parallelization - near-linear scaling!\n")
        elif efficiency > 60:
            f.write("✓ Good parallelization - acceptable overhead.\n")
        else:
            f.write("⚠️ Suboptimal parallelization - consider tuning job allocation.\n")
    
    logger.info(f"Comparison report saved to {report_path}")


if __name__ == '__main__':
    main()
