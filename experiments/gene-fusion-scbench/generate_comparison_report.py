#!/usr/bin/env python3
"""
Generate comparison report between Gene Embedding Fusion and scBench baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# scBench baseline accuracies from the paper
BASELINES = {
    'Myeloid': 38.62,
    'pancread': 76.24,
    'lupus': 59.77,
    'scfoundation': None,  # No baseline available
    'Multiple_Sclerosis': None  # No baseline available
}

def generate_report(results_csv, output_md):
    """Generate markdown comparison report"""
    
    # Load results
    df = pd.read_csv(results_csv)
    
    # Filter out errors
    df_valid = df[df['accuracy'] != 'ERROR'].copy()
    df_valid['accuracy'] = df_valid['accuracy'].astype(float)
    
    # Calculate statistics
    summary = df_valid.groupby(['dataset', 'model', 'fusion'])['accuracy'].agg(
        ['mean', 'std', 'count', 'min', 'max']
    ).reset_index()
    
    # Add baseline comparison
    summary['baseline'] = summary['dataset'].map(BASELINES)
    summary['delta'] = summary['mean'] - summary['baseline']
    summary['improvement'] = (summary['delta'] / summary['baseline'] * 100)
    
    # Generate markdown report
    with open(output_md, 'w') as f:
        f.write("# Gene Embedding Fusion vs scBench Baseline\n\n")
        f.write("**Evaluation Protocol**: 70/30 random split, 5 trials, Linear classifier, Adam lr=0.005, 50 epochs\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        # Best results per dataset
        f.write("### Best Results Per Dataset\n\n")
        f.write("| Dataset | Best Method | Model | Accuracy | Baseline | Δ | Improvement |\n")
        f.write("|---------|-------------|-------|----------|----------|---|-------------|\n")
        
        for dataset in sorted(summary['dataset'].unique()):
            dataset_df = summary[summary['dataset'] == dataset]
            best_row = dataset_df.loc[dataset_df['mean'].idxmax()]
            
            baseline = BASELINES[dataset]
            if baseline is not None:
                delta = best_row['mean'] - baseline
                improvement = (delta / baseline) * 100
                status = "✅" if delta > 0 else "❌"
                f.write(f"| {dataset} | {best_row['fusion']} | {best_row['model']} | "
                       f"{best_row['mean']:.2f}±{best_row['std']:.2f}% | {baseline:.2f}% | "
                       f"{delta:+.2f}% {status} | {improvement:+.1f}% |\n")
            else:
                f.write(f"| {dataset} | {best_row['fusion']} | {best_row['model']} | "
                       f"{best_row['mean']:.2f}±{best_row['std']:.2f}% | N/A | N/A | N/A |\n")
        
        f.write("\n")
        
        # Full results table
        f.write("## Complete Results\n\n")
        f.write("### Sorted by Dataset\n\n")
        
        for dataset in sorted(summary['dataset'].unique()):
            f.write(f"#### {dataset}\n\n")
            baseline = BASELINES[dataset]
            if baseline is not None:
                f.write(f"**scBench Baseline**: {baseline:.2f}%\n\n")
            
            f.write("| Model | Fusion Method | Accuracy | Min-Max | vs Baseline | Δ |\n")
            f.write("|-------|---------------|----------|---------|-------------|---|\n")
            
            dataset_df = summary[summary['dataset'] == dataset].sort_values('mean', ascending=False)
            for _, row in dataset_df.iterrows():
                if baseline is not None:
                    delta = row['mean'] - baseline
                    status = "✅" if delta > 0 else "❌"
                    f.write(f"| {row['model']} | {row['fusion']} | "
                           f"{row['mean']:.2f}±{row['std']:.2f}% | "
                           f"{row['min']:.2f}-{row['max']:.2f}% | "
                           f"{baseline:.2f}% | {delta:+.2f}% {status} |\n")
                else:
                    f.write(f"| {row['model']} | {row['fusion']} | "
                           f"{row['mean']:.2f}±{row['std']:.2f}% | "
                           f"{row['min']:.2f}-{row['max']:.2f}% | N/A | N/A |\n")
            f.write("\n")
        
        # Analysis by fusion method
        f.write("## Analysis by Fusion Method\n\n")
        
        for fusion in sorted(summary['fusion'].unique()):
            f.write(f"### {fusion.replace('_', ' ').title()}\n\n")
            
            fusion_df = summary[summary['fusion'] == fusion]
            
            # Calculate average performance across datasets with baselines
            fusion_with_baseline = fusion_df[fusion_df['baseline'].notna()]
            if len(fusion_with_baseline) > 0:
                avg_improvement = fusion_with_baseline['improvement'].mean()
                f.write(f"**Average improvement over baseline**: {avg_improvement:+.1f}%\n\n")
            
            f.write("| Dataset | Model | Accuracy | vs Baseline |\n")
            f.write("|---------|-------|----------|-------------|\n")
            
            for _, row in fusion_df.iterrows():
                if row['baseline'] is not None:
                    delta = row['mean'] - row['baseline']
                    status = "✅" if delta > 0 else "❌"
                    f.write(f"| {row['dataset']} | {row['model']} | "
                           f"{row['mean']:.2f}±{row['std']:.2f}% | "
                           f"{delta:+.2f}% {status} |\n")
                else:
                    f.write(f"| {row['dataset']} | {row['model']} | "
                           f"{row['mean']:.2f}±{row['std']:.2f}% | N/A |\n")
            f.write("\n")
        
        # Model comparison
        f.write("## Model Comparison: difference_v3 vs genept\n\n")
        
        for fusion in sorted(summary['fusion'].unique()):
            f.write(f"### {fusion.replace('_', ' ').title()}\n\n")
            f.write("| Dataset | difference_v3 | genept | Winner |\n")
            f.write("|---------|---------------|--------|--------|\n")
            
            for dataset in sorted(summary['dataset'].unique()):
                diff_row = summary[(summary['dataset'] == dataset) & 
                                  (summary['fusion'] == fusion) & 
                                  (summary['model'] == 'difference_v3')]
                gene_row = summary[(summary['dataset'] == dataset) & 
                                  (summary['fusion'] == fusion) & 
                                  (summary['model'] == 'genept')]
                
                if len(diff_row) > 0 and len(gene_row) > 0:
                    diff_acc = diff_row.iloc[0]['mean']
                    diff_std = diff_row.iloc[0]['std']
                    gene_acc = gene_row.iloc[0]['mean']
                    gene_std = gene_row.iloc[0]['std']
                    
                    if diff_acc > gene_acc:
                        winner = "difference_v3 ✅"
                    elif gene_acc > diff_acc:
                        winner = "genept ✅"
                    else:
                        winner = "Tie"
                    
                    f.write(f"| {dataset} | {diff_acc:.2f}±{diff_std:.2f}% | "
                           f"{gene_acc:.2f}±{gene_std:.2f}% | {winner} |\n")
            f.write("\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        
        # 1. Best fusion method overall
        with_baseline = summary[summary['baseline'].notna()].copy()
        if len(with_baseline) > 0:
            best_fusion_overall = with_baseline.groupby('fusion')['improvement'].mean().idxmax()
            best_fusion_improvement = with_baseline.groupby('fusion')['improvement'].mean().max()
            f.write(f"1. **Best fusion method overall**: {best_fusion_overall} "
                   f"(avg +{best_fusion_improvement:.1f}% improvement)\n")
        
        # 2. Most robust fusion method (lowest std)
        avg_std = summary.groupby('fusion')['std'].mean()
        most_robust = avg_std.idxmin()
        f.write(f"2. **Most robust fusion method**: {most_robust} "
               f"(avg std={avg_std[most_robust]:.2f}%)\n")
        
        # 3. Best model overall
        if len(with_baseline) > 0:
            best_model = with_baseline.groupby('model')['improvement'].mean().idxmax()
            best_model_improvement = with_baseline.groupby('model')['improvement'].mean().max()
            f.write(f"3. **Best model overall**: {best_model} "
                   f"(avg +{best_model_improvement:.1f}% improvement)\n")
        
        # 4. Datasets where fusion helps most
        if len(with_baseline) > 0:
            dataset_improvements = with_baseline.groupby('dataset')['improvement'].mean().sort_values(ascending=False)
            f.write(f"4. **Datasets where fusion helps most**:\n")
            for dataset, improvement in dataset_improvements.items():
                f.write(f"   - {dataset}: +{improvement:.1f}%\n")
        
        f.write("\n")
        
        # Detailed trial results
        f.write("## Detailed Trial Results\n\n")
        f.write("Individual trial results showing consistency across runs:\n\n")
        
        for dataset in sorted(df_valid['dataset'].unique()):
            f.write(f"### {dataset}\n\n")
            dataset_trials = df_valid[df_valid['dataset'] == dataset].sort_values(
                ['model', 'fusion', 'trial']
            )
            f.write("| Model | Fusion | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Mean±Std |\n")
            f.write("|-------|--------|---------|---------|---------|---------|---------|----------|\n")
            
            for (model, fusion), group in dataset_trials.groupby(['model', 'fusion']):
                trials = group.sort_values('trial')['accuracy'].values
                if len(trials) == 5:
                    mean = np.mean(trials)
                    std = np.std(trials)
                    f.write(f"| {model} | {fusion} | "
                           f"{trials[0]:.2f} | {trials[1]:.2f} | {trials[2]:.2f} | "
                           f"{trials[3]:.2f} | {trials[4]:.2f} | {mean:.2f}±{std:.2f} |\n")
            f.write("\n")
        
        # Experimental setup
        f.write("## Experimental Setup\n\n")
        f.write("### Data Split\n")
        f.write("- **Train/Test**: 70/30 random split (matching scBench)\n")
        f.write("- **Trials**: 5 independent trials with different random seeds\n")
        f.write("- **Split method**: PyTorch `random_split()` (not stratified)\n\n")
        
        f.write("### Classifier Training\n")
        f.write("- **Architecture**: Linear classifier `nn.Linear(emb_dim, n_classes)`\n")
        f.write("- **Optimizer**: Adam with lr=0.005\n")
        f.write("- **Training**: 50 epochs\n")
        f.write("- **Batch size**: 64\n")
        f.write("- **Evaluation**: Every 5 epochs, report best\n\n")
        
        f.write("### Fusion Methods\n\n")
        f.write("1. **weighted_avg**: Expression-weighted average of gene embeddings\n")
        f.write("   - `weights = expr / (expr.sum() + 1e-8)`\n")
        f.write("   - `cell_emb = gene_embs.T @ weights`\n\n")
        
        f.write("2. **concatenation**: Weighted average + statistical features\n")
        f.write("   - Combines weighted average with [mean, std, max, count] of expression\n")
        f.write("   - Output dim: `emb_dim + 4`\n\n")
        
        f.write("3. **gating**: Sigmoid gating based on normalized expression\n")
        f.write("   - `gates = sigmoid((expr - mean) / std)`\n")
        f.write("   - `cell_emb = (gene_embs * gates).mean()`\n\n")
        
        f.write("4. **attention**: Softmax attention weights\n")
        f.write("   - `attn_weights = softmax(expr)`\n")
        f.write("   - `cell_emb = gene_embs.T @ attn_weights`\n\n")
        
        f.write("### Models\n\n")
        f.write("- **difference_v3**: `/root/autodl-tmp/scbenchmark/save_pretrain/difference_aligned_v3/best_model.pt`\n")
        f.write("- **genept**: `/root/autodl-tmp/scbenchmark/save_pretrain/genept/best_model.pt`\n\n")
        
        f.write("### Datasets\n\n")
        for dataset, baseline in BASELINES.items():
            if baseline is not None:
                f.write(f"- **{dataset}**: scBench baseline = {baseline:.2f}%\n")
            else:
                f.write(f"- **{dataset}**: No baseline available\n")
    
    print(f"✅ Comparison report generated: {output_md}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_csv = sys.argv[1]
    else:
        results_csv = "/root/autodl-tmp/gene-fusion-scbench/results_scbench.csv"
    
    output_md = results_csv.replace('.csv', '_COMPARISON.md')
    
    generate_report(results_csv, output_md)
