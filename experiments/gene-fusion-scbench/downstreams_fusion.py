#!/usr/bin/env python3
"""
Gene Embedding Fusion Benchmark - scBench Compatible Evaluation
Modified from scBench's downstreams_cls.py to use gene embeddings with fusion methods
"""

import scanpy as sc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.metrics import accuracy_score
import warnings
import datetime
import random
import csv
import sys
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Add the scbenchmark directory to path for imports
sys.path.insert(0, '/root/autodl-tmp/scbenchmark')
from util_ours.utils import GeneVocab


class ExpressionDataset(Dataset):
    """Dataset for single-cell expression data"""
    def __init__(self, data, max_length, pad_token_id, pad_value, args=None):
        self.data = data
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.args = args
        self.preprocess_mode = args.preprocess_mode if args else "none"
        self.class_label_map = {label: idx for idx, label in enumerate(sorted(set(self.data['cls_name'])))}
    
    def __len__(self):
        return len(self.data['cls_name'])
    
    def _binning(self, row):
        """Binning the row into n_bins using a quantile-based digitization."""
        if torch.all(row == 0):
            return torch.zeros_like(row)
        n_bins = self.args.n_bins
        bins = torch.quantile(row, torch.linspace(0, 1, n_bins - 1))
        left_digits = torch.bucketize(row, bins, right=False) - 1
        right_digits = torch.bucketize(row, bins, right=True) - 1
        rands = torch.rand_like(row)
        digits = rands * (right_digits - left_digits) + left_digits
        digits = torch.ceil(digits).to(torch.int64)
        return digits
    
    def _preprocess(self, genes, expressions):
        genes = torch.tensor(genes, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float)
        
        if len(genes) < self.max_length:
            padding_length = self.max_length - len(genes)
            genes = torch.cat([genes, torch.full((padding_length,), self.pad_token_id, dtype=torch.long)])
            original_exps = torch.cat([expressions, torch.full((padding_length,), self.pad_value, dtype=torch.float)])
        else:
            if self.args and self.args.use_weighted_sampling:
                weights = expressions - expressions.min() + 1e-5
                indices = torch.multinomial(weights, self.max_length, replacement=False)
            else:
                indices = torch.randperm(len(genes))[:self.max_length]
            genes = genes[indices]
            original_exps = expressions[indices]
            padding_length = 0
        
        if self.preprocess_mode == "bin":
            expressions = self._binning(original_exps)
        else:
            expressions = original_exps
        
        return genes, expressions, padding_length
    
    def __getitem__(self, index):
        genes = self.data['genes'][index]
        expressions = self.data['expressions'][index]
        cls_label = self.data['cls_name'][index]
        
        genes, expressions, padding_length = self._preprocess(genes, expressions)
        cls_label_idx = self.class_label_map[cls_label]
        
        return {
            'genes': genes,
            'expr': expressions,
            'cls_label': cls_label_idx
        }


def load_gene_embeddings(model_path, vocab_size):
    """Load gene embeddings from pretrained model checkpoint"""
    print(f"Loading gene embeddings from {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Try different possible keys for gene embeddings
        possible_keys = [
            'module.embedding.weight',  # For DataParallel models
            'embedding.weight',
            'encoder.embedding.weight',
            'gene_embedding.weight',
            'encoder.gene_embedding.weight',
            'value_encoder.weight'
        ]
        
        gene_embeddings = None
        for key in possible_keys:
            if key in checkpoint:
                gene_embeddings = checkpoint[key]
                print(f"  Found gene embeddings at key: {key}")
                break
        
        if gene_embeddings is None:
            # Try to find in state_dict if checkpoint is a dict with 'state_dict' or 'model'
            for meta_key in ['state_dict', 'model']:
                if meta_key in checkpoint:
                    state_dict = checkpoint[meta_key]
                    for key in possible_keys:
                        if key in state_dict:
                            gene_embeddings = state_dict[key]
                            print(f"  Found gene embeddings at {meta_key}.{key}")
                            break
                    if gene_embeddings is not None:
                        break
        
        if gene_embeddings is None:
            # Last resort: print all keys and try to find embedding-like tensors
            print("  Available keys in checkpoint:")
            if isinstance(checkpoint, dict):
                for k in list(checkpoint.keys())[:20]:
                    print(f"    {k}: {type(checkpoint[k])}")
            raise KeyError("Could not find gene embeddings in checkpoint")
        
        # Verify shape
        print(f"  Gene embeddings shape: {gene_embeddings.shape}")
        if gene_embeddings.shape[0] < vocab_size:
            print(f"  WARNING: Embedding size {gene_embeddings.shape[0]} < vocab size {vocab_size}")
            print(f"  Padding with zeros to match vocab size")
            padding = torch.zeros(vocab_size - gene_embeddings.shape[0], gene_embeddings.shape[1])
            gene_embeddings = torch.cat([gene_embeddings, padding], dim=0)
        
        return gene_embeddings
    
    except Exception as e:
        print(f"  ERROR loading gene embeddings: {e}")
        raise


def extract_features_fusion(gene_embeddings, data_loader, fusion_method, vocab):
    """
    Extract cell embeddings using gene embeddings + fusion method
    
    Args:
        gene_embeddings: [vocab_size, emb_dim] - Pretrained gene embeddings
        data_loader: DataLoader with gene indices and expression values
        fusion_method: 'weighted_avg', 'concatenation', 'gating', 'attention'
        vocab: GeneVocab object
    
    Returns:
        embeddings: [n_cells, emb_dim] - Cell embeddings
        labels: [n_cells] - Cell type labels
    """
    embeddings = []
    labels = []
    emb_dim = gene_embeddings.size(1)
    
    # Determine output dimension based on fusion method
    if fusion_method == 'concatenation':
        output_dim = emb_dim + 4  # avg_emb + [mean, std, max, count]
    else:
        output_dim = emb_dim
    
    with torch.no_grad():
        for data_dict in data_loader:
            genes = data_dict['genes']  # [batch, max_len]
            expressions = data_dict['expr']  # [batch, max_len]
            
            batch_emb = []
            for i in range(len(genes)):
                g = genes[i]  # [max_len]
                e = expressions[i]  # [max_len]
                
                # Filter padding and out-of-vocab
                mask = (g >= 0) & (g < len(gene_embeddings)) & (g != vocab["<pad>"])
                g_valid = g[mask]
                e_valid = e[mask]
                
                if len(g_valid) == 0:
                    # If no valid genes, use zero vector
                    emb = torch.zeros(output_dim)
                else:
                    # Get gene embeddings for valid genes
                    gene_embs = gene_embeddings[g_valid]  # [n_genes, emb_dim]
                    
                    # Apply fusion method
                    if fusion_method == 'weighted_avg':
                        # Normalize expression values to weights
                        weights = e_valid / (e_valid.sum() + 1e-8)
                        emb = (gene_embs.T @ weights)
                    
                    elif fusion_method == 'concatenation':
                        # Weighted average + statistical features
                        weights = e_valid / (e_valid.sum() + 1e-8)
                        avg_emb = (gene_embs.T @ weights)
                        mean_val = e_valid.mean()
                        std_val = e_valid.std()
                        max_val = e_valid.max()
                        count_val = torch.tensor(len(e_valid), dtype=torch.float)
                        emb = torch.cat([avg_emb, torch.tensor([mean_val, std_val, max_val, count_val])])
                    
                    elif fusion_method == 'gating':
                        # Normalize expression then sigmoid gating
                        e_norm = (e_valid - e_valid.mean()) / (e_valid.std() + 1e-8)
                        gates = torch.sigmoid(e_norm)
                        gated_emb = gene_embs * gates.unsqueeze(1)
                        emb = gated_emb.mean(dim=0)
                    
                    elif fusion_method == 'attention':
                        # Softmax attention weights
                        attn_weights = torch.softmax(e_valid, dim=0)
                        emb = (gene_embs.T @ attn_weights)
                    
                    else:
                        raise ValueError(f"Unknown fusion method: {fusion_method}")
                
                batch_emb.append(emb)
            
            embeddings.append(torch.stack(batch_emb))
            labels.append(data_dict['cls_label'])
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return embeddings, labels


def evaluate_classifier(classifier, eval_loader):
    """Evaluate linear classifier on evaluation set"""
    classifier.eval()
    total = 0
    correct = 0
    device = 'cpu'  # No GPU on this server
    
    with torch.no_grad():
        for cell_embeddings, cls_labels in eval_loader:
            cell_embeddings = cell_embeddings.to(device)
            cls_labels = cls_labels.to(device)
            
            logits = classifier(cell_embeddings)
            _, predicted = torch.max(logits.data, 1)
            total += cls_labels.size(0)
            correct += (predicted == cls_labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def run_single_trial(gene_embeddings, dataset, fusion_method, vocab, args, trial_seed):
    """
    Run a single trial with the scBench evaluation protocol
    
    Returns:
        best_accuracy: Best accuracy achieved during training
    """
    # Set seed for this trial
    torch.manual_seed(trial_seed)
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    
    # 70/30 split using random_split
    num_train = int(len(dataset) * args.train_ratio)
    train_dataset, eval_dataset = random_split(dataset, [num_train, len(dataset) - num_train])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Extract features using fusion method
    print(f"    Extracting train features with {fusion_method}...")
    train_embeddings, train_labels = extract_features_fusion(gene_embeddings, train_loader, fusion_method, vocab)
    
    print(f"    Extracting eval features with {fusion_method}...")
    eval_embeddings, eval_labels = extract_features_fusion(gene_embeddings, eval_loader, fusion_method, vocab)
    
    # Move to device (CPU) and create tensor datasets
    device = 'cpu'  # No GPU on this server
    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)
    eval_embeddings = eval_embeddings.to(device)
    eval_labels = eval_labels.to(device)
    
    train_dataset_tensors = TensorDataset(train_embeddings, train_labels)
    eval_dataset_tensors = TensorDataset(eval_embeddings, eval_labels)
    
    train_loader = DataLoader(train_dataset_tensors, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset_tensors, batch_size=args.batch_size, shuffle=False)
    
    # Create linear classifier
    emb_dim = train_embeddings.size(1)
    n_classes = len(dataset.class_label_map)
    device = 'cpu'  # No GPU on this server
    classifier = nn.Linear(emb_dim, n_classes).to(device)
    
    # Optimizer and loss (matching scBench settings)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0
    best_epoch = 0
    
    for epoch in range(1, args.num_epochs + 1):
        classifier.train()
        total_loss = 0
        num_batches = 0
        
        for cell_embeddings, cls_labels in train_loader:
            logits = classifier(cell_embeddings)
            loss = criterion(logits, cls_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Evaluate every args.eval_epoch epochs
        if epoch % args.eval_epoch == 0:
            accuracy = evaluate_classifier(classifier, eval_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            
            if epoch % args.print_epoch == 0:
                print(f"      Epoch {epoch}/{args.num_epochs}: Loss={total_loss/num_batches:.4f}, "
                      f"Eval Acc={accuracy:.2f}%, Best={best_accuracy:.2f}% (epoch {best_epoch})")
    
    return best_accuracy


def run_experiment(model_name, model_path, dataset_name, fusion_method, args, vocab):
    """
    Run complete experiment: 5 trials for one model+dataset+fusion combination
    
    Returns:
        results: List of (trial_id, accuracy) tuples
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {dataset_name} | {model_name} | {fusion_method}")
    print(f"{'='*80}")
    
    # Load gene embeddings
    gene_embeddings = load_gene_embeddings(model_path, len(vocab))
    
    # Load dataset
    input_path = Path(args.input_directory)
    input_file = input_path / f"{dataset_name}_data.pt"
    data_saved_all = torch.load(input_file)
    dataset = ExpressionDataset(data_saved_all, args.test_maxseq, vocab["<pad>"], args.pad_value, args=args)
    
    print(f"Dataset: {len(dataset)} cells, {len(dataset.class_label_map)} classes")
    print(f"Gene embeddings: {gene_embeddings.shape}")
    
    # Run 5 trials
    results = []
    trial_seeds = [42 + i * 1000 for i in range(args.num_trials)]  # Different seeds for each trial
    
    for trial in range(args.num_trials):
        print(f"\n  Trial {trial+1}/{args.num_trials} (seed={trial_seeds[trial]})...")
        accuracy = run_single_trial(gene_embeddings, dataset, fusion_method, vocab, args, trial_seeds[trial])
        results.append((trial + 1, accuracy))
        print(f"  Trial {trial+1} accuracy: {accuracy:.2f}%")
    
    # Calculate statistics
    accuracies = [acc for _, acc in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n  Results: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Individual: {accuracies}")
    
    return results, mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser()
    
    # Model paths
    parser.add_argument("--model_name", type=str, required=True, choices=['difference_v3', 'genept'])
    parser.add_argument("--model_path", type=str, required=True)
    
    # Dataset
    parser.add_argument("--dataset_name", type=str, required=True,
                       choices=['Myeloid', 'pancread', 'lupus', 'scfoundation', 'Multiple_Sclerosis'])
    parser.add_argument("--input_directory", type=str,
                       default="/root/autodl-tmp/scbenchmark/data/downstreams/classification/processed_data")
    
    # Fusion method
    parser.add_argument("--fusion_method", type=str, required=True,
                       choices=['weighted_avg', 'concatenation', 'gating', 'attention'])
    
    # Vocabulary
    parser.add_argument("--vocab_path", type=str,
                       default="/root/autodl-tmp/scbenchmark/vocab.json")
    
    # Training settings (matching scBench)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--eval_epoch", type=int, default=5)
    parser.add_argument("--print_epoch", type=int, default=10)
    
    # Data split settings (matching scBench)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--num_trials", type=int, default=5)
    
    # Dataset settings
    parser.add_argument("--test_maxseq", type=int, default=512)
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--preprocess_mode", type=str, default="none")
    parser.add_argument("--use_weighted_sampling", action="store_true")
    
    # Output
    parser.add_argument("--output_csv", type=str,
                       default="/root/autodl-tmp/gene-fusion-scbench/results_scbench.csv")
    
    args = parser.parse_args()
    args.pad_value = 0
    
    # Load vocabulary
    vocab = GeneVocab.from_file(Path(args.vocab_path))
    
    # Run experiment
    results, mean_acc, std_acc = run_experiment(
        args.model_name, args.model_path, args.dataset_name,
        args.fusion_method, args, vocab
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Write individual trial results
    file_exists = os.path.exists(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['dataset', 'model', 'fusion', 'trial', 'accuracy'])
        
        for trial, accuracy in results:
            writer.writerow([args.dataset_name, args.model_name, args.fusion_method, trial, f"{accuracy:.4f}"])
    
    print(f"\nResults saved to {args.output_csv}")
    print(f"\nFinal: {args.dataset_name} | {args.model_name} | {args.fusion_method}")
    print(f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")


if __name__ == "__main__":
    main()
