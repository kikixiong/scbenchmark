#!/usr/bin/env python3
"""Evaluate gene-embedding fusion on .pt or .h5ad datasets.

Features:
- Single dataset or directory/glob batch mode for .h5ad
- Automatic label key detection for common single-cell metadata columns
- Gene-name to vocab-id mapping via vocab.json
- Probe metrics with stratified K-fold (KNN + LR accuracy mean/std)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

COMMON_LABEL_KEYS = [
    "cell_type",
    "celltype",
    "cell_type_label",
    "CellType",
    "cell_label",
    "label",
    "labels",
    "condition",
    "perturbation",
    "guide",
]


class GeneEmbeddingFusion:
    def __init__(self, gene_embeddings: torch.Tensor):
        self.gene_embeddings = gene_embeddings
        self.embedding_dim = gene_embeddings.shape[1]

    def weighted_avg(self, genes: np.ndarray, exprs: np.ndarray) -> torch.Tensor:
        if len(genes) == 0:
            return torch.zeros(self.embedding_dim)
        gene_embs = self.gene_embeddings[genes]
        expr_t = torch.from_numpy(exprs).float()
        weights = expr_t / (expr_t.sum() + 1e-8)
        return (weights.unsqueeze(1) * gene_embs).sum(dim=0)

    def concatenation(self, genes: np.ndarray, exprs: np.ndarray) -> torch.Tensor:
        avg_emb = self.weighted_avg(genes, exprs)
        if len(exprs) == 0:
            stats = torch.tensor([0.0, 0.0, 0.0, 0.0])
        else:
            stats = torch.tensor([
                float(exprs.mean()),
                float(exprs.std() if len(exprs) > 1 else 0.0),
                float(exprs.max()),
                float(len(exprs)),
            ])
        return torch.cat([avg_emb, stats], dim=0)

    def gating(self, genes: np.ndarray, exprs: np.ndarray) -> torch.Tensor:
        if len(genes) == 0:
            return torch.zeros(self.embedding_dim)
        gene_embs = self.gene_embeddings[genes]
        expr_t = torch.from_numpy(exprs).float()
        if len(exprs) > 1:
            expr_t = (expr_t - expr_t.mean()) / (expr_t.std() + 1e-8)
        return (torch.sigmoid(expr_t).unsqueeze(1) * gene_embs).mean(dim=0)

    def attention(self, genes: np.ndarray, exprs: np.ndarray) -> torch.Tensor:
        if len(genes) == 0:
            return torch.zeros(self.embedding_dim)
        gene_embs = self.gene_embeddings[genes]
        expr_t = torch.from_numpy(exprs).float()
        attn = torch.softmax(expr_t, dim=0)
        return (attn.unsqueeze(1) * gene_embs).sum(dim=0)


def load_gene_embeddings(model_path: Path, embedding_key: str) -> torch.Tensor:
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    candidate_keys = [embedding_key, "weights.embedding", "module.embedding.weight", "embedding.weight"]

    if isinstance(state, dict):
        for key in candidate_keys:
            val = state.get(key)
            if isinstance(val, torch.Tensor):
                return val
        for container in ("state_dict", "model", "weights"):
            nested = state.get(container)
            if isinstance(nested, dict):
                for key in candidate_keys:
                    val = nested.get(key)
                    if isinstance(val, torch.Tensor):
                        return val
        available = list(state.keys())[:30]
    else:
        available = []
    raise KeyError(
        "Could not locate embedding tensor in checkpoint. "
        f"Tried keys: {candidate_keys}. Top-level keys sample: {available}"
    )


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    vocab_obj = json.loads(vocab_path.read_text(encoding="utf-8"))
    if "token_to_idx" in vocab_obj and isinstance(vocab_obj["token_to_idx"], dict):
        return vocab_obj["token_to_idx"]
    return vocab_obj


def load_sparse_pt_dataset(dataset_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    genes = [np.asarray(g, dtype=np.int64) for g in data["genes"]]
    exprs = [np.asarray(e, dtype=np.float32) for e in data["expressions"]]
    labels = np.asarray(data["cls_name"])
    return genes, exprs, labels


def _resolve_label_key(adata, user_label_key: str | None) -> str:
    columns = list(adata.obs.columns)
    if user_label_key:
        if user_label_key not in columns:
            raise KeyError(f"label_key '{user_label_key}' not found in adata.obs. Available sample: {columns[:20]}")
        return user_label_key

    # Only auto-select semantically meaningful label columns to avoid
    # accidentally using QC/statistics columns (e.g., nperts/ngenes/ncounts).
    preferred = [
        "perturbation",
        "guide_id",
        "guide",
        "label",
        "labels",
        "condition",
        "cell_type",
        "celltype",
        "cell_type_label",
        "CellType",
        "cell_label",
    ]

    def valid_for_cv(col: str) -> bool:
        vc = adata.obs[col].astype(str).value_counts(dropna=True)
        if vc.shape[0] < 2:
            return False
        return vc.min() >= 2

    for key in preferred:
        if key in columns and valid_for_cv(key):
            return key

    existing_preferred = [k for k in preferred if k in columns]
    raise KeyError(
        "Could not auto-detect a valid label column from preferred keys "
        f"{preferred}. Existing preferred keys: {existing_preferred}. "
        "Please pass --label_key explicitly."
    )


def load_h5ad_dataset(dataset_path: Path, vocab_path: Path, label_key: str | None) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, str]:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Reading .h5ad requires scanpy. Please install scanpy first.") from exc

    adata = sc.read_h5ad(dataset_path)
    effective_label_key = _resolve_label_key(adata, label_key)

    vocab = load_vocab(vocab_path)
    gene_names = adata.var_names.astype(str).tolist()
    mapped_ids = np.array([vocab.get(g, -1) for g in gene_names], dtype=np.int64)
    valid_gene_mask = mapped_ids >= 0

    X = adata.X
    is_sparse = hasattr(X, "tocsr")
    if is_sparse:
        X = X.tocsr()

    genes, exprs = [], []
    for i in range(adata.n_obs):
        row = X[i]
        if is_sparse:
            idx = np.asarray(row.indices, dtype=np.int64)
            val = np.asarray(row.data, dtype=np.float32)
        else:
            row_arr = np.asarray(row).reshape(-1)
            idx = np.nonzero(row_arr)[0].astype(np.int64)
            val = row_arr[idx].astype(np.float32)

        if len(idx) == 0:
            genes.append(np.array([], dtype=np.int64))
            exprs.append(np.array([], dtype=np.float32))
            continue

        keep = valid_gene_mask[idx]
        idx = idx[keep]
        val = val[keep]

        genes.append(mapped_ids[idx].astype(np.int64))
        exprs.append(val)

    labels = adata.obs[effective_label_key].astype(str).to_numpy()
    return genes, exprs, labels, effective_label_key


def load_dataset(dataset_path: Path, vocab_path: Path, label_key: str | None):
    suffix = dataset_path.suffix.lower()
    if suffix == ".pt":
        genes, exprs, labels = load_sparse_pt_dataset(dataset_path)
        return genes, exprs, labels, "cls_name"
    if suffix == ".h5ad":
        return load_h5ad_dataset(dataset_path, vocab_path, label_key)
    raise ValueError(f"Unsupported dataset format: {dataset_path}. Use .pt or .h5ad")


def build_cell_embeddings(
    genes: List[np.ndarray],
    exprs: List[np.ndarray],
    gene_embeddings: torch.Tensor,
    fusion: str,
    pad_id: int,
) -> np.ndarray:
    fusion_impl = GeneEmbeddingFusion(gene_embeddings)
    fusion_fn = getattr(fusion_impl, fusion)
    out = []
    with torch.no_grad():
        for g, e in zip(genes, exprs):
            valid = (g >= 0) & (g < gene_embeddings.shape[0]) & (g != pad_id)
            out.append(fusion_fn(g[valid], e[valid]))
    return torch.stack(out).numpy()


def _filter_rare_classes(X: np.ndarray, y: np.ndarray, min_class_count: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if min_class_count <= 1:
        return X, y, {"dropped_classes": 0, "dropped_samples": 0}
    class_counts = np.bincount(y)
    keep_classes = np.where(class_counts >= min_class_count)[0]
    keep_mask = np.isin(y, keep_classes)
    dropped_samples = int((~keep_mask).sum())
    dropped_classes = int((class_counts > 0).sum() - len(keep_classes))
    return X[keep_mask], y[keep_mask], {"dropped_classes": dropped_classes, "dropped_samples": dropped_samples}


def _safe_n_splits(y: np.ndarray, requested_splits: int) -> int:
    class_counts = np.bincount(y)
    class_counts = class_counts[class_counts > 0]
    if len(class_counts) < 2:
        raise ValueError("Need at least 2 classes in the whole dataset for classification evaluation")
    min_count = int(class_counts.min())
    if min_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 samples; stratified CV cannot run. "
            "Please filter labels or merge rare classes first."
        )
    return min(requested_splits, min_count)


def evaluate_label_baseline(y: np.ndarray, n_splits: int, seed: int, min_class_count: int, baseline_type: str) -> Dict[str, Dict[str, float]]:
    # Keep same filtering/CV protocol as fusion probes.
    dummy_x = np.zeros((len(y), 1), dtype=np.float32)
    _, y, filter_info = _filter_rare_classes(dummy_x, y, min_class_count)
    effective_splits = _safe_n_splits(y, n_splits)
    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    scores = []
    rng = np.random.default_rng(seed)
    for tr_idx, te_idx in cv.split(np.zeros(len(y)), y):
        y_tr, y_te = y[tr_idx], y[te_idx]
        if baseline_type == "majority_baseline":
            pred = np.full_like(y_te, fill_value=np.bincount(y_tr).argmax())
        elif baseline_type == "random_baseline":
            probs = np.bincount(y_tr) / len(y_tr)
            classes = np.arange(len(probs))
            pred = rng.choice(classes, size=len(y_te), p=probs)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        scores.append(float((pred == y_te).mean()))

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return {
        "KNN": {"mean_acc": mean_acc, "std_acc": std_acc},
        "LR": {"mean_acc": mean_acc, "std_acc": std_acc},
        "cv": {
            "requested_splits": int(n_splits),
            "effective_splits": int(effective_splits),
            "used_folds": int(len(scores)),
            "min_class_count": int(min_class_count),
            "baseline_type": baseline_type,
        },
        "filter": filter_info,
    }


def evaluate_probes(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int, n_neighbors: int, min_class_count: int) -> Dict[str, Dict[str, float]]:
    X, y, filter_info = _filter_rare_classes(X, y, min_class_count)
    effective_splits = _safe_n_splits(y, n_splits)
    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    acc_knn, acc_lr = [], []

    for tr_idx, te_idx in cv.split(X, y):
        x_tr, x_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Extremely imbalanced datasets can still produce one-class training folds
        # after user-provided filtering; skip such folds safely.
        if np.unique(y_tr).size < 2:
            continue

        knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=n_neighbors))])
        knn.fit(x_tr, y_tr)
        acc_knn.append(accuracy_score(y_te, knn.predict(x_te)))

        lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1))])
        lr.fit(x_tr, y_tr)
        acc_lr.append(accuracy_score(y_te, lr.predict(x_te)))

    if not acc_knn or not acc_lr:
        raise ValueError(
            "No valid CV folds with at least 2 training classes. "
            "Try reducing --n_splits or checking label distribution."
        )

    return {
        "KNN": {"mean_acc": float(np.mean(acc_knn)), "std_acc": float(np.std(acc_knn))},
        "LR": {"mean_acc": float(np.mean(acc_lr)), "std_acc": float(np.std(acc_lr))},
        "cv": {"requested_splits": int(n_splits), "effective_splits": int(effective_splits), "used_folds": int(len(acc_lr)), "min_class_count": int(min_class_count)},
        "filter": filter_info,
    }


def evaluate_one(dataset_path: Path, args: argparse.Namespace, gene_emb: torch.Tensor) -> Dict[str, Any]:
    base = {
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.stem,
        "model_path": str(args.model_path),
        "embedding_key": args.embedding_key,
        "fusion": args.fusion,
    }
    try:
        genes, exprs, labels, effective_label_key = load_dataset(dataset_path, args.vocab_path, args.label_key)
        y = LabelEncoder().fit_transform(labels)
        if args.fusion in ["majority_baseline", "random_baseline"]:
            metrics = evaluate_label_baseline(y, args.n_splits, args.seed, args.min_class_count, args.fusion)
            dim = 0
            n_cells = int(len(y))
        else:
            X = build_cell_embeddings(genes, exprs, gene_emb, args.fusion, args.pad_id)
            metrics = evaluate_probes(X, y, args.n_splits, args.seed, args.n_neighbors, args.min_class_count)
            dim = int(X.shape[1])
            n_cells = int(X.shape[0])
        base.update({
            "status": "ok",
            "label_key": effective_label_key,
            "n_cells": n_cells,
            "dim": dim,
            "metrics": metrics,
        })
    except Exception as exc:
        base.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        if not args.skip_invalid:
            raise
    return base


def collect_datasets(dataset_path: Path) -> List[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    if dataset_path.is_dir():
        files = sorted(dataset_path.glob("*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No .h5ad files found under directory: {dataset_path}")
        return files
    # allow shell-like wildcard path passed as string
    files = sorted(Path().glob(str(dataset_path)))
    if files:
        return files
    raise FileNotFoundError(f"dataset_path not found: {dataset_path}")


def write_summary(results: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    # default csv
    fields = [
        "dataset_name", "dataset_path", "status", "error_type", "error", "label_key", "fusion", "n_cells", "dim",
        "knn_mean", "knn_std", "lr_mean", "lr_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dataset_name": r.get("dataset_name"),
                "dataset_path": r.get("dataset_path"),
                "status": r.get("status", "ok"),
                "error_type": r.get("error_type", ""),
                "error": r.get("error", ""),
                "label_key": r.get("label_key", ""),
                "fusion": r.get("fusion"),
                "n_cells": r.get("n_cells", ""),
                "dim": r.get("dim", ""),
                "knn_mean": r.get("metrics", {}).get("KNN", {}).get("mean_acc", ""),
                "knn_std": r.get("metrics", {}).get("KNN", {}).get("std_acc", ""),
                "lr_mean": r.get("metrics", {}).get("LR", {}).get("mean_acc", ""),
                "lr_std": r.get("metrics", {}).get("LR", {}).get("std_acc", ""),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style embedding evaluation (.pt/.h5ad, single or batch)")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Path to one dataset file, a directory of .h5ad files, or a wildcard")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--embedding_key", type=str, default="weights.embedding",
                        help="Primary embedding key. Also auto-tries weights.embedding/module.embedding.weight/embedding.weight")
    parser.add_argument("--vocab_path", type=Path, default=Path("vocab.json"),
                        help="Gene vocab JSON for .h5ad mapping")
    parser.add_argument("--label_key", type=str, default=None,
                        help="Label column in adata.obs for .h5ad. If omitted, auto-detect common names")
    parser.add_argument("--fusion", choices=["weighted_avg", "concatenation", "gating", "attention", "majority_baseline", "random_baseline"], default="weighted_avg")
    parser.add_argument("--pad_id", type=int, default=60696)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_class_count", type=int, default=2,
                        help="Drop classes with fewer than this many samples before CV (default: 2)")
    parser.add_argument("--output_json", type=Path, default=None,
                        help="Write single-result json or batch summary (.json/.csv)")
    parser.add_argument("--skip_invalid", action="store_true", default=True,
                        help="Skip datasets that cannot be evaluated (e.g., only one class).")
    parser.add_argument("--no-skip_invalid", dest="skip_invalid", action="store_false",
                        help="Fail immediately when any dataset errors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = collect_datasets(args.dataset_path)
    gene_emb = None if args.fusion in ["majority_baseline", "random_baseline"] else load_gene_embeddings(args.model_path, args.embedding_key)

    results = []
    for ds in dataset_paths:
        result = evaluate_one(ds, args, gene_emb)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    if args.output_json is not None:
        if len(results) == 1 and args.output_json.suffix.lower() == ".json":
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(results[0], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            write_summary(results, args.output_json)


if __name__ == "__main__":
    main()
