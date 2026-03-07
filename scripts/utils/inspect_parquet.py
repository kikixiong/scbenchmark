#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def inspect_with_pyarrow(path: Path, sample_rows: int) -> None:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    print(f"File: {path}")
    print("Backend: pyarrow")
    print(f"Num row groups: {parquet_file.num_row_groups}")
    print(f"Schema columns: {parquet_file.schema.names}")

    arrow_schema = parquet_file.schema_arrow
    for field in arrow_schema:
        print(f"- {field.name}: {field.type}")

    batch_iter = parquet_file.iter_batches(batch_size=max(sample_rows, 1))
    first_batch = next(batch_iter, None)
    if first_batch is None:
        print("No rows in parquet.")
        return

    first = first_batch.to_pydict()
    n = min(sample_rows, len(next(iter(first.values()))) if first else 0)
    print(f"Sample rows inspected: {n}")
    for col, values in first.items():
        head = values[:n]
        first_val = head[0] if head else None
        first_type = type(first_val).__name__ if first_val is not None else "None"
        first_len = _safe_len(first_val)
        len_text = f", first_len={first_len}" if first_len is not None else ""
        print(f"  * {col}: first_type={first_type}{len_text}")


def inspect_with_pandas(path: Path, sample_rows: int) -> None:
    import pandas as pd

    frame = pd.read_parquet(path)
    print(f"File: {path}")
    print("Backend: pandas")
    print(f"Shape: {frame.shape}")
    print("Columns:")
    for col, dtype in frame.dtypes.items():
        series = frame[col]
        first_val = series.iloc[0] if len(series) > 0 else None
        first_type = type(first_val).__name__ if first_val is not None else "None"
        first_len = _safe_len(first_val)
        len_text = f", first_len={first_len}" if first_len is not None else ""
        print(f"- {col}: dtype={dtype}, first_type={first_type}{len_text}")

    if sample_rows > 0 and len(frame) > 0:
        print("Head:")
        print(frame.head(sample_rows).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect parquet file structure (schema/columns/sample types).")
    parser.add_argument("parquet_path", type=Path, help="Path to parquet file")
    parser.add_argument("--sample-rows", type=int, default=3, help="How many rows to sample for type hints")
    parser.add_argument(
        "--backend",
        choices=["auto", "pyarrow", "pandas"],
        default="auto",
        help="Inspection backend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.parquet_path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if args.backend == "pyarrow":
        inspect_with_pyarrow(path, args.sample_rows)
        return
    if args.backend == "pandas":
        inspect_with_pandas(path, args.sample_rows)
        return

    try:
        inspect_with_pyarrow(path, args.sample_rows)
    except Exception as pyarrow_error:
        print(f"pyarrow inspection failed: {pyarrow_error}")
        print("Falling back to pandas...")
        inspect_with_pandas(path, args.sample_rows)


if __name__ == "__main__":
    main()
