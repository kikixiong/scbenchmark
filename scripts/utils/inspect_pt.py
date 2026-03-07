#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _describe(obj: Any, indent: int, max_items: int) -> None:
    import torch

    prefix = " " * indent
    if torch.is_tensor(obj):
        print(f"{prefix}Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})")
        return

    if isinstance(obj, dict):
        print(f"{prefix}dict(len={len(obj)})")
        items = list(obj.items())
        for idx, (key, value) in enumerate(items):
            if idx >= max_items:
                print(f"{prefix}  ... ({len(items) - max_items} more items)")
                break
            print(f"{prefix}  [{key!r}]", end=" ")
            _describe(value, indent + 2, max_items)
        return

    if isinstance(obj, (list, tuple)):
        name = type(obj).__name__
        print(f"{prefix}{name}(len={len(obj)})")
        for idx, value in enumerate(obj[:max_items]):
            print(f"{prefix}  [{idx}]", end=" ")
            _describe(value, indent + 2, max_items)
        if len(obj) > max_items:
            print(f"{prefix}  ... ({len(obj) - max_items} more items)")
        return

    print(f"{prefix}{type(obj).__name__}: {obj}")


def inspect_pt(path: Path, max_items: int, weights_only: bool) -> None:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=weights_only)
    print(f"File: {path}")
    print(f"Top-level type: {type(obj).__name__}")
    _describe(obj, indent=0, max_items=max_items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect .pt/.pth structure and tensor metadata")
    parser.add_argument("pt_path", type=Path, help="Path to .pt/.pth file")
    parser.add_argument("--max-items", type=int, default=30, help="Maximum items shown per container")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Use torch.load(..., weights_only=True)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pt_path.exists():
        raise FileNotFoundError(f"File not found: {args.pt_path}")
    inspect_pt(path=args.pt_path, max_items=args.max_items, weights_only=args.weights_only)


if __name__ == "__main__":
    main()
