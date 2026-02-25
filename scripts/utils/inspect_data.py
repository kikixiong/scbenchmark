#!/usr/bin/env python
"""Inspect the structure of dataset files"""

import torch
from pathlib import Path

datasets = {
    'Myeloid': 'data/downstreams/classification/processed_data/Myeloid_data.pt',
    'pancread': 'data/downstreams/classification/processed_data/pancread_data.pt',
    'lupus': 'data/downstreams/classification/processed_data/lupus_data.pt'
}

for name, path in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Path: {path}")
    print(f"{'='*60}")
    
    try:
        data = torch.load(path, map_location='cpu')
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: type={type(value)}, ", end='')
                if hasattr(value, 'shape'):
                    print(f"shape={value.shape}")
                elif isinstance(value, list):
                    print(f"len={len(value)}, first_elem_type={type(value[0]) if value else None}")
                else:
                    print(f"value={value}")
        elif isinstance(data, (list, tuple)):
            print(f"Length: {len(data)}")
            for i, item in enumerate(data[:3]):  # First 3 items
                print(f"  Item {i}: type={type(item)}, ", end='')
                if hasattr(item, 'shape'):
                    print(f"shape={item.shape}")
                else:
                    print(f"value={item}")
        else:
            print(f"Unknown data structure")
            if hasattr(data, 'shape'):
                print(f"Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading: {e}")

# Also inspect the model
print(f"\n{'='*60}")
print("Model: save_pretrain/difference_aligned_v3/best_model.pt")
print(f"{'='*60}")

try:
    model = torch.load('save_pretrain/difference_aligned_v3/best_model.pt', map_location='cpu')
    print(f"Type: {type(model)}")
    
    if isinstance(model, dict):
        print(f"Top-level keys: {list(model.keys())}")
        
        # Look for model state
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in model:
                print(f"\nState dict keys ({key}):")
                state_dict = model[key]
                for k in list(state_dict.keys())[:30]:
                    v = state_dict[k]
                    if hasattr(v, 'shape'):
                        print(f"  {k}: {v.shape}")
                    else:
                        print(f"  {k}: {type(v)}")
except Exception as e:
    print(f"Error loading model: {e}")
