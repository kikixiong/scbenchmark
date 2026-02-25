import torch
import sys

model_path = sys.argv[1]
print(f'Loading {model_path}...')
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

if isinstance(checkpoint, dict):
    print(f'\nCheckpoint type: dict with {len(checkpoint)} keys')
    print('\nTop-level keys:')
    for k in list(checkpoint.keys())[:30]:
        v = checkpoint[k]
        if isinstance(v, torch.Tensor):
            print(f'  {k}: Tensor{list(v.shape)}')
        elif isinstance(v, dict):
            print(f'  {k}: dict with {len(v)} keys')
        else:
            print(f'  {k}: {type(v)}')
    
    # Check for module prefix
    has_module = any(k.startswith('module.') for k in checkpoint.keys())
    if has_module:
        print('\nHas module. prefix - stripping...')
        # Find embedding keys
        emb_keys = [k for k in checkpoint.keys() if 'emb' in k.lower() and isinstance(checkpoint[k], torch.Tensor)]
        print(f'\nEmbedding-related keys ({len(emb_keys)}):')
        for k in emb_keys[:20]:
            print(f'  {k}: {list(checkpoint[k].shape)}')
else:
    print(f'Checkpoint type: {type(checkpoint)}')
