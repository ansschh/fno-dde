#!/usr/bin/env python3
"""Check the data structure and loss_mask to verify history/future split."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.sharded_dataset import ShardedDDEDataset
import torch

ds = ShardedDDEDataset("data_baseline_v1", "hutch", "test")
sample = ds[0]

print("=== DATA STRUCTURE ===")
print(f"Input shape:  {sample['input'].shape}")
print(f"Target shape: {sample['target'].shape}")
print(f"Loss mask shape: {sample['loss_mask'].shape}")

mask = sample['loss_mask']
print(f"\n=== LOSS MASK ===")
print(f"Mask sum: {mask.sum().item()} / {len(mask)} timesteps")
print(f"First 10 mask values: {mask[:10].tolist()}")
print(f"Last 10 mask values: {mask[-10:].tolist()}")

# Find where mask transitions
transitions = torch.where(mask[:-1] != mask[1:])[0]
print(f"Mask transitions at indices: {transitions.tolist()}")

if len(transitions) > 0:
    t_idx = transitions[0].item()
    print(f"\nMask is 0 for t[0:{t_idx+1}], then 1 for t[{t_idx+1}:end]")
    print(f"History region: {t_idx+1} timesteps (NOT supervised)")
    print(f"Future region: {len(mask) - t_idx - 1} timesteps (supervised)")
else:
    print("\nMask is all 1s - entire target is supervised")
    if mask[0] == 0:
        print("Or mask is all 0s - nothing supervised (BUG!)")
