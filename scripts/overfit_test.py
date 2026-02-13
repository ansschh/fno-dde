#!/usr/bin/env python3
"""
Overfit Test: Verify model can memorize a small dataset.

If FNO cannot overfit 64 samples to near-zero error, there's a bug in:
- encoding
- loss function  
- normalization
- architecture

This is a critical sanity check before trusting any benchmark results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual


def masked_mse_loss(pred, target, mask):
    """MSE loss only on masked (future) region."""
    mask_exp = mask.unsqueeze(-1)
    diff = (pred - target) * mask_exp
    return (diff ** 2).sum() / (mask_exp.sum() * target.shape[-1] + 1e-8)


def compute_rel_l2(pred, target, mask, target_mean, target_std):
    """Compute relL2 in original space with mask."""
    pred_orig = pred * target_std + target_mean
    target_orig = target * target_std + target_mean
    
    mask_exp = mask.unsqueeze(-1)
    diff = (pred_orig - target_orig) * mask_exp
    target_masked = target_orig * mask_exp
    
    diff_l2 = torch.sqrt((diff ** 2).sum(dim=(1, 2)) + 1e-8)
    target_l2 = torch.sqrt((target_masked ** 2).sum(dim=(1, 2)) + 1e-8)
    
    return (diff_l2 / target_l2).mean().item()


def run_overfit_test(family: str, n_samples: int = 64, epochs: int = 500, device: str = "cuda"):
    """Run overfit test for a family."""
    
    # Load small subset of data
    data_dir = "data_baseline_v2" if family == "dist_exp" else "data_baseline_v1"
    dataset = ShardedDDEDataset(data_dir, family, "train")
    
    # Take first n_samples
    subset_indices = list(range(min(n_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=min(32, n_samples), shuffle=True)
    
    # Get a sample to determine dimensions
    sample = dataset[0]
    in_channels = sample["input"].shape[-1]
    out_channels = sample["target"].shape[-1]
    
    # Create model (small for fast overfit)
    model = FNO1dResidual(
        modes=12,
        width=32,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=2,
        activation="gelu",
        dropout=0.0,  # No dropout for overfitting
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n{'='*60}")
    print(f"Overfit Test: {family.upper()}")
    print(f"  Samples: {n_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    
    history = {"loss": [], "rel_l2": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_rel_l2 = 0
        n_batches = 0
        
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            mask = batch["loss_mask"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = masked_mse_loss(pred, y, mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rel_l2 += compute_rel_l2(pred, y, mask, target_mean, target_std)
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_batches
        avg_rel_l2 = epoch_rel_l2 / n_batches
        history["loss"].append(avg_loss)
        history["rel_l2"].append(avg_rel_l2)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}: loss={avg_loss:.6f}, relL2={avg_rel_l2:.4f}")
    
    final_rel_l2 = history["rel_l2"][-1]
    passed = final_rel_l2 < 0.01  # Pass if < 1% error
    
    print(f"\n  Final relL2: {final_rel_l2:.4f}")
    print(f"  Status: {'PASS' if passed else 'FAIL'} (threshold: 0.01)")
    
    return {
        "family": family,
        "n_samples": n_samples,
        "epochs": epochs,
        "final_loss": history["loss"][-1],
        "final_rel_l2": final_rel_l2,
        "passed": passed,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="hutch")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    families = args.families if args.families else [args.family]
    
    results = {}
    output_dir = Path("reports/overfit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for family in families:
        result = run_overfit_test(
            family, 
            n_samples=args.n_samples,
            epochs=args.epochs,
            device=args.device
        )
        results[family] = result
        
        # Save individual result
        with open(output_dir / f"{family}_overfit.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("OVERFIT TEST SUMMARY")
    print("="*60)
    print(f"{'Family':<15} | {'Final relL2':>12} | {'Status':>8}")
    print("-"*45)
    
    all_passed = True
    for family, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{family:<15} | {result['final_rel_l2']:>12.4f} | {status:>8}")
        if not result["passed"]:
            all_passed = False
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
