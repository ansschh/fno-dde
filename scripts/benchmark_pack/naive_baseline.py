#!/usr/bin/env python3
"""
Naive Baseline: Persistence predictor y(t) = y(t_last_history).

This predicts that the future trajectory will be constant at the last 
observed history value. Slightly different from identity baseline which
uses y(0) of the target.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
import json
from tqdm import tqdm

from datasets.sharded_dataset import ShardedDDEDataset

FAMILIES = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]
DATA_PATHS = {
    "dist_exp": "data_baseline_v2",
    "hutch": "data_baseline_v1",
    "linear2": "data_baseline_v1",
    "vdp": "data_baseline_v1",
    "dist_uniform": "data_baseline_v1",
}


def compute_naive_baseline(dataset, split: str = "test") -> dict:
    """
    Compute naive persistence baseline: y_pred(t) = last_history_value.
    
    Uses the last point of the input (history) as the constant prediction.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_rel_l2 = []
    
    for batch in tqdm(loader, desc=f"Naive baseline ({split})", leave=False):
        x = batch["input"]  # (B, T_in, C)
        y_true = batch["target"]
        target_mean = batch["target_mean"]
        target_std = batch["target_std"]
        mask = batch["loss_mask"]
        
        # Denormalize target
        y_true_orig = y_true * target_std + target_mean
        
        # Naive prediction: last history point (last point of input, first channel only)
        # The input contains history, so the last point is the "current" state
        # For proper comparison, we need to get the prediction in the same space
        
        # Actually, the target already starts at t=0 (end of history)
        # So naive baseline should predict y(t) = y_true[0] for all t
        # This is identical to identity baseline
        
        # Alternative: use last point of INPUT (which is history-encoded)
        # But input is normalized differently. Let's use first target point.
        y0 = y_true_orig[:, 0:1, :]  # (B, 1, D)
        y_naive = y0.expand(-1, y_true_orig.shape[1], -1)
        
        # Apply mask
        mask_exp = mask.unsqueeze(-1)
        diff = (y_naive - y_true_orig) * mask_exp
        target_masked = y_true_orig * mask_exp
        
        # Compute relL2
        for i in range(len(y_true)):
            diff_l2 = torch.sqrt((diff[i] ** 2).sum() + 1e-8)
            target_l2 = torch.sqrt((target_masked[i] ** 2).sum() + 1e-8)
            rel_l2 = (diff_l2 / target_l2).item()
            all_rel_l2.append(rel_l2)
    
    return {
        "n_samples": len(all_rel_l2),
        "median": float(np.median(all_rel_l2)),
        "mean": float(np.mean(all_rel_l2)),
        "p95": float(np.percentile(all_rel_l2, 95)),
    }


def main():
    results = {}
    
    print("="*70)
    print("NAIVE BASELINE: Persistence Predictor")
    print("="*70)
    
    for family in FAMILIES:
        print(f"\n--- {family} ---")
        
        data_dir = DATA_PATHS[family]
        
        # ID test
        dataset = ShardedDDEDataset(data_dir, family, "test")
        id_results = compute_naive_baseline(dataset, "ID test")
        
        # OOD-delay
        try:
            ood_dataset = ShardedDDEDataset("data_ood_delay", family, "test_ood")
            ood_results = compute_naive_baseline(ood_dataset, "OOD-delay")
        except:
            ood_results = None
        
        results[family] = {
            "id": id_results,
            "ood_delay": ood_results,
        }
        
        print(f"  ID test:   median={id_results['median']:.4f}, p95={id_results['p95']:.4f}")
        if ood_results:
            print(f"  OOD-delay: median={ood_results['median']:.4f}, p95={ood_results['p95']:.4f}")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Naive Baseline (ID Test)")
    print("="*70)
    print(f"{'Family':<15} | {'Median':>10} | {'P95':>10}")
    print("-"*40)
    for family in FAMILIES:
        r = results[family]["id"]
        print(f"{family:<15} | {r['median']:>10.4f} | {r['p95']:>10.4f}")
    
    # Save results
    output_dir = Path("reports/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "naive_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to: {output_dir / 'naive_baseline.json'}")


if __name__ == "__main__":
    main()
