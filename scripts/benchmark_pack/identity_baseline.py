#!/usr/bin/env python3
"""
Identity Baseline: y(t) = y(0) for all t.

This is the simplest possible "predictor" - just repeat the initial condition.
If FNO isn't beating this by a significant margin, there's a training issue.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
import json
from tqdm import tqdm
from typing import Dict

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual


FAMILY_ORDER = ["dist_exp", "hutch", "dist_uniform", "vdp", "linear2"]
FAMILY_DISPLAY = {
    "dist_exp": "DistExp",
    "hutch": "Hutchinson",
    "linear2": "Linear2",
    "vdp": "Van der Pol",
    "dist_uniform": "DistUniform",
}

MODEL_PATHS = {
    "dist_exp": "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403",
    "hutch": "outputs/baseline_v1/hutch_seed42_20251228_131919",
    "linear2": "outputs/baseline_v1/linear2_seed42_20251228_142839",
    "vdp": "outputs/baseline_v1/vdp_seed42_20251229_020516",
    "dist_uniform": "outputs/baseline_v1/dist_uniform_seed42_20251229_030851",
}

DATA_PATHS = {
    "dist_exp": "data_baseline_v2",
    "hutch": "data_baseline_v1",
    "linear2": "data_baseline_v1",
    "vdp": "data_baseline_v1",
    "dist_uniform": "data_baseline_v1",
}


def load_model(model_dir: Path, device: str = "cuda"):
    """Load trained model from checkpoint."""
    checkpoint_path = model_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]
    
    in_channels = state_dict["lift.weight"].shape[1]
    out_channels = state_dict["proj2.weight"].shape[0]
    
    model = FNO1dResidual(
        modes=config["model"]["modes"],
        width=config["model"]["width"],
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=config["model"]["n_layers"],
        activation=config["model"].get("activation", "gelu"),
        dropout=config["model"].get("dropout", 0.1),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, config


def compute_identity_baseline(dataset, device: str = "cpu") -> Dict:
    """
    Compute identity baseline: y_pred(t) = y(0) for all t.
    
    y(0) is the last point of the history (first point of future).
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_rel_l2_identity = []
    
    for batch in tqdm(loader, desc="Identity baseline", leave=False):
        # Target in original space
        y_true = batch["target"]  # (B, T, D) normalized
        target_mean = batch["target_mean"]
        target_std = batch["target_std"]
        
        # Denormalize target
        y_true_orig = y_true * target_std + target_mean  # (B, T, D)
        
        # Identity baseline: y(0) repeated for all t
        # y(0) is the first time point of the target (continuation from history)
        y0 = y_true_orig[:, 0:1, :]  # (B, 1, D)
        y_identity = y0.expand(-1, y_true_orig.shape[1], -1)  # (B, T, D)
        
        # Compute relL2 for identity baseline
        for i in range(len(y_true)):
            diff = y_identity[i] - y_true_orig[i]
            rel_l2 = torch.norm(diff) / (torch.norm(y_true_orig[i]) + 1e-10)
            all_rel_l2_identity.append(rel_l2.item())
    
    return {
        "rel_l2": np.array(all_rel_l2_identity),
        "median": np.median(all_rel_l2_identity),
        "p95": np.percentile(all_rel_l2_identity, 95),
        "mean": np.mean(all_rel_l2_identity),
    }


def compute_fno_performance(model, dataset, device: str = "cuda") -> Dict:
    """Compute FNO performance on test set."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_rel_l2 = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="FNO inference", leave=False):
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            
            y_pred = model(x)
            
            # Denormalize
            y_pred_orig = y_pred * target_std + target_mean
            y_true_orig = y_true * target_std + target_mean
            
            for i in range(len(y_pred)):
                diff = y_pred_orig[i] - y_true_orig[i]
                rel_l2 = torch.norm(diff) / (torch.norm(y_true_orig[i]) + 1e-10)
                all_rel_l2.append(rel_l2.item())
    
    return {
        "rel_l2": np.array(all_rel_l2),
        "median": np.median(all_rel_l2),
        "p95": np.percentile(all_rel_l2, 95),
        "mean": np.mean(all_rel_l2),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    results = {}
    
    print("\n" + "="*80)
    print("IDENTITY BASELINE COMPARISON: y(t) = y(0)")
    print("="*80)
    
    for family in FAMILY_ORDER:
        print(f"\n--- {FAMILY_DISPLAY[family]} ---")
        
        # Load dataset
        dataset = ShardedDDEDataset(DATA_PATHS[family], family, "test")
        
        # Compute identity baseline
        identity_results = compute_identity_baseline(dataset, "cpu")
        
        # Load model and compute FNO performance
        model_dir = Path(MODEL_PATHS[family])
        model, _ = load_model(model_dir, args.device)
        fno_results = compute_fno_performance(model, dataset, args.device)
        
        # Compute improvement ratio
        ratio_median = identity_results["median"] / (fno_results["median"] + 1e-10)
        ratio_p95 = identity_results["p95"] / (fno_results["p95"] + 1e-10)
        
        results[family] = {
            "identity": identity_results,
            "fno": fno_results,
            "ratio_median": ratio_median,
            "ratio_p95": ratio_p95,
        }
        
        print(f"  Identity baseline: median={identity_results['median']:.4f}, p95={identity_results['p95']:.4f}")
        print(f"  FNO:               median={fno_results['median']:.4f}, p95={fno_results['p95']:.4f}")
        print(f"  Improvement:       {ratio_median:.1f}x (median), {ratio_p95:.1f}x (p95)")
        
        if ratio_median < 2:
            print(f"  ⚠️  WARNING: FNO is less than 2x better than identity baseline!")
        elif ratio_median < 10:
            print(f"  ⚡ FNO is {ratio_median:.1f}x better (moderate improvement)")
        else:
            print(f"  ✓  FNO is {ratio_median:.1f}x better (strong improvement)")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Family':<15} | {'Identity Med':>12} | {'FNO Med':>10} | {'Ratio':>8} | {'Status':<20}")
    print("-"*75)
    
    for family in FAMILY_ORDER:
        r = results[family]
        status = "✓ Strong" if r["ratio_median"] >= 10 else ("⚡ Moderate" if r["ratio_median"] >= 2 else "⚠️ WEAK")
        print(f"{FAMILY_DISPLAY[family]:<15} | {r['identity']['median']:>12.4f} | {r['fno']['median']:>10.4f} | {r['ratio_median']:>7.1f}x | {status:<20}")
    
    # Save results
    output_dir = Path("reports/model_viz/all5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy to native for JSON
    def to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        return obj
    
    with open(output_dir / "identity_baseline_comparison.json", "w") as f:
        json.dump(to_native(results), f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'identity_baseline_comparison.json'}")


if __name__ == "__main__":
    main()
