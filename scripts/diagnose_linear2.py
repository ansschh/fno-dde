#!/usr/bin/env python3
"""
Diagnose why Linear2 is harder than Hutch.

Analyzes:
1. Error vs delay magnitude
2. Error vs stability margin
3. Error vs trajectory complexity (oscillation frequency)
4. Parameter sensitivity analysis
"""
import numpy as np
import torch
import yaml
import json
import sys
from pathlib import Path
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fno1d import FNO1d
from datasets.sharded_dataset import ShardedDDEDataset
from torch.utils.data import DataLoader


def compute_trajectory_complexity(y):
    """Compute trajectory complexity metrics."""
    # Number of zero crossings
    y_centered = y - np.mean(y)
    zero_crossings = np.sum(np.diff(np.sign(y_centered)) != 0)
    
    # Number of peaks
    peaks, _ = find_peaks(y.flatten())
    n_peaks = len(peaks)
    
    # Total variation (smoothness measure)
    total_variation = np.sum(np.abs(np.diff(y.flatten())))
    
    # Dynamic range
    dynamic_range = np.max(y) - np.min(y)
    
    return {
        "zero_crossings": zero_crossings,
        "n_peaks": n_peaks,
        "total_variation": total_variation,
        "dynamic_range": dynamic_range,
    }


def analyze_family(family_name, data_dir, model_dir, device="cuda"):
    """Analyze error patterns for a family."""
    # Load data
    ds = ShardedDDEDataset(data_dir, family_name, "test")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Load model
    with open(Path(model_dir) / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    
    in_ch = ds[0]["input"].shape[1]
    out_ch = ds[0]["target"].shape[1]
    
    model = FNO1d(
        modes=model_cfg["modes"],
        width=model_cfg["width"],
        in_channels=in_ch,
        out_channels=out_ch,
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)
    
    ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    
    # Collect metrics
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Analyzing {family_name}"):
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            params = batch["params"].numpy()[0]
            
            pred = model(x)
            
            # Compute error
            diff = pred - y
            l2_diff = torch.sqrt(torch.sum(diff**2)).item()
            l2_target = torch.sqrt(torch.sum(y**2)).item()
            rel_l2 = l2_diff / (l2_target + 1e-8)
            
            # Get target trajectory
            y_np = y.cpu().numpy()[0, :, 0]
            
            # Compute complexity
            complexity = compute_trajectory_complexity(y_np)
            
            # Extract delay info
            if family_name == "hutch":
                tau = params[2]
                max_tau = tau
            else:  # linear2
                tau1, tau2 = params[3], params[4]
                max_tau = max(tau1, tau2)
            
            results.append({
                "rel_l2": rel_l2,
                "max_tau": max_tau,
                "params": params.tolist(),
                **complexity,
            })
    
    return results


def main():
    print("="*70)
    print("Linear2 Difficulty Diagnosis")
    print("="*70)
    
    # Analyze both families
    hutch_results = analyze_family(
        "hutch", 
        "data_scale/n10240",
        "outputs/scale_curve/hutch_seed42_20251227_204858"
    )
    
    linear2_results = analyze_family(
        "linear2",
        "data_scale/n10240",
        "outputs/scale_curve_linear2/linear2_seed42_20251227_235230"
    )
    
    # Convert to arrays for analysis
    hutch_err = np.array([r["rel_l2"] for r in hutch_results])
    hutch_tau = np.array([r["max_tau"] for r in hutch_results])
    hutch_tv = np.array([r["total_variation"] for r in hutch_results])
    hutch_dr = np.array([r["dynamic_range"] for r in hutch_results])
    
    linear2_err = np.array([r["rel_l2"] for r in linear2_results])
    linear2_tau = np.array([r["max_tau"] for r in linear2_results])
    linear2_tv = np.array([r["total_variation"] for r in linear2_results])
    linear2_dr = np.array([r["dynamic_range"] for r in linear2_results])
    
    print("\n" + "="*70)
    print("1. Overall Error Comparison")
    print("="*70)
    print(f"Hutch:   median={np.median(hutch_err):.4f}, p95={np.percentile(hutch_err, 95):.4f}")
    print(f"Linear2: median={np.median(linear2_err):.4f}, p95={np.percentile(linear2_err, 95):.4f}")
    print(f"Ratio:   {np.median(linear2_err)/np.median(hutch_err):.2f}x")
    
    print("\n" + "="*70)
    print("2. Error vs Delay Magnitude")
    print("="*70)
    
    # Bin by delay
    for family, err, tau in [("Hutch", hutch_err, hutch_tau), ("Linear2", linear2_err, linear2_tau)]:
        bins = [(0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
        print(f"\n{family}:")
        for lo, hi in bins:
            mask = (tau >= lo) & (tau < hi)
            if np.sum(mask) > 0:
                print(f"  τ ∈ [{lo:.1f}, {hi:.1f}): n={np.sum(mask):3d}, median={np.median(err[mask]):.4f}")
    
    print("\n" + "="*70)
    print("3. Trajectory Complexity Comparison")
    print("="*70)
    print(f"Total Variation - Hutch: {np.median(hutch_tv):.2f}, Linear2: {np.median(linear2_tv):.2f}")
    print(f"Dynamic Range   - Hutch: {np.median(hutch_dr):.2f}, Linear2: {np.median(linear2_dr):.2f}")
    
    print("\n" + "="*70)
    print("4. Correlation Analysis")
    print("="*70)
    
    # Correlation between error and metrics
    from scipy.stats import spearmanr
    
    for family, err, tau, tv, dr in [
        ("Hutch", hutch_err, hutch_tau, hutch_tv, hutch_dr),
        ("Linear2", linear2_err, linear2_tau, linear2_tv, linear2_dr)
    ]:
        print(f"\n{family}:")
        corr_tau, _ = spearmanr(err, tau)
        corr_tv, _ = spearmanr(err, tv)
        corr_dr, _ = spearmanr(err, dr)
        print(f"  Error vs τ:           ρ = {corr_tau:.3f}")
        print(f"  Error vs TotalVar:    ρ = {corr_tv:.3f}")
        print(f"  Error vs DynamicRange: ρ = {corr_dr:.3f}")
    
    print("\n" + "="*70)
    print("5. Worst-case Analysis (p95 errors)")
    print("="*70)
    
    # Find worst cases
    hutch_worst_idx = np.argsort(hutch_err)[-10:]
    linear2_worst_idx = np.argsort(linear2_err)[-10:]
    
    print("\nHutch worst cases (top 10):")
    for i in hutch_worst_idx[::-1]:
        r = hutch_results[i]
        print(f"  err={r['rel_l2']:.3f}, τ={r['max_tau']:.2f}, TV={r['total_variation']:.1f}")
    
    print("\nLinear2 worst cases (top 10):")
    for i in linear2_worst_idx[::-1]:
        r = linear2_results[i]
        print(f"  err={r['rel_l2']:.3f}, τ={r['max_tau']:.2f}, TV={r['total_variation']:.1f}")
    
    # Save results
    results = {
        "hutch": {
            "median": float(np.median(hutch_err)),
            "p95": float(np.percentile(hutch_err, 95)),
            "corr_tau": float(spearmanr(hutch_err, hutch_tau)[0]),
            "corr_tv": float(spearmanr(hutch_err, hutch_tv)[0]),
        },
        "linear2": {
            "median": float(np.median(linear2_err)),
            "p95": float(np.percentile(linear2_err, 95)),
            "corr_tau": float(spearmanr(linear2_err, linear2_tau)[0]),
            "corr_tv": float(spearmanr(linear2_err, linear2_tv)[0]),
        }
    }
    
    with open("reports/linear2_diagnosis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to reports/linear2_diagnosis.json")


if __name__ == "__main__":
    main()
