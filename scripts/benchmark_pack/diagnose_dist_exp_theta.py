#!/usr/bin/env python3
"""
Step 2: Diagnose dist_exp θ = λτ distribution

This script analyzes the current dist_exp dataset to show why
delay sensitivity is low: most samples have θ >> 3 where exp(-θ) ≈ 0.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_params(data_dir: Path, family: str = "dist_exp"):
    """Load all parameters from dist_exp dataset."""
    family_dir = data_dir / family
    
    all_params = {"tau": [], "lam": [], "r": [], "K": []}
    
    # Load from split subdirectories
    for split in ["train", "val", "test"]:
        split_dir = family_dir / split
        if not split_dir.exists():
            continue
        
        for shard_path in sorted(split_dir.glob("*.npz")):
            data = np.load(shard_path)
            params = data["params"]
            
            # Handle different param formats
            if hasattr(params, 'dtype') and params.dtype.names:
                for name in ["tau", "lam", "r", "K"]:
                    if name in params.dtype.names:
                        all_params[name].extend(params[name].tolist())
            else:
                # Assume order from manifest
                manifest_path = family_dir / "manifest.json"
                with open(manifest_path) as f:
                    manifest = json.load(f)
                param_names = manifest.get("param_names", ["r", "K", "lam", "tau"])
                
                for i, name in enumerate(param_names):
                    if name in all_params:
                        all_params[name].extend(params[:, i].tolist())
    
    return {k: np.array(v) for k, v in all_params.items()}


def diagnose_theta(data_dir: Path, output_dir: Path):
    """Analyze θ = λτ distribution and create diagnostic report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Diagnosing dist_exp θ = λτ distribution")
    print("="*70)
    
    params = load_params(data_dir)
    
    tau = params["tau"]
    lam = params["lam"]
    theta = lam * tau
    exp_neg_theta = np.exp(-theta)
    
    print(f"\nSamples analyzed: {len(tau)}")
    
    # Statistics
    print(f"\nτ (tau):")
    print(f"  Range: [{tau.min():.3f}, {tau.max():.3f}]")
    print(f"  Mean: {tau.mean():.3f}, Median: {np.median(tau):.3f}")
    
    print(f"\nλ (lambda):")
    print(f"  Range: [{lam.min():.3f}, {lam.max():.3f}]")
    print(f"  Mean: {lam.mean():.3f}, Median: {np.median(lam):.3f}")
    
    print(f"\nθ = λτ (dimensionless product):")
    print(f"  Range: [{theta.min():.3f}, {theta.max():.3f}]")
    print(f"  Mean: {theta.mean():.3f}, Median: {np.median(theta):.3f}")
    print(f"  p5: {np.percentile(theta, 5):.3f}, p95: {np.percentile(theta, 95):.3f}")
    
    print(f"\nexp(-θ) (delay weight):")
    print(f"  Range: [{exp_neg_theta.min():.4f}, {exp_neg_theta.max():.4f}]")
    print(f"  Mean: {exp_neg_theta.mean():.4f}, Median: {np.median(exp_neg_theta):.4f}")
    
    # Key diagnostic: what fraction is in "good" vs "bad" regime
    good_regime = (theta >= 0.3) & (theta <= 2.5)
    ode_regime = theta > 3.0
    
    print(f"\n--- REGIME ANALYSIS ---")
    print(f"  θ ∈ [0.3, 2.5] (good delay regime): {100*good_regime.mean():.1f}%")
    print(f"  θ > 3.0 (ODE-ish, exp(-θ) < 5%):    {100*ode_regime.mean():.1f}%")
    print(f"  θ > 5.0 (very ODE-ish):             {100*(theta > 5.0).mean():.1f}%")
    
    # Save results
    results = {
        "n_samples": len(tau),
        "tau": {
            "min": float(tau.min()),
            "max": float(tau.max()),
            "mean": float(tau.mean()),
            "median": float(np.median(tau)),
        },
        "lambda": {
            "min": float(lam.min()),
            "max": float(lam.max()),
            "mean": float(lam.mean()),
            "median": float(np.median(lam)),
        },
        "theta": {
            "min": float(theta.min()),
            "max": float(theta.max()),
            "mean": float(theta.mean()),
            "median": float(np.median(theta)),
            "p5": float(np.percentile(theta, 5)),
            "p95": float(np.percentile(theta, 95)),
        },
        "exp_neg_theta": {
            "min": float(exp_neg_theta.min()),
            "max": float(exp_neg_theta.max()),
            "mean": float(exp_neg_theta.mean()),
            "median": float(np.median(exp_neg_theta)),
        },
        "regime_analysis": {
            "good_regime_fraction": float(good_regime.mean()),
            "ode_regime_fraction": float(ode_regime.mean()),
            "very_ode_fraction": float((theta > 5.0).mean()),
        },
        "diagnosis": "PROBLEM: Most samples in ODE-ish regime (θ >> 3)" if ode_regime.mean() > 0.3 else "OK",
    }
    
    with open(output_dir / "theta_distribution.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # τ histogram
    axes[0, 0].hist(tau, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel("τ (tau)")
    axes[0, 0].set_title(f"τ Distribution (n={len(tau)})")
    axes[0, 0].axvline(np.median(tau), color='r', linestyle='--', label=f"median={np.median(tau):.2f}")
    axes[0, 0].legend()
    
    # λ histogram
    axes[0, 1].hist(lam, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel("λ (lambda)")
    axes[0, 1].set_title(f"λ Distribution")
    axes[0, 1].axvline(np.median(lam), color='r', linestyle='--', label=f"median={np.median(lam):.2f}")
    axes[0, 1].legend()
    
    # θ histogram with regime zones
    axes[1, 0].hist(theta, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvspan(0.3, 2.5, alpha=0.2, color='green', label='Good regime [0.3, 2.5]')
    axes[1, 0].axvspan(3.0, theta.max()+0.5, alpha=0.2, color='red', label='ODE-ish (θ > 3)')
    axes[1, 0].set_xlabel("θ = λτ")
    axes[1, 0].set_title(f"θ Distribution — {100*ode_regime.mean():.0f}% in ODE-ish regime!")
    axes[1, 0].legend()
    
    # exp(-θ) histogram
    axes[1, 1].hist(exp_neg_theta, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0.05, color='r', linestyle='--', label='5% threshold')
    axes[1, 1].set_xlabel("exp(-θ) — delay weight")
    axes[1, 1].set_title(f"Delay Weight Distribution")
    axes[1, 1].legend()
    
    plt.suptitle("dist_exp: Why Delay Sensitivity is Low", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "theta_distribution.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_baseline_v1")
    parser.add_argument("--output_dir", default="reports/data_quality/dist_exp")
    args = parser.parse_args()
    
    diagnose_theta(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
