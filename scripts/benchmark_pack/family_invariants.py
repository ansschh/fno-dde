#!/usr/bin/env python3
"""
Part A: Family-Specific Invariant Checks

A1) dist_uniform auxiliary identity: m'(t) ≈ (x(t) - x(t-τ))/τ
A2) dist_exp delay sensitivity: does τ actually affect trajectories?
A3) VdP history consistency: v ≈ dx/dt in histories
A4) Linear2 difficulty profiling

These checks verify that the DDEs are doing what we think they're doing.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dde.families import DDE_FAMILIES
from dde.solve_python.dde_solver import solve_dde


def load_samples(data_dir: Path, family: str, split: str = "test", n_samples: int = 100):
    """Load samples from sharded dataset."""
    family_dir = data_dir / family
    manifest_path = family_dir / "manifest.json"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Handle both old format (shards list) and new format (split subdirs)
    if "shards" in manifest:
        split_shards = [s for s in manifest["shards"] if s["split"] == split]
        shard_paths = [family_dir / s["filename"] for s in split_shards]
    else:
        # New format: data is in split subdirectories
        split_dir = family_dir / split
        if split_dir.exists():
            shard_paths = sorted(split_dir.glob("*.npz"))
        else:
            shard_paths = []
    
    samples = []
    param_names = manifest.get("param_names", [])
    
    for shard_path in shard_paths:
        if not shard_path.exists():
            continue
        data = np.load(shard_path)
        
        phi = data["phi"]
        y = data["y"]
        params = data["params"]
        t_hist = data["t_hist"]
        t_out = data["t_out"]
        
        for i in range(min(len(phi), n_samples - len(samples))):
            # Handle different param formats
            if hasattr(params, 'dtype') and params.dtype.names:
                p = {k: float(params[k][i]) for k in params.dtype.names}
            elif len(param_names) > 0:
                p = {name: float(params[i, j]) for j, name in enumerate(param_names)}
            else:
                p = {f"p{j}": float(params[i, j]) for j in range(params.shape[1])}
            
            samples.append({
                "phi": phi[i],
                "y": y[i],
                "params": p,
                "t_hist": t_hist,
                "t_out": t_out,
            })
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break
    
    return samples


# =============================================================================
# A1: dist_uniform auxiliary identity check
# =============================================================================
def check_dist_uniform_auxiliary(data_dir: Path, output_dir: Path, n_samples: int = 100):
    """
    Verify: m'(t) ≈ (x(t) - x(t-τ))/τ
    
    The auxiliary state m tracks the moving average, so its derivative
    should equal the difference quotient.
    """
    print("\n" + "="*70)
    print("A1: dist_uniform auxiliary identity check")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(data_dir, "dist_uniform", n_samples=n_samples)
    
    residuals_median = []
    residuals_p95 = []
    
    for i, sample in enumerate(samples):
        t_out = sample["t_out"]
        y = sample["y"]  # shape (n_out, 2) = [x, m]
        tau = sample["params"]["tau"]
        
        x = y[:, 0]
        m = y[:, 1]
        dt = t_out[1] - t_out[0]
        
        # Compute m'(t) via finite difference (central)
        m_dot = np.gradient(m, dt)
        
        # Compute (x(t) - x(t-τ))/τ
        # Need to interpolate x at t-τ
        # Combine history and output for full trajectory
        t_hist = sample["t_hist"]
        phi = sample["phi"][:, 0]  # x history
        
        t_full = np.concatenate([t_hist, t_out])
        x_full = np.concatenate([phi, x])
        x_interp = CubicSpline(t_full, x_full)
        
        # Evaluate at t - τ (only where t - τ >= t_hist[0])
        valid_mask = t_out - tau >= t_hist[0]
        t_valid = t_out[valid_mask]
        x_delayed = x_interp(t_valid - tau)
        x_current = x[valid_mask]
        
        rhs = (x_current - x_delayed) / tau
        m_dot_valid = m_dot[valid_mask]
        
        # Compute residual
        residual = np.abs(m_dot_valid - rhs)
        residuals_median.append(np.median(residual))
        residuals_p95.append(np.percentile(residual, 95))
    
    results = {
        "check": "dist_uniform_auxiliary_identity",
        "equation": "m'(t) = (x(t) - x(t-τ))/τ",
        "n_samples": len(samples),
        "residual_median": {
            "mean": float(np.mean(residuals_median)),
            "std": float(np.std(residuals_median)),
            "max": float(np.max(residuals_median)),
        },
        "residual_p95": {
            "mean": float(np.mean(residuals_p95)),
            "std": float(np.std(residuals_p95)),
            "max": float(np.max(residuals_p95)),
        },
        "pass": bool(np.mean(residuals_median) < 0.1),
    }
    
    print(f"  Samples checked: {len(samples)}")
    print(f"  Residual median (across samples): {np.mean(residuals_median):.6f} ± {np.std(residuals_median):.6f}")
    print(f"  Residual p95 (across samples): {np.mean(residuals_p95):.6f} ± {np.std(residuals_p95):.6f}")
    print(f"  PASS: {results['pass']}")
    
    # Save results
    with open(output_dir / "aux_identity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot residual distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals_median, bins=30, alpha=0.7, label="Median residual per sample")
    ax.axvline(np.mean(residuals_median), color='r', linestyle='--', label=f"Mean: {np.mean(residuals_median):.4f}")
    ax.set_xlabel("Residual |m'(t) - (x(t)-x(t-τ))/τ|")
    ax.set_ylabel("Count")
    ax.set_title("dist_uniform: Auxiliary Identity Residuals")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "aux_identity_hist.png", dpi=150)
    plt.close()
    
    return results


# =============================================================================
# A2: dist_exp delay sensitivity check (CRITICAL)
# =============================================================================
def check_dist_exp_delay_sensitivity(data_dir: Path, output_dir: Path, n_samples: int = 100):
    """
    CRITICAL CHECK: Does τ actually affect dist_exp trajectories?
    
    For each sample:
    1. Hold history φ and non-delay params fixed
    2. Solve with τ_low=0.5 and τ_high=1.8
    3. Compute relative difference
    
    If sensitivity is near zero, dist_exp is NOT a proper delay benchmark.
    """
    print("\n" + "="*70)
    print("A2: dist_exp delay sensitivity check (CRITICAL)")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    family = DDE_FAMILIES["dist_exp"]()
    samples = load_samples(data_dir, "dist_exp", n_samples=n_samples)
    
    tau_low, tau_high = 0.5, 1.8
    sensitivities = []
    
    for i, sample in enumerate(samples):
        try:
            phi = sample["phi"]
            t_hist = sample["t_hist"]
            params = dict(sample["params"])
            
            # Solve with tau_low
            params_low = params.copy()
            params_low["tau"] = tau_low
            sol_low = solve_dde(family, params_low, phi, t_hist, T=15.0, n_points=256)
            
            # Solve with tau_high
            params_high = params.copy()
            params_high["tau"] = tau_high
            sol_high = solve_dde(family, params_high, phi, t_hist, T=15.0, n_points=256)
            
            # Compute relative difference
            y_low = sol_low.y
            y_high = sol_high.y
            
            diff = np.linalg.norm(y_low - y_high) / (np.linalg.norm(y_low) + 1e-10)
            sensitivities.append(diff)
            
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue
    
    sensitivities = np.array(sensitivities)
    
    results = {
        "check": "dist_exp_delay_sensitivity",
        "tau_low": tau_low,
        "tau_high": tau_high,
        "n_samples": len(sensitivities),
        "sensitivity": {
            "mean": float(np.mean(sensitivities)),
            "std": float(np.std(sensitivities)),
            "median": float(np.median(sensitivities)),
            "p5": float(np.percentile(sensitivities, 5)),
            "p95": float(np.percentile(sensitivities, 95)),
            "min": float(np.min(sensitivities)),
            "max": float(np.max(sensitivities)),
        },
        "pass": bool(np.median(sensitivities) > 0.05),  # At least 5% relative change
        "warning": "LOW SENSITIVITY - delay may not be tested!" if np.median(sensitivities) < 0.05 else None,
    }
    
    print(f"  Samples checked: {len(sensitivities)}")
    print(f"  τ range tested: [{tau_low}, {tau_high}]")
    print(f"  Sensitivity (relL2 diff):")
    print(f"    Mean: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"    Median: {np.median(sensitivities):.4f}")
    print(f"    Range: [{np.min(sensitivities):.4f}, {np.max(sensitivities):.4f}]")
    print(f"  PASS: {results['pass']}")
    if results["warning"]:
        print(f"  ⚠️  WARNING: {results['warning']}")
    
    # Save results
    with open(output_dir / "delay_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot sensitivity distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sensitivities, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.median(sensitivities), color='r', linestyle='--', 
               label=f"Median: {np.median(sensitivities):.4f}")
    ax.axvline(0.05, color='orange', linestyle=':', label="5% threshold")
    ax.set_xlabel(f"Relative trajectory change |x(τ={tau_low}) - x(τ={tau_high})| / |x|")
    ax.set_ylabel("Count")
    ax.set_title("dist_exp: Delay Sensitivity Check")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "delay_sensitivity_hist.png", dpi=150)
    plt.close()
    
    return results


# =============================================================================
# A3: VdP history consistency check
# =============================================================================
def check_vdp_history_consistency(data_dir: Path, output_dir: Path, n_samples: int = 200):
    """
    VdP is 2D with state [x, v] where physically v = dx/dt.
    
    Check that histories satisfy this constraint:
    Δ(t) = v(t) - d/dt[x(t)]
    
    Compare ID vs OOD-history to see if OOD-history is unfairly inconsistent.
    """
    print("\n" + "="*70)
    print("A3: VdP history consistency check")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_consistency(samples):
        """Compute RMS(Δ) for each sample's history."""
        rms_deltas = []
        for sample in samples:
            phi = sample["phi"]  # shape (n_hist, 2) = [x_hist, v_hist]
            t_hist = sample["t_hist"]
            dt = t_hist[1] - t_hist[0]
            
            x_hist = phi[:, 0]
            v_hist = phi[:, 1]
            
            # Compute dx/dt via finite difference
            dx_dt = np.gradient(x_hist, dt)
            
            # Consistency error
            delta = v_hist - dx_dt
            rms = np.sqrt(np.mean(delta**2))
            rms_deltas.append(rms)
        
        return np.array(rms_deltas)
    
    # Load ID samples
    id_samples = load_samples(data_dir, "vdp", split="test", n_samples=n_samples)
    id_rms = compute_consistency(id_samples)
    
    # Load OOD-history samples
    ood_dir = data_dir.parent / "data_ood_history"
    if ood_dir.exists():
        ood_samples = load_samples(ood_dir, "vdp", split="test", n_samples=n_samples)
        ood_rms = compute_consistency(ood_samples)
    else:
        ood_rms = None
        print("  WARNING: OOD-history data not found")
    
    results = {
        "check": "vdp_history_consistency",
        "constraint": "v(t) = dx/dt",
        "id_samples": len(id_samples),
        "id_rms_delta": {
            "mean": float(np.mean(id_rms)),
            "std": float(np.std(id_rms)),
            "median": float(np.median(id_rms)),
            "p95": float(np.percentile(id_rms, 95)),
        },
    }
    
    print(f"  ID samples: {len(id_samples)}")
    print(f"  ID RMS(Δ): {np.mean(id_rms):.6f} ± {np.std(id_rms):.6f}")
    
    if ood_rms is not None and len(ood_rms) > 0:
        results["ood_samples"] = len(ood_samples)
        results["ood_rms_delta"] = {
            "mean": float(np.mean(ood_rms)),
            "std": float(np.std(ood_rms)),
            "median": float(np.median(ood_rms)),
            "p95": float(np.percentile(ood_rms, 95)),
        }
        results["ood_vs_id_ratio"] = float(np.mean(ood_rms) / (np.mean(id_rms) + 1e-10))
        results["pass"] = True  # No strict pass/fail for consistency check
        
        print(f"  OOD samples: {len(ood_samples)}")
        print(f"  OOD RMS(Δ): {np.mean(ood_rms):.6f} ± {np.std(ood_rms):.6f}")
        print(f"  OOD/ID ratio: {results['ood_vs_id_ratio']:.2f}x")
        
        # Flag if OOD is much more inconsistent
        if results["ood_vs_id_ratio"] > 5:
            results["warning"] = "OOD-history may have unfair inconsistency!"
            print(f"  ⚠️  WARNING: {results['warning']}")
    else:
        results["pass"] = True
        print("  OOD-history data not available for comparison")
    
    # Save results
    with open(output_dir / "history_consistency.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(id_rms, bins=30, alpha=0.7, label=f"ID (mean={np.mean(id_rms):.4f})")
    if ood_rms is not None:
        ax.hist(ood_rms, bins=30, alpha=0.7, label=f"OOD-history (mean={np.mean(ood_rms):.4f})")
    ax.set_xlabel("RMS(v - dx/dt) over history")
    ax.set_ylabel("Count")
    ax.set_title("VdP: History Consistency (v = dx/dt)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "history_consistency_hist.png", dpi=150)
    plt.close()
    
    return results


# =============================================================================
# A4: Linear2 difficulty profiling
# =============================================================================
def check_linear2_difficulty(data_dir: Path, output_dir: Path, n_samples: int = 500):
    """
    Profile Linear2 difficulty to understand why it's the hardest family.
    
    Compute:
    - ||y||_2 distribution (relL2 denominator)
    - max|y| distribution
    - Growth indicator: max(y) / y(0)
    - Frequency content
    """
    print("\n" + "="*70)
    print("A4: Linear2 difficulty profiling")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(data_dir, "linear2", split="test", n_samples=n_samples)
    
    norms = []
    max_vals = []
    growth_ratios = []
    
    for sample in samples:
        y = sample["y"][:, 0]  # 1D state
        
        norms.append(np.linalg.norm(y))
        max_vals.append(np.max(np.abs(y)))
        growth_ratios.append(np.max(np.abs(y)) / (np.abs(y[0]) + 1e-10))
    
    results = {
        "check": "linear2_difficulty_profiling",
        "n_samples": len(samples),
        "norm_l2": {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "median": float(np.median(norms)),
            "p5": float(np.percentile(norms, 5)),
            "p95": float(np.percentile(norms, 95)),
        },
        "max_amplitude": {
            "mean": float(np.mean(max_vals)),
            "std": float(np.std(max_vals)),
            "median": float(np.median(max_vals)),
            "p95": float(np.percentile(max_vals, 95)),
        },
        "growth_ratio": {
            "mean": float(np.mean(growth_ratios)),
            "std": float(np.std(growth_ratios)),
            "median": float(np.median(growth_ratios)),
            "p95": float(np.percentile(growth_ratios, 95)),
        },
    }
    
    print(f"  Samples: {len(samples)}")
    print(f"  ||y||_2: {np.median(norms):.2f} (p95: {np.percentile(norms, 95):.2f})")
    print(f"  max|y|: {np.median(max_vals):.2f} (p95: {np.percentile(max_vals, 95):.2f})")
    print(f"  Growth ratio: {np.median(growth_ratios):.2f}x (p95: {np.percentile(growth_ratios, 95):.2f}x)")
    
    # Save results
    with open(output_dir / "difficulty_proxies.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].hist(norms, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel("||y||_2")
    axes[0].set_title("L2 Norm Distribution")
    
    axes[1].hist(max_vals, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel("max|y|")
    axes[1].set_title("Max Amplitude Distribution")
    
    axes[2].hist(np.clip(growth_ratios, 0, 100), bins=30, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel("max|y| / |y(0)|")
    axes[2].set_title("Growth Ratio Distribution")
    
    plt.suptitle("Linear2: Difficulty Proxies")
    plt.tight_layout()
    plt.savefig(output_dir / "difficulty_proxies.png", dpi=150)
    plt.close()
    
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Family-specific invariant checks")
    parser.add_argument("--data_dir", default="data_baseline_v1", help="Base data directory")
    parser.add_argument("--output_dir", default="reports/data_quality", help="Output directory")
    parser.add_argument("--check", choices=["all", "dist_uniform", "dist_exp", "vdp", "linear2"],
                        default="all", help="Which check to run")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    results = {}
    
    if args.check in ["all", "dist_uniform"]:
        results["dist_uniform"] = check_dist_uniform_auxiliary(
            data_dir, output_dir / "dist_uniform"
        )
    
    if args.check in ["all", "dist_exp"]:
        results["dist_exp"] = check_dist_exp_delay_sensitivity(
            data_dir, output_dir / "dist_exp"
        )
    
    if args.check in ["all", "vdp"]:
        results["vdp"] = check_vdp_history_consistency(
            data_dir, output_dir / "vdp"
        )
    
    if args.check in ["all", "linear2"]:
        results["linear2"] = check_linear2_difficulty(
            data_dir, output_dir / "linear2"
        )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for family, res in results.items():
        status = "✓ PASS" if res.get("pass", True) else "✗ FAIL"
        warning = f" ⚠️ {res.get('warning', '')}" if res.get("warning") else ""
        print(f"  {family}: {status}{warning}")


if __name__ == "__main__":
    main()
