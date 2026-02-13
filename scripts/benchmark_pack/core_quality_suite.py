#!/usr/bin/env python3
"""
Part B: Core Data Quality Suite (Same for All Families)

B1) Integrity: shapes, NaNs, continuity, positivity
B2) Label fidelity: fast vs reference solver comparison
B3) Residual benchmark: physics check for all 5 families
B4) Diversity metrics: amplitude, oscillations, history roughness
B5) Reproducibility: shard hash verification

Run for each family and split to ensure data quality.
"""
import numpy as np
import json
import hashlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dde.families import DDE_FAMILIES
from dde.solve_python.dde_solver import solve_dde


def load_shard(shard_path: Path):
    """Load a single shard file."""
    data = np.load(shard_path)
    return {
        "phi": data["phi"],
        "y": data["y"],
        "params": data["params"],
        "t_hist": data["t_hist"],
        "t_out": data["t_out"],
    }


def load_manifest(family_dir: Path):
    """Load manifest for a family."""
    with open(family_dir / "manifest.json") as f:
        return json.load(f)


# =============================================================================
# B1: Integrity Checks
# =============================================================================
def get_shard_paths(family_dir: Path, manifest: dict):
    """Get shard paths handling both old and new data formats."""
    if "shards" in manifest:
        return [family_dir / s["filename"] for s in manifest["shards"]]
    else:
        # New format: data in split subdirectories
        shard_paths = []
        for split in ["train", "val", "test"]:
            split_dir = family_dir / split
            if split_dir.exists():
                shard_paths.extend(sorted(split_dir.glob("*.npz")))
        return shard_paths


def check_integrity(family_dir: Path, output_dir: Path):
    """
    Integrity checks:
    - Shapes match manifest
    - No NaNs/Infs
    - Continuity at t=0 (phi[-1] ≈ y[0])
    - Positivity where required
    """
    print("\n  B1: Integrity checks...")
    
    manifest = load_manifest(family_dir)
    family = manifest["family"]
    
    results = {
        "family": family,
        "total_samples": 0,
        "nan_count": 0,
        "inf_count": 0,
        "shape_errors": 0,
        "continuity_violations": 0,
        "positivity_violations": 0,
        "continuity_gaps": [],
    }
    
    def to_int(x):
        return int(x) if hasattr(x, 'item') else x
    
    requires_positive = family in ["hutch", "dist_uniform", "dist_exp"]
    
    shard_paths = get_shard_paths(family_dir, manifest)
    for shard_path in shard_paths:
        data = load_shard(shard_path)
        
        n = len(data["phi"])
        results["total_samples"] += n
        
        # NaN/Inf checks
        results["nan_count"] += np.isnan(data["phi"]).sum() + np.isnan(data["y"]).sum()
        results["inf_count"] += np.isinf(data["phi"]).sum() + np.isinf(data["y"]).sum()
        
        # Continuity at t=0
        for i in range(n):
            gap = np.abs(data["phi"][i, -1] - data["y"][i, 0])
            max_gap = np.max(gap)
            results["continuity_gaps"].append(float(max_gap))
            if max_gap > 0.1:  # Threshold for discontinuity
                results["continuity_violations"] += 1
        
        # Positivity check
        if requires_positive:
            # Check first state dimension
            neg_phi = (data["phi"][:, :, 0] < 0).sum()
            neg_y = (data["y"][:, :, 0] < 0).sum()
            results["positivity_violations"] += neg_phi + neg_y
    
    # Compute continuity stats
    gaps = np.array(results["continuity_gaps"])
    results["continuity_stats"] = {
        "mean": float(np.mean(gaps)),
        "median": float(np.median(gaps)),
        "p95": float(np.percentile(gaps, 95)),
        "max": float(np.max(gaps)),
    }
    del results["continuity_gaps"]  # Don't save raw list
    
    # Convert numpy types to Python types for JSON
    results["total_samples"] = int(results["total_samples"])
    results["nan_count"] = int(results["nan_count"])
    results["inf_count"] = int(results["inf_count"])
    results["continuity_violations"] = int(results["continuity_violations"])
    results["positivity_violations"] = int(results["positivity_violations"])
    
    results["pass"] = bool(
        results["nan_count"] == 0 and
        results["inf_count"] == 0 and
        results["continuity_violations"] == 0 and
        results["positivity_violations"] == 0
    )
    
    print(f"      Total samples: {results['total_samples']}")
    print(f"      NaN count: {results['nan_count']}")
    print(f"      Inf count: {results['inf_count']}")
    print(f"      Continuity violations: {results['continuity_violations']}")
    print(f"      Positivity violations: {results['positivity_violations']}")
    print(f"      PASS: {results['pass']}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "integrity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# B2: Label Fidelity (spot-check vs reference solver)
# =============================================================================
def check_label_fidelity(family_dir: Path, output_dir: Path, n_samples: int = 50):
    """
    Spot-check labels by re-solving with reference solver.
    Compare fast solver output vs fresh solve.
    """
    print("\n  B2: Label fidelity (spot-check)...")
    
    manifest = load_manifest(family_dir)
    family_name = manifest["family"]
    
    if family_name not in DDE_FAMILIES:
        print(f"      SKIP: {family_name} not in DDE_FAMILIES")
        return {"skip": True, "reason": "family not available"}
    
    family = DDE_FAMILIES[family_name]()
    
    # Load test split samples
    test_shards = [s for s in manifest["shards"] if s["split"] == "test"]
    if not test_shards:
        test_shards = manifest["shards"][:1]
    
    shard_path = family_dir / test_shards[0]["filename"]
    data = load_shard(shard_path)
    
    rel_errors = []
    n_checked = min(n_samples, len(data["phi"]))
    
    for i in range(n_checked):
        try:
            phi = data["phi"][i]
            y_stored = data["y"][i]
            t_hist = data["t_hist"]
            t_out = data["t_out"]
            
            # Extract params
            params_arr = data["params"][i]
            if hasattr(params_arr, 'dtype') and params_arr.dtype.names:
                params = {k: float(params_arr[k]) for k in params_arr.dtype.names}
            else:
                param_names = manifest.get("param_names", family.config.param_names)
                params = {name: float(params_arr[j]) for j, name in enumerate(param_names)}
            
            # Re-solve
            T = t_out[-1]
            sol = solve_dde(family, params, phi, t_hist, T=T, n_points=len(t_out))
            y_resolved = sol.y
            
            # Compare
            rel_err = np.linalg.norm(y_stored - y_resolved) / (np.linalg.norm(y_stored) + 1e-10)
            rel_errors.append(rel_err)
            
        except Exception as e:
            print(f"      Sample {i} failed: {e}")
            continue
    
    rel_errors = np.array(rel_errors)
    
    results = {
        "family": family_name,
        "n_checked": len(rel_errors),
        "rel_error": {
            "mean": float(np.mean(rel_errors)),
            "std": float(np.std(rel_errors)),
            "median": float(np.median(rel_errors)),
            "p95": float(np.percentile(rel_errors, 95)),
            "max": float(np.max(rel_errors)),
        },
        "pass": np.median(rel_errors) < 0.01,  # Should match within 1%
    }
    
    print(f"      Samples checked: {len(rel_errors)}")
    print(f"      Rel error median: {np.median(rel_errors):.6f}")
    print(f"      Rel error p95: {np.percentile(rel_errors, 95):.6f}")
    print(f"      PASS: {results['pass']}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_fidelity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# B3: Residual Benchmark (Physics Check)
# =============================================================================
def compute_rhs_residual(family_name: str, y: np.ndarray, phi: np.ndarray, 
                         t_out: np.ndarray, t_hist: np.ndarray, params: dict):
    """
    Compute |y'(t) - RHS(t, y, y_delayed)| for a trajectory.
    Returns residual over time.
    """
    family = DDE_FAMILIES[family_name]()
    dt = t_out[1] - t_out[0]
    
    # Compute y'(t) via finite difference
    y_dot = np.gradient(y, dt, axis=0)
    
    # Build full trajectory for interpolation
    t_full = np.concatenate([t_hist, t_out])
    y_full = np.concatenate([phi, y], axis=0)
    y_interp = CubicSpline(t_full, y_full)
    
    residuals = []
    delays = family.get_delays(params)
    
    # Skip first few points where numerical derivatives are poor
    for i in range(5, len(t_out) - 5):
        t = t_out[i]
        
        # Get delayed values
        y_delayed = {}
        if "tau" in params:
            tau = params["tau"]
            y_delayed["tau"] = y_interp(t - tau)
        if "tau1" in params:
            y_delayed["tau1"] = y_interp(t - params["tau1"])
        if "tau2" in params:
            y_delayed["tau2"] = y_interp(t - params["tau2"])
        
        # Compute RHS
        try:
            rhs = family.rhs(t, y[i], y_delayed, params)
            residual = np.linalg.norm(y_dot[i] - rhs)
            residuals.append(residual)
        except Exception:
            residuals.append(np.nan)
    
    return np.array(residuals)


def check_residual_benchmark(family_dir: Path, output_dir: Path, n_samples: int = 100):
    """
    Physics check: verify RHS consistency for all families.
    """
    print("\n  B3: Residual benchmark (physics check)...")
    
    manifest = load_manifest(family_dir)
    family_name = manifest["family"]
    
    if family_name not in DDE_FAMILIES:
        print(f"      SKIP: {family_name} not in DDE_FAMILIES")
        return {"skip": True, "reason": "family not available"}
    
    # Load test samples
    test_shards = [s for s in manifest["shards"] if s["split"] == "test"]
    if not test_shards:
        test_shards = manifest["shards"][:1]
    
    shard_path = family_dir / test_shards[0]["filename"]
    data = load_shard(shard_path)
    
    all_residuals = []
    n_checked = min(n_samples, len(data["phi"]))
    
    for i in range(n_checked):
        try:
            phi = data["phi"][i]
            y = data["y"][i]
            t_hist = data["t_hist"]
            t_out = data["t_out"]
            
            # Extract params
            params_arr = data["params"][i]
            if hasattr(params_arr, 'dtype') and params_arr.dtype.names:
                params = {k: float(params_arr[k]) for k in params_arr.dtype.names}
            else:
                param_names = manifest.get("param_names", DDE_FAMILIES[family_name]().config.param_names)
                params = {name: float(params_arr[j]) for j, name in enumerate(param_names)}
            
            residuals = compute_rhs_residual(family_name, y, phi, t_out, t_hist, params)
            all_residuals.append(np.nanmedian(residuals))
            
        except Exception as e:
            continue
    
    all_residuals = np.array(all_residuals)
    
    results = {
        "family": family_name,
        "n_checked": len(all_residuals),
        "residual": {
            "mean": float(np.nanmean(all_residuals)),
            "std": float(np.nanstd(all_residuals)),
            "median": float(np.nanmedian(all_residuals)),
            "p95": float(np.nanpercentile(all_residuals, 95)),
        },
        "pass": np.nanmedian(all_residuals) < 0.5,
    }
    
    print(f"      Samples checked: {len(all_residuals)}")
    print(f"      Residual median: {np.nanmedian(all_residuals):.6f}")
    print(f"      Residual p95: {np.nanpercentile(all_residuals, 95):.6f}")
    print(f"      PASS: {results['pass']}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "residual.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# B4: Diversity Metrics
# =============================================================================
def check_diversity(family_dir: Path, output_dir: Path, n_samples: int = 500):
    """
    Diversity metrics:
    - Amplitude range
    - Oscillation count (zero crossings)
    - History roughness (gradient energy)
    - Trajectory energy
    """
    print("\n  B4: Diversity metrics...")
    
    manifest = load_manifest(family_dir)
    family_name = manifest["family"]
    
    # Load all samples from test split
    test_shards = [s for s in manifest["shards"] if s["split"] == "test"]
    if not test_shards:
        test_shards = manifest["shards"]
    
    amplitudes = []
    oscillations = []
    history_roughness = []
    trajectory_energy = []
    
    count = 0
    for shard_info in test_shards:
        if count >= n_samples:
            break
        shard_path = family_dir / shard_info["filename"]
        data = load_shard(shard_path)
        
        for i in range(min(len(data["phi"]), n_samples - count)):
            phi = data["phi"][i]
            y = data["y"][i]
            t_hist = data["t_hist"]
            
            # Amplitude (max - min of first state dim)
            y0 = y[:, 0] if y.ndim > 1 else y
            amplitudes.append(np.max(y0) - np.min(y0))
            
            # Oscillations (zero crossings of detrended signal)
            y_detrend = y0 - np.mean(y0)
            zero_crossings = np.sum(np.diff(np.sign(y_detrend)) != 0)
            oscillations.append(zero_crossings)
            
            # History roughness (L2 norm of gradient)
            phi0 = phi[:, 0] if phi.ndim > 1 else phi
            dt_hist = t_hist[1] - t_hist[0]
            phi_grad = np.gradient(phi0, dt_hist)
            history_roughness.append(np.sqrt(np.mean(phi_grad**2)))
            
            # Trajectory energy
            trajectory_energy.append(np.linalg.norm(y0))
            
            count += 1
    
    results = {
        "family": family_name,
        "n_samples": count,
        "amplitude": {
            "mean": float(np.mean(amplitudes)),
            "std": float(np.std(amplitudes)),
            "median": float(np.median(amplitudes)),
            "p5": float(np.percentile(amplitudes, 5)),
            "p95": float(np.percentile(amplitudes, 95)),
        },
        "oscillations": {
            "mean": float(np.mean(oscillations)),
            "std": float(np.std(oscillations)),
            "median": float(np.median(oscillations)),
        },
        "history_roughness": {
            "mean": float(np.mean(history_roughness)),
            "std": float(np.std(history_roughness)),
            "median": float(np.median(history_roughness)),
        },
        "trajectory_energy": {
            "mean": float(np.mean(trajectory_energy)),
            "std": float(np.std(trajectory_energy)),
            "median": float(np.median(trajectory_energy)),
        },
    }
    
    print(f"      Samples: {count}")
    print(f"      Amplitude: {np.median(amplitudes):.3f} (p5-p95: {np.percentile(amplitudes, 5):.3f}-{np.percentile(amplitudes, 95):.3f})")
    print(f"      Oscillations: {np.median(oscillations):.1f}")
    print(f"      History roughness: {np.median(history_roughness):.4f}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "diversity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot diversity distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].hist(amplitudes, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel("Amplitude (max - min)")
    axes[0, 0].set_title(f"Amplitude Distribution (median={np.median(amplitudes):.2f})")
    
    axes[0, 1].hist(oscillations, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel("Zero crossings")
    axes[0, 1].set_title(f"Oscillation Count (median={np.median(oscillations):.0f})")
    
    axes[1, 0].hist(history_roughness, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel("History gradient RMS")
    axes[1, 0].set_title(f"History Roughness (median={np.median(history_roughness):.3f})")
    
    axes[1, 1].hist(trajectory_energy, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel("||y||_2")
    axes[1, 1].set_title(f"Trajectory Energy (median={np.median(trajectory_energy):.2f})")
    
    plt.suptitle(f"{family_name}: Diversity Metrics")
    plt.tight_layout()
    plt.savefig(output_dir / "diversity.png", dpi=150)
    plt.close()
    
    return results


# =============================================================================
# B5: Reproducibility (shard hash)
# =============================================================================
def check_reproducibility(family_dir: Path, output_dir: Path):
    """
    Compute and store shard hashes for reproducibility verification.
    """
    print("\n  B5: Reproducibility (shard hashes)...")
    
    manifest = load_manifest(family_dir)
    
    hashes = {}
    for shard_info in manifest["shards"]:
        shard_path = family_dir / shard_info["filename"]
        with open(shard_path, "rb") as f:
            content = f.read()
        h = hashlib.sha256(content).hexdigest()[:16]
        hashes[shard_info["filename"]] = h
    
    results = {
        "family": manifest["family"],
        "n_shards": len(hashes),
        "hashes": hashes,
    }
    
    print(f"      Shards hashed: {len(hashes)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "repro.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Also save as simple text
    with open(output_dir / "repro.txt", "w") as f:
        f.write(f"Family: {manifest['family']}\n")
        f.write(f"Shards: {len(hashes)}\n\n")
        for fname, h in hashes.items():
            f.write(f"{h}  {fname}\n")
    
    return results


# =============================================================================
# Run All Checks for a Family
# =============================================================================
def run_quality_suite(family: str, data_dir: Path, output_dir: Path):
    """Run all quality checks for a single family."""
    print(f"\n{'='*70}")
    print(f"Quality Suite: {family}")
    print(f"{'='*70}")
    
    family_dir = data_dir / family
    family_output = output_dir / family
    
    if not family_dir.exists():
        print(f"  ERROR: {family_dir} not found")
        return None
    
    results = {}
    results["integrity"] = check_integrity(family_dir, family_output)
    results["label_fidelity"] = check_label_fidelity(family_dir, family_output)
    results["residual"] = check_residual_benchmark(family_dir, family_output)
    results["diversity"] = check_diversity(family_dir, family_output)
    results["reproducibility"] = check_reproducibility(family_dir, family_output)
    
    # Summary
    all_pass = all(
        r.get("pass", True) for r in results.values() 
        if isinstance(r, dict) and "pass" in r
    )
    
    print(f"\n  Overall: {'✓ PASS' if all_pass else '✗ FAIL'}")
    
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Core data quality suite")
    parser.add_argument("--data_dir", default="data_baseline_v1")
    parser.add_argument("--output_dir", default="reports/data_quality")
    parser.add_argument("--family", default="all", 
                        help="Family to check (or 'all')")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    families = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]
    if args.family != "all":
        families = [args.family]
    
    all_results = {}
    for family in families:
        all_results[family] = run_quality_suite(family, data_dir, output_dir)
    
    # Save summary
    summary = {
        family: {
            "integrity": r["integrity"].get("pass", None) if r else None,
            "label_fidelity": r["label_fidelity"].get("pass", None) if r else None,
            "residual": r["residual"].get("pass", None) if r else None,
        }
        for family, r in all_results.items()
        if r is not None
    }
    
    with open(output_dir / "quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("QUALITY SUITE SUMMARY")
    print("="*70)
    for family, s in summary.items():
        status = "✓" if all(v for v in s.values() if v is not None) else "✗"
        print(f"  {family}: {status}")


if __name__ == "__main__":
    main()
