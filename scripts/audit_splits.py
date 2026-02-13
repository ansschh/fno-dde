#!/usr/bin/env python3
"""
OOD Split Audit Script

Audits all dataset splits to verify OOD definitions and identify distribution shifts.
Critical for explaining anomalies like "Linear2 OOD-delay better than ID".

Computes:
A) Delay distributions (tau histograms, ranges)
B) Parameter distributions (family-specific)
C) Difficulty proxies from ground-truth y (max_abs, l2_norm, amplitude)
D) History proxies (phi stats, roughness)
E) Normalization sanity checks

Output:
- reports/split_audit/{family}_split_audit.json
- reports/split_audit/{family}_split_audit.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
from collections import defaultdict
from datetime import datetime


def load_shard_data(shard_path: Path) -> dict:
    """Load data from a single shard file."""
    data = np.load(shard_path)
    return {k: data[k] for k in data.keys()}


def load_split_data(data_dir: Path, family: str, split: str) -> dict:
    """Load all shards for a given split."""
    split_dir = data_dir / family / split
    if not split_dir.exists():
        return None
    
    all_data = defaultdict(list)
    shard_files = sorted(split_dir.glob("shard_*.npz"))
    
    if not shard_files:
        return None
    
    for shard_file in shard_files:
        shard_data = load_shard_data(shard_file)
        for k, v in shard_data.items():
            if v.ndim > 0:  # Skip scalars
                all_data[k].append(v)
    
    # Concatenate all shards
    result = {}
    for k, v_list in all_data.items():
        if v_list:
            result[k] = np.concatenate(v_list, axis=0)
    
    return result


def compute_delay_stats_hutch(params: np.ndarray, lags: np.ndarray) -> dict:
    """Compute delay statistics for Hutch family (single tau)."""
    tau = lags.flatten() if lags is not None else params[:, 2]  # tau is 3rd param
    
    return {
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "tau_mean": float(tau.mean()),
        "tau_std": float(tau.std()),
        "tau_median": float(np.median(tau)),
        "tau_histogram": {
            "bins": np.linspace(0, 2.5, 26).tolist(),
            "counts": np.histogram(tau, bins=np.linspace(0, 2.5, 26))[0].tolist()
        }
    }


def compute_delay_stats_linear2(params: np.ndarray, lags: np.ndarray) -> dict:
    """Compute delay statistics for Linear2 family (two taus)."""
    # Linear2 params: [a, b1, b2, tau1, tau2]
    if lags is not None and lags.shape[1] >= 2:
        tau1 = lags[:, 0]
        tau2 = lags[:, 1]
    else:
        tau1 = params[:, 3]
        tau2 = params[:, 4]
    
    max_tau = np.maximum(tau1, tau2)
    tau_diff = np.abs(tau1 - tau2)
    
    return {
        "tau1_min": float(tau1.min()),
        "tau1_max": float(tau1.max()),
        "tau1_mean": float(tau1.mean()),
        "tau2_min": float(tau2.min()),
        "tau2_max": float(tau2.max()),
        "tau2_mean": float(tau2.mean()),
        "max_tau_min": float(max_tau.min()),
        "max_tau_max": float(max_tau.max()),
        "max_tau_mean": float(max_tau.mean()),
        "tau_diff_mean": float(tau_diff.mean()),
        "tau_diff_std": float(tau_diff.std()),
        "max_tau_histogram": {
            "bins": np.linspace(0, 2.5, 26).tolist(),
            "counts": np.histogram(max_tau, bins=np.linspace(0, 2.5, 26))[0].tolist()
        },
        "tau_diff_histogram": {
            "bins": np.linspace(0, 2.0, 21).tolist(),
            "counts": np.histogram(tau_diff, bins=np.linspace(0, 2.0, 21))[0].tolist()
        }
    }


def compute_param_stats_hutch(params: np.ndarray) -> dict:
    """Compute parameter statistics for Hutch family."""
    # Hutch params: [r, K, tau]
    r = params[:, 0]
    K = params[:, 1]
    
    return {
        "r_min": float(r.min()),
        "r_max": float(r.max()),
        "r_mean": float(r.mean()),
        "r_std": float(r.std()),
        "K_min": float(K.min()),
        "K_max": float(K.max()),
        "K_mean": float(K.mean()),
        "K_std": float(K.std()),
    }


def compute_param_stats_linear2(params: np.ndarray) -> dict:
    """Compute parameter statistics for Linear2 family."""
    # Linear2 params: [a, b1, b2, tau1, tau2]
    a = params[:, 0]
    b1 = params[:, 1]
    b2 = params[:, 2]
    
    # Sign patterns
    b1_pos = (b1 > 0).sum()
    b2_pos = (b2 > 0).sum()
    both_pos = ((b1 > 0) & (b2 > 0)).sum()
    both_neg = ((b1 < 0) & (b2 < 0)).sum()
    mixed = len(b1) - both_pos - both_neg
    
    return {
        "a_min": float(a.min()),
        "a_max": float(a.max()),
        "a_mean": float(a.mean()),
        "a_std": float(a.std()),
        "b1_min": float(b1.min()),
        "b1_max": float(b1.max()),
        "b1_mean": float(b1.mean()),
        "b2_min": float(b2.min()),
        "b2_max": float(b2.max()),
        "b2_mean": float(b2.mean()),
        "sign_patterns": {
            "b1_positive_frac": float(b1_pos / len(b1)),
            "b2_positive_frac": float(b2_pos / len(b2)),
            "both_positive_frac": float(both_pos / len(b1)),
            "both_negative_frac": float(both_neg / len(b1)),
            "mixed_signs_frac": float(mixed / len(b1)),
        }
    }


def compute_difficulty_proxies(y: np.ndarray) -> dict:
    """Compute difficulty proxies from ground-truth trajectories."""
    # y shape: (n_samples, n_time, n_dim)
    n_samples = y.shape[0]
    
    # Per-sample statistics
    max_abs_y = np.abs(y).max(axis=(1, 2))  # Max absolute value per sample
    l2_norm_y = np.sqrt((y ** 2).sum(axis=(1, 2)))  # L2 norm per sample
    amplitude = y.max(axis=(1, 2)) - y.min(axis=(1, 2))  # Peak-to-peak amplitude
    
    # Detect potential blowups (very large values)
    blowup_threshold = 100.0
    n_blowups = (max_abs_y > blowup_threshold).sum()
    
    return {
        "max_abs_y": {
            "min": float(max_abs_y.min()),
            "max": float(max_abs_y.max()),
            "mean": float(max_abs_y.mean()),
            "median": float(np.median(max_abs_y)),
            "p95": float(np.percentile(max_abs_y, 95)),
            "p99": float(np.percentile(max_abs_y, 99)),
        },
        "l2_norm_y": {
            "min": float(l2_norm_y.min()),
            "max": float(l2_norm_y.max()),
            "mean": float(l2_norm_y.mean()),
            "median": float(np.median(l2_norm_y)),
            "p95": float(np.percentile(l2_norm_y, 95)),
        },
        "amplitude": {
            "min": float(amplitude.min()),
            "max": float(amplitude.max()),
            "mean": float(amplitude.mean()),
            "median": float(np.median(amplitude)),
        },
        "blowup_count": int(n_blowups),
        "blowup_frac": float(n_blowups / n_samples),
    }


def compute_history_proxies(phi: np.ndarray) -> dict:
    """Compute history function proxies."""
    # phi shape: (n_samples, n_hist, n_dim)
    
    max_abs_phi = np.abs(phi).max(axis=(1, 2))
    l2_norm_phi = np.sqrt((phi ** 2).sum(axis=(1, 2)))
    
    # Roughness: RMS of finite differences (derivative proxy)
    dphi = np.diff(phi, axis=1)
    roughness = np.sqrt((dphi ** 2).mean(axis=(1, 2)))
    
    return {
        "max_abs_phi": {
            "min": float(max_abs_phi.min()),
            "max": float(max_abs_phi.max()),
            "mean": float(max_abs_phi.mean()),
            "median": float(np.median(max_abs_phi)),
        },
        "l2_norm_phi": {
            "min": float(l2_norm_phi.min()),
            "max": float(l2_norm_phi.max()),
            "mean": float(l2_norm_phi.mean()),
            "median": float(np.median(l2_norm_phi)),
        },
        "roughness": {
            "min": float(roughness.min()),
            "max": float(roughness.max()),
            "mean": float(roughness.mean()),
            "median": float(np.median(roughness)),
            "p95": float(np.percentile(roughness, 95)),
        }
    }


def audit_split(data_dir: Path, family: str, split: str) -> dict:
    """Run full audit on a single split."""
    data = load_split_data(data_dir, family, split)
    
    if data is None:
        return {"status": "not_found", "path": str(data_dir / family / split)}
    
    result = {
        "status": "ok",
        "path": str(data_dir / family / split),
        "n_samples": len(data.get("y", data.get("params", []))),
    }
    
    # A) Delay distributions
    if family == "hutch":
        result["delay_stats"] = compute_delay_stats_hutch(
            data.get("params"), data.get("lags")
        )
    elif family == "linear2":
        result["delay_stats"] = compute_delay_stats_linear2(
            data.get("params"), data.get("lags")
        )
    
    # B) Parameter distributions
    if "params" in data:
        if family == "hutch":
            result["param_stats"] = compute_param_stats_hutch(data["params"])
        elif family == "linear2":
            result["param_stats"] = compute_param_stats_linear2(data["params"])
    
    # C) Difficulty proxies
    if "y" in data:
        result["difficulty"] = compute_difficulty_proxies(data["y"])
    
    # D) History proxies
    if "phi" in data:
        result["history"] = compute_history_proxies(data["phi"])
    
    return result


def compare_splits(audit_results: dict, id_key: str = "id_test") -> dict:
    """Compare OOD splits against ID baseline."""
    comparisons = {}
    
    if id_key not in audit_results:
        return comparisons
    
    id_data = audit_results[id_key]
    
    for split_name, split_data in audit_results.items():
        if split_name == id_key or split_data.get("status") != "ok":
            continue
        
        comparison = {"split": split_name}
        
        # Compare difficulty
        if "difficulty" in id_data and "difficulty" in split_data:
            id_l2 = id_data["difficulty"]["l2_norm_y"]["median"]
            ood_l2 = split_data["difficulty"]["l2_norm_y"]["median"]
            comparison["l2_norm_ratio"] = ood_l2 / id_l2 if id_l2 > 0 else None
            
            id_amp = id_data["difficulty"]["amplitude"]["median"]
            ood_amp = split_data["difficulty"]["amplitude"]["median"]
            comparison["amplitude_ratio"] = ood_amp / id_amp if id_amp > 0 else None
        
        # Compare history roughness
        if "history" in id_data and "history" in split_data:
            id_rough = id_data["history"]["roughness"]["median"]
            ood_rough = split_data["history"]["roughness"]["median"]
            comparison["roughness_ratio"] = ood_rough / id_rough if id_rough > 0 else None
        
        comparisons[split_name] = comparison
    
    return comparisons


def generate_markdown_report(family: str, audit_results: dict, comparisons: dict) -> str:
    """Generate markdown audit report."""
    lines = [
        f"# Split Audit Report: {family.upper()}",
        f"",
        f"Generated: {datetime.now().isoformat()}",
        f"",
        "## Summary Table",
        "",
        "| Split | N | τ Range | L2 Norm (median) | Amplitude (median) | Roughness (median) |",
        "|-------|---|---------|------------------|--------------------|--------------------|",
    ]
    
    for split_name, data in audit_results.items():
        if data.get("status") != "ok":
            lines.append(f"| {split_name} | - | NOT FOUND | - | - | - |")
            continue
        
        n = data.get("n_samples", "-")
        
        # Tau range
        if "delay_stats" in data:
            ds = data["delay_stats"]
            if "tau_min" in ds:
                tau_range = f"[{ds['tau_min']:.2f}, {ds['tau_max']:.2f}]"
            elif "max_tau_min" in ds:
                tau_range = f"max∈[{ds['max_tau_min']:.2f}, {ds['max_tau_max']:.2f}]"
            else:
                tau_range = "-"
        else:
            tau_range = "-"
        
        # Difficulty
        l2_norm = data.get("difficulty", {}).get("l2_norm_y", {}).get("median", "-")
        amplitude = data.get("difficulty", {}).get("amplitude", {}).get("median", "-")
        roughness = data.get("history", {}).get("roughness", {}).get("median", "-")
        
        l2_str = f"{l2_norm:.2f}" if isinstance(l2_norm, float) else str(l2_norm)
        amp_str = f"{amplitude:.2f}" if isinstance(amplitude, float) else str(amplitude)
        rough_str = f"{roughness:.4f}" if isinstance(roughness, float) else str(roughness)
        
        lines.append(f"| {split_name} | {n} | {tau_range} | {l2_str} | {amp_str} | {rough_str} |")
    
    # Comparisons
    if comparisons:
        lines.extend([
            "",
            "## OOD vs ID Comparisons",
            "",
            "| Split | L2 Norm Ratio | Amplitude Ratio | Roughness Ratio |",
            "|-------|---------------|-----------------|-----------------|",
        ])
        
        for split_name, comp in comparisons.items():
            l2_ratio = comp.get("l2_norm_ratio")
            amp_ratio = comp.get("amplitude_ratio")
            rough_ratio = comp.get("roughness_ratio")
            
            l2_str = f"{l2_ratio:.3f}" if l2_ratio else "-"
            amp_str = f"{amp_ratio:.3f}" if amp_ratio else "-"
            rough_str = f"{rough_ratio:.3f}" if rough_ratio else "-"
            
            lines.append(f"| {split_name} | {l2_str} | {amp_str} | {rough_str} |")
        
        lines.extend([
            "",
            "**Interpretation:**",
            "- L2 Norm Ratio < 1.0 means OOD trajectories have *lower* energy → potentially easier",
            "- Amplitude Ratio < 1.0 means OOD has smaller excursions → potentially easier",
            "- Roughness Ratio > 1.0 means OOD history is rougher → potentially harder for model",
        ])
    
    # Delay distribution details
    lines.extend([
        "",
        "## Delay Distribution Details",
        "",
    ])
    
    for split_name, data in audit_results.items():
        if data.get("status") != "ok" or "delay_stats" not in data:
            continue
        
        ds = data["delay_stats"]
        lines.append(f"### {split_name}")
        lines.append("")
        
        if "tau_mean" in ds:
            lines.append(f"- τ: mean={ds['tau_mean']:.3f}, std={ds['tau_std']:.3f}, median={ds['tau_median']:.3f}")
        if "tau1_mean" in ds:
            lines.append(f"- τ1: mean={ds['tau1_mean']:.3f}, range=[{ds['tau1_min']:.3f}, {ds['tau1_max']:.3f}]")
            lines.append(f"- τ2: mean={ds['tau2_mean']:.3f}, range=[{ds['tau2_min']:.3f}, {ds['tau2_max']:.3f}]")
            lines.append(f"- max(τ1,τ2): mean={ds['max_tau_mean']:.3f}")
            lines.append(f"- |τ1-τ2|: mean={ds['tau_diff_mean']:.3f}, std={ds['tau_diff_std']:.3f}")
        
        lines.append("")
    
    return "\n".join(lines)


def compute_delay_stats_generic(params: np.ndarray, lags: np.ndarray, param_names: list = None) -> dict:
    """Compute delay statistics for generic families."""
    if lags is not None and len(lags) > 0:
        tau = lags.flatten() if lags.ndim > 1 else lags
    elif params is not None and param_names:
        if "tau" in param_names:
            tau_idx = param_names.index("tau")
            tau = params[:, tau_idx]
        else:
            return {"note": "No tau parameter found"}
    else:
        return {"note": "No delay data available"}
    
    return {
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "tau_mean": float(tau.mean()),
        "tau_std": float(tau.std()),
        "tau_median": float(np.median(tau)),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit OOD splits for distribution shifts")
    parser.add_argument("--family", default=None, choices=["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"])
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--output_dir", default="reports/split_audit")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine families to audit
    if args.families:
        families = args.families
    elif args.family:
        families = [args.family]
    else:
        families = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]
    
    for family in families:
        print(f"\n{'=' * 70}")
        print(f"Split Audit: {family.upper()}")
        print(f"{'=' * 70}")
        
        run_family_audit(family, output_dir)


def run_family_audit(family: str, output_dir: Path):
    """Run audit for a single family."""
    # Define data paths based on family
    if family == "dist_exp":
        base_data = "data_baseline_v2"
    else:
        base_data = "data_baseline_v1"
    
    # Define all splits to audit
    splits_config = {
        "id_train": (base_data, "train"),
        "id_val": (base_data, "val"),
        "id_test": (base_data, "test"),
        "ood_delay": ("data_ood_delay", "test_ood"),
        "ood_history": ("data_ood_history", "test_spline"),
        "ood_horizon": ("data_ood_horizon", "test"),
    }
    
    audit_results = {}
    
    for split_name, (data_dir, split) in splits_config.items():
        print(f"\nAuditing {split_name}...")
        result = audit_split(Path(data_dir), family, split)
        audit_results[split_name] = result
        
        if result["status"] == "ok":
            print(f"  ✓ {result['n_samples']} samples")
            if "difficulty" in result:
                d = result["difficulty"]
                print(f"  L2 norm median: {d['l2_norm_y']['median']:.2f}")
                print(f"  Amplitude median: {d['amplitude']['median']:.2f}")
        else:
            print(f"  ✗ Not found: {result['path']}")
    
    # Compute comparisons
    comparisons = compare_splits(audit_results, "id_test")
    
    # Save JSON
    json_path = output_dir / f"{family}_split_audit.json"
    with open(json_path, "w") as f:
        json.dump({
            "family": family,
            "generated_at": datetime.now().isoformat(),
            "splits": audit_results,
            "comparisons": comparisons,
        }, f, indent=2)
    print(f"\nSaved: {json_path}")
    
    # Save markdown
    md_report = generate_markdown_report(family, audit_results, comparisons)
    md_path = output_dir / f"{family}_split_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Saved: {md_path}")
    
    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    if "ood_delay" in comparisons:
        l2_ratio = comparisons["ood_delay"].get("l2_norm_ratio")
        if l2_ratio and l2_ratio < 1.0:
            print(f"  OOD-delay has LOWER L2 norm than ID ({l2_ratio:.3f}x)")
            print("   This may explain why OOD-delay appears 'easier' than ID!")
    
    if "ood_history" in comparisons:
        rough_ratio = comparisons["ood_history"].get("roughness_ratio")
        if rough_ratio and rough_ratio > 1.0:
            print(f"  OOD-history has HIGHER roughness than ID ({rough_ratio:.3f}x)")
            print("   Spline histories may be harder for the model.")


if __name__ == "__main__":
    main()
