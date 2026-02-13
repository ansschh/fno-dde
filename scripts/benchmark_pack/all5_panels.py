#!/usr/bin/env python3
"""
Generate multi-family panel figures for Baseline-All-5 comparison.

Produces:
1. all5_id_bar_median_p95.png - ID performance bar chart
2. all5_id_cdf_relL2.png - CDF panels for each family
3. all5_id_error_vs_time.png - Error vs time panels
4. all5_ood_gap_bars.png - OOD gaps grouped bar chart
5. all5_error_vs_delay.png - Error vs delay scatter panels
6. all5_error_vs_roughness.png - Error vs history roughness panels

Also saves baseline_all5_metrics_full.json with per-sample arrays.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual


# Family display names and order (easiest to hardest based on v2)
FAMILY_ORDER = ["dist_exp", "hutch", "dist_uniform", "vdp", "linear2"]
FAMILY_DISPLAY = {
    "dist_exp": "DistExp",
    "hutch": "Hutch", 
    "linear2": "Linear2",
    "vdp": "VdP",
    "dist_uniform": "DistUniform",
}

# Model paths for baseline v2
MODEL_PATHS = {
    "dist_exp": "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403",
    "hutch": "outputs/baseline_v1/hutch_seed42_20251228_131919",
    "linear2": "outputs/baseline_v1/linear2_seed42_20251228_142839",
    "vdp": "outputs/baseline_v1/vdp_seed42_20251229_020516",
    "dist_uniform": "outputs/baseline_v1/dist_uniform_seed42_20251229_030851",
}

# Data paths
DATA_PATHS = {
    "dist_exp": "data_baseline_v2",
    "hutch": "data_baseline_v1",
    "linear2": "data_baseline_v1",
    "vdp": "data_baseline_v1",
    "dist_uniform": "data_baseline_v1",
}

OOD_SPLITS = {
    "ood_delay": ("data_ood_delay", "test_ood"),
    "ood_history": ("data_ood_history", "test_spline"),
    "ood_horizon": ("data_ood_horizon", "test_horizon"),
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


def evaluate_split(model, dataset, device: str, batch_size: int = 64) -> Dict:
    """Run inference and collect metrics using loss_mask for consistency with training."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_rel_l2 = []
    all_pointwise_err = []  # (n_samples, n_time)
    all_taus = []
    all_roughness = []
    
    # Get tau index from param_names
    param_names = dataset.param_names
    tau_idx = None
    if "tau" in param_names:
        tau_idx = param_names.index("tau")
    elif "tau1" in param_names:
        tau_idx = param_names.index("tau1")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            mask = batch["loss_mask"].to(device)  # Future region mask
            
            y_pred = model(x)
            
            # Denormalize
            y_pred_orig = y_pred * target_std + target_mean
            y_true_orig = y_true * target_std + target_mean
            
            # Apply mask for future region only (consistent with training)
            mask_exp = mask.unsqueeze(-1)  # (B, T, 1)
            
            # Extract params (tensor of shape [batch, n_params])
            params_batch = batch.get("params", None)
            
            for i in range(len(y_pred)):
                # Masked difference (future region only)
                diff_masked = (y_pred_orig[i] - y_true_orig[i]) * mask_exp[i]
                target_masked = y_true_orig[i] * mask_exp[i]
                
                # Global relL2 with mask (consistent with training evaluation)
                diff_l2 = torch.sqrt((diff_masked ** 2).sum() + 1e-8)
                target_l2 = torch.sqrt((target_masked ** 2).sum() + 1e-8)
                rel_l2 = (diff_l2 / target_l2).item()
                all_rel_l2.append(rel_l2)
                
                # Pointwise error (per time step) - only on masked region
                # Use trajectory-level normalization to avoid denominator collapse
                # when y(t) passes through zero
                diff_per_t = torch.norm(diff_masked, dim=-1)  # (T,)
                # Use global trajectory norm instead of per-timestep (prevents 1e8 spikes)
                global_norm = torch.sqrt((target_masked ** 2).sum() / (mask[i].sum() + 1e-8))
                target_per_t = torch.clamp(torch.norm(target_masked, dim=-1), min=global_norm * 0.01)
                pointwise_err = (diff_per_t / (target_per_t + 1e-8)).cpu().numpy()
                # Zero out non-masked timesteps for clean aggregation
                pointwise_err = pointwise_err * mask[i].cpu().numpy()
                all_pointwise_err.append(pointwise_err)
                
                # Extract tau from params tensor
                if params_batch is not None and tau_idx is not None:
                    tau_val = params_batch[i, tau_idx].item()
                    all_taus.append(tau_val)
                
                # Compute history roughness from input (first ~64 points are history)
                hist = x[i, :64].cpu().numpy()  # (n_hist, channels)
                if len(hist) > 1:
                    fd = np.diff(hist, axis=0)
                    roughness = np.sqrt(np.mean(fd**2))
                    all_roughness.append(roughness)
    
    rel_l2 = np.array(all_rel_l2)
    pointwise_err = np.array(all_pointwise_err)
    
    # Compute time-aggregated metrics
    n_time = pointwise_err.shape[1] if len(pointwise_err) > 0 else 0
    t_grid = np.linspace(0, 15.0, n_time) if n_time > 0 else np.array([])
    
    err_vs_time = {
        "t": t_grid.tolist(),
        "mean": np.mean(pointwise_err, axis=0).tolist() if len(pointwise_err) > 0 else [],
        "median": np.median(pointwise_err, axis=0).tolist() if len(pointwise_err) > 0 else [],
        "p90": np.percentile(pointwise_err, 90, axis=0).tolist() if len(pointwise_err) > 0 else [],
    }
    
    return {
        "n_samples": len(rel_l2),
        "median": float(np.median(rel_l2)),
        "mean": float(np.mean(rel_l2)),
        "std": float(np.std(rel_l2)),
        "p90": float(np.percentile(rel_l2, 90)),
        "p95": float(np.percentile(rel_l2, 95)),
        "per_sample": rel_l2.tolist(),
        "err_vs_time": err_vs_time,
        "taus": all_taus if all_taus else None,
        "roughness": all_roughness if all_roughness else None,
    }


def collect_all_metrics(device: str = "cuda") -> Dict:
    """Collect metrics for all 5 families on all splits."""
    all_metrics = {}
    
    for family in FAMILY_ORDER:
        print(f"\n{'='*60}")
        print(f"Processing: {family}")
        print(f"{'='*60}")
        
        model_dir = Path(MODEL_PATHS[family])
        data_dir = DATA_PATHS[family]
        
        # Load model
        print("Loading model...")
        model, config = load_model(model_dir, device)
        
        family_metrics = {"config": {
            "modes": config["model"]["modes"],
            "width": config["model"]["width"],
            "n_layers": config["model"]["n_layers"],
        }}
        
        # ID test
        print("Evaluating ID test...")
        try:
            id_dataset = ShardedDDEDataset(data_dir, family, "test")
            family_metrics["id"] = evaluate_split(model, id_dataset, device)
        except Exception as e:
            print(f"  Warning: {e}")
            family_metrics["id"] = None
        
        # OOD splits
        for split_name, (ood_dir, split_key) in OOD_SPLITS.items():
            print(f"Evaluating {split_name}...")
            try:
                ood_path = Path(ood_dir)
                if ood_path.exists() and (ood_path / family).exists():
                    ood_dataset = ShardedDDEDataset(str(ood_path), family, split_key)
                    family_metrics[split_name] = evaluate_split(model, ood_dataset, device)
                else:
                    family_metrics[split_name] = None
            except Exception as e:
                print(f"  Warning: {e}")
                family_metrics[split_name] = None
        
        all_metrics[family] = family_metrics
    
    return all_metrics


# =============================================================================
# PANEL FIGURE GENERATORS
# =============================================================================

def plot_id_bar_chart(metrics: Dict, output_dir: Path):
    """Figure 1: ID median + p95 bar chart for all 5 families."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    families = []
    medians = []
    p95s = []
    
    for family in FAMILY_ORDER:
        if metrics[family]["id"] is not None:
            families.append(FAMILY_DISPLAY[family])
            medians.append(metrics[family]["id"]["median"])
            p95s.append(metrics[family]["id"]["p95"])
    
    x = np.arange(len(families))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, medians, width, label='Median', color='steelblue')
    bars2 = ax.bar(x + width/2, p95s, width, label='P95', color='coral', alpha=0.8)
    
    ax.set_ylabel('Relative L2 Error')
    ax.set_title('Baseline FNO: ID Test Performance (All 5 Families)')
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, medians):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "all5_id_bar_median_p95.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_id_bar_median_p95.png")


def plot_id_cdf_panels(metrics: Dict, output_dir: Path):
    """Figure 2: CDF panels for each family (ID test)."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=True)
    
    # Determine common x-axis limit
    all_errors = []
    for family in FAMILY_ORDER:
        if metrics[family]["id"] is not None:
            all_errors.extend(metrics[family]["id"]["per_sample"])
    xlim_max = min(np.percentile(all_errors, 99), 2.0)
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if metrics[family]["id"] is None:
            ax.set_visible(False)
            continue
            
        per_sample = np.array(metrics[family]["id"]["per_sample"])
        sorted_err = np.sort(per_sample)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        
        ax.plot(sorted_err, cdf, linewidth=2, color='steelblue')
        
        # Vertical lines at median and p95
        median = metrics[family]["id"]["median"]
        p95 = metrics[family]["id"]["p95"]
        ax.axvline(median, color='red', linestyle='--', alpha=0.7, label=f'Med={median:.3f}')
        ax.axvline(p95, color='orange', linestyle='--', alpha=0.7, label=f'P95={p95:.3f}')
        
        ax.set_xlim(0, xlim_max)
        ax.set_xlabel('relL2')
        ax.set_title(FAMILY_DISPLAY[family])
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('CDF')
    fig.suptitle('Baseline FNO: Per-Sample Error Distribution (ID Test)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "all5_id_cdf_relL2.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_id_cdf_relL2.png")


def plot_error_vs_time_panels(metrics: Dict, output_dir: Path):
    """Figure 3: Error vs time panels for each family (ID test)."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=True)
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if metrics[family]["id"] is None or not metrics[family]["id"]["err_vs_time"]["t"]:
            ax.set_visible(False)
            continue
            
        evt = metrics[family]["id"]["err_vs_time"]
        t = np.array(evt["t"])
        median = np.array(evt["median"])
        p90 = np.array(evt["p90"])
        
        ax.plot(t, median, linewidth=2, label='Median', color='steelblue')
        ax.fill_between(t, median, p90, alpha=0.3, color='steelblue', label='P50-P90')
        
        ax.set_xlabel('Time')
        ax.set_title(FAMILY_DISPLAY[family])
        ax.grid(True, alpha=0.3)
        if family == FAMILY_ORDER[0]:
            ax.legend(fontsize=7)
    
    axes[0].set_ylabel('Relative Error')
    fig.suptitle('Baseline FNO: Error vs Time (ID Test)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "all5_id_error_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_id_error_vs_time.png")


def plot_ood_gap_bars(metrics: Dict, output_dir: Path):
    """Figure 4: OOD gaps grouped bar chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    families = []
    gaps_delay = []
    gaps_history = []
    gaps_horizon = []
    
    for family in FAMILY_ORDER:
        if metrics[family]["id"] is None:
            continue
        families.append(FAMILY_DISPLAY[family])
        id_median = metrics[family]["id"]["median"]
        
        # Delay gap
        if metrics[family].get("ood_delay") is not None:
            gaps_delay.append(metrics[family]["ood_delay"]["median"] / (id_median + 1e-10))
        else:
            gaps_delay.append(np.nan)
        
        # History gap
        if metrics[family].get("ood_history") is not None:
            gaps_history.append(metrics[family]["ood_history"]["median"] / (id_median + 1e-10))
        else:
            gaps_history.append(np.nan)
        
        # Horizon gap
        if metrics[family].get("ood_horizon") is not None:
            gaps_horizon.append(metrics[family]["ood_horizon"]["median"] / (id_median + 1e-10))
        else:
            gaps_horizon.append(np.nan)
    
    x = np.arange(len(families))
    width = 0.25
    
    bars1 = ax.bar(x - width, gaps_delay, width, label='OOD-Delay', color='#e74c3c')
    bars2 = ax.bar(x, gaps_history, width, label='OOD-History', color='#3498db')
    bars3 = ax.bar(x + width, gaps_horizon, width, label='OOD-Horizon', color='#2ecc71')
    
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No gap (1×)')
    
    ax.set_ylabel('OOD/ID Ratio')
    ax.set_title('Baseline FNO: OOD Generalization Gaps')
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars, gaps in [(bars1, gaps_delay), (bars2, gaps_history), (bars3, gaps_horizon)]:
        for bar, val in zip(bars, gaps):
            if not np.isnan(val):
                ax.annotate(f'{val:.1f}×', xy=(bar.get_x() + bar.get_width()/2, val),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "all5_ood_gap_bars.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_ood_gap_bars.png")


def plot_error_vs_delay_panels(metrics: Dict, output_dir: Path):
    """Figure 5: Error vs delay scatter panels."""
    # Collect all errors to determine shared y-axis limit
    all_errors = []
    for family in FAMILY_ORDER:
        if metrics[family]["id"] is not None:
            all_errors.extend(metrics[family]["id"]["per_sample"])
    ylim_max = min(np.percentile(all_errors, 98), 3.0) if all_errors else 2.0
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=True)
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if metrics[family]["id"] is None:
            ax.set_visible(False)
            continue
        
        taus = metrics[family]["id"].get("taus")
        per_sample = metrics[family]["id"]["per_sample"]
        
        if taus is None or len(taus) == 0:
            ax.text(0.5, 0.5, 'No τ data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(FAMILY_DISPLAY[family])
            continue
        
        taus = np.array(taus)
        errors = np.array(per_sample)[:len(taus)]
        
        ax.scatter(taus, errors, alpha=0.3, s=10, color='steelblue')
        
        # Binned mean
        bins = np.linspace(taus.min(), taus.max(), 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        for i in range(len(bins)-1):
            mask = (taus >= bins[i]) & (taus < bins[i+1])
            if mask.sum() > 0:
                bin_means.append(errors[mask].mean())
            else:
                bin_means.append(np.nan)
        
        ax.plot(bin_centers, bin_means, 'r-', linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('τ')
        ax.set_title(FAMILY_DISPLAY[family])
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('relL2')
    axes[0].set_ylim(0, ylim_max)
    fig.suptitle('Baseline FNO: Error vs Delay (ID Test)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "all5_error_vs_delay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_error_vs_delay.png")


def plot_error_vs_roughness_panels(metrics: Dict, output_dir: Path):
    """Figure 6: Error vs history roughness scatter panels."""
    # Collect all errors to determine shared y-axis limit
    all_errors = []
    for family in FAMILY_ORDER:
        if metrics[family]["id"] is not None:
            all_errors.extend(metrics[family]["id"]["per_sample"])
    ylim_max = min(np.percentile(all_errors, 98), 3.0) if all_errors else 2.0
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=True)
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if metrics[family]["id"] is None:
            ax.set_visible(False)
            continue
        
        roughness = metrics[family]["id"].get("roughness")
        per_sample = metrics[family]["id"]["per_sample"]
        
        if roughness is None or len(roughness) == 0:
            ax.text(0.5, 0.5, 'No roughness data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(FAMILY_DISPLAY[family])
            continue
        
        roughness = np.array(roughness)
        errors = np.array(per_sample)[:len(roughness)]
        
        ax.scatter(roughness, errors, alpha=0.3, s=10, color='steelblue')
        
        # Binned mean
        bins = np.linspace(roughness.min(), roughness.max(), 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        for i in range(len(bins)-1):
            mask = (roughness >= bins[i]) & (roughness < bins[i+1])
            if mask.sum() > 0:
                bin_means.append(errors[mask].mean())
            else:
                bin_means.append(np.nan)
        
        ax.plot(bin_centers, bin_means, 'r-', linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Roughness')
        ax.set_title(FAMILY_DISPLAY[family])
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('relL2')
    axes[0].set_ylim(0, ylim_max)
    fig.suptitle('Baseline FNO: Error vs History Roughness (ID Test)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "all5_error_vs_roughness.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_error_vs_roughness.png")


def generate_all_panels(metrics: Dict, output_dir: Path):
    """Generate all multi-family panel figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating multi-family panel figures...")
    plot_id_bar_chart(metrics, output_dir)
    plot_id_cdf_panels(metrics, output_dir)
    plot_error_vs_time_panels(metrics, output_dir)
    plot_ood_gap_bars(metrics, output_dir)
    plot_error_vs_delay_panels(metrics, output_dir)
    plot_error_vs_roughness_panels(metrics, output_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate multi-family panel figures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="reports/model_viz/all5")
    parser.add_argument("--skip_eval", action="store_true", 
                        help="Skip evaluation, load from cached metrics")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "baseline_all5_metrics_full.json"
    
    if args.skip_eval and metrics_path.exists():
        print("Loading cached metrics...")
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        print("Collecting metrics for all 5 families...")
        metrics = collect_all_metrics(args.device)
        
        # Save full metrics (excluding large per_sample arrays for JSON size)
        print(f"\nSaving metrics to {metrics_path}...")
        
        # Convert numpy types to Python native for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(x) for x in obj]
            return obj
        
        metrics_native = convert_to_native(metrics)
        
        with open(metrics_path, "w") as f:
            json.dump(metrics_native, f, indent=2)
    
    # Generate all panel figures
    generate_all_panels(metrics, output_dir)
    
    print(f"\n✓ All panel figures saved to: {output_dir}")
    print(f"  Generated {len(list(output_dir.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
