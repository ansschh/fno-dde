#!/usr/bin/env python3
"""
High-signal diagnostic plots for DDE-FNO model evaluation.

Generates plots that:
1. Summarize performance cleanly ("bad" is obvious in one glance)
2. Diagnose failure modes (learn something, not just "it's bad")

Plots generated:
A. Quantile band trajectories (truth vs prediction)
C. ECDF of per-sample relL2 with threshold lines
D. Waterfall/sorted error curve
E. Success rate vs time at threshold ε
G. Amplitude ratio histogram (pred/true)
J. Residual on predictions distribution

Also saves diagnostic NPZ/NPY artifacts per family/run.
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
from typing import Dict, List, Tuple, Optional
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual


FAMILY_ORDER = ["dist_exp", "hutch", "dist_uniform", "vdp", "linear2"]
FAMILY_DISPLAY = {
    "dist_exp": "DistExp",
    "hutch": "Hutch",
    "linear2": "Linear2",
    "vdp": "VdP",
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


def collect_predictions(model, dataset, device: str, batch_size: int = 64) -> Dict:
    """Collect all predictions and compute diagnostics using loss_mask."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    all_rel_l2 = []
    all_pointwise_err = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting predictions", leave=False):
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            mask = batch["loss_mask"].to(device)  # Future region mask
            
            y_pred = model(x)
            
            # Denormalize
            y_pred_orig = y_pred * target_std + target_mean
            y_true_orig = y_true * target_std + target_mean
            
            # Apply mask for future region only
            mask_exp = mask.unsqueeze(-1)  # (B, T, 1)
            
            for i in range(len(y_pred)):
                all_y_true.append(y_true_orig[i].cpu().numpy())
                all_y_pred.append(y_pred_orig[i].cpu().numpy())
                
                # Masked difference (future region only)
                diff_masked = (y_pred_orig[i] - y_true_orig[i]) * mask_exp[i]
                target_masked = y_true_orig[i] * mask_exp[i]
                
                # Global relL2 with mask (consistent with training)
                diff_l2 = torch.sqrt((diff_masked ** 2).sum() + 1e-8)
                target_l2 = torch.sqrt((target_masked ** 2).sum() + 1e-8)
                rel_l2 = (diff_l2 / target_l2).item()
                all_rel_l2.append(rel_l2)
                
                # Pointwise error (per time step) - use trajectory-level normalization
                diff_per_t = torch.norm(diff_masked, dim=-1)
                global_norm = torch.sqrt((target_masked ** 2).sum() / (mask[i].sum() + 1e-8))
                target_per_t = torch.clamp(torch.norm(target_masked, dim=-1), min=global_norm * 0.01)
                pointwise_err = (diff_per_t / (target_per_t + 1e-8)).cpu().numpy()
                all_pointwise_err.append(pointwise_err)
    
    y_true = np.array(all_y_true)  # (N, T, D)
    y_pred = np.array(all_y_pred)
    rel_l2 = np.array(all_rel_l2)
    pointwise_err = np.array(all_pointwise_err)
    
    n_time = y_true.shape[1]
    t = np.linspace(0, 15.0, n_time)
    
    # Compute amplitude ratio (pred/true)
    amplitude_true = y_true.max(axis=1) - y_true.min(axis=1)  # (N, D)
    amplitude_pred = y_pred.max(axis=1) - y_pred.min(axis=1)
    amplitude_ratio = amplitude_pred / (amplitude_true + 1e-10)
    
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "t": t,
        "rel_l2": rel_l2,
        "pointwise_err": pointwise_err,
        "amplitude_true": amplitude_true,
        "amplitude_pred": amplitude_pred,
        "amplitude_ratio": amplitude_ratio,
    }


# =============================================================================
# PLOT A: Quantile Band Trajectories
# =============================================================================

def plot_quantile_band_trajectories(data: Dict, output_dir: Path, family: str, split: str):
    """Plot A: Quantile bands for truth vs prediction."""
    y_true = data["y_true"]  # (N, T, D)
    y_pred = data["y_pred"]
    t = data["t"]
    
    state_dim = y_true.shape[2]
    fig, axes = plt.subplots(1, state_dim, figsize=(6*state_dim, 4), squeeze=False)
    axes = axes.flatten()
    
    for dim in range(state_dim):
        ax = axes[dim]
        true_d = y_true[:, :, dim]  # (N, T)
        pred_d = y_pred[:, :, dim]
        
        # Compute quantiles
        true_median = np.median(true_d, axis=0)
        true_p10 = np.percentile(true_d, 10, axis=0)
        true_p90 = np.percentile(true_d, 90, axis=0)
        
        pred_median = np.median(pred_d, axis=0)
        pred_p10 = np.percentile(pred_d, 10, axis=0)
        pred_p90 = np.percentile(pred_d, 90, axis=0)
        
        # Plot truth
        ax.fill_between(t, true_p10, true_p90, alpha=0.2, color='blue', label='True P10-P90')
        ax.plot(t, true_median, linewidth=2, color='blue', label='True Median')
        
        # Plot prediction
        ax.fill_between(t, pred_p10, pred_p90, alpha=0.2, color='red', label='Pred P10-P90')
        ax.plot(t, pred_median, linewidth=2, color='red', linestyle='--', label='Pred Median')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'State dim {dim}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{FAMILY_DISPLAY[family]} — Quantile Band Trajectories ({split})', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"quantile_bands_{split}.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PLOT C: ECDF of per-sample relL2
# =============================================================================

def plot_ecdf_with_thresholds(data: Dict, output_dir: Path, family: str, split: str,
                               thresholds: List[float] = [0.1, 0.2, 0.5, 1.0]):
    """Plot C: ECDF of per-sample relL2 with threshold lines."""
    rel_l2 = data["rel_l2"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sorted_err = np.sort(rel_l2)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    
    ax.plot(sorted_err, cdf, linewidth=2, color='steelblue')
    
    # Add threshold lines
    colors = ['green', 'orange', 'red', 'darkred']
    for thresh, color in zip(thresholds, colors):
        ax.axvline(thresh, color=color, linestyle='--', alpha=0.7, label=f'ε={thresh}')
        # Annotate fraction below threshold
        frac_below = (rel_l2 < thresh).mean()
        ax.annotate(f'{frac_below:.1%}', xy=(thresh, frac_below), 
                   xytext=(thresh+0.05, frac_below+0.05),
                   fontsize=9, color=color)
    
    # Annotate median and p95
    median = np.median(rel_l2)
    p95 = np.percentile(rel_l2, 95)
    ax.axvline(median, color='black', linestyle='-', alpha=0.5)
    ax.axvline(p95, color='black', linestyle=':', alpha=0.5)
    ax.text(0.98, 0.02, f'Median: {median:.3f}\nP95: {p95:.3f}', 
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Relative L2 Error')
    ax.set_ylabel('CDF (fraction of samples ≤ x)')
    ax.set_title(f'{FAMILY_DISPLAY[family]} — Error ECDF ({split})')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(2.0, np.percentile(rel_l2, 99)))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ecdf_relL2_{split}.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PLOT D: Waterfall / Sorted Error Curve
# =============================================================================

def plot_waterfall_errors(data: Dict, output_dir: Path, family: str, split: str):
    """Plot D: Sorted per-sample error curve (waterfall)."""
    rel_l2 = data["rel_l2"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sorted_err = np.sort(rel_l2)
    ranks = np.arange(len(sorted_err))
    
    ax.fill_between(ranks, 0, sorted_err, alpha=0.3, color='steelblue')
    ax.plot(ranks, sorted_err, linewidth=1, color='steelblue')
    
    # Mark percentiles
    for p, color in [(50, 'green'), (90, 'orange'), (95, 'red')]:
        idx = int(len(sorted_err) * p / 100)
        val = sorted_err[idx]
        ax.axhline(val, color=color, linestyle='--', alpha=0.7, label=f'P{p}: {val:.3f}')
        ax.axvline(idx, color=color, linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Sample Rank (sorted by error)')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title(f'{FAMILY_DISPLAY[family]} — Sorted Error Curve ({split})')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"waterfall_{split}.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PLOT E: Success Rate vs Time
# =============================================================================

def plot_success_rate_vs_time(data: Dict, output_dir: Path, family: str, split: str,
                               thresholds: List[float] = [0.1, 0.2, 0.5]):
    """Plot E: Fraction of samples with error < threshold vs time."""
    pointwise_err = data["pointwise_err"]  # (N, T)
    t = data["t"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['green', 'orange', 'red']
    for thresh, color in zip(thresholds, colors):
        success_rate = (pointwise_err < thresh).mean(axis=0)  # (T,)
        ax.plot(t, success_rate, linewidth=2, color=color, label=f'ε = {thresh}')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Success Rate (fraction with error < ε)')
    ax.set_title(f'{FAMILY_DISPLAY[family]} — Success Rate vs Time ({split})')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add annotation for key insight
    mid_time_idx = len(t) // 2
    late_time_idx = int(len(t) * 0.8)
    for thresh, color in zip(thresholds[:1], colors[:1]):  # Just annotate first threshold
        success_late = (pointwise_err[:, late_time_idx] < thresh).mean()
        ax.annotate(f'{success_late:.0%} at t={t[late_time_idx]:.1f}',
                   xy=(t[late_time_idx], success_late),
                   xytext=(t[late_time_idx]-2, success_late-0.15),
                   arrowprops=dict(arrowstyle='->', color=color),
                   fontsize=9, color=color)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"success_rate_{split}.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PLOT G: Amplitude Ratio Histogram
# =============================================================================

def plot_amplitude_ratio_histogram(data: Dict, output_dir: Path, family: str, split: str):
    """Plot G: Histogram of amplitude_pred / amplitude_true."""
    amplitude_ratio = data["amplitude_ratio"]  # (N, D)
    
    state_dim = amplitude_ratio.shape[1]
    fig, axes = plt.subplots(1, state_dim, figsize=(5*state_dim, 4), squeeze=False)
    axes = axes.flatten()
    
    for dim in range(state_dim):
        ax = axes[dim]
        ratios = amplitude_ratio[:, dim]
        ratios = ratios[(ratios > 0.1) & (ratios < 3.0)]  # Filter outliers for visualization
        
        ax.hist(ratios, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (1.0)')
        
        median_ratio = np.median(ratios)
        ax.axvline(median_ratio, color='green', linestyle='-', linewidth=2, label=f'Median: {median_ratio:.2f}')
        
        ax.set_xlabel(f'Amplitude Ratio (Pred/True) — dim {dim}')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Annotate interpretation
        if median_ratio < 0.9:
            ax.text(0.02, 0.98, 'Under-predicted\n(damped)', transform=ax.transAxes,
                   va='top', fontsize=9, color='orange')
        elif median_ratio > 1.1:
            ax.text(0.02, 0.98, 'Over-predicted', transform=ax.transAxes,
                   va='top', fontsize=9, color='red')
    
    fig.suptitle(f'{FAMILY_DISPLAY[family]} — Amplitude Ratio ({split})', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"amplitude_ratio_{split}.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PLOT H: Pred vs True Scatter at Key Times
# =============================================================================

def plot_pred_vs_true_scatter(data: Dict, output_dir: Path, family: str, split: str,
                               time_fracs: List[float] = [0.1, 0.25, 0.5, 0.75]):
    """Plot H: Scatter of pred vs true at key time points."""
    y_true = data["y_true"]  # (N, T, D)
    y_pred = data["y_pred"]
    t = data["t"]
    
    state_dim = y_true.shape[2]
    n_times = len(time_fracs)
    
    fig, axes = plt.subplots(state_dim, n_times, figsize=(4*n_times, 4*state_dim), squeeze=False)
    
    for dim in range(state_dim):
        for col, frac in enumerate(time_fracs):
            ax = axes[dim, col]
            t_idx = int(frac * len(t))
            t_val = t[t_idx]
            
            true_vals = y_true[:, t_idx, dim]
            pred_vals = y_pred[:, t_idx, dim]
            
            ax.scatter(true_vals, pred_vals, alpha=0.3, s=10, color='steelblue')
            
            # Diagonal line
            lims = [min(true_vals.min(), pred_vals.min()), 
                    max(true_vals.max(), pred_vals.max())]
            ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
            
            # Correlation
            corr = np.corrcoef(true_vals, pred_vals)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if dim == 0:
                ax.set_title(f't = {t_val:.1f}', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'Pred (dim {dim})')
            ax.set_xlabel(f'True (dim {dim})')
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{FAMILY_DISPLAY[family]} — Pred vs True Scatter ({split})', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_pred_vs_true_{split}.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MULTI-FAMILY 5-PANEL PLOTS
# =============================================================================

def plot_5panel_ecdf(all_data: Dict[str, Dict], output_dir: Path, split: str,
                     thresholds: List[float] = [0.1, 0.2, 0.5, 1.0]):
    """5-panel ECDF across all families."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    
    # Compute shared x-limit
    all_errors = np.concatenate([d["rel_l2"] for d in all_data.values()])
    xlim = min(2.0, np.percentile(all_errors, 99))
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if family not in all_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(FAMILY_DISPLAY[family])
            continue
        
        rel_l2 = all_data[family]["rel_l2"]
        sorted_err = np.sort(rel_l2)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        
        ax.plot(sorted_err, cdf, linewidth=2, color='steelblue')
        
        # Threshold lines
        colors = ['green', 'orange', 'red', 'darkred']
        for thresh, color in zip(thresholds, colors):
            ax.axvline(thresh, color=color, linestyle='--', alpha=0.5)
        
        # Annotate median
        median = np.median(rel_l2)
        p95 = np.percentile(rel_l2, 95)
        ax.text(0.95, 0.05, f'Med: {median:.3f}\nP95: {p95:.3f}',
               transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(FAMILY_DISPLAY[family])
        ax.set_xlim(0, xlim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('relL2')
    
    axes[0].set_ylabel('CDF')
    fig.suptitle(f'Error ECDF — {split.upper()}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"all5_ecdf_{split}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_ecdf_{split}.png")


def plot_5panel_success_rate(all_data: Dict[str, Dict], output_dir: Path, split: str,
                              threshold: float = 0.2):
    """5-panel success rate vs time."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if family not in all_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(FAMILY_DISPLAY[family])
            continue
        
        pointwise_err = all_data[family]["pointwise_err"]
        t = all_data[family]["t"]
        
        success_rate = (pointwise_err < threshold).mean(axis=0)
        ax.plot(t, success_rate, linewidth=2, color='steelblue')
        ax.fill_between(t, 0, success_rate, alpha=0.2, color='steelblue')
        
        # Annotate late-time success
        late_idx = int(len(t) * 0.8)
        late_success = success_rate[late_idx]
        ax.annotate(f'{late_success:.0%}', xy=(t[late_idx], late_success),
                   xytext=(t[late_idx]-2, late_success+0.1),
                   fontsize=9, color='red')
        
        ax.set_title(FAMILY_DISPLAY[family])
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    axes[0].set_ylabel(f'Success Rate (err < {threshold})')
    fig.suptitle(f'Success Rate vs Time (ε={threshold}) — {split.upper()}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"all5_success_rate_{split}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_success_rate_{split}.png")


def plot_5panel_amplitude_ratio(all_data: Dict[str, Dict], output_dir: Path, split: str):
    """5-panel amplitude ratio histogram."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if family not in all_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(FAMILY_DISPLAY[family])
            continue
        
        amplitude_ratio = all_data[family]["amplitude_ratio"]
        # Use first state dimension or mean across dims
        ratios = amplitude_ratio.mean(axis=1) if amplitude_ratio.ndim > 1 else amplitude_ratio
        ratios = ratios[(ratios > 0.1) & (ratios < 3.0)]
        
        ax.hist(ratios, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2)
        
        median_ratio = np.median(ratios)
        ax.axvline(median_ratio, color='green', linestyle='-', linewidth=2)
        ax.text(0.95, 0.95, f'Med: {median_ratio:.2f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9)
        
        ax.set_title(FAMILY_DISPLAY[family])
        ax.set_xlabel('Amp. Ratio')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('Count')
    fig.suptitle(f'Amplitude Ratio (Pred/True) — {split.upper()}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"all5_amplitude_ratio_{split}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_amplitude_ratio_{split}.png")


def plot_5panel_waterfall(all_data: Dict[str, Dict], output_dir: Path, split: str):
    """5-panel sorted error curves."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    
    for ax, family in zip(axes, FAMILY_ORDER):
        if family not in all_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(FAMILY_DISPLAY[family])
            continue
        
        rel_l2 = all_data[family]["rel_l2"]
        sorted_err = np.sort(rel_l2)
        ranks = np.arange(len(sorted_err)) / len(sorted_err)  # Normalize to [0, 1]
        
        ax.fill_between(ranks, sorted_err, alpha=0.3, color='steelblue')
        ax.plot(ranks, sorted_err, linewidth=1, color='steelblue')
        
        # Mark p50 and p95
        p50 = np.median(rel_l2)
        p95 = np.percentile(rel_l2, 95)
        ax.axhline(p50, color='green', linestyle='--', alpha=0.7)
        ax.axhline(p95, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(FAMILY_DISPLAY[family])
        ax.set_xlabel('Sample Percentile')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('relL2')
    axes[0].set_yscale('log')
    fig.suptitle(f'Sorted Error Curve — {split.upper()}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"all5_waterfall_{split}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all5_waterfall_{split}.png")


# =============================================================================
# MAIN
# =============================================================================

def generate_diagnostics_for_family(family: str, device: str = "cuda"):
    """Generate all diagnostic plots for a single family."""
    model_dir = Path(MODEL_PATHS[family])
    data_dir = DATA_PATHS[family]
    run_id = model_dir.name
    
    output_dir = Path("reports/model_viz") / family / run_id / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating diagnostics: {family}")
    print(f"{'='*60}")
    
    # Load model
    model, config = load_model(model_dir, device)
    
    # ID test split
    print("Processing ID test split...")
    try:
        dataset = ShardedDDEDataset(data_dir, family, "test")
        data = collect_predictions(model, dataset, device)
        
        # Save diagnostic arrays
        np.save(output_dir / "per_sample_relL2_id.npy", data["rel_l2"])
        np.save(output_dir / "amplitude_ratio_id.npy", data["amplitude_ratio"])
        np.savez(output_dir / "error_curves_id.npz",
                 t=data["t"],
                 pointwise_err_mean=data["pointwise_err"].mean(axis=0),
                 pointwise_err_median=np.median(data["pointwise_err"], axis=0),
                 pointwise_err_p90=np.percentile(data["pointwise_err"], 90, axis=0),
                 pointwise_err_p95=np.percentile(data["pointwise_err"], 95, axis=0))
        
        # Generate per-family plots
        plot_quantile_band_trajectories(data, output_dir, family, "id")
        plot_ecdf_with_thresholds(data, output_dir, family, "id")
        plot_waterfall_errors(data, output_dir, family, "id")
        plot_success_rate_vs_time(data, output_dir, family, "id")
        plot_amplitude_ratio_histogram(data, output_dir, family, "id")
        plot_pred_vs_true_scatter(data, output_dir, family, "id")
        
        print(f"  Saved diagnostics to: {output_dir}")
        return data
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def generate_all5_diagnostics(device: str = "cuda"):
    """Generate multi-family 5-panel diagnostic plots."""
    all_data = {}
    
    # Collect data for all families
    for family in FAMILY_ORDER:
        data = generate_diagnostics_for_family(family, device)
        if data is not None:
            all_data[family] = data
    
    # Generate 5-panel plots
    output_dir = Path("reports/model_viz/all5_panels")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating 5-panel diagnostic plots")
    print(f"{'='*60}")
    
    plot_5panel_ecdf(all_data, output_dir, "id")
    plot_5panel_success_rate(all_data, output_dir, "id")
    plot_5panel_amplitude_ratio(all_data, output_dir, "id")
    plot_5panel_waterfall(all_data, output_dir, "id")
    
    print(f"\n✓ All diagnostic plots saved")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate diagnostic plots")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--family", default=None, help="Single family or 'all'")
    args = parser.parse_args()
    
    if args.family and args.family != "all":
        generate_diagnostics_for_family(args.family, args.device)
    else:
        generate_all5_diagnostics(args.device)


if __name__ == "__main__":
    main()
