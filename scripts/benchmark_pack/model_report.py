#!/usr/bin/env python3
"""
Comprehensive Model Report Generator

Generates standardized visualizations for baseline FNO models:
- A. Training behavior plots
- B. Aggregate performance plots (ID + OOD)
- C. Sample-level trajectory overlays
- D. Diagnostic plots (error vs delay/params/roughness)
- E. Family-specific interpretability plots

Output structure:
reports/model_viz/{family}/{run_id}/
  training_curves.png
  lr_and_grad.png
  error_hist_{split}.png
  error_cdf_{split}.png
  error_vs_time_{split}.png
  split_boxplot.png
  overlays_random_{split}.png
  overlays_worst_{split}.png
  error_vs_delay.png
  error_vs_params.png
  error_vs_roughness.png
  phase_portrait_{split}.png (vdp only)
  summary.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
import json
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual


@dataclass
class EvalResults:
    """Container for evaluation results on a single split."""
    split_name: str
    y_true: np.ndarray  # (n_samples, n_time, state_dim)
    y_pred: np.ndarray
    phi: np.ndarray     # (n_samples, n_hist, state_dim) - history
    t_hist: np.ndarray  # (n_hist,)
    t_out: np.ndarray   # (n_time,)
    params: List[Dict]  # per-sample parameters
    rel_l2: np.ndarray  # (n_samples,) per-sample relL2
    pointwise_rel_err: np.ndarray  # (n_samples, n_time) per time point


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


def evaluate_split(model, dataset, device: str, batch_size: int = 64) -> EvalResults:
    """Run inference and collect per-sample results."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    all_phi = []
    all_params = []
    all_rel_l2 = []
    all_pointwise_err = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            
            y_pred = model(x)
            
            # Denormalize
            y_pred_orig = y_pred * target_std + target_mean
            y_true_orig = y_true * target_std + target_mean
            
            # Per-sample metrics
            for i in range(len(y_pred)):
                diff = y_pred_orig[i] - y_true_orig[i]
                rel_l2 = torch.norm(diff) / (torch.norm(y_true_orig[i]) + 1e-10)
                
                # Pointwise relative error (per time step)
                # Shape: (n_time,)
                pointwise_err = torch.norm(diff, dim=-1) / (torch.norm(y_true_orig[i], dim=-1) + 1e-10)
                
                all_y_true.append(y_true_orig[i].cpu().numpy())
                all_y_pred.append(y_pred_orig[i].cpu().numpy())
                all_rel_l2.append(rel_l2.item())
                all_pointwise_err.append(pointwise_err.cpu().numpy())
            
            # Collect params
            if "params" in batch:
                for p in batch["params"]:
                    if isinstance(p, dict):
                        all_params.append(p)
                    else:
                        all_params.append({})
    
    # Build time grids (approximate)
    n_time = all_y_true[0].shape[0]
    t_out = np.linspace(0, 15.0, n_time)  # Approximate
    t_hist = np.linspace(-2.0, 0, 64)     # Approximate
    
    return EvalResults(
        split_name="",
        y_true=np.array(all_y_true),
        y_pred=np.array(all_y_pred),
        phi=np.zeros((len(all_y_true), 64, all_y_true[0].shape[-1])),  # Placeholder
        t_hist=t_hist,
        t_out=t_out,
        params=all_params if all_params else [{}] * len(all_y_true),
        rel_l2=np.array(all_rel_l2),
        pointwise_rel_err=np.array(all_pointwise_err),
    )


# =============================================================================
# SECTION A: Training Behavior Plots
# =============================================================================

def plot_training_curves(history: Dict, output_dir: Path, best_epoch: int = None):
    """A1: Learning curves (train/val loss, val relL2)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train", alpha=0.8)
    axes[0].plot(epochs, history["val_loss"], label="Val", alpha=0.8)
    if best_epoch:
        axes[0].axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f"Best: {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Val relL2
    axes[1].plot(epochs, history["val_rel_l2"], color='green', alpha=0.8)
    if best_epoch:
        axes[1].axvline(best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val relL2 (median)")
    axes[1].set_title("Validation Relative L2")
    axes[1].grid(True, alpha=0.3)
    
    # Train-val gap
    gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
    axes[2].plot(epochs, gap, color='purple', alpha=0.8)
    axes[2].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Val - Train Loss")
    axes[2].set_title("Generalization Gap")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_lr_schedule(history: Dict, output_dir: Path):
    """A2: Learning rate schedule."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    epochs = np.arange(1, len(history["lr"]) + 1)
    ax.plot(epochs, history["lr"], color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lr_schedule.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION B: Aggregate Performance Plots
# =============================================================================

def plot_error_histogram(results: EvalResults, output_dir: Path, xlim: Tuple[float, float] = None):
    """B1: Error distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    rel_l2 = results.rel_l2
    median = np.median(rel_l2)
    p90 = np.percentile(rel_l2, 90)
    p95 = np.percentile(rel_l2, 95)
    
    # Histogram
    axes[0].hist(rel_l2, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].axvline(median, color='r', linestyle='--', label=f"Median: {median:.4f}")
    axes[0].axvline(p90, color='orange', linestyle='--', label=f"P90: {p90:.4f}")
    axes[0].axvline(p95, color='purple', linestyle='--', label=f"P95: {p95:.4f}")
    axes[0].set_xlabel("Relative L2 Error")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{results.split_name}: Error Distribution")
    axes[0].legend()
    if xlim:
        axes[0].set_xlim(xlim)
    
    # CDF
    sorted_err = np.sort(rel_l2)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[1].plot(sorted_err, cdf, linewidth=2)
    axes[1].axhline(0.5, color='r', linestyle='--', alpha=0.5, label="Median")
    axes[1].axhline(0.9, color='orange', linestyle='--', alpha=0.5, label="P90")
    axes[1].axhline(0.95, color='purple', linestyle='--', alpha=0.5, label="P95")
    axes[1].set_xlabel("Relative L2 Error")
    axes[1].set_ylabel("CDF")
    axes[1].set_title(f"{results.split_name}: Cumulative Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    if xlim:
        axes[1].set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"error_hist_{results.split_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_split_comparison(all_results: Dict[str, EvalResults], output_dir: Path):
    """B2: Box/violin plot comparing splits."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    data = []
    labels = []
    for split_name, results in all_results.items():
        data.append(results.rel_l2)
        labels.append(split_name)
    
    # Violin plot
    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Performance Across Splits")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add median annotations
    for i, d in enumerate(data):
        median = np.median(d)
        ax.annotate(f"{median:.3f}", (i, median), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "split_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_vs_time(results: EvalResults, output_dir: Path):
    """B3: Error vs time curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    t = results.t_out
    err = results.pointwise_rel_err  # (n_samples, n_time)
    
    mean_err = np.mean(err, axis=0)
    median_err = np.median(err, axis=0)
    p90_err = np.percentile(err, 90, axis=0)
    p95_err = np.percentile(err, 95, axis=0)
    
    # Mean and percentiles
    axes[0].plot(t, mean_err, label="Mean", linewidth=2)
    axes[0].plot(t, median_err, label="Median", linewidth=2)
    axes[0].fill_between(t, median_err, p90_err, alpha=0.3, label="P50-P90")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title(f"{results.split_name}: Error vs Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # P95 (tail behavior)
    axes[1].plot(t, p95_err, color='red', linewidth=2, label="P95")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("P95 Relative Error")
    axes[1].set_title(f"{results.split_name}: Tail Error vs Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"error_vs_time_{results.split_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_vs_time_comparison(all_results: Dict[str, EvalResults], output_dir: Path):
    """B3 extended: Compare error vs time across splits."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (split_name, results), color in zip(all_results.items(), colors):
        t = results.t_out
        median_err = np.median(results.pointwise_rel_err, axis=0)
        ax.plot(t, median_err, label=split_name, linewidth=2, color=color)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Median Relative Error")
    ax.set_title("Error vs Time: All Splits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_time_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION C: Sample-Level Trajectory Overlays
# =============================================================================

def plot_trajectory_overlays(results: EvalResults, output_dir: Path, 
                             n_samples: int = 9, selection: str = "random",
                             state_labels: List[str] = None):
    """C1/C2: Trajectory overlays (random, best, median, worst)."""
    n = min(n_samples, len(results.rel_l2))
    
    if selection == "random":
        indices = np.random.choice(len(results.rel_l2), n, replace=False)
        title_suffix = "Random Samples"
    elif selection == "best":
        indices = np.argsort(results.rel_l2)[:n]
        title_suffix = "Best Samples (Lowest Error)"
    elif selection == "worst":
        indices = np.argsort(results.rel_l2)[-n:][::-1]
        title_suffix = "Worst Samples (Highest Error)"
    elif selection == "median":
        mid = len(results.rel_l2) // 2
        indices = np.argsort(results.rel_l2)[mid-n//2:mid+n//2+1][:n]
        title_suffix = "Median Samples"
    else:
        indices = range(n)
        title_suffix = ""
    
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    state_dim = results.y_true.shape[-1]
    
    if state_labels is None:
        state_labels = [f"x_{i}" for i in range(state_dim)]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    t = results.t_out
    
    for ax_idx, sample_idx in enumerate(indices):
        ax = axes[ax_idx]
        y_true = results.y_true[sample_idx]
        y_pred = results.y_pred[sample_idx]
        err = results.rel_l2[sample_idx]
        
        for dim in range(state_dim):
            ax.plot(t, y_true[:, dim], label=f"True {state_labels[dim]}", 
                   linestyle='-', alpha=0.8)
            ax.plot(t, y_pred[:, dim], label=f"Pred {state_labels[dim]}", 
                   linestyle='--', alpha=0.8)
        
        ax.axvline(0, color='k', linestyle=':', alpha=0.3)
        ax.set_title(f"Sample {sample_idx} (relL2={err:.4f})", fontsize=9)
        ax.set_xlabel("Time", fontsize=8)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for ax_idx in range(len(indices), len(axes)):
        axes[ax_idx].set_visible(False)
    
    fig.suptitle(f"{results.split_name}: {title_suffix}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"overlays_{selection}_{results.split_name}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION D: Diagnostic Plots
# =============================================================================

def plot_error_vs_delay(results: EvalResults, output_dir: Path):
    """D1: Error vs delay scatter."""
    if not results.params or "tau" not in results.params[0]:
        return  # Skip if no delay params
    
    taus = [p.get("tau", p.get("tau1", 1.0)) for p in results.params]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(taus, results.rel_l2, alpha=0.5, s=20)
    
    # Binned mean
    tau_arr = np.array(taus)
    bins = np.linspace(tau_arr.min(), tau_arr.max(), 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    for i in range(len(bins)-1):
        mask = (tau_arr >= bins[i]) & (tau_arr < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(results.rel_l2[mask].mean())
        else:
            bin_means.append(np.nan)
    
    ax.plot(bin_centers, bin_means, 'r-', linewidth=2, marker='o', label="Binned Mean")
    
    ax.set_xlabel("Delay τ")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title(f"{results.split_name}: Error vs Delay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"error_vs_delay_{results.split_name}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_constraint_violations(results: EvalResults, output_dir: Path, 
                                requires_positive: bool = True):
    """B6: Constraint violation analysis for positive families."""
    if not requires_positive:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Minimum predicted value per sample
    min_preds = results.y_pred.min(axis=(1, 2))
    
    axes[0].hist(min_preds, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].axvline(0, color='r', linestyle='--', label="Zero threshold")
    axes[0].set_xlabel("Minimum Predicted Value")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{results.split_name}: Min Prediction Distribution")
    axes[0].legend()
    
    # Negativity fraction per sample
    neg_frac = (results.y_pred < 0).mean(axis=(1, 2))
    n_violated = (min_preds < 0).sum()
    pct_violated = 100 * n_violated / len(min_preds)
    
    axes[1].hist(neg_frac[neg_frac > 0], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel("Fraction of Negative Predictions")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{results.split_name}: Negativity ({pct_violated:.1f}% samples violated)")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"constraint_violations_{results.split_name}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION E: Family-Specific Plots
# =============================================================================

def plot_vdp_phase_portrait(results: EvalResults, output_dir: Path, n_samples: int = 6):
    """VdP: Phase portrait (x vs v)."""
    if results.y_true.shape[-1] != 2:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # Select diverse samples
    indices = np.argsort(results.rel_l2)
    selected = [indices[0], indices[len(indices)//4], indices[len(indices)//2],
                indices[3*len(indices)//4], indices[-2], indices[-1]][:n_samples]
    
    for ax, idx in zip(axes, selected):
        y_true = results.y_true[idx]
        y_pred = results.y_pred[idx]
        err = results.rel_l2[idx]
        
        ax.plot(y_true[:, 0], y_true[:, 1], 'b-', label="True", alpha=0.8, linewidth=1.5)
        ax.plot(y_pred[:, 0], y_pred[:, 1], 'r--', label="Pred", alpha=0.8, linewidth=1.5)
        ax.plot(y_true[0, 0], y_true[0, 1], 'go', markersize=8, label="Start")
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title(f"Sample {idx} (relL2={err:.4f})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"{results.split_name}: Phase Portraits (x vs v)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"phase_portrait_{results.split_name}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN REPORT GENERATOR
# =============================================================================

def generate_summary_md(family: str, config: Dict, all_results: Dict[str, EvalResults],
                        output_dir: Path, best_epoch: int):
    """Generate summary markdown file."""
    lines = [
        f"# Model Report: {family}",
        "",
        f"**Generated:** {np.datetime64('today')}",
        "",
        "## Configuration",
        "",
        f"- **Model:** FNO1dResidual",
        f"- **Modes:** {config['model']['modes']}",
        f"- **Width:** {config['model']['width']}",
        f"- **Layers:** {config['model']['n_layers']}",
        f"- **Epochs:** {config['epochs']}",
        f"- **Best Epoch:** {best_epoch}",
        "",
        "## Performance Summary",
        "",
        "| Split | N | Median | P90 | P95 |",
        "|-------|---|--------|-----|-----|",
    ]
    
    for split_name, results in all_results.items():
        n = len(results.rel_l2)
        median = np.median(results.rel_l2)
        p90 = np.percentile(results.rel_l2, 90)
        p95 = np.percentile(results.rel_l2, 95)
        lines.append(f"| {split_name} | {n} | {median:.4f} | {p90:.4f} | {p95:.4f} |")
    
    if "id" in all_results:
        id_median = np.median(all_results["id"].rel_l2)
        lines.extend([
            "",
            "## OOD Gaps",
            "",
            "| Split | Gap (vs ID) |",
            "|-------|-------------|",
        ])
        for split_name, results in all_results.items():
            if split_name != "id":
                gap = np.median(results.rel_l2) / (id_median + 1e-10)
                lines.append(f"| {split_name} | {gap:.2f}x |")
    
    lines.extend([
        "",
        "## Plots",
        "",
        "- `training_curves.png` - Loss and validation metrics over training",
        "- `split_boxplot.png` - Performance comparison across splits",
        "- `error_vs_time_comparison.png` - Error evolution over time",
        "- `overlays_random_*.png` - Random trajectory samples",
        "- `overlays_worst_*.png` - Worst-case trajectories",
    ])
    
    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))


def generate_model_report(
    model_dir: Path,
    family: str,
    data_dir: Path,
    output_dir: Path,
    ood_data_dirs: Dict[str, Path] = None,
    device: str = "cuda",
):
    """Generate complete model report."""
    print(f"\n{'='*60}")
    print(f"Generating Model Report: {family}")
    print(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and config
    print("Loading model...")
    model, config = load_model(model_dir, device)
    
    # Load training history
    history_path = model_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        best_epoch = np.argmin(history["val_loss"]) + 1
        
        print("Generating training plots...")
        plot_training_curves(history, output_dir, best_epoch)
        plot_lr_schedule(history, output_dir)
    else:
        history = None
        best_epoch = 0
    
    # Evaluate on splits
    all_results = {}
    
    # ID test
    print("Evaluating ID test split...")
    try:
        id_dataset = ShardedDDEDataset(str(data_dir), family, "test")
        id_results = evaluate_split(model, id_dataset, device)
        id_results.split_name = "id"
        all_results["id"] = id_results
    except Exception as e:
        print(f"  Warning: Could not load ID test: {e}")
    
    # OOD splits
    ood_splits = {
        "ood_delay": ("data_ood_delay", "test_ood"),
        "ood_history": ("data_ood_history", "test_spline"),
        "ood_horizon": ("data_ood_horizon", "test_horizon"),
    }
    
    for split_name, (ood_dir, split_key) in ood_splits.items():
        print(f"Evaluating {split_name}...")
        try:
            ood_path = Path(ood_dir)
            if ood_path.exists():
                ood_dataset = ShardedDDEDataset(str(ood_path), family, split_key)
                ood_results = evaluate_split(model, ood_dataset, device)
                ood_results.split_name = split_name
                all_results[split_name] = ood_results
        except Exception as e:
            print(f"  Warning: Could not load {split_name}: {e}")
    
    # Determine x-axis limits for histograms (shared across splits)
    all_errors = np.concatenate([r.rel_l2 for r in all_results.values()])
    xlim = (0, min(np.percentile(all_errors, 99), 2.0))
    
    # Generate plots
    print("\nGenerating performance plots...")
    
    # B1: Error histograms per split
    for results in all_results.values():
        plot_error_histogram(results, output_dir, xlim=xlim)
    
    # B2: Split comparison
    if len(all_results) > 1:
        plot_split_comparison(all_results, output_dir)
    
    # B3: Error vs time
    for results in all_results.values():
        plot_error_vs_time(results, output_dir)
    
    if len(all_results) > 1:
        plot_error_vs_time_comparison(all_results, output_dir)
    
    # C: Trajectory overlays
    print("Generating trajectory overlays...")
    state_labels = ["x", "v"] if family == "vdp" else None
    for results in all_results.values():
        plot_trajectory_overlays(results, output_dir, n_samples=9, 
                                selection="random", state_labels=state_labels)
        plot_trajectory_overlays(results, output_dir, n_samples=9, 
                                selection="worst", state_labels=state_labels)
    
    # D: Diagnostic plots
    print("Generating diagnostic plots...")
    for results in all_results.values():
        plot_error_vs_delay(results, output_dir)
        
        # Constraint violations for positive families
        if family in ["hutch", "dist_uniform", "dist_exp"]:
            plot_constraint_violations(results, output_dir, requires_positive=True)
    
    # E: Family-specific plots
    if family == "vdp":
        print("Generating VdP phase portraits...")
        for results in all_results.values():
            plot_vdp_phase_portrait(results, output_dir)
    
    # Summary
    print("Generating summary...")
    generate_summary_md(family, config, all_results, output_dir, best_epoch)
    
    print(f"\n✓ Report saved to: {output_dir}")
    print(f"  Generated {len(list(output_dir.glob('*.png')))} plots")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate comprehensive model report")
    parser.add_argument("--model_dir", required=True, help="Model checkpoint directory")
    parser.add_argument("--family", required=True, 
                        choices=["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"])
    parser.add_argument("--data_dir", default="data_baseline_v1", help="ID data directory")
    parser.add_argument("--output_dir", default="reports/model_viz", help="Output directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    run_id = model_dir.name
    output_dir = Path(args.output_dir) / args.family / run_id
    
    generate_model_report(
        model_dir=model_dir,
        family=args.family,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
