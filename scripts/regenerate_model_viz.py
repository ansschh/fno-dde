#!/usr/bin/env python3
"""
Regenerate model visualization plots with seaborn styling for all families.
Generates:
1. Error vs Time: All Splits
2. Per-split: Error vs Time (Mean, Median, P50-P90) + Tail Error (P95)
3. Training curves: 3 columns (Train/Val Loss, Val relL2, Gen Gap)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual

# Seaborn styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'

FAMILIES = {
    "hutch": {
        "data_dir": "data_baseline_v1",
        "ckpt_dir": "outputs/baseline_v1/hutch_seed42_20251228_131919",
        "title": "Hutchinson",
    },
    "linear2": {
        "data_dir": "data_baseline_v1",
        "ckpt_dir": "outputs/baseline_v1/linear2_seed42_20251228_142839",
        "title": "Linear Two-Delay",
    },
    "vdp": {
        "data_dir": "data_baseline_v1",
        "ckpt_dir": "outputs/baseline_v1/vdp_seed42_20251229_020516",
        "title": "Van der Pol",
    },
    "dist_uniform": {
        "data_dir": "data_baseline_v1",
        "ckpt_dir": "outputs/baseline_v1/dist_uniform_seed42_20251229_030851",
        "title": "Distributed Uniform",
    },
    "dist_exp": {
        "data_dir": "data_baseline_v2",
        "ckpt_dir": "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403",
        "title": "Distributed Exponential",
    },
}

SPLIT_COLORS = {
    "id": "#1f77b4",
    "ood_delay": "#d62728",
    "ood_history": "#e377c2",
    "ood_horizon": "#17becf",
}

SPLITS = ["id", "ood_delay", "ood_history", "ood_horizon"]


def load_model(ckpt_path: Path, device: str = "cuda"):
    """Load trained FNO model."""
    ckpt_file = ckpt_path / "best_model.pt"
    if not ckpt_file.exists():
        ckpt_file = ckpt_path / "final_model.pt"
    
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    state_dict = checkpoint["model_state_dict"]
    
    model = FNO1dResidual(
        modes=model_cfg.get("modes", 16),
        width=model_cfg.get("width", 48),
        in_channels=state_dict["lift.weight"].shape[1],
        out_channels=state_dict["proj2.weight"].shape[0],
        n_layers=model_cfg.get("n_layers", 4),
        activation=model_cfg.get("activation", "gelu"),
        dropout=0.0,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def compute_pointwise_errors(model, dataset, device: str = "cuda"):
    """Compute pointwise errors for all samples."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_pointwise_err = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            mask = batch["loss_mask"].to(device)
            
            y_pred = model(x)
            
            y_pred_orig = y_pred * target_std + target_mean
            y_true_orig = y_true * target_std + target_mean
            
            mask_exp = mask.unsqueeze(-1)
            
            for i in range(len(y_pred)):
                diff_masked = (y_pred_orig[i] - y_true_orig[i]) * mask_exp[i]
                target_masked = y_true_orig[i] * mask_exp[i]
                
                diff_per_t = torch.norm(diff_masked, dim=-1)
                global_norm = torch.sqrt((target_masked ** 2).sum() / (mask[i].sum() + 1e-8))
                target_per_t = torch.clamp(torch.norm(target_masked, dim=-1), min=global_norm * 0.01)
                pointwise_err = (diff_per_t / (target_per_t + 1e-8)).cpu().numpy()
                pointwise_err = pointwise_err * mask[i].cpu().numpy()
                all_pointwise_err.append(pointwise_err)
    
    return np.array(all_pointwise_err)


def get_split_data(family: str, split: str, device: str = "cuda"):
    """Get data for a specific split."""
    cfg = FAMILIES[family]
    
    if split == "id":
        data_dir = ROOT / cfg["data_dir"]
        split_name = "test"
    elif split == "ood_delay":
        data_dir = ROOT / "data_ood_delay"
        split_name = "test_ood"
    elif split == "ood_history":
        data_dir = ROOT / "data_ood_history"
        split_name = "test_spline"  # Correct split name
    elif split == "ood_horizon":
        data_dir = ROOT / "data_ood_horizon"
        split_name = "test_horizon"  # Correct split name
    else:
        return None
    
    try:
        dataset = ShardedDDEDataset(data_dir, family, split_name)
        return dataset
    except Exception as e:
        print(f"      Error loading {split}: {e}")
        return None


def plot_error_vs_time_all_splits(family: str, model, device: str = "cuda"):
    """Plot Error vs Time for all splits on one figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for split in SPLITS:
        dataset = get_split_data(family, split, device)
        if dataset is None:
            continue
        
        print(f"    {split}...")
        pointwise_err = compute_pointwise_errors(model, dataset, device)
        
        n_time = pointwise_err.shape[1]
        t = np.linspace(0, 15, n_time)
        
        # Compute median per timestep
        median = np.median(pointwise_err, axis=0)
        
        # Find future start
        future_start = np.argmax(median > 0)
        
        ax.plot(t[future_start:], median[future_start:], 
                color=SPLIT_COLORS[split], linewidth=2, label=split)
    
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Median Relative Error", fontsize=11)
    ax.set_title(f"{FAMILIES[family]['title']}: Error vs Time (All Splits)", fontsize=12)
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_split_error_and_tail(family: str, split: str, model, device: str = "cuda"):
    """Plot Error vs Time (Mean, Median, P50-P90) and Tail Error (P95) for a split."""
    dataset = get_split_data(family, split, device)
    if dataset is None:
        return None
    
    pointwise_err = compute_pointwise_errors(model, dataset, device)
    
    n_time = pointwise_err.shape[1]
    t = np.linspace(0, 15, n_time)
    
    # Compute statistics
    mean = np.mean(pointwise_err, axis=0)
    median = np.median(pointwise_err, axis=0)
    p50 = np.percentile(pointwise_err, 50, axis=0)
    p90 = np.percentile(pointwise_err, 90, axis=0)
    p95 = np.percentile(pointwise_err, 95, axis=0)
    
    # Find future start
    future_start = np.argmax(median > 0)
    t_plot = t[future_start:]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Error vs Time with band
    ax1 = axes[0]
    ax1.fill_between(t_plot, p50[future_start:], p90[future_start:], 
                     alpha=0.3, color='#1f77b4', label='P50-P90')
    ax1.plot(t_plot, mean[future_start:], color='#1f77b4', linewidth=2, label='Mean')
    ax1.plot(t_plot, median[future_start:], color='#ff7f0e', linewidth=2, label='Median')
    
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("Relative Error", fontsize=11)
    ax1.set_title(f"{split}: Error vs Time", fontsize=12)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Tail Error (P95)
    ax2 = axes[1]
    ax2.plot(t_plot, p95[future_start:], color='#d62728', linewidth=2, label='P95')
    
    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_ylabel("P95 Relative Error", fontsize=11)
    ax2.set_title(f"{split}: Tail Error vs Time", fontsize=12)
    ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_training_curves(family: str, ckpt_path: Path):
    """Plot training curves: 3 columns (Train/Val Loss, Val relL2, Gen Gap)."""
    # Try to load history from JSON file
    history_file = ckpt_path / "history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        return None
    
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_rel_l2 = history.get("val_rel_l2", history.get("val_relL2", []))
    
    if not train_loss or not val_loss:
        return None
    
    epochs = list(range(1, len(train_loss) + 1))
    
    # Compute generalization gap
    gen_gap = [v - t for t, v in zip(train_loss, val_loss)]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Col 1: Train and Val Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, color='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_loss, color='#ff7f0e', linewidth=2, label='Val Loss')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training and Validation Loss", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Col 2: Val relL2
    ax2 = axes[1]
    if val_rel_l2:
        ax2.plot(epochs[:len(val_rel_l2)], val_rel_l2, color='#2ca02c', linewidth=2, label='Val relL2')
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Validation relL2", fontsize=11)
        ax2.set_title("Validation Relative L2 Error", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No relL2 data", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Validation Relative L2 Error", fontsize=12)
    
    # Col 3: Generalization Gap
    ax3 = axes[2]
    ax3.plot(epochs, gen_gap, color='#9467bd', linewidth=2, label='Gen Gap (Val - Train)')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Gap", fontsize=11)
    ax3.set_title("Generalization Gap", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f"{FAMILIES[family]['title']}: Training Curves", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def save_figure(fig, output_dir: Path, name: str):
    """Save figure as both PNG and PDF."""
    fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f"{name}.pdf", format='pdf', bbox_inches='tight')
    print(f"    Saved: {name}.png, {name}.pdf")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    base_output_dir = ROOT / "reports/visualizations/clean_plots"
    
    for family, cfg in FAMILIES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {family}")
        print(f"{'='*60}")
        
        output_dir = base_output_dir / family
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        ckpt_path = ROOT / cfg["ckpt_dir"]
        print(f"  Loading model from {ckpt_path}...")
        model, checkpoint = load_model(ckpt_path, device)
        
        # 1. Error vs Time: All Splits
        print("  Generating Error vs Time (All Splits)...")
        fig = plot_error_vs_time_all_splits(family, model, device)
        save_figure(fig, output_dir, "error_vs_time_all_splits")
        plt.close(fig)
        
        # 2. Per-split: Error vs Time + Tail Error
        for split in SPLITS:
            print(f"  Generating {split} plots...")
            fig = plot_split_error_and_tail(family, split, model, device)
            if fig is not None:
                save_figure(fig, output_dir, f"{split}_error_and_tail")
                plt.close(fig)
            else:
                print(f"    Skipped (no data for {split})")
        
        # 3. Training curves
        print("  Generating training curves...")
        fig = plot_training_curves(family, ckpt_path)
        if fig is not None:
            save_figure(fig, output_dir, "training_curves")
            plt.close(fig)
        else:
            print("    Skipped (no training history)")
    
    print(f"\n{'='*60}")
    print(f"✓ All plots saved to: {base_output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
