#!/usr/bin/env python3
"""
Clean visualization: 1 plot per family showing supervised vs unsupervised regions.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual

FAMILIES = {
    "hutch": ("data_baseline_v1", "outputs/baseline_v1/hutch_seed42_20251228_131919", "Hutchinson (Population Dynamics)"),
    "linear2": ("data_baseline_v1", "outputs/baseline_v1/linear2_seed42_20251228_142839", "Linear2 (Two Delays)"),
    "vdp": ("data_baseline_v1", "outputs/baseline_v1/vdp_seed42_20251229_020516", "Van der Pol (Oscillator)"),
    "dist_uniform": ("data_baseline_v1", "outputs/baseline_v1/dist_uniform_seed42_20251229_030851", "Distributed Uniform"),
    "dist_exp": ("data_baseline_v2", "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403", "Distributed Exponential"),
}

SAMPLE_INDICES = {"hutch": 42, "linear2": 100, "vdp": 55, "dist_uniform": 30, "dist_exp": 20}
SAMPLE_INDICES_ALT = {"hutch": 150, "linear2": 200, "vdp": 180, "dist_uniform": 75, "dist_exp": 90}


def load_model(ckpt_dir: str, device: str = "cuda"):
    """Load trained FNO model from checkpoint."""
    ckpt_path = ROOT / ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT / ckpt_dir / "final_model.pt"
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    
    state_dict = checkpoint["model_state_dict"]
    in_channels = state_dict["lift.weight"].shape[1]
    out_channels = state_dict["proj2.weight"].shape[0]
    
    model = FNO1dResidual(
        modes=model_cfg.get("modes", 16),
        width=model_cfg.get("width", 48),
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=model_cfg.get("n_layers", 4),
        activation=model_cfg.get("activation", "gelu"),
        dropout=model_cfg.get("dropout", 0.0),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def plot_with_regions(family: str, data_dir: str, ckpt_dir: str, title: str, 
                      sample_idx: int, device: str = "cuda"):
    """Create a single clean plot showing supervised/unsupervised regions."""
    dataset = ShardedDDEDataset(ROOT / data_dir, family, "test")
    model = load_model(ckpt_dir, device)
    
    sample = dataset[sample_idx]
    
    with torch.no_grad():
        x = sample["input"].unsqueeze(0).to(device)
        y_true = sample["target"]
        target_mean = sample["target_mean"]
        target_std = sample["target_std"]
        mask = sample["loss_mask"]
        
        y_pred = model(x).squeeze(0).cpu()
        
        y_true_orig = y_true * target_std + target_mean
        y_pred_orig = y_pred * target_std + target_mean
    
    # Find boundary
    transitions = torch.where(mask[:-1] != mask[1:])[0]
    boundary_idx = transitions[0].item() if len(transitions) > 0 else 0
    
    # Time axis
    t = np.linspace(0, 15, len(y_true))
    t_boundary = t[boundary_idx]
    
    # Compute future-only error
    mask_exp = mask.unsqueeze(-1)
    diff = (y_pred_orig - y_true_orig) * mask_exp
    target_masked = y_true_orig * mask_exp
    rel_l2 = torch.sqrt((diff**2).sum()) / torch.sqrt((target_masked**2).sum() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Shade regions
    ax.axvspan(t[0], t_boundary, alpha=0.2, color='lightgray', zorder=0)
    ax.axvspan(t_boundary, t[-1], alpha=0.1, color='lightgreen', zorder=0)
    ax.axvline(x=t_boundary, color='black', linestyle='--', linewidth=1.5, zorder=1)
    
    # Plot trajectories
    ax.plot(t, y_true_orig[:, 0].numpy(), 'b-', linewidth=2.5, label='Ground Truth', zorder=2)
    ax.plot(t, y_pred_orig[:, 0].numpy(), 'r--', linewidth=2.5, label='FNO Prediction', zorder=2)
    
    # Title and labels
    ax.set_title(f'{title}\nFuture Region relL2: {rel_l2:.1%}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    
    # Legend at top right, outside plot area
    ax.legend(fontsize=11, loc='upper right')
    
    # Region labels - positioned to not overlap
    y_range = ax.get_ylim()
    y_top = y_range[1] - 0.05 * (y_range[1] - y_range[0])
    
    ax.text(t_boundary/2, y_top, 'History\n(Unsupervised)', 
            ha='center', va='top', fontsize=10, color='gray', fontweight='bold')
    ax.text((t_boundary + t[-1])/2, y_top, 'Future\n(Supervised)', 
            ha='center', va='top', fontsize=10, color='darkgreen', fontweight='bold')
    
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(t[0], t[-1])
    
    plt.tight_layout()
    return fig


def plot_future_only(family: str, data_dir: str, ckpt_dir: str, title: str, 
                     sample_idx: int, device: str = "cuda"):
    """Create a single clean plot showing ONLY the future (supervised) region."""
    dataset = ShardedDDEDataset(ROOT / data_dir, family, "test")
    model = load_model(ckpt_dir, device)
    
    sample = dataset[sample_idx]
    
    with torch.no_grad():
        x = sample["input"].unsqueeze(0).to(device)
        y_true = sample["target"]
        target_mean = sample["target_mean"]
        target_std = sample["target_std"]
        mask = sample["loss_mask"]
        
        y_pred = model(x).squeeze(0).cpu()
        
        y_true_orig = y_true * target_std + target_mean
        y_pred_orig = y_pred * target_std + target_mean
    
    # Find boundary
    transitions = torch.where(mask[:-1] != mask[1:])[0]
    boundary_idx = transitions[0].item() if len(transitions) > 0 else 0
    
    # Time axis - only future
    t_full = np.linspace(0, 15, len(y_true))
    t = t_full[boundary_idx:]
    y_true_future = y_true_orig[boundary_idx:, 0].numpy()
    y_pred_future = y_pred_orig[boundary_idx:, 0].numpy()
    
    # Compute future-only error
    mask_exp = mask.unsqueeze(-1)
    diff = (y_pred_orig - y_true_orig) * mask_exp
    target_masked = y_true_orig * mask_exp
    rel_l2 = torch.sqrt((diff**2).sum()) / torch.sqrt((target_masked**2).sum() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot trajectories
    ax.plot(t, y_true_future, 'b-', linewidth=2.5, label='Ground Truth')
    ax.plot(t, y_pred_future, 'r--', linewidth=2.5, label='FNO Prediction')
    
    # Title and labels
    ax.set_title(f'{title} — Future Region Only\nrelL2: {rel_l2:.1%}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = ROOT / "reports/visualizations/clean_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Part 1: Full plots with regions shown
    print("\n=== Generating plots with supervised/unsupervised regions ===")
    for family, (data_dir, ckpt_dir, title) in FAMILIES.items():
        print(f"  {family}...")
        fig = plot_with_regions(family, data_dir, ckpt_dir, title, 
                                SAMPLE_INDICES[family], device)
        fig.savefig(output_dir / f"{family}_with_regions.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Saved to: {output_dir}")
    
    # Part 2: Future-only plots (different samples)
    print("\n=== Generating future-only plots (different samples) ===")
    for family, (data_dir, ckpt_dir, title) in FAMILIES.items():
        print(f"  {family}...")
        fig = plot_future_only(family, data_dir, ckpt_dir, title, 
                               SAMPLE_INDICES_ALT[family], device)
        fig.savefig(output_dir / f"{family}_future_only.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Saved to: {output_dir}")


if __name__ == "__main__":
    main()
