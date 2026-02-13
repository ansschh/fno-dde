#!/usr/bin/env python3
"""
Clean visualization v2: Seaborn styling, structured legends, 3 samples per family.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm import tqdm

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual

# Set seaborn style
sns.set_theme(style="whitegrid", palette="muted")

# Try to use Open Sans, fallback to sans-serif
try:
    plt.rcParams['font.family'] = 'Open Sans'
except:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'

FAMILIES = {
    "hutch": ("data_baseline_v1", "outputs/baseline_v1/hutch_seed42_20251228_131919", "Hutchinson"),
    "linear2": ("data_baseline_v1", "outputs/baseline_v1/linear2_seed42_20251228_142839", "Linear Two-Delay"),
    "vdp": ("data_baseline_v1", "outputs/baseline_v1/vdp_seed42_20251229_020516", "Van der Pol"),
    "dist_uniform": ("data_baseline_v1", "outputs/baseline_v1/dist_uniform_seed42_20251229_030851", "Distributed Uniform"),
    "dist_exp": ("data_baseline_v2", "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403", "Distributed Exponential"),
}

COLORS = {
    "truth": "#1f77b4",
    "pred": "#d62728",
    "history": "#e0e0e0",
    "future": "#d4edda",
}


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


def compute_all_errors(model, dataset, device: str = "cuda"):
    """Compute relL2 errors for all samples in dataset."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_errors = []
    idx = 0
    
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
                
                diff_l2 = torch.sqrt((diff_masked ** 2).sum() + 1e-8)
                target_l2 = torch.sqrt((target_masked ** 2).sum() + 1e-8)
                rel_l2 = (diff_l2 / target_l2).item()
                all_errors.append((idx, rel_l2))
                idx += 1
    
    return all_errors


def get_sample_indices(errors, seed=42):
    """Get indices for random, worst, and best samples."""
    sorted_by_error = sorted(errors, key=lambda x: x[1])
    
    best_idx = sorted_by_error[0][0]
    worst_idx = sorted_by_error[-1][0]
    
    # Random from middle 50%
    n = len(sorted_by_error)
    mid_start = n // 4
    mid_end = 3 * n // 4
    np.random.seed(seed)
    random_idx = sorted_by_error[np.random.randint(mid_start, mid_end)][0]
    
    return {
        "best": best_idx,
        "random": random_idx,
        "worst": worst_idx,
    }


def get_sample_data(model, dataset, idx, device):
    """Get prediction and ground truth for a single sample."""
    sample = dataset[idx]
    
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
    
    # Compute error
    mask_exp = mask.unsqueeze(-1)
    diff = (y_pred_orig - y_true_orig) * mask_exp
    target_masked = y_true_orig * mask_exp
    rel_l2 = torch.sqrt((diff**2).sum()) / torch.sqrt((target_masked**2).sum() + 1e-8)
    
    return {
        "y_true": y_true_orig[:, 0].numpy(),
        "y_pred": y_pred_orig[:, 0].numpy(),
        "boundary_idx": boundary_idx,
        "rel_l2": rel_l2.item(),
        "n_time": len(y_true),
    }


def create_family_plot_with_regions(family: str, data_dir: str, ckpt_dir: str, 
                                     title: str, device: str = "cuda"):
    """Create 3-column plot showing best, random, worst samples with regions."""
    dataset = ShardedDDEDataset(ROOT / data_dir, family, "test")
    model = load_model(ckpt_dir, device)
    
    print(f"  Computing errors for {family}...")
    errors = compute_all_errors(model, dataset, device)
    indices = get_sample_indices(errors)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    sample_types = ["best", "random", "worst"]
    
    for col, stype in enumerate(sample_types):
        idx = indices[stype]
        data = get_sample_data(model, dataset, idx, device)
        
        t = np.linspace(0, 15, data["n_time"])
        t_boundary = t[data["boundary_idx"]]
        
        ax = axes[col]
        
        # Shade regions (no text inside)
        ax.axvspan(t[0], t_boundary, alpha=0.3, color=COLORS["history"], zorder=0)
        ax.axvspan(t_boundary, t[-1], alpha=0.3, color=COLORS["future"], zorder=0)
        ax.axvline(x=t_boundary, color='#666666', linestyle='--', linewidth=1, zorder=1)
        
        # Plot trajectories
        ax.plot(t, data["y_true"], color=COLORS["truth"], linewidth=2, label='Ground Truth')
        ax.plot(t, data["y_pred"], color=COLORS["pred"], linewidth=2, linestyle='--', label='Prediction')
        
        # Column title
        ax.set_title(f"{stype.capitalize()} Sample (relL2 = {data['rel_l2']:.1%})", fontsize=11)
        ax.set_xlabel('Time', fontsize=10)
        if col == 0:
            ax.set_ylabel('State', fontsize=10)
        ax.set_xlim(t[0], t[-1])
    
    # Main title - short, informative, one line
    fig.suptitle(f"{title} DDE Family — Prediction vs Ground Truth (n = {len(dataset)} test samples)", 
                 fontsize=12, y=1.02)
    
    # Structured legend below the plots
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color=COLORS["truth"], linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color=COLORS["pred"], linewidth=2, linestyle='--', label='FNO Prediction'),
        Patch(facecolor=COLORS["history"], alpha=0.3, label='History Region (input, not supervised)'),
        Patch(facecolor=COLORS["future"], alpha=0.3, label='Future Region (target, supervised)'),
        Line2D([0], [0], color='#666666', linewidth=1, linestyle='--', label='History/Future Boundary'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fancybox=False, edgecolor='#cccccc')
    
    plt.tight_layout()
    return fig


def create_family_plot_future_only(family: str, data_dir: str, ckpt_dir: str, 
                                    title: str, device: str = "cuda"):
    """Create 3-column plot showing only future region."""
    dataset = ShardedDDEDataset(ROOT / data_dir, family, "test")
    model = load_model(ckpt_dir, device)
    
    print(f"  Computing errors for {family}...")
    errors = compute_all_errors(model, dataset, device)
    indices = get_sample_indices(errors, seed=123)  # Different seed for variety
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    sample_types = ["best", "random", "worst"]
    
    for col, stype in enumerate(sample_types):
        idx = indices[stype]
        data = get_sample_data(model, dataset, idx, device)
        
        t_full = np.linspace(0, 15, data["n_time"])
        boundary = data["boundary_idx"]
        
        # Only future region
        t = t_full[boundary:]
        y_true = data["y_true"][boundary:]
        y_pred = data["y_pred"][boundary:]
        
        ax = axes[col]
        
        ax.plot(t, y_true, color=COLORS["truth"], linewidth=2, label='Ground Truth')
        ax.plot(t, y_pred, color=COLORS["pred"], linewidth=2, linestyle='--', label='Prediction')
        
        ax.set_title(f"{stype.capitalize()} Sample (relL2 = {data['rel_l2']:.1%})", fontsize=11)
        ax.set_xlabel('Time', fontsize=10)
        if col == 0:
            ax.set_ylabel('State', fontsize=10)
    
    fig.suptitle(f"{title} DDE Family — Future Region Only (supervised, n = {len(dataset)} test samples)", 
                 fontsize=12, y=1.02)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["truth"], linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color=COLORS["pred"], linewidth=2, linestyle='--', label='FNO Prediction'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=True, fancybox=False, edgecolor='#cccccc')
    
    plt.tight_layout()
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = ROOT / "reports/visualizations/clean_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Generating plots with supervised/unsupervised regions ===")
    for family, (data_dir, ckpt_dir, title) in FAMILIES.items():
        print(f"\n{family}:")
        fig = create_family_plot_with_regions(family, data_dir, ckpt_dir, title, device)
        fig.savefig(output_dir / f"{family}_with_regions.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {family}_with_regions.pdf")
    
    print("\n=== Generating future-only plots ===")
    for family, (data_dir, ckpt_dir, title) in FAMILIES.items():
        print(f"\n{family}:")
        fig = create_family_plot_future_only(family, data_dir, ckpt_dir, title, device)
        fig.savefig(output_dir / f"{family}_future_only.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {family}_future_only.pdf")
    
    print(f"\n✓ All plots saved to: {output_dir}")
    print(f"  Generated {len(FAMILIES) * 2} figures for {len(FAMILIES)} families")


if __name__ == "__main__":
    main()
