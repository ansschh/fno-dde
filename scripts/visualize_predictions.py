#!/usr/bin/env python3
"""
Visualize FNO predictions vs ground truth for all 5 families.
Shows how well the model captures each type of dynamics.
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
    "hutch": ("data_baseline_v1", "outputs/baseline_v1/hutch_seed42_20251228_131919", "Hutchinson"),
    "linear2": ("data_baseline_v1", "outputs/baseline_v1/linear2_seed42_20251228_142839", "Linear2"),
    "vdp": ("data_baseline_v1", "outputs/baseline_v1/vdp_seed42_20251229_020516", "Van der Pol"),
    "dist_uniform": ("data_baseline_v1", "outputs/baseline_v1/dist_uniform_seed42_20251229_030851", "Dist Uniform"),
    "dist_exp": ("data_baseline_v2", "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403", "Dist Exp"),
}


def load_model(ckpt_dir: str, device: str = "cuda"):
    """Load trained FNO model from checkpoint."""
    ckpt_path = ROOT / ckpt_dir / "best_model.pt"
    print(f"    Loading: {ckpt_path}")
    if not ckpt_path.exists():
        ckpt_path = ROOT / ckpt_dir / "final_model.pt"
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Get model config
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    
    # Infer dimensions from state dict if not in config
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def plot_predictions(family: str, data_dir: str, ckpt_dir: str, title: str, 
                     axes, n_samples: int = 3, device: str = "cuda"):
    """Plot predictions vs ground truth for a family."""
    dataset = ShardedDDEDataset(data_dir, family, "test")
    
    try:
        model = load_model(ckpt_dir, device)
    except Exception as e:
        print(f"    ERROR loading model: {e}")
        axes[0].text(0.5, 0.5, f"Model error:\n{e}", 
                     ha='center', va='center', transform=axes[0].transAxes, fontsize=8)
        axes[0].set_title(title)
        return
    
    # Get samples with varying difficulty
    np.random.seed(123)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[int(idx)]
            
            x = sample["input"].unsqueeze(0).to(device)
            y_true = sample["target"]
            target_mean = sample["target_mean"]
            target_std = sample["target_std"]
            
            # Model prediction
            y_pred = model(x).squeeze(0).cpu()
            
            # Denormalize
            y_true_orig = y_true * target_std + target_mean
            y_pred_orig = y_pred * target_std + target_mean
            
            # Get mask to find boundary
            mask = sample["loss_mask"]
            transitions = torch.where(mask[:-1] != mask[1:])[0]
            boundary_idx = transitions[0].item() if len(transitions) > 0 else 0
            
            # Time axis
            t = np.linspace(0, 15, len(y_true))
            t_boundary = t[boundary_idx] if boundary_idx > 0 else 0
            
            ax = axes[i]
            
            # Shade unsupervised history region
            ax.axvspan(t[0], t_boundary, alpha=0.15, color='gray', label='History (unsupervised)')
            ax.axvline(x=t_boundary, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.plot(t, y_true_orig[:, 0].numpy(), 'b-', linewidth=2, label='Ground Truth')
            ax.plot(t, y_pred_orig[:, 0].numpy(), 'r--', linewidth=2, label='FNO Prediction')
            
            # Compute error ONLY on future region (matching training)
            mask_exp = mask.unsqueeze(-1)
            diff = (y_pred_orig - y_true_orig) * mask_exp
            target_masked = y_true_orig * mask_exp
            rel_l2 = torch.sqrt((diff**2).sum()) / torch.sqrt((target_masked**2).sum() + 1e-8)
            
            ax.set_title(f'{title} (future relL2: {rel_l2:.1%})', fontsize=10)
            ax.set_xlabel('Time', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(fontsize=7, loc='upper right')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    fig, axes = plt.subplots(5, 3, figsize=(14, 16))
    
    print("Generating prediction visualizations...")
    
    for row, (family, (data_dir, ckpt_dir, title)) in enumerate(FAMILIES.items()):
        print(f"  {family}...")
        plot_predictions(family, data_dir, ckpt_dir, title, axes[row], n_samples=3, device=device)
    
    # Y-axis labels
    for row, family in enumerate(FAMILIES.keys()):
        axes[row, 0].set_ylabel('State', fontsize=9)
    
    plt.suptitle('FNO Predictions vs Ground Truth (3 samples per family)', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Save
    output_dir = Path("reports/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "predictions_vs_truth.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_dir / 'predictions_vs_truth.png'}")
    
    plt.show()


if __name__ == "__main__":
    main()
