#!/usr/bin/env python3
"""
Visualize BEST FNO predictions vs ground truth.
Picks samples with lowest error to show what success looks like.
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


def find_best_samples(model, dataset, device, n_best=3):
    """Find samples with lowest error."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_errors = []
    all_indices = []
    idx = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"]
            target_mean = batch["target_mean"]
            target_std = batch["target_std"]
            mask = batch["loss_mask"]
            
            pred = model(x).cpu()
            
            for i in range(len(pred)):
                y_true = y[i] * target_std[i] + target_mean[i]
                y_pred = pred[i] * target_std[i] + target_mean[i]
                m = mask[i].unsqueeze(-1)
                
                diff = (y_pred - y_true) * m
                rel_l2 = torch.sqrt((diff**2).sum()) / torch.sqrt(((y_true * m)**2).sum() + 1e-8)
                all_errors.append(rel_l2.item())
                all_indices.append(idx)
                idx += 1
    
    # Get indices of best samples
    sorted_idx = np.argsort(all_errors)
    return [all_indices[i] for i in sorted_idx[:n_best]], [all_errors[i] for i in sorted_idx[:n_best]]


def plot_best_predictions(family: str, data_dir: str, ckpt_dir: str, title: str, 
                          axes, n_samples: int = 3, device: str = "cuda"):
    """Plot best predictions for a family."""
    dataset = ShardedDDEDataset(ROOT / data_dir, family, "test")
    
    try:
        model = load_model(ckpt_dir, device)
    except Exception as e:
        print(f"    ERROR: {e}")
        return
    
    print(f"  {family}: finding best samples...")
    best_indices, best_errors = find_best_samples(model, dataset, device, n_samples)
    
    with torch.no_grad():
        for i, (idx, err) in enumerate(zip(best_indices, best_errors)):
            sample = dataset[idx]
            
            x = sample["input"].unsqueeze(0).to(device)
            y_true = sample["target"]
            target_mean = sample["target_mean"]
            target_std = sample["target_std"]
            
            y_pred = model(x).squeeze(0).cpu()
            
            y_true_orig = y_true * target_std + target_mean
            y_pred_orig = y_pred * target_std + target_mean
            
            t = np.linspace(0, 15, len(y_true))
            
            ax = axes[i]
            ax.plot(t, y_true_orig[:, 0].numpy(), 'b-', linewidth=2, label='Ground Truth')
            ax.plot(t, y_pred_orig[:, 0].numpy(), 'r--', linewidth=2, label='FNO Prediction')
            ax.set_title(f'{title} (relL2: {err:.1%})', fontsize=10)
            ax.set_xlabel('Time', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(fontsize=8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    fig, axes = plt.subplots(5, 3, figsize=(14, 16))
    
    print("Finding and plotting BEST predictions...")
    
    for row, (family, (data_dir, ckpt_dir, title)) in enumerate(FAMILIES.items()):
        plot_best_predictions(family, data_dir, ckpt_dir, title, axes[row], n_samples=3, device=device)
    
    for row in range(5):
        axes[row, 0].set_ylabel('State', fontsize=9)
    
    plt.suptitle('FNO BEST Predictions vs Ground Truth (lowest error samples)', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_dir = Path("reports/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "best_predictions.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_dir / 'best_predictions.png'}")
    
    plt.show()


if __name__ == "__main__":
    main()
