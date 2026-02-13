#!/usr/bin/env python3
"""
Generate P95 Tail Error vs Time plots for OOD splits.
Uses fixed epsilon (trajectory-level normalization) to avoid denominator collapse.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual

FAMILIES = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]
FAMILY_COLORS = {
    "hutch": "#1f77b4",
    "linear2": "#ff7f0e", 
    "vdp": "#2ca02c",
    "dist_uniform": "#9467bd",
    "dist_exp": "#d62728",
}

CKPT_PATHS = {
    "hutch": ROOT / "outputs/baseline_v1/hutch_seed42_20251228_131919",
    "linear2": ROOT / "outputs/baseline_v1/linear2_seed42_20251228_142839",
    "vdp": ROOT / "outputs/baseline_v1/vdp_seed42_20251229_020516",
    "dist_uniform": ROOT / "outputs/baseline_v1/dist_uniform_seed42_20251229_030851",
    "dist_exp": ROOT / "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403",
}

DATA_PATHS = {
    "hutch": "data_baseline_v1",
    "linear2": "data_baseline_v1",
    "vdp": "data_baseline_v1",
    "dist_uniform": "data_baseline_v1",
    "dist_exp": "data_baseline_v2",
}


def load_model(ckpt_path: Path, device: str = "cuda"):
    """Load trained FNO model."""
    ckpt_file = ckpt_path / "best_model.pt"
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
    return model


def compute_pointwise_errors(model, dataset, device: str = "cuda"):
    """Compute pointwise errors with fixed epsilon."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_pointwise_err = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing", leave=False):
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
                # Fixed epsilon: use trajectory-level normalization
                global_norm = torch.sqrt((target_masked ** 2).sum() / (mask[i].sum() + 1e-8))
                target_per_t = torch.clamp(torch.norm(target_masked, dim=-1), min=global_norm * 0.01)
                pointwise_err = (diff_per_t / (target_per_t + 1e-8)).cpu().numpy()
                pointwise_err = pointwise_err * mask[i].cpu().numpy()
                all_pointwise_err.append(pointwise_err)
    
    return np.array(all_pointwise_err)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = ROOT / "reports/model_viz/all5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_to_plot = [
        ("ood_delay", "data_ood_delay", "test_ood", "OOD-Delay"),
        ("ood_horizon", "data_ood_horizon", "test", "OOD-Horizon"),
        ("id", None, "test", "ID Test"),  # Uses family-specific data path
    ]
    
    for split_key, data_dir_override, split_name, title in splits_to_plot:
        print(f"\n=== {title} ===")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for family in FAMILIES:
            print(f"  {family}...")
            
            model = load_model(CKPT_PATHS[family], device)
            
            data_dir = data_dir_override if data_dir_override else DATA_PATHS[family]
            
            try:
                dataset = ShardedDDEDataset(ROOT / data_dir, family, split_name)
            except:
                print(f"    Skip (no data)")
                continue
            
            pointwise_err = compute_pointwise_errors(model, dataset, device)
            
            # Compute P95 per timestep
            n_time = pointwise_err.shape[1]
            t = np.linspace(0, 15, n_time)
            
            # Only compute P95 where we have valid data (mask=1)
            p95 = np.percentile(pointwise_err, 95, axis=0)
            
            # Find where future starts (first non-zero in average)
            future_start = np.argmax(p95 > 0)
            
            ax.plot(t[future_start:], p95[future_start:], 
                    color=FAMILY_COLORS[family], linewidth=2, label=family)
        
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Pointwise Error (P95)", fontsize=12)
        ax.set_title(f"Tail Error vs Time (P95) — {title}\n(Fixed epsilon: trajectory-level normalization)", 
                     fontsize=14, fontweight='bold')
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"p95_tail_{split_key}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: p95_tail_{split_key}.png")
    
    print(f"\n✓ All P95 tail plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
