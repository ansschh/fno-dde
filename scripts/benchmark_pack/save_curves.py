#!/usr/bin/env python3
"""
Save curve data as NPZ files for all families and splits.

This extracts:
1. Error vs time curves for each split (t, mean, median, p50, p90, p95)
2. Training curves (epoch, train_loss, val_loss, val_rel_l2, best_epoch)

Output structure:
reports/model_viz/{family}/{run_id}/curves/
  error_vs_time_id.npz
  error_vs_time_ood_delay.npz
  error_vs_time_ood_history.npz
  error_vs_time_ood_horizon.npz
  training_curves.npz
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
import json
from tqdm import tqdm
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual


FAMILY_ORDER = ["dist_exp", "hutch", "dist_uniform", "vdp", "linear2"]

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


def compute_error_vs_time(model, dataset, device: str, batch_size: int = 64) -> Dict:
    """Compute error vs time curves for a split using loss_mask."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_pointwise_err = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing curves", leave=False):
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
                diff_masked = (y_pred_orig[i] - y_true_orig[i]) * mask_exp[i]
                target_masked = y_true_orig[i] * mask_exp[i]
                
                # Pointwise error (per time step) - use trajectory-level normalization
                # to avoid denominator collapse when y(t) approaches zero
                diff_per_t = torch.norm(diff_masked, dim=-1)  # (T,)
                global_norm = torch.sqrt((target_masked ** 2).sum() / (mask[i].sum() + 1e-8))
                target_per_t = torch.clamp(torch.norm(target_masked, dim=-1), min=global_norm * 0.01)
                pointwise_err = (diff_per_t / (target_per_t + 1e-8)).cpu().numpy()
                all_pointwise_err.append(pointwise_err)
                all_masks.append(mask[i].cpu().numpy())
    
    pointwise_err = np.array(all_pointwise_err)  # (n_samples, n_time)
    n_time = pointwise_err.shape[1]
    t = np.linspace(0, 15.0, n_time)
    
    return {
        "t": t,
        "mean": np.mean(pointwise_err, axis=0),
        "median": np.median(pointwise_err, axis=0),
        "p50": np.percentile(pointwise_err, 50, axis=0),
        "p90": np.percentile(pointwise_err, 90, axis=0),
        "p95": np.percentile(pointwise_err, 95, axis=0),
    }


def save_training_curves(model_dir: Path, output_dir: Path):
    """Save training curves from history.json."""
    history_path = model_dir / "history.json"
    if not history_path.exists():
        print(f"  Warning: No history.json in {model_dir}")
        return
    
    with open(history_path) as f:
        history = json.load(f)
    
    n_epochs = len(history["train_loss"])
    epoch = np.arange(1, n_epochs + 1)
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    val_rel_l2 = np.array(history.get("val_rel_l2", history.get("val_rell2", [0]*n_epochs)))
    best_epoch = np.argmin(val_loss) + 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "training_curves.npz",
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        val_rel_l2=val_rel_l2,
        best_epoch=best_epoch,
    )
    print(f"  Saved: training_curves.npz")


def save_error_curves_for_family(family: str, device: str = "cuda"):
    """Save all error vs time curves for a family."""
    model_dir = Path(MODEL_PATHS[family])
    data_dir = DATA_PATHS[family]
    run_id = model_dir.name
    
    output_dir = Path("reports/model_viz") / family / run_id / "curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {family} ({run_id})")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    model, config = load_model(model_dir, device)
    
    # Save training curves
    save_training_curves(model_dir, output_dir)
    
    # ID test split
    print("Computing ID test curves...")
    try:
        id_dataset = ShardedDDEDataset(data_dir, family, "test")
        curves = compute_error_vs_time(model, id_dataset, device)
        np.savez(output_dir / "error_vs_time_id.npz", **curves)
        print(f"  Saved: error_vs_time_id.npz")
    except Exception as e:
        print(f"  Warning: Could not compute ID curves: {e}")
    
    # OOD splits
    for split_name, (ood_dir, split_key) in OOD_SPLITS.items():
        print(f"Computing {split_name} curves...")
        try:
            ood_path = Path(ood_dir)
            if ood_path.exists() and (ood_path / family).exists():
                ood_dataset = ShardedDDEDataset(str(ood_path), family, split_key)
                curves = compute_error_vs_time(model, ood_dataset, device)
                np.savez(output_dir / f"error_vs_time_{split_name}.npz", **curves)
                print(f"  Saved: error_vs_time_{split_name}.npz")
            else:
                print(f"  Skipped: {ood_dir}/{family} not found")
        except Exception as e:
            print(f"  Warning: Could not compute {split_name} curves: {e}")
    
    return run_id


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Save curve data as NPZ files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--families", nargs="*", default=FAMILY_ORDER)
    args = parser.parse_args()
    
    run_map = {}
    
    for family in args.families:
        run_id = save_error_curves_for_family(family, args.device)
        run_map[family] = run_id
    
    # Save run map
    run_map_path = Path("reports/model_viz/run_map_baseline_v2.json")
    with open(run_map_path, "w") as f:
        json.dump(run_map, f, indent=2)
    print(f"\n✓ Saved run map to: {run_map_path}")
    
    print("\nCurves saved for all families. Ready for plot_all5_panels.py")


if __name__ == "__main__":
    main()
