#!/usr/bin/env python3
"""
Generate error-vs-time curves for paper figures.

For each family and split, compute mean and p90 error vs time on [0, T].
Output: reports/baseline_eval/figs/{family}_error_vs_time_{split}.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.fno1d import FNO1d, FNO1dResidual
from datasets.sharded_dataset import ShardedDDEDataset


def load_model(model_dir: Path, device: str):
    """Load trained model from checkpoint."""
    config_path = model_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_cfg = config.get("model", {})
    
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "final_model.pt"
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    in_channels = ckpt["model_state_dict"]["lift.weight"].shape[1]
    if "proj2.weight" in ckpt["model_state_dict"]:
        out_channels = ckpt["model_state_dict"]["proj2.weight"].shape[0]
    else:
        out_channels = 1
    
    use_residual = config.get("use_residual", False)
    model_class = FNO1dResidual if use_residual else FNO1d
    
    model = model_class(
        modes=model_cfg.get("modes", 12),
        width=model_cfg.get("width", 48),
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=model_cfg.get("n_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    
    return model, config


def compute_error_vs_time(
    model,
    dataset,
    device: str,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict:
    """Compute per-timestep error statistics."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_errors = []  # List of (n_samples, n_time) arrays
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing errors"):
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            mask = batch["loss_mask"].to(device)
            
            pred = model(x)
            
            # Denormalize
            y_mean_t = torch.from_numpy(y_mean).float().to(device)
            y_std_t = torch.from_numpy(y_std).float().to(device)
            
            pred_orig = pred * y_std_t + y_mean_t
            y_orig = y * y_std_t + y_mean_t
            
            # Per-timestep absolute error (averaged over state dims)
            error = torch.abs(pred_orig - y_orig).mean(dim=-1)  # (batch, time)
            all_errors.append(error.cpu().numpy())
    
    # Concatenate all batches
    all_errors = np.concatenate(all_errors, axis=0)  # (n_samples, n_time)
    
    # Compute statistics per timestep
    mean_error = all_errors.mean(axis=0)
    p50_error = np.percentile(all_errors, 50, axis=0)
    p90_error = np.percentile(all_errors, 90, axis=0)
    p10_error = np.percentile(all_errors, 10, axis=0)
    
    return {
        "mean": mean_error,
        "p50": p50_error,
        "p90": p90_error,
        "p10": p10_error,
        "n_samples": len(all_errors),
    }


def plot_error_vs_time(
    results: dict,
    t_out: np.ndarray,
    family: str,
    output_path: Path,
    n_hist: int = 256,
):
    """Create error-vs-time plot for multiple splits."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "id": "#2E86AB",
        "ood_delay": "#E94F37",
        "ood_delay_hole": "#F39C12",
        "ood_history": "#8E44AD",
    }
    
    labels = {
        "id": "ID (test)",
        "ood_delay": "OOD-delay",
        "ood_delay_hole": "OOD-delay-hole",
        "ood_history": "OOD-history",
    }
    
    # Only plot future region (after history)
    t_future = t_out[n_hist:]
    
    for split_name, data in results.items():
        if data is None:
            continue
        
        color = colors.get(split_name, "#333333")
        label = labels.get(split_name, split_name)
        
        mean_future = data["mean"][n_hist:]
        p90_future = data["p90"][n_hist:]
        p10_future = data["p10"][n_hist:]
        
        ax.plot(t_future, mean_future, color=color, linewidth=2, label=f"{label} (mean)")
        ax.fill_between(t_future, p10_future, p90_future, color=color, alpha=0.2)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='t=0 (prediction start)')
    
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Absolute Error (original space)", fontsize=12)
    ax.set_title(f"{family.upper()} - Error vs Time", fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_future[0], t_future[-1]])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate error-vs-time curves")
    parser.add_argument("--family", required=True, choices=["hutch", "linear2"])
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", default="reports/baseline_eval/figs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Error vs Time Analysis: {args.family.upper()}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(model_dir, args.device)
    
    # Load training stats
    print("Loading training stats...")
    train_ds = ShardedDDEDataset("data_baseline_v1", args.family, "train")
    y_mean = train_ds.y_mean
    y_std = train_ds.y_std
    
    # Define splits to evaluate
    splits_config = {
        "id": ("data_baseline_v1", "test"),
        "ood_delay": ("data_ood_delay", "test_ood"),
        "ood_delay_hole": ("data_ood_delay_hole", "test_hole"),
        "ood_history": ("data_ood_history", "test_spline"),
    }
    
    results = {}
    t_out = None
    n_hist = None
    
    for split_name, (data_dir, split) in splits_config.items():
        print(f"\n  Processing {split_name}...")
        
        try:
            dataset = ShardedDDEDataset(data_dir, args.family, split)
            # Share training stats
            dataset.y_mean = y_mean
            dataset.y_std = y_std
            dataset.phi_mean = train_ds.phi_mean
            dataset.phi_std = train_ds.phi_std
            dataset.param_mean = train_ds.param_mean
            dataset.param_std = train_ds.param_std
            
            if t_out is None:
                t_out = dataset.t_combined
                n_hist = dataset.n_hist
            
            error_data = compute_error_vs_time(model, dataset, args.device, y_mean, y_std)
            results[split_name] = error_data
            print(f"    Mean error at T: {error_data['mean'][-1]:.4f}")
            
        except Exception as e:
            print(f"    [SKIP] {e}")
            results[split_name] = None
    
    # Create combined plot
    output_path = output_dir / f"{args.family}_error_vs_time.png"
    plot_error_vs_time(results, t_out, args.family, output_path, n_hist)
    
    # Save data as JSON
    json_data = {}
    for split_name, data in results.items():
        if data is not None:
            json_data[split_name] = {
                "mean": data["mean"].tolist(),
                "p50": data["p50"].tolist(),
                "p90": data["p90"].tolist(),
                "n_samples": data["n_samples"],
            }
    
    json_path = output_dir / f"{args.family}_error_vs_time.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
