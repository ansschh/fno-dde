#!/usr/bin/env python3
"""
Part E: Enhanced Evaluation Suite with Error-vs-Time Curves

Unified evaluation producing:
- relL2 median/p95 per split
- Error-vs-time curves (mean + p90 envelope)
- Per-sample error distributions
- Constraint violation rates (where applicable)

Output JSON files ready for paper tables and figures.
"""
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.fno1d import FNO1d, FNO1dResidual
from datasets.sharded_dataset import ShardedDDEDataset


def load_model(model_dir: Path, device: str = "cuda"):
    """Load trained model from checkpoint."""
    checkpoint_path = model_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type from config or checkpoint
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        use_residual = config.get("model", {}).get("use_residual", True)
    else:
        use_residual = True
    
    # Get model dimensions from checkpoint
    state_dict = checkpoint["model_state_dict"]
    
    # Infer dimensions from state dict
    # fc0 weight shape: (width, in_channels)
    fc0_weight = state_dict.get("fc0.weight", state_dict.get("fno.fc0.weight"))
    width = fc0_weight.shape[0]
    in_channels = fc0_weight.shape[1]
    
    # fc2 weight shape: (out_channels, width)
    fc2_weight = state_dict.get("fc2.weight", state_dict.get("fno.fc2.weight"))
    out_channels = fc2_weight.shape[0]
    
    # Count spectral layers
    n_layers = sum(1 for k in state_dict.keys() if "spectral" in k and "weight" in k)
    
    # Get modes from first spectral layer
    for k, v in state_dict.items():
        if "spectral" in k and "weight" in k:
            modes = v.shape[-1]
            break
    
    if use_residual:
        model = FNO1dResidual(
            modes=modes,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )
    else:
        model = FNO1d(
            modes=modes,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def compute_error_vs_time(y_pred: np.ndarray, y_true: np.ndarray):
    """
    Compute error metrics at each time step.
    
    Args:
        y_pred: (batch, time, state_dim)
        y_true: (batch, time, state_dim)
    
    Returns:
        Dictionary with error curves
    """
    # Absolute error at each time
    abs_err = np.abs(y_pred - y_true)  # (batch, time, state_dim)
    
    # Sum over state dims for total error
    if abs_err.ndim == 3:
        abs_err_total = np.sqrt(np.sum(abs_err**2, axis=-1))  # (batch, time)
    else:
        abs_err_total = abs_err
    
    # Statistics over batch at each time
    mean_err = np.mean(abs_err_total, axis=0)
    p50_err = np.percentile(abs_err_total, 50, axis=0)
    p90_err = np.percentile(abs_err_total, 90, axis=0)
    
    # Relative error at each time
    y_norm = np.sqrt(np.sum(y_true**2, axis=-1)) if y_true.ndim == 3 else np.abs(y_true)
    rel_err = abs_err_total / (y_norm + 1e-10)
    
    mean_rel = np.mean(rel_err, axis=0)
    p90_rel = np.percentile(rel_err, 90, axis=0)
    
    return {
        "abs_mean": mean_err,
        "abs_p50": p50_err,
        "abs_p90": p90_err,
        "rel_mean": mean_rel,
        "rel_p90": p90_rel,
    }


def evaluate_split(
    model,
    data_dir: Path,
    family: str,
    split: str,
    stats: dict,
    device: str = "cuda",
    batch_size: int = 64,
):
    """Evaluate model on a single split."""
    
    # Load dataset
    dataset = ShardedDDEDataset(
        data_dir=str(data_dir),
        family=family,
        split=split if split != "test" else "test",
        normalize=True,
        train_stats=stats,
    )
    
    if len(dataset) == 0:
        return None
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    all_y_pred = []
    all_y_true = []
    all_rel_l2 = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split}", leave=False):
            phi = batch["phi"].to(device)
            y_true = batch["y"].to(device)
            params = batch["params"].to(device)
            
            # Forward pass
            y_pred = model(phi, params)
            
            # Compute per-sample relL2
            diff = y_pred - y_true
            rel_l2 = torch.sqrt(torch.sum(diff**2, dim=(1, 2))) / (
                torch.sqrt(torch.sum(y_true**2, dim=(1, 2))) + 1e-10
            )
            
            all_y_pred.append(y_pred.cpu().numpy())
            all_y_true.append(y_true.cpu().numpy())
            all_rel_l2.extend(rel_l2.cpu().numpy().tolist())
    
    y_pred = np.concatenate(all_y_pred, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    rel_l2 = np.array(all_rel_l2)
    
    # Denormalize for original-space metrics
    y_mean = np.array(stats["y_mean"])
    y_std = np.array(stats["y_std"])
    
    y_pred_orig = y_pred * y_std + y_mean
    y_true_orig = y_true * y_std + y_mean
    
    # Original space relL2
    diff_orig = y_pred_orig - y_true_orig
    rel_l2_orig = np.sqrt(np.sum(diff_orig**2, axis=(1, 2))) / (
        np.sqrt(np.sum(y_true_orig**2, axis=(1, 2))) + 1e-10
    )
    
    # Error vs time curves
    error_curves = compute_error_vs_time(y_pred_orig, y_true_orig)
    
    results = {
        "split": split,
        "n_samples": len(rel_l2),
        "normalized": {
            "rel_l2_mean": float(np.mean(rel_l2)),
            "rel_l2_std": float(np.std(rel_l2)),
            "rel_l2_median": float(np.median(rel_l2)),
            "rel_l2_p95": float(np.percentile(rel_l2, 95)),
        },
        "original": {
            "rel_l2_mean": float(np.mean(rel_l2_orig)),
            "rel_l2_std": float(np.std(rel_l2_orig)),
            "rel_l2_median": float(np.median(rel_l2_orig)),
            "rel_l2_p95": float(np.percentile(rel_l2_orig, 95)),
        },
        "error_vs_time": {
            k: v.tolist() for k, v in error_curves.items()
        },
    }
    
    return results, y_pred_orig, y_true_orig


def evaluate_family(
    model_dir: Path,
    family: str,
    output_dir: Path,
    base_data_dir: Path = Path("data_baseline_v1"),
    device: str = "cuda",
):
    """Full evaluation of a family across all splits."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {family}")
    print(f"{'='*70}")
    
    output_dir = output_dir / family
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(model_dir, device)
    print(f"  Model loaded from {model_dir.name}")
    
    # Load training stats
    stats_path = model_dir / "training_stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    
    # Define splits and their data directories
    splits = {
        "id": (base_data_dir, "test"),
        "ood_delay": (base_data_dir.parent / "data_ood_delay", "test"),
        "ood_delay_hole": (base_data_dir.parent / "data_ood_delay_hole", "test"),
        "ood_history": (base_data_dir.parent / "data_ood_history", "test"),
        "ood_horizon": (base_data_dir.parent / "data_ood_horizon", "test"),
    }
    
    all_results = {}
    
    for split_name, (data_dir, split_type) in splits.items():
        family_data_dir = data_dir / family
        if not family_data_dir.exists():
            print(f"  {split_name}: SKIP (data not found)")
            continue
        
        try:
            results, y_pred, y_true = evaluate_split(
                model, data_dir, family, split_type, stats, device
            )
            all_results[split_name] = results
            
            median = results["original"]["rel_l2_median"]
            p95 = results["original"]["rel_l2_p95"]
            print(f"  {split_name}: median={median:.4f}, p95={p95:.4f}")
            
            # Save per-split results
            with open(output_dir / f"metrics_{split_name}.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Generate error-vs-time plot
            t_out = np.linspace(0, 20, len(results["error_vs_time"]["abs_mean"]))
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Absolute error
            axes[0].plot(t_out, results["error_vs_time"]["abs_mean"], 
                        label="Mean", linewidth=2)
            axes[0].fill_between(t_out, 
                                results["error_vs_time"]["abs_p50"],
                                results["error_vs_time"]["abs_p90"],
                                alpha=0.3, label="p50-p90")
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel("Absolute Error")
            axes[0].set_title(f"{split_name}: Absolute Error vs Time")
            axes[0].legend()
            
            # Relative error
            axes[1].plot(t_out, results["error_vs_time"]["rel_mean"],
                        label="Mean", linewidth=2)
            axes[1].fill_between(t_out,
                                np.zeros_like(t_out),
                                results["error_vs_time"]["rel_p90"],
                                alpha=0.3, label="p90 envelope")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Relative Error")
            axes[1].set_title(f"{split_name}: Relative Error vs Time")
            axes[1].legend()
            
            plt.suptitle(f"{family}: Error vs Time ({split_name})")
            plt.tight_layout()
            plt.savefig(output_dir / f"error_vs_time_{split_name}.png", dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"  {split_name}: ERROR - {e}")
            continue
    
    # Compute OOD gaps
    if "id" in all_results:
        id_median = all_results["id"]["original"]["rel_l2_median"]
        gaps = {}
        for split_name, results in all_results.items():
            if split_name != "id":
                ood_median = results["original"]["rel_l2_median"]
                gaps[split_name] = ood_median / (id_median + 1e-10)
        all_results["ood_gaps"] = gaps
        
        print(f"\n  OOD Gaps:")
        for split_name, gap in gaps.items():
            print(f"    {split_name}: {gap:.2f}x")
    
    # Save combined results
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary comparison plot
    if len(all_results) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        splits = [k for k in all_results.keys() if k not in ["ood_gaps"]]
        medians = [all_results[k]["original"]["rel_l2_median"] for k in splits]
        p95s = [all_results[k]["original"]["rel_l2_p95"] for k in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        ax.bar(x - width/2, medians, width, label="Median")
        ax.bar(x + width/2, p95s, width, label="P95", alpha=0.7)
        
        ax.set_ylabel("Relative L2 Error")
        ax.set_title(f"{family}: Performance Across Splits")
        ax.set_xticks(x)
        ax.set_xticklabels(splits, rotation=45, ha="right")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "split_comparison.png", dpi=150)
        plt.close()
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced evaluation with error curves")
    parser.add_argument("--model_dir", required=True, help="Model checkpoint directory")
    parser.add_argument("--family", required=True, 
                        choices=["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"])
    parser.add_argument("--output_dir", default="reports/baseline_eval")
    parser.add_argument("--data_dir", default="data_baseline_v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    evaluate_family(
        Path(args.model_dir),
        args.family,
        Path(args.output_dir),
        Path(args.data_dir),
        args.device,
    )


if __name__ == "__main__":
    main()
