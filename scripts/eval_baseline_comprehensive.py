#!/usr/bin/env python3
"""
Comprehensive Baseline Evaluation Script

Evaluates a trained model on all available splits:
- ID test
- OOD-delay (extrapolation)
- OOD-delay-hole (interpolation)
- OOD-history
- OOD-horizon

All metrics are computed in ORIGINAL SPACE using the unified evaluation module.
Outputs consistent JSON files for paper-ready tables.

Usage:
    python scripts/eval_baseline_comprehensive.py \
        --model_dir outputs/baseline_v1/hutch_seed42_... \
        --family hutch \
        --output_dir reports/baseline_eval/hutch/run_001
"""
import argparse
import json
import yaml
import torch
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fno1d import FNO1d, FNO1dResidual
from datasets.sharded_dataset import ShardedDDEDataset
from eval.unified_eval import (
    EvalMetrics,
    evaluate_model_on_dataset,
    save_metrics_json,
    print_metrics_table,
)


# Dataset paths for each split type
# Note: ShardedDDEDataset expects data_dir where manifest is at data_dir/{family}/manifest.json
# So we pass the parent directory, not the family-specific directory
SPLIT_CONFIGS = {
    "id": {
        "data_template": "data_baseline_v1",
        "split": "test",
        "description": "In-distribution test",
    },
    "ood_delay": {
        "data_template": "data_ood_delay",
        "split": "test_ood",
        "description": "OOD-delay extrapolation (τ > 1.3)",
    },
    "ood_delay_hole": {
        "data_template": "data_ood_delay_hole",
        "split": "test_hole",
        "description": "OOD-delay interpolation (τ ∈ [0.9, 1.1])",
    },
    "ood_history": {
        "data_template": "data_ood_history",
        "split": "test_spline",
        "description": "OOD-history (spline instead of Fourier)",
    },
    "ood_horizon": {
        "data_template": "data_ood_horizon",
        "split": "test_horizon",
        "description": "OOD-horizon (T=40 instead of T=20)",
    },
}


def load_model(model_dir: Path, device: str) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    # Load config
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
    else:
        model_cfg = {"modes": 12, "width": 48, "n_layers": 3, "dropout": 0.1}
    
    # Load checkpoint to get input channels
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "final_model.pt"
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Detect input/output channels from checkpoint
    in_channels = ckpt["model_state_dict"]["lift.weight"].shape[1]
    # Handle different checkpoint formats
    if "proj.2.weight" in ckpt["model_state_dict"]:
        out_channels = ckpt["model_state_dict"]["proj.2.weight"].shape[0]
    elif "proj2.weight" in ckpt["model_state_dict"]:
        out_channels = ckpt["model_state_dict"]["proj2.weight"].shape[0]
    else:
        out_channels = 1  # Default fallback
    
    # Check if use_residual was enabled (FNO1dResidual has built-in layer_norm)
    use_residual = config.get("use_residual", False)
    
    # Select model class
    model_class = FNO1dResidual if use_residual else FNO1d
    
    # Build model
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
    
    return model


def evaluate_split(
    model: torch.nn.Module,
    family: str,
    split_key: str,
    device: str,
    base_dir: Path,
    train_stats: dict = None,
) -> EvalMetrics:
    """Evaluate model on a specific split."""
    config = SPLIT_CONFIGS[split_key]
    data_dir = base_dir / config["data_template"]
    
    if not data_dir.exists():
        print(f"  [SKIP] {split_key}: {data_dir} not found")
        return None
    
    try:
        dataset = ShardedDDEDataset(str(data_dir), family, config["split"])
        # Use training stats for consistent normalization
        if train_stats is not None:
            dataset.y_mean = train_stats["y_mean"]
            dataset.y_std = train_stats["y_std"]
            dataset.phi_mean = train_stats["phi_mean"]
            dataset.phi_std = train_stats["phi_std"]
            dataset.param_mean = train_stats["param_mean"]
            dataset.param_std = train_stats["param_std"]
        metrics = evaluate_model_on_dataset(model, dataset, device, split_key)
        return metrics
    except Exception as e:
        print(f"  [ERROR] {split_key}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Comprehensive baseline evaluation")
    parser.add_argument("--model_dir", required=True, help="Model checkpoint directory")
    parser.add_argument("--family", required=True, choices=["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"])
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_dir", default=".", help="Base directory for data paths")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    base_dir = Path(args.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Comprehensive Baseline Evaluation")
    print("=" * 70)
    print(f"Model: {model_dir}")
    print(f"Family: {args.family}")
    print(f"Output: {output_dir}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(model_dir, args.device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Load training dataset to get normalization stats
    print("Loading training stats for consistent normalization...")
    train_ds = ShardedDDEDataset(str(base_dir / "data_baseline_v1"), args.family, "train")
    train_stats = {
        "y_mean": train_ds.y_mean,
        "y_std": train_ds.y_std,
        "phi_mean": train_ds.phi_mean,
        "phi_std": train_ds.phi_std,
        "param_mean": train_ds.param_mean,
        "param_std": train_ds.param_std,
    }
    print(f"  y_mean={train_stats['y_mean']}, y_std={train_stats['y_std']}")
    print()
    
    # Evaluate on all splits
    print("Evaluating splits...")
    all_metrics = {}
    
    for split_key, config in SPLIT_CONFIGS.items():
        print(f"\n  {split_key}: {config['description']}")
        metrics = evaluate_split(model, args.family, split_key, args.device, base_dir, train_stats)
        if metrics is not None:
            all_metrics[split_key] = metrics
            print(f"    n={metrics.n_samples}, median={metrics.relL2_orig_median:.4f}, "
                  f"p95={metrics.relL2_orig_p95:.4f}")
    
    # Print summary table
    print("\n")
    print(print_metrics_table(all_metrics))
    
    # Save metrics
    save_metrics_json(all_metrics, output_dir)
    
    # Save run metadata
    metadata = {
        "model_dir": str(model_dir),
        "family": args.family,
        "timestamp": datetime.now().isoformat(),
        "splits_evaluated": list(all_metrics.keys()),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Compute OOD gaps
    if "id" in all_metrics:
        print("\nOOD Generalization Gaps (vs ID):")
        id_median = all_metrics["id"].relL2_orig_median
        for name, m in all_metrics.items():
            if name != "id":
                gap = m.relL2_orig_median / id_median
                print(f"  {name}: {gap:.2f}x")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
