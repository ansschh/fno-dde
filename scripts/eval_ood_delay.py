#!/usr/bin/env python3
"""
Evaluate trained OOD-delay models on both ID and OOD test sets.
"""
import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fno1d import FNO1d
from datasets.sharded_dataset import ShardedDDEDataset
from torch.utils.data import DataLoader


def relative_l2_error(pred, target, mask=None):
    """Compute relative L2 error per sample."""
    if mask is not None:
        pred = pred * mask
        target = target * mask
    
    diff = pred - target
    l2_diff = torch.sqrt(torch.sum(diff ** 2, dim=(1, 2)))
    l2_target = torch.sqrt(torch.sum(target ** 2, dim=(1, 2)))
    return l2_diff / (l2_target + 1e-8)


def evaluate_split(model, data_dir, family, split, device, batch_size=32):
    """Evaluate model on a specific split."""
    dataset = ShardedDDEDataset(data_dir, family, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_rel_l2 = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            mask = batch["loss_mask"].to(device)
            
            pred = model(x)
            
            # Denormalize
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            
            pred_orig = pred * target_std + target_mean
            y_orig = y * target_std + target_mean
            
            rel_l2 = relative_l2_error(pred_orig, y_orig, mask)
            all_rel_l2.extend(rel_l2.cpu().numpy())
    
    all_rel_l2 = np.array(all_rel_l2)
    
    return {
        "n_samples": len(all_rel_l2),
        "mean": float(np.mean(all_rel_l2)),
        "std": float(np.std(all_rel_l2)),
        "median": float(np.median(all_rel_l2)),
        "p95": float(np.percentile(all_rel_l2, 95)),
        "p5": float(np.percentile(all_rel_l2, 5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory with trained model")
    ap.add_argument("--data_dir", required=True, help="OOD dataset directory")
    ap.add_argument("--family", required=True, choices=["hutch", "linear2"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    
    # Load model config from YAML
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_config = config.get("model", {})
    else:
        # Default config
        model_config = {
            "modes": 16,
            "width": 32,
            "n_layers": 4,
            "dropout": 0.1,
        }
    
    # Determine input/output channels from data
    test_ds = ShardedDDEDataset(data_dir, args.family, "test")
    in_channels = test_ds[0]["input"].shape[1]
    out_channels = test_ds[0]["target"].shape[1]
    
    # Build model
    model = FNO1d(
        modes=model_config.get("modes", 16),
        width=model_config.get("width", 32),
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=model_config.get("n_layers", 4),
        dropout=model_config.get("dropout", 0.1),
    ).to(args.device)
    
    # Load weights
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "final_model.pt"
    
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model from {ckpt_path}")
    print(f"Family: {args.family}")
    print(f"Data dir: {data_dir}")
    print()
    
    # Evaluate on ID test
    print("=" * 60)
    print("ID Test (restricted τ range)")
    print("=" * 60)
    id_results = evaluate_split(model, data_dir, args.family, "test", args.device)
    print(f"  N samples: {id_results['n_samples']}")
    print(f"  relL2_orig median: {id_results['median']:.4f}")
    print(f"  relL2_orig p95:    {id_results['p95']:.4f}")
    print(f"  relL2_orig mean:   {id_results['mean']:.4f} ± {id_results['std']:.4f}")
    print()
    
    # Evaluate on OOD test
    print("=" * 60)
    print("OOD Test (extended τ range)")
    print("=" * 60)
    ood_results = evaluate_split(model, data_dir, args.family, "test_ood", args.device)
    print(f"  N samples: {ood_results['n_samples']}")
    print(f"  relL2_orig median: {ood_results['median']:.4f}")
    print(f"  relL2_orig p95:    {ood_results['p95']:.4f}")
    print(f"  relL2_orig mean:   {ood_results['mean']:.4f} ± {ood_results['std']:.4f}")
    print()
    
    # Summary
    print("=" * 60)
    print("OOD Generalization Gap")
    print("=" * 60)
    gap_median = ood_results['median'] / id_results['median']
    gap_p95 = ood_results['p95'] / id_results['p95']
    print(f"  Median ratio (OOD/ID): {gap_median:.2f}x")
    print(f"  P95 ratio (OOD/ID):    {gap_p95:.2f}x")
    
    # Save results
    results = {
        "family": args.family,
        "model_dir": str(model_dir),
        "id_test": id_results,
        "ood_test": ood_results,
        "gap_median_ratio": gap_median,
        "gap_p95_ratio": gap_p95,
    }
    
    out_path = model_dir / "ood_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
