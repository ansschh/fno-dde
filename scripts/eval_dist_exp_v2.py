#!/usr/bin/env python3
"""Evaluate dist_exp_v2 model on ID and OOD splits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import json
from tqdm import tqdm

from datasets.sharded_dataset import ShardedDDEDataset, create_sharded_dataloaders


def evaluate_split(model, dataset, device, batch_size=64):
    """Evaluate model on a dataset split."""
    model.eval()
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_rel_l2 = []
    all_rmse = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            
            y_pred = model(x)
            
            # Denormalize using per-sample stats
            y_pred_orig = y_pred * target_std + target_mean
            y_true_orig = y_true * target_std + target_mean
            
            # Per-sample metrics
            for i in range(len(y_pred)):
                diff = y_pred_orig[i] - y_true_orig[i]
                rel_l2 = torch.norm(diff) / (torch.norm(y_true_orig[i]) + 1e-10)
                rmse = torch.sqrt(torch.mean(diff**2))
                
                all_rel_l2.append(rel_l2.item())
                all_rmse.append(rmse.item())
    
    rel_l2 = np.array(all_rel_l2)
    rmse = np.array(all_rmse)
    
    return {
        "rel_l2_mean": float(np.mean(rel_l2)),
        "rel_l2_std": float(np.std(rel_l2)),
        "rel_l2_median": float(np.median(rel_l2)),
        "rel_l2_p95": float(np.percentile(rel_l2, 95)),
        "rmse_median": float(np.median(rmse)),
        "n_samples": len(rel_l2),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model_dir = Path("outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403")
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]
    
    # Infer dimensions from state dict
    in_channels = state_dict["lift.weight"].shape[1]
    out_channels = state_dict["proj2.weight"].shape[0]
    
    # Rebuild model
    from models.fno1d import FNO1dResidual
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
    
    results = {"family": "dist_exp_v2"}
    
    # ID test
    print("\n=== ID Test ===")
    id_dataset = ShardedDDEDataset("data_baseline_v2", "dist_exp", "test")
    results["id_test"] = evaluate_split(model, id_dataset, device)
    print(f"  relL2 median: {results['id_test']['rel_l2_median']:.4f}")
    print(f"  relL2 p95: {results['id_test']['rel_l2_p95']:.4f}")
    
    # OOD-delay
    print("\n=== OOD-delay ===")
    try:
        ood_delay_ds = ShardedDDEDataset("data_ood_delay", "dist_exp", "test_ood")
        # Share normalization from ID
        ood_delay_ds.phi_mean, ood_delay_ds.phi_std = id_dataset.phi_mean, id_dataset.phi_std
        ood_delay_ds.y_mean, ood_delay_ds.y_std = id_dataset.y_mean, id_dataset.y_std
        results["ood_delay"] = evaluate_split(model, ood_delay_ds, device)
        print(f"  relL2 median: {results['ood_delay']['rel_l2_median']:.4f}")
        gap = results["ood_delay"]["rel_l2_median"] / (results["id_test"]["rel_l2_median"] + 1e-10)
        print(f"  OOD/ID gap: {gap:.2f}x")
    except Exception as e:
        print(f"  Error: {e}")
    
    # OOD-history
    print("\n=== OOD-history ===")
    try:
        ood_hist_ds = ShardedDDEDataset("data_ood_history", "dist_exp", "test_spline")
        ood_hist_ds.phi_mean, ood_hist_ds.phi_std = id_dataset.phi_mean, id_dataset.phi_std
        ood_hist_ds.y_mean, ood_hist_ds.y_std = id_dataset.y_mean, id_dataset.y_std
        results["ood_history"] = evaluate_split(model, ood_hist_ds, device)
        print(f"  relL2 median: {results['ood_history']['rel_l2_median']:.4f}")
        gap = results["ood_history"]["rel_l2_median"] / (results["id_test"]["rel_l2_median"] + 1e-10)
        print(f"  OOD/ID gap: {gap:.2f}x")
    except Exception as e:
        print(f"  Error: {e}")
    
    # OOD-horizon
    print("\n=== OOD-horizon ===")
    try:
        ood_hor_ds = ShardedDDEDataset("data_ood_horizon", "dist_exp", "test_horizon")
        ood_hor_ds.phi_mean, ood_hor_ds.phi_std = id_dataset.phi_mean, id_dataset.phi_std
        ood_hor_ds.y_mean, ood_hor_ds.y_std = id_dataset.y_mean, id_dataset.y_std
        results["ood_horizon"] = evaluate_split(model, ood_hor_ds, device)
        print(f"  relL2 median: {results['ood_horizon']['rel_l2_median']:.4f}")
        gap = results["ood_horizon"]["rel_l2_median"] / (results["id_test"]["rel_l2_median"] + 1e-10)
        print(f"  OOD/ID gap: {gap:.2f}x")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Compute OOD gaps summary
    id_median = results["id_test"]["rel_l2_median"]
    results["ood_gaps"] = {}
    for split in ["ood_delay", "ood_history", "ood_horizon"]:
        if split in results:
            results["ood_gaps"][split] = results[split]["rel_l2_median"] / (id_median + 1e-10)
    
    # Save results
    output_dir = Path("reports/baseline_eval_v2/dist_exp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results saved to {output_dir}/all_metrics.json ===")
    
    # Print summary
    print("\n=== DIST_EXP_V2 SUMMARY ===")
    print(f"ID test relL2 median: {results['id_test']['rel_l2_median']:.4f}")
    for split, gap in results.get("ood_gaps", {}).items():
        print(f"{split} gap: {gap:.2f}x")


if __name__ == "__main__":
    main()
