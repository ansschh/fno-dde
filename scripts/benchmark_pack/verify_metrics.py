#!/usr/bin/env python3
"""
Verify metrics using the original training evaluation approach.
Compare FNO vs Identity baseline properly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np
from tqdm import tqdm
from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import FNO1dResidual

FAMILIES = {
    "hutch": ("outputs/baseline_v1/hutch_seed42_20251228_131919", "data_baseline_v1"),
    "linear2": ("outputs/baseline_v1/linear2_seed42_20251228_142839", "data_baseline_v1"),
    "dist_exp": ("outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403", "data_baseline_v2"),
    "vdp": ("outputs/baseline_v1/vdp_seed42_20251229_020516", "data_baseline_v1"),
    "dist_uniform": ("outputs/baseline_v1/dist_uniform_seed42_20251229_030851", "data_baseline_v1"),
}


def load_model(model_dir: Path, device: str = "cuda"):
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
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
    model.to(device).eval()
    return model


def evaluate_with_mask(model, loader, device):
    """Evaluate using the same approach as training (with loss_mask)."""
    all_rel_l2_fno = []
    all_rel_l2_identity = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            mask = batch["loss_mask"].to(device)
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            
            # FNO prediction
            outputs = model(inputs)
            
            # Denormalize to original space
            outputs_orig = outputs * target_std + target_mean
            targets_orig = targets * target_std + target_mean
            
            # Apply mask (future region only)
            mask_exp = mask.unsqueeze(-1)
            
            # FNO error
            diff_fno = (outputs_orig - targets_orig) * mask_exp
            target_masked = targets_orig * mask_exp
            
            diff_l2_fno = torch.sqrt((diff_fno ** 2).sum(dim=(1, 2)) + 1e-8)
            target_l2 = torch.sqrt((target_masked ** 2).sum(dim=(1, 2)) + 1e-8)
            rel_l2_fno = (diff_l2_fno / target_l2).cpu().numpy()
            all_rel_l2_fno.extend(rel_l2_fno)
            
            # Identity baseline: y(t) = y(0) for all t
            # y(0) is the first point of the target
            y0 = targets_orig[:, 0:1, :]  # (B, 1, D)
            identity_pred = y0.expand(-1, targets_orig.shape[1], -1)
            
            diff_identity = (identity_pred - targets_orig) * mask_exp
            diff_l2_identity = torch.sqrt((diff_identity ** 2).sum(dim=(1, 2)) + 1e-8)
            rel_l2_identity = (diff_l2_identity / target_l2).cpu().numpy()
            all_rel_l2_identity.extend(rel_l2_identity)
    
    return {
        "fno_median": np.median(all_rel_l2_fno),
        "fno_p95": np.percentile(all_rel_l2_fno, 95),
        "identity_median": np.median(all_rel_l2_identity),
        "identity_p95": np.percentile(all_rel_l2_identity, 95),
    }


def main():
    print("="*70)
    print("METRIC VERIFICATION: FNO vs Identity Baseline (with loss_mask)")
    print("="*70)
    
    results = {}
    
    for fam, (model_path, data_dir) in FAMILIES.items():
        model_dir = Path(model_path)
        print(f"\n--- {fam} ---")
        
        model = load_model(model_dir, "cuda")
        dataset = ShardedDDEDataset(data_dir, fam, "test")
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        
        metrics = evaluate_with_mask(model, loader, "cuda")
        results[fam] = metrics
        
        ratio = metrics["identity_median"] / (metrics["fno_median"] + 1e-10)
        
        print(f"  Identity: median={metrics['identity_median']:.4f}, p95={metrics['identity_p95']:.4f}")
        print(f"  FNO:      median={metrics['fno_median']:.4f}, p95={metrics['fno_p95']:.4f}")
        print(f"  Ratio:    {ratio:.1f}x improvement")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Family':<15} | {'Identity':>10} | {'FNO':>10} | {'Ratio':>8}")
    print("-"*50)
    for fam in FAMILIES:
        r = results[fam]
        ratio = r["identity_median"] / (r["fno_median"] + 1e-10)
        print(f"{fam:<15} | {r['identity_median']:>10.4f} | {r['fno_median']:>10.4f} | {ratio:>7.1f}x")


if __name__ == "__main__":
    main()
