"""
Diagnostic audit script for FNO training pipeline.

Checks:
A) Metric + masking audit
   - Verify mask sum corresponds to t>0 only
   - Print distribution of ||target|| in denominator
   - Check for small denominator explosions

B) Baselines
   - Naive constant continuation
   - Compare to FNO results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from datasets.sharded_dataset import ShardedDDEDataset, create_sharded_dataloaders
from models import create_fno1d


def audit_metric_and_masking(data_dir: str, family: str):
    """Audit the metric computation and masking."""
    print(f"\n{'='*60}")
    print(f"METRIC & MASKING AUDIT: {family}")
    print(f"{'='*60}")
    
    # Load dataset
    train_ds = ShardedDDEDataset(data_dir, family, "train")
    test_ds = ShardedDDEDataset(data_dir, family, "test")
    
    # Share normalization
    test_ds.phi_mean = train_ds.phi_mean
    test_ds.phi_std = train_ds.phi_std
    test_ds.y_mean = train_ds.y_mean
    test_ds.y_std = train_ds.y_std
    test_ds.param_mean = train_ds.param_mean
    test_ds.param_std = train_ds.param_std
    
    print(f"\n--- Dataset Info ---")
    print(f"n_hist: {test_ds.n_hist}, n_out: {test_ds.n_out}, n_total: {test_ds.n_total}")
    print(f"state_dim: {test_ds.state_dim}")
    print(f"Test samples: {len(test_ds)}")
    
    print(f"\n--- Normalization Stats ---")
    print(f"y_mean: {train_ds.y_mean}")
    print(f"y_std: {train_ds.y_std}")
    print(f"phi_mean: {train_ds.phi_mean}")
    print(f"phi_std: {train_ds.phi_std}")
    
    # Check mask
    sample = test_ds[0]
    mask = sample['loss_mask']
    print(f"\n--- Mask Check ---")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum: {mask.sum().item()} (expected: {test_ds.n_out})")
    print(f"Mask[:n_hist].sum(): {mask[:test_ds.n_hist].sum().item()} (expected: 0)")
    print(f"Mask[n_hist:].sum(): {mask[test_ds.n_hist:].sum().item()} (expected: {test_ds.n_out})")
    
    # Collect target norms (normalized space)
    target_norms_normalized = []
    target_norms_original = []
    
    for i in range(len(test_ds)):
        sample = test_ds[i]
        target = sample['target']  # normalized
        mask = sample['loss_mask'].unsqueeze(-1)
        
        # Masked target norm in normalized space
        target_masked = target * mask
        norm_normalized = torch.sqrt((target_masked ** 2).sum() + 1e-8).item()
        target_norms_normalized.append(norm_normalized)
        
        # Denormalize and compute original space norm
        target_original = target * torch.tensor(train_ds.y_std) + torch.tensor(train_ds.y_mean)
        target_original_masked = target_original * mask
        norm_original = torch.sqrt((target_original_masked ** 2).sum() + 1e-8).item()
        target_norms_original.append(norm_original)
    
    target_norms_normalized = np.array(target_norms_normalized)
    target_norms_original = np.array(target_norms_original)
    
    print(f"\n--- Target Norm Distribution (NORMALIZED space - used in relL2) ---")
    print(f"  min:    {target_norms_normalized.min():.4f}")
    print(f"  p5:     {np.percentile(target_norms_normalized, 5):.4f}")
    print(f"  median: {np.median(target_norms_normalized):.4f}")
    print(f"  p95:    {np.percentile(target_norms_normalized, 95):.4f}")
    print(f"  max:    {target_norms_normalized.max():.4f}")
    
    print(f"\n--- Target Norm Distribution (ORIGINAL space) ---")
    print(f"  min:    {target_norms_original.min():.6f}")
    print(f"  p5:     {np.percentile(target_norms_original, 5):.6f}")
    print(f"  median: {np.median(target_norms_original):.6f}")
    print(f"  p95:    {np.percentile(target_norms_original, 95):.6f}")
    print(f"  max:    {target_norms_original.max():.6f}")
    
    # Check for small denominator samples
    small_denom_threshold = 1.0
    n_small = (target_norms_normalized < small_denom_threshold).sum()
    print(f"\n--- Small Denominator Check ---")
    print(f"Samples with ||target||_normalized < {small_denom_threshold}: {n_small}/{len(test_ds)} ({100*n_small/len(test_ds):.1f}%)")
    
    return train_ds, test_ds


def compute_baselines(train_ds, test_ds, family: str):
    """Compute naive and other baselines."""
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON: {family}")
    print(f"{'='*60}")
    
    naive_rel_l2_normalized = []
    naive_rel_l2_original = []
    zero_rel_l2_normalized = []
    
    for i in range(len(test_ds)):
        sample = test_ds[i]
        target = sample['target']  # normalized, (n_total, d_state)
        mask = sample['loss_mask'].unsqueeze(-1)  # (n_total, 1)
        n_hist = test_ds.n_hist
        
        # --- Naive baseline: constant continuation from phi(0) ---
        # In normalized space, phi(0) is the last history point
        phi_0_normalized = target[n_hist - 1:n_hist, :]  # (1, d_state)
        naive_pred = phi_0_normalized.expand(target.shape[0], -1)  # broadcast
        
        # relL2 in normalized space
        diff = (naive_pred - target) * mask
        target_masked = target * mask
        diff_norm = torch.sqrt((diff ** 2).sum() + 1e-8)
        target_norm = torch.sqrt((target_masked ** 2).sum() + 1e-8)
        naive_rel_l2_normalized.append((diff_norm / target_norm).item())
        
        # relL2 in original space
        target_orig = target * torch.tensor(train_ds.y_std) + torch.tensor(train_ds.y_mean)
        naive_orig = naive_pred * torch.tensor(train_ds.y_std) + torch.tensor(train_ds.y_mean)
        diff_orig = (naive_orig - target_orig) * mask
        target_orig_masked = target_orig * mask
        diff_norm_orig = torch.sqrt((diff_orig ** 2).sum() + 1e-8)
        target_norm_orig = torch.sqrt((target_orig_masked ** 2).sum() + 1e-8)
        naive_rel_l2_original.append((diff_norm_orig / target_norm_orig).item())
        
        # --- Zero baseline (predict zeros in normalized space) ---
        zero_pred = torch.zeros_like(target)
        diff_zero = (zero_pred - target) * mask
        diff_norm_zero = torch.sqrt((diff_zero ** 2).sum() + 1e-8)
        zero_rel_l2_normalized.append((diff_norm_zero / target_norm).item())
    
    naive_rel_l2_normalized = np.array(naive_rel_l2_normalized)
    naive_rel_l2_original = np.array(naive_rel_l2_original)
    zero_rel_l2_normalized = np.array(zero_rel_l2_normalized)
    
    print(f"\n--- Naive Baseline (constant continuation from phi(0)) ---")
    print(f"  [NORMALIZED space]")
    print(f"    mean:   {naive_rel_l2_normalized.mean():.4f} ± {naive_rel_l2_normalized.std():.4f}")
    print(f"    median: {np.median(naive_rel_l2_normalized):.4f}")
    print(f"    p95:    {np.percentile(naive_rel_l2_normalized, 95):.4f}")
    print(f"  [ORIGINAL space]")
    print(f"    mean:   {naive_rel_l2_original.mean():.4f} ± {naive_rel_l2_original.std():.4f}")
    print(f"    median: {np.median(naive_rel_l2_original):.4f}")
    print(f"    p95:    {np.percentile(naive_rel_l2_original, 95):.4f}")
    
    print(f"\n--- Zero Baseline (predict zeros in normalized space) ---")
    print(f"  [NORMALIZED space]")
    print(f"    mean:   {zero_rel_l2_normalized.mean():.4f} ± {zero_rel_l2_normalized.std():.4f}")
    print(f"    median: {np.median(zero_rel_l2_normalized):.4f}")
    print(f"    p95:    {np.percentile(zero_rel_l2_normalized, 95):.4f}")
    
    return {
        'naive_normalized': naive_rel_l2_normalized,
        'naive_original': naive_rel_l2_original,
        'zero_normalized': zero_rel_l2_normalized,
    }


def load_and_evaluate_fno(data_dir: str, family: str, checkpoint_dir: str, device: str = 'cuda'):
    """Load trained FNO and evaluate with proper metrics."""
    print(f"\n{'='*60}")
    print(f"FNO EVALUATION: {family}")
    print(f"{'='*60}")
    
    # Find the checkpoint
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    best_model_path = checkpoint_path / 'best_model.pt'
    if not best_model_path.exists():
        print(f"Best model not found: {best_model_path}")
        return None
    
    # Load config
    import yaml
    with open(checkpoint_path / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    train_ds = ShardedDDEDataset(data_dir, family, "train")
    test_ds = ShardedDDEDataset(data_dir, family, "test")
    test_ds.phi_mean = train_ds.phi_mean
    test_ds.phi_std = train_ds.phi_std
    test_ds.y_mean = train_ds.y_mean
    test_ds.y_std = train_ds.y_std
    test_ds.param_mean = train_ds.param_mean
    test_ds.param_std = train_ds.param_std
    
    # Get dimensions
    sample = test_ds[0]
    in_channels = sample['input'].shape[-1]
    out_channels = sample['target'].shape[-1]
    
    # Create model
    model = create_fno1d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=config.get('model', {}),
        use_residual=config.get('use_residual', False),
    )
    
    # Load weights
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    fno_rel_l2_normalized = []
    fno_rel_l2_original = []
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            inputs = sample['input'].unsqueeze(0).to(device)
            target = sample['target']
            mask = sample['loss_mask'].unsqueeze(-1)
            
            pred = model(inputs).squeeze(0).cpu()
            
            # relL2 in normalized space
            diff = (pred - target) * mask
            target_masked = target * mask
            diff_norm = torch.sqrt((diff ** 2).sum() + 1e-8)
            target_norm = torch.sqrt((target_masked ** 2).sum() + 1e-8)
            fno_rel_l2_normalized.append((diff_norm / target_norm).item())
            
            # relL2 in original space
            target_orig = target * torch.tensor(train_ds.y_std) + torch.tensor(train_ds.y_mean)
            pred_orig = pred * torch.tensor(train_ds.y_std) + torch.tensor(train_ds.y_mean)
            diff_orig = (pred_orig - target_orig) * mask
            target_orig_masked = target_orig * mask
            diff_norm_orig = torch.sqrt((diff_orig ** 2).sum() + 1e-8)
            target_norm_orig = torch.sqrt((target_orig_masked ** 2).sum() + 1e-8)
            fno_rel_l2_original.append((diff_norm_orig / target_norm_orig).item())
    
    fno_rel_l2_normalized = np.array(fno_rel_l2_normalized)
    fno_rel_l2_original = np.array(fno_rel_l2_original)
    
    print(f"\n--- FNO Results ---")
    print(f"  [NORMALIZED space]")
    print(f"    mean:   {fno_rel_l2_normalized.mean():.4f} ± {fno_rel_l2_normalized.std():.4f}")
    print(f"    median: {np.median(fno_rel_l2_normalized):.4f}")
    print(f"    p95:    {np.percentile(fno_rel_l2_normalized, 95):.4f}")
    print(f"  [ORIGINAL space]")
    print(f"    mean:   {fno_rel_l2_original.mean():.4f} ± {fno_rel_l2_original.std():.4f}")
    print(f"    median: {np.median(fno_rel_l2_original):.4f}")
    print(f"    p95:    {np.percentile(fno_rel_l2_original, 95):.4f}")
    
    return {
        'fno_normalized': fno_rel_l2_normalized,
        'fno_original': fno_rel_l2_original,
    }


def print_comparison_table(family: str, baselines: dict, fno_results: dict = None):
    """Print a comparison table."""
    print(f"\n{'='*60}")
    print(f"COMPARISON TABLE: {family}")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<25} {'Median':>10} {'Mean':>10} {'P95':>10}")
    print("-" * 55)
    
    print(f"{'Naive (normalized)':<25} {np.median(baselines['naive_normalized']):>10.4f} "
          f"{baselines['naive_normalized'].mean():>10.4f} "
          f"{np.percentile(baselines['naive_normalized'], 95):>10.4f}")
    
    print(f"{'Naive (original)':<25} {np.median(baselines['naive_original']):>10.4f} "
          f"{baselines['naive_original'].mean():>10.4f} "
          f"{np.percentile(baselines['naive_original'], 95):>10.4f}")
    
    print(f"{'Zero (normalized)':<25} {np.median(baselines['zero_normalized']):>10.4f} "
          f"{baselines['zero_normalized'].mean():>10.4f} "
          f"{np.percentile(baselines['zero_normalized'], 95):>10.4f}")
    
    if fno_results:
        print(f"{'FNO (normalized)':<25} {np.median(fno_results['fno_normalized']):>10.4f} "
              f"{fno_results['fno_normalized'].mean():>10.4f} "
              f"{np.percentile(fno_results['fno_normalized'], 95):>10.4f}")
        
        print(f"{'FNO (original)':<25} {np.median(fno_results['fno_original']):>10.4f} "
              f"{fno_results['fno_original'].mean():>10.4f} "
              f"{np.percentile(fno_results['fno_original'], 95):>10.4f}")
    
    # Compute improvement over naive
    if fno_results:
        naive_median = np.median(baselines['naive_normalized'])
        fno_median = np.median(fno_results['fno_normalized'])
        improvement = (naive_median - fno_median) / naive_median * 100
        print(f"\nFNO improvement over Naive (normalized): {improvement:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_medium')
    parser.add_argument('--families', type=str, nargs='+', default=['hutch', 'linear2'])
    parser.add_argument('--checkpoint_base', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    for family in args.families:
        print(f"\n\n{'#'*70}")
        print(f"# DIAGNOSTIC AUDIT: {family.upper()}")
        print(f"{'#'*70}")
        
        # A) Metric + masking audit
        train_ds, test_ds = audit_metric_and_masking(args.data_dir, family)
        
        # B) Baselines
        baselines = compute_baselines(train_ds, test_ds, family)
        
        # C) FNO evaluation (find latest checkpoint)
        checkpoint_dirs = list(Path(args.checkpoint_base).glob(f"{family}_*"))
        if checkpoint_dirs:
            latest_checkpoint = sorted(checkpoint_dirs)[-1]
            fno_results = load_and_evaluate_fno(
                args.data_dir, family, str(latest_checkpoint), args.device
            )
        else:
            print(f"\nNo FNO checkpoints found for {family}")
            fno_results = None
        
        # Comparison table
        print_comparison_table(family, baselines, fno_results)


if __name__ == '__main__':
    main()
