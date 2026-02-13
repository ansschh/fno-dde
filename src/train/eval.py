"""
Evaluation Script for Trained FNO Models

Provides detailed evaluation metrics, error analysis, and visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import yaml
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import DDEDataset, create_dataloaders
from models import create_fno1d
from train_fno import masked_mse_loss, relative_l2_error


@torch.no_grad()
def compute_error_vs_time(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute prediction error as a function of time.
    
    Returns:
        Dictionary with 't', 'mean_error', 'std_error', 'rel_error'
    """
    model.eval()
    
    all_errors = []
    all_targets = []
    t_grid = None
    n_hist = None
    
    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        mask = batch['loss_mask'].to(device)
        t = batch['t'].numpy()
        
        if t_grid is None:
            t_grid = t[0]  # Same for all samples
            n_hist = int((mask[0] == 0).sum().item())
        
        outputs = model(inputs)
        
        # Compute pointwise squared error (batch, length, channels)
        sq_error = ((outputs - targets) ** 2).cpu().numpy()
        all_errors.append(sq_error)
        all_targets.append(targets.cpu().numpy())
    
    # Stack all errors
    all_errors = np.concatenate(all_errors, axis=0)  # (N, length, channels)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute statistics over samples (axis 0) and channels (axis 2)
    mean_sq_error = all_errors.mean(axis=(0, 2))  # (length,)
    mean_error = np.sqrt(mean_sq_error)
    std_error = np.sqrt(all_errors.mean(axis=2)).std(axis=0)  # std over samples
    
    # Relative error
    target_magnitude = np.sqrt((all_targets ** 2).mean(axis=(0, 2)))
    rel_error = mean_error / (target_magnitude + 1e-8)
    
    return {
        't': t_grid,
        'mean_error': mean_error,
        'std_error': std_error,
        'rel_error': rel_error,
        'n_hist': n_hist,
    }


@torch.no_grad()
def compute_generalization_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute various generalization metrics.
    
    Returns:
        Dictionary with 'mse', 'rel_l2', 'max_error', 'median_error'
    """
    model.eval()
    
    all_losses = []
    all_rel_l2 = []
    all_max_errors = []
    
    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        mask = batch['loss_mask'].to(device)
        
        outputs = model(inputs)
        
        # Per-sample metrics
        mask_expanded = mask.unsqueeze(-1)
        sq_error = ((outputs - targets) ** 2) * mask_expanded
        
        # MSE per sample
        sample_mse = sq_error.sum(dim=(1, 2)) / (mask_expanded.sum(dim=(1, 2)) + 1e-8)
        all_losses.extend(sample_mse.cpu().numpy())
        
        # Relative L2 per sample
        diff_norm = torch.sqrt((sq_error).sum(dim=(1, 2)) + 1e-8)
        target_norm = torch.sqrt(((targets * mask_expanded) ** 2).sum(dim=(1, 2)) + 1e-8)
        rel_l2 = diff_norm / target_norm
        all_rel_l2.extend(rel_l2.cpu().numpy())
        
        # Max error per sample
        abs_error = torch.abs(outputs - targets) * mask_expanded
        max_error = abs_error.max(dim=1)[0].max(dim=1)[0]
        all_max_errors.extend(max_error.cpu().numpy())
    
    return {
        'mse': float(np.mean(all_losses)),
        'mse_std': float(np.std(all_losses)),
        'rel_l2': float(np.mean(all_rel_l2)),
        'rel_l2_std': float(np.std(all_rel_l2)),
        'max_error': float(np.mean(all_max_errors)),
        'median_error': float(np.median(all_max_errors)),
    }


def plot_predictions(
    model: nn.Module,
    dataset: DDEDataset,
    device: torch.device,
    n_samples: int = 5,
    output_dir: Optional[Path] = None,
):
    """
    Plot example predictions vs ground truth.
    """
    model.eval()
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for ax, idx in zip(axes, indices):
        sample = dataset[idx]
        inputs = sample['input'].unsqueeze(0).to(device)
        targets = sample['target'].numpy()
        t = sample['t'].numpy()
        mask = sample['loss_mask'].numpy()
        
        with torch.no_grad():
            outputs = model(inputs).cpu().numpy()[0]
        
        n_hist = int((mask == 0).sum())
        
        # Plot each channel
        for ch in range(targets.shape[-1]):
            ax.plot(t, targets[:, ch], 'b-', label=f'Target (ch {ch})' if ch == 0 else None, alpha=0.8)
            ax.plot(t, outputs[:, ch], 'r--', label=f'Prediction (ch {ch})' if ch == 0 else None, alpha=0.8)
        
        # Mark history/future boundary
        ax.axvline(x=t[n_hist], color='gray', linestyle=':', alpha=0.5, label='t=0')
        ax.axvspan(t[0], t[n_hist], alpha=0.1, color='gray', label='History')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        ax.legend()
        ax.set_title(f'Sample {idx}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'predictions.png', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_error_vs_time(
    error_data: Dict[str, np.ndarray],
    output_dir: Optional[Path] = None,
):
    """
    Plot error as a function of time.
    """
    t = error_data['t']
    mean_error = error_data['mean_error']
    std_error = error_data['std_error']
    rel_error = error_data['rel_error']
    n_hist = error_data['n_hist']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Absolute error
    ax1.plot(t[n_hist:], mean_error[n_hist:], 'b-', label='Mean RMSE')
    ax1.fill_between(
        t[n_hist:],
        mean_error[n_hist:] - std_error[n_hist:],
        mean_error[n_hist:] + std_error[n_hist:],
        alpha=0.3
    )
    ax1.set_xlabel('Time')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Absolute Error vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Relative error
    ax2.plot(t[n_hist:], rel_error[n_hist:], 'r-', label='Relative Error')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Relative Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'error_vs_time.png', dpi=150)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained FNO model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots and results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to evaluate on')
    parser.add_argument('--n_plot', type=int, default=5,
                        help='Number of samples to plot')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        family=config['family'],
        batch_size=config.get('batch_size', 32),
        num_workers=0,  # Simpler for evaluation
    )
    
    # Get dimensions
    sample = test_loader.dataset[0]
    in_channels = sample['input'].shape[-1]
    out_channels = sample['target'].shape[-1]
    
    # Create and load model
    model = create_fno1d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=config.get('model', {}),
        use_residual=config.get('use_residual', False),
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_generalization_metrics(model, test_loader, device)
    
    print(f"\nTest Metrics:")
    print(f"  MSE: {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"  Relative L2: {metrics['rel_l2']:.4f} ± {metrics['rel_l2_std']:.4f}")
    print(f"  Max Error (mean): {metrics['max_error']:.4f}")
    print(f"  Max Error (median): {metrics['median_error']:.4f}")
    
    # Compute error vs time
    print("\nComputing error vs time...")
    error_data = compute_error_vs_time(model, test_loader, device)
    
    # Plot results
    print("\nGenerating plots...")
    plot_predictions(model, test_loader.dataset, device, args.n_plot, output_dir)
    plot_error_vs_time(error_data, output_dir)
    
    # Save results
    if output_dir:
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        np.savez(
            output_dir / 'error_vs_time.npz',
            t=error_data['t'],
            mean_error=error_data['mean_error'],
            std_error=error_data['std_error'],
            rel_error=error_data['rel_error'],
        )
        
        print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
