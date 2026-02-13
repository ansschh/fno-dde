#!/usr/bin/env python3
"""
Unified Evaluation Module for DDE-FNO

Provides consistent metrics in ORIGINAL SPACE across all evaluation scripts.
All reported metrics use the same schema and denormalization.

Core Metrics (always computed):
- relL2_orig_median: Median relative L2 error in original space
- relL2_orig_p95: 95th percentile relative L2 error
- relL2_orig_mean: Mean relative L2 error
- relL2_orig_std: Standard deviation of relative L2 error
- n_samples: Number of samples evaluated

Optional Metrics:
- error_vs_time: Error as function of time (for plots)
- per_sample_errors: Full list of per-sample errors (for analysis)
"""
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List, Union
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader


@dataclass
class EvalMetrics:
    """Standardized evaluation metrics container."""
    split_name: str
    n_samples: int
    relL2_orig_median: float
    relL2_orig_p95: float
    relL2_orig_mean: float
    relL2_orig_std: float
    relL2_orig_p5: float = 0.0
    relL2_orig_min: float = 0.0
    relL2_orig_max: float = 0.0
    # Frequency-binned RMSE (a la Localized FNO paper)
    frmse_low: float = 0.0
    frmse_mid: float = 0.0
    frmse_high: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_errors(cls, errors: np.ndarray, split_name: str,
                    frmse: Optional[Dict[str, float]] = None) -> "EvalMetrics":
        """Create metrics from array of per-sample errors."""
        frmse = frmse or {}
        return cls(
            split_name=split_name,
            n_samples=len(errors),
            relL2_orig_median=float(np.median(errors)),
            relL2_orig_p95=float(np.percentile(errors, 95)),
            relL2_orig_mean=float(np.mean(errors)),
            relL2_orig_std=float(np.std(errors)),
            relL2_orig_p5=float(np.percentile(errors, 5)),
            relL2_orig_min=float(np.min(errors)),
            relL2_orig_max=float(np.max(errors)),
            frmse_low=frmse.get("low", 0.0),
            frmse_mid=frmse.get("mid", 0.0),
            frmse_high=frmse.get("high", 0.0),
        )


def compute_frequency_binned_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute frequency-binned RMSE in original space.

    Splits the frequency axis into 3 equal bands (low/mid/high) and computes
    RMSE for each band, following the fRMSE diagnostic from the Localized FNO paper.

    Args:
        pred: Predictions in normalized space (batch, length, channels)
        target: Targets in normalized space (batch, length, channels)
        mask: Loss mask (batch, length) or (batch, length, 1)
        target_mean: Denormalization mean (batch, 1, channels)
        target_std: Denormalization std (batch, 1, channels)

    Returns:
        Dict with keys "low", "mid", "high" containing per-band fRMSE values.
    """
    # Denormalize
    pred_orig = pred * target_std + target_mean
    target_orig = target * target_std + target_mean

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    # Apply mask and work on future region
    error = (pred_orig - target_orig) * mask
    target_masked = target_orig * mask

    # Average over batch and channels: (length,) power spectrum
    B, L, C = error.shape

    # FFT along time dimension (dim=1)
    error_fft = torch.fft.rfft(error, dim=1)   # (B, L//2+1, C)
    target_fft = torch.fft.rfft(target_masked, dim=1)

    # Spectral power: |F|^2, averaged over batch and channels
    error_power = (error_fft.abs() ** 2).mean(dim=(0, 2))     # (L//2+1,)
    target_power = (target_fft.abs() ** 2).mean(dim=(0, 2))

    n_freqs = error_power.shape[0]
    band_size = max(n_freqs // 3, 1)

    bands = {
        "low": (0, band_size),
        "mid": (band_size, 2 * band_size),
        "high": (2 * band_size, n_freqs),
    }

    result = {}
    for band_name, (start, end) in bands.items():
        err_band = error_power[start:end].sum()
        tgt_band = target_power[start:end].sum()
        # fRMSE = sqrt(error_energy / target_energy) per band
        if tgt_band > 1e-12:
            result[band_name] = float(torch.sqrt(err_band / tgt_band).cpu())
        else:
            result[band_name] = 0.0

    return result


def compute_relative_l2_original_space(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """
    Compute relative L2 error in ORIGINAL SPACE.
    
    Args:
        pred: Predictions in normalized space (batch, length, channels)
        target: Targets in normalized space (batch, length, channels)
        mask: Loss mask, 1 for future region (batch, length) or (batch, length, 1)
        target_mean: Per-sample mean for denormalization (batch, 1, channels)
        target_std: Per-sample std for denormalization (batch, 1, channels)
    
    Returns:
        Per-sample relative L2 errors (batch,)
    """
    # Denormalize to original space
    pred_orig = pred * target_std + target_mean
    target_orig = target * target_std + target_mean
    
    # Ensure mask has correct shape
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)  # (batch, length, 1)
    
    # Apply mask (only compute on future region)
    pred_masked = pred_orig * mask
    target_masked = target_orig * mask
    
    # Compute L2 norms
    diff = pred_masked - target_masked
    l2_diff = torch.sqrt(torch.sum(diff ** 2, dim=(1, 2)) + 1e-12)
    l2_target = torch.sqrt(torch.sum(target_masked ** 2, dim=(1, 2)) + 1e-12)
    
    return l2_diff / l2_target


@torch.no_grad()
def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
) -> EvalMetrics:
    """
    Evaluate model on a dataloader with unified metrics.
    
    IMPORTANT: All metrics are computed in ORIGINAL SPACE after denormalization.
    If normalization stats are not available, assumes data is already in original space.
    
    Args:
        model: Trained model
        loader: DataLoader for evaluation
        device: Device to evaluate on
        split_name: Name of split (for reporting)
    
    Returns:
        EvalMetrics with all standard metrics
    """
    model.eval()
    all_errors = []
    # Accumulators for frequency-binned RMSE
    frmse_accum = {"low": [], "mid": [], "high": []}

    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        mask = batch["loss_mask"].to(device)

        # Forward pass
        pred = model(x)

        # Get normalization stats if available, otherwise use identity transform
        if "target_mean" in batch and "target_std" in batch:
            target_mean = batch["target_mean"].to(device)
            target_std = batch["target_std"].to(device)
            # Compute relative L2 in original space
            rel_l2 = compute_relative_l2_original_space(
                pred, y, mask, target_mean, target_std
            )
            # Compute frequency-binned RMSE
            frmse_batch = compute_frequency_binned_rmse(
                pred, y, mask, target_mean, target_std
            )
            for k in frmse_accum:
                frmse_accum[k].append(frmse_batch[k])
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            pred_masked = pred * mask
            target_masked = y * mask
            diff = pred_masked - target_masked
            l2_diff = torch.sqrt(torch.sum(diff ** 2, dim=(1, 2)) + 1e-12)
            l2_target = torch.sqrt(torch.sum(target_masked ** 2, dim=(1, 2)) + 1e-12)
            rel_l2 = l2_diff / l2_target

        all_errors.extend(rel_l2.cpu().numpy())

    # Average fRMSE across batches
    frmse_avg = {}
    for k, vals in frmse_accum.items():
        frmse_avg[k] = float(np.mean(vals)) if vals else 0.0

    return EvalMetrics.from_errors(np.array(all_errors), split_name, frmse=frmse_avg)


@torch.no_grad()
def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    split_name: str = "test",
    batch_size: int = 32,
) -> EvalMetrics:
    """
    Evaluate model on a dataset (creates loader internally).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return evaluate_model_on_loader(model, loader, device, split_name)


@torch.no_grad()
def compute_error_vs_time(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute relative L2 error as a function of time.
    
    Returns dict with:
        - t: time grid
        - rel_error_mean: mean relative error at each time
        - rel_error_std: std of relative error at each time
    """
    model.eval()
    
    all_sq_errors = []
    all_sq_targets = []
    t_grid = None
    
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        mask = batch["loss_mask"].to(device)
        target_mean = batch["target_mean"].to(device)
        target_std = batch["target_std"].to(device)
        
        if t_grid is None and "t" in batch:
            t_grid = batch["t"][0].numpy()
        
        pred = model(x)
        
        # Denormalize
        pred_orig = pred * target_std + target_mean
        target_orig = y * target_std + target_mean
        
        # Squared errors at each point
        sq_error = ((pred_orig - target_orig) ** 2).cpu().numpy()
        sq_target = (target_orig ** 2).cpu().numpy()
        
        all_sq_errors.append(sq_error)
        all_sq_targets.append(sq_target)
    
    all_sq_errors = np.concatenate(all_sq_errors, axis=0)
    all_sq_targets = np.concatenate(all_sq_targets, axis=0)
    
    # Mean over samples and channels
    mean_sq_error = all_sq_errors.mean(axis=(0, 2))
    mean_sq_target = all_sq_targets.mean(axis=(0, 2))
    
    rel_error = np.sqrt(mean_sq_error) / (np.sqrt(mean_sq_target) + 1e-8)
    
    return {
        "t": t_grid,
        "rel_error_mean": rel_error,
        "rmse": np.sqrt(mean_sq_error),
    }


def save_metrics_json(
    metrics_dict: Dict[str, EvalMetrics],
    output_dir: Path,
    prefix: str = "metrics",
) -> None:
    """
    Save all metrics to JSON files with consistent naming.
    
    Creates files like:
        - metrics_id.json
        - metrics_ood_delay.json
        - etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, metrics in metrics_dict.items():
        filename = f"{prefix}_{split_name}.json"
        metrics.to_json(output_dir / filename)
    
    # Also save combined summary
    summary = {name: m.to_dict() for name, m in metrics_dict.items()}
    with open(output_dir / f"{prefix}_all.json", "w") as f:
        json.dump(summary, f, indent=2)


def print_metrics_table(metrics_dict: Dict[str, EvalMetrics]) -> str:
    """Print metrics in a formatted table."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Split':<20} {'N':>8} {'Median':>10} {'P95':>10} {'Mean±Std':>15}")
    lines.append("=" * 70)
    
    for name, m in metrics_dict.items():
        lines.append(
            f"{name:<20} {m.n_samples:>8} {m.relL2_orig_median:>10.4f} "
            f"{m.relL2_orig_p95:>10.4f} {m.relL2_orig_mean:>7.4f}±{m.relL2_orig_std:.4f}"
        )
    
    lines.append("=" * 70)
    return "\n".join(lines)
