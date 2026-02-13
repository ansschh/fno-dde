"""
Model Performance Metrics

Comprehensive metrics for evaluating FNO model performance:
- Relative L2 error
- Time-resolved error curves
- Spectral error (for oscillatory families)
- Constraint violation metrics
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from scipy.fft import rfft, rfftfreq


@dataclass
class SampleMetrics:
    """Metrics for a single sample."""
    sample_id: int
    rel_l2: float
    mse: float
    max_error: float
    error_at_t0: float  # History adherence
    min_predicted: float  # For positivity check
    frac_negative: float  # Fraction of negative values
    spectral_error: Optional[float] = None


class ModelMetrics:
    """
    Comprehensive model evaluation metrics.
    """
    
    def __init__(self, family: str, requires_positive: bool = False):
        self.family = family
        self.requires_positive = requires_positive
        self.sample_metrics: List[SampleMetrics] = []
        self.error_vs_time: List[np.ndarray] = []
        self.t_grid: Optional[np.ndarray] = None
    
    def add_sample(
        self,
        sample_id: int,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        t: np.ndarray,
        n_hist: int,
        compute_spectral: bool = False,
    ):
        """
        Add metrics for a single sample.
        
        Args:
            sample_id: Sample identifier
            y_pred: Predicted solution (N_total, d_state)
            y_true: True solution (N_total, d_state)
            t: Time grid (N_total,)
            n_hist: Number of history points
            compute_spectral: Whether to compute spectral error
        """
        if self.t_grid is None:
            self.t_grid = t
        
        # Get future portion only
        y_pred_future = y_pred[n_hist:]
        y_true_future = y_true[n_hist:]
        t_future = t[n_hist:]
        
        # Relative L2
        diff = y_pred_future - y_true_future
        diff_norm = np.sqrt(np.sum(diff ** 2))
        true_norm = np.sqrt(np.sum(y_true_future ** 2)) + 1e-10
        rel_l2 = diff_norm / true_norm
        
        # MSE
        mse = np.mean(diff ** 2)
        
        # Max error
        max_error = np.max(np.abs(diff))
        
        # Error at t=0 (history adherence)
        error_at_t0 = np.max(np.abs(y_pred[n_hist] - y_true[n_hist]))
        
        # Positivity metrics
        min_predicted = np.min(y_pred_future)
        frac_negative = np.mean(y_pred_future < 0) if y_pred_future.size > 0 else 0.0
        
        # Spectral error (optional)
        spectral_error = None
        if compute_spectral and y_pred_future.shape[1] >= 1:
            spectral_error = self._compute_spectral_error(
                y_pred_future[:, 0], y_true_future[:, 0], t_future
            )
        
        self.sample_metrics.append(SampleMetrics(
            sample_id=sample_id,
            rel_l2=rel_l2,
            mse=mse,
            max_error=max_error,
            error_at_t0=error_at_t0,
            min_predicted=min_predicted,
            frac_negative=frac_negative,
            spectral_error=spectral_error,
        ))
        
        # Error vs time
        pointwise_error = np.sqrt(np.sum(diff ** 2, axis=1))
        self.error_vs_time.append(pointwise_error)
    
    def _compute_spectral_error(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        t: np.ndarray,
    ) -> float:
        """Compute L2 distance between FFT magnitudes."""
        if len(y_pred) < 16:
            return 0.0
        
        # Detrend
        y_pred_detrend = y_pred - np.mean(y_pred)
        y_true_detrend = y_true - np.mean(y_true)
        
        # FFT
        fft_pred = np.abs(rfft(y_pred_detrend))
        fft_true = np.abs(rfft(y_true_detrend))
        
        # Normalize
        fft_pred = fft_pred / (np.sum(fft_pred) + 1e-10)
        fft_true = fft_true / (np.sum(fft_true) + 1e-10)
        
        # L2 distance
        return float(np.sqrt(np.sum((fft_pred - fft_true) ** 2)))
    
    def compute_stats(self) -> Dict:
        """Compute summary statistics."""
        if len(self.sample_metrics) == 0:
            return {"error": "No samples"}
        
        rel_l2s = np.array([m.rel_l2 for m in self.sample_metrics])
        mses = np.array([m.mse for m in self.sample_metrics])
        max_errors = np.array([m.max_error for m in self.sample_metrics])
        errors_t0 = np.array([m.error_at_t0 for m in self.sample_metrics])
        
        stats = {
            "n_samples": len(self.sample_metrics),
            "rel_l2": {
                "mean": float(np.mean(rel_l2s)),
                "median": float(np.median(rel_l2s)),
                "std": float(np.std(rel_l2s)),
                "p95": float(np.percentile(rel_l2s, 95)),
                "max": float(np.max(rel_l2s)),
            },
            "mse": {
                "mean": float(np.mean(mses)),
                "median": float(np.median(mses)),
            },
            "max_error": {
                "mean": float(np.mean(max_errors)),
                "p95": float(np.percentile(max_errors, 95)),
            },
            "history_adherence": {
                "error_at_t0_mean": float(np.mean(errors_t0)),
                "error_at_t0_max": float(np.max(errors_t0)),
            },
        }
        
        # Positivity metrics
        if self.requires_positive:
            min_preds = np.array([m.min_predicted for m in self.sample_metrics])
            frac_negs = np.array([m.frac_negative for m in self.sample_metrics])
            
            stats["positivity"] = {
                "min_predicted": float(np.min(min_preds)),
                "frac_samples_with_negative": float(np.mean(min_preds < 0)),
                "mean_frac_negative_per_sample": float(np.mean(frac_negs)),
            }
        
        # Spectral error
        spectral_errors = [m.spectral_error for m in self.sample_metrics if m.spectral_error is not None]
        if spectral_errors:
            stats["spectral_error"] = {
                "mean": float(np.mean(spectral_errors)),
                "median": float(np.median(spectral_errors)),
            }
        
        # Error vs time
        if self.error_vs_time:
            error_curves = np.stack(self.error_vs_time)
            stats["error_vs_time"] = {
                "mean": error_curves.mean(axis=0).tolist(),
                "p90": np.percentile(error_curves, 90, axis=0).tolist(),
            }
            if self.t_grid is not None:
                n_hist = len(self.t_grid) - len(self.error_vs_time[0])
                stats["error_vs_time"]["t"] = self.t_grid[n_hist:].tolist()
        
        return stats
    
    def save(self, path: Path):
        """Save results to JSON."""
        stats = self.compute_stats()
        stats["family"] = self.family
        
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def compute_all_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    family: str,
    requires_positive: bool = False,
    compute_spectral: bool = False,
) -> ModelMetrics:
    """
    Compute all metrics for a model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Torch device
        family: DDE family name
        requires_positive: Whether family requires positive solutions
        compute_spectral: Whether to compute spectral error
        
    Returns:
        ModelMetrics object with all computed metrics
    """
    model.eval()
    metrics = ModelMetrics(family, requires_positive)
    
    sample_id = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].numpy()
            t = batch["t"].numpy()
            loss_mask = batch["loss_mask"].numpy()
            
            outputs = model(inputs).cpu().numpy()
            
            # Find n_hist from mask
            n_hist = int((loss_mask[0] == 0).sum())
            
            for i in range(outputs.shape[0]):
                metrics.add_sample(
                    sample_id=sample_id,
                    y_pred=outputs[i],
                    y_true=targets[i],
                    t=t[i],
                    n_hist=n_hist,
                    compute_spectral=compute_spectral,
                )
                sample_id += 1
    
    return metrics
