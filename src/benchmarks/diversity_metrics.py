"""
Diversity Metrics

Computes dataset diversity statistics to ensure training data
is not trivial (everything goes to equilibrium).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from scipy import signal
from scipy.fft import rfft, rfftfreq


@dataclass
class TrajectoryStats:
    """Statistics for a single trajectory."""
    sample_id: int
    # Dynamics
    amplitude_range: float      # max - min
    n_oscillations: int         # number of zero crossings / 2
    settling_time: Optional[float]  # time to reach steady state
    dominant_freq: Optional[float]  # FFT peak frequency
    # State bounds
    max_val: float
    min_val: float
    final_val: float


class DiversityMetrics:
    """
    Computes and aggregates diversity metrics for a dataset.
    
    Metrics:
    - Dynamics: settling time, oscillation count, dominant frequency
    - Amplitude range
    - History roughness
    - Parameter coverage
    """
    
    def __init__(self, family: str):
        self.family = family
        self.trajectory_stats: List[TrajectoryStats] = []
        self.history_roughness: List[float] = []
        self.params_collected: List[np.ndarray] = []
        self.lags_collected: List[np.ndarray] = []
    
    def add_sample(
        self,
        sample_id: int,
        t: np.ndarray,
        y: np.ndarray,
        phi: np.ndarray,
        params: np.ndarray,
        lags: np.ndarray,
    ):
        """
        Add a sample for diversity analysis.
        
        Args:
            sample_id: Sample identifier
            t: Time grid (future portion)
            y: Solution (N, d_state) - future portion only
            phi: History (N_hist, d_hist)
            params: Parameter vector
            lags: Delay vector
        """
        # Use first state dimension for scalar metrics
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_scalar = y[:, 0]
        
        if phi.ndim == 1:
            phi = phi.reshape(-1, 1)
        phi_scalar = phi[:, 0]
        
        # Amplitude range
        amplitude_range = np.max(y_scalar) - np.min(y_scalar)
        
        # Oscillation count (zero crossings around mean)
        y_centered = y_scalar - np.mean(y_scalar)
        zero_crossings = np.sum(np.diff(np.sign(y_centered)) != 0)
        n_oscillations = zero_crossings // 2
        
        # Settling time
        settling_time = self._compute_settling_time(t, y_scalar)
        
        # Dominant frequency
        dominant_freq = self._compute_dominant_frequency(t, y_scalar)
        
        # State bounds
        max_val = np.max(y_scalar)
        min_val = np.min(y_scalar)
        final_val = y_scalar[-1]
        
        self.trajectory_stats.append(TrajectoryStats(
            sample_id=sample_id,
            amplitude_range=amplitude_range,
            n_oscillations=n_oscillations,
            settling_time=settling_time,
            dominant_freq=dominant_freq,
            max_val=max_val,
            min_val=min_val,
            final_val=final_val,
        ))
        
        # History roughness (RMS of numerical derivative)
        if len(phi_scalar) > 1:
            dt_hist = 1.0 / len(phi_scalar)  # normalized
            dphi = np.diff(phi_scalar) / dt_hist
            roughness = np.sqrt(np.mean(dphi ** 2))
            self.history_roughness.append(roughness)
        
        # Collect params and lags
        self.params_collected.append(params)
        self.lags_collected.append(lags)
    
    def _compute_settling_time(
        self,
        t: np.ndarray,
        y: np.ndarray,
        eps: float = 0.05,
    ) -> Optional[float]:
        """
        Compute time to settle (first time |y - y_final| < eps * amplitude stays small).
        """
        y_final = np.mean(y[-max(1, len(y)//10):])  # mean of last 10%
        amplitude = np.max(np.abs(y - y_final))
        
        if amplitude < 1e-6:
            return 0.0  # Already at steady state
        
        threshold = eps * amplitude
        
        # Find settling time
        settled = np.abs(y - y_final) < threshold
        
        # Find first time it's settled and stays settled
        for i in range(len(settled) - 1, -1, -1):
            if not settled[i]:
                if i < len(t) - 1:
                    return t[i + 1]
                else:
                    return None
        
        return t[0]  # Was always settled
    
    def _compute_dominant_frequency(
        self,
        t: np.ndarray,
        y: np.ndarray,
    ) -> Optional[float]:
        """Compute dominant frequency from FFT."""
        if len(y) < 16:
            return None
        
        # Detrend
        y_detrend = y - np.mean(y)
        
        # FFT
        dt = t[1] - t[0]
        freqs = rfftfreq(len(y), dt)
        fft_mag = np.abs(rfft(y_detrend))
        
        # Find peak (skip DC)
        if len(fft_mag) < 2:
            return None
        
        peak_idx = np.argmax(fft_mag[1:]) + 1
        return float(freqs[peak_idx])
    
    def compute_stats(self) -> Dict:
        """Compute summary statistics."""
        if len(self.trajectory_stats) == 0:
            return {"error": "No samples"}
        
        # Dynamics stats
        amplitudes = [s.amplitude_range for s in self.trajectory_stats]
        oscillations = [s.n_oscillations for s in self.trajectory_stats]
        settling_times = [s.settling_time for s in self.trajectory_stats if s.settling_time is not None]
        dom_freqs = [s.dominant_freq for s in self.trajectory_stats if s.dominant_freq is not None]
        
        stats = {
            "n_samples": len(self.trajectory_stats),
            "dynamics": {
                "amplitude_range": {
                    "mean": float(np.mean(amplitudes)),
                    "std": float(np.std(amplitudes)),
                    "min": float(np.min(amplitudes)),
                    "max": float(np.max(amplitudes)),
                },
                "n_oscillations": {
                    "mean": float(np.mean(oscillations)),
                    "median": float(np.median(oscillations)),
                    "max": int(np.max(oscillations)),
                    "frac_zero": float(np.mean(np.array(oscillations) == 0)),
                },
            },
        }
        
        if settling_times:
            stats["dynamics"]["settling_time"] = {
                "mean": float(np.mean(settling_times)),
                "median": float(np.median(settling_times)),
                "p90": float(np.percentile(settling_times, 90)),
            }
        
        if dom_freqs:
            stats["dynamics"]["dominant_freq"] = {
                "mean": float(np.mean(dom_freqs)),
                "std": float(np.std(dom_freqs)),
            }
        
        # History stats
        if self.history_roughness:
            stats["history"] = {
                "roughness_mean": float(np.mean(self.history_roughness)),
                "roughness_std": float(np.std(self.history_roughness)),
            }
        
        # Parameter coverage
        if self.params_collected:
            params_arr = np.stack(self.params_collected)
            stats["params"] = {
                "means": params_arr.mean(axis=0).tolist(),
                "stds": params_arr.std(axis=0).tolist(),
                "mins": params_arr.min(axis=0).tolist(),
                "maxs": params_arr.max(axis=0).tolist(),
            }
        
        if self.lags_collected:
            lags_arr = np.stack(self.lags_collected)
            stats["lags"] = {
                "means": lags_arr.mean(axis=0).tolist(),
                "stds": lags_arr.std(axis=0).tolist(),
                "mins": lags_arr.min(axis=0).tolist(),
                "maxs": lags_arr.max(axis=0).tolist(),
            }
        
        return stats
    
    def save(self, path: Path):
        """Save results to JSON."""
        stats = self.compute_stats()
        stats["family"] = self.family
        
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def run_diversity_benchmark(
    data_dir: Path,
    family: str,
    split: str = "train",
    max_samples: int = 1000,
) -> DiversityMetrics:
    """
    Run diversity benchmark on dataset.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.sharded_dataset import ShardedDDEDataset
    
    metrics = DiversityMetrics(family)
    
    try:
        dataset = ShardedDDEDataset(str(data_dir), family, split, normalize=False)
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return metrics
    
    n_check = min(max_samples, len(dataset))
    
    for idx in range(n_check):
        sample = dataset[idx]
        
        t = sample["t"].numpy()
        target = sample["target"].numpy()
        params = sample["params"].numpy()
        
        n_hist = dataset.n_hist
        phi = target[:n_hist]
        y = target[n_hist:]
        t_out = t[n_hist:]
        
        # Get lags from dataset
        lags = np.array([params[i] for i, name in enumerate(dataset.param_names) 
                        if "tau" in name.lower()])
        if len(lags) == 0:
            lags = np.array([1.0])  # Default
        
        metrics.add_sample(
            sample_id=idx,
            t=t_out,
            y=y,
            phi=phi,
            params=params,
            lags=lags,
        )
    
    return metrics
