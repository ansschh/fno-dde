"""
Residual Benchmark

Computes DDE residuals as a physics-based quality check.
Catches subtle bugs even when solver returns "Success".
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ResidualResult:
    """Residual check result for a single sample."""
    sample_id: int
    mean_residual: float
    max_residual: float
    residual_vs_time: np.ndarray


class ResidualBenchmark:
    """
    Computes DDE residuals for generated solutions.
    
    DDE-aware residual checking:
    - Uses 5-point stencil for higher accuracy derivatives
    - Skips time points near breaking points (multiples of τ)
    
    For discrete-delay families:
        R(t) ≈ dy/dt - f(t, x(t), x(t-τ), ...)
    
    This catches:
    - Wrong lag indexing
    - Wrong units/scaling
    - Interpolation bugs
    """
    
    def __init__(self, family: str):
        self.family = family
        self.results: List[ResidualResult] = []
    
    def add_result(
        self,
        sample_id: int,
        t: np.ndarray,
        y: np.ndarray,
        rhs_values: np.ndarray,
        lags: Optional[List[float]] = None,
    ):
        """
        Add a residual check result with DDE-aware processing.
        
        Args:
            sample_id: Sample identifier
            t: Time grid (N,)
            y: Solution (N, d_state)
            rhs_values: RHS evaluated at each time point (N, d_state)
            lags: List of delay values (for breaking point detection)
        """
        dt = t[1] - t[0]
        n = len(t)
        
        # Use 5-point stencil for interior: (-y[i+2] + 8y[i+1] - 8y[i-1] + y[i-2]) / (12*dt)
        dy_dt = np.zeros_like(y)
        
        # 5-point stencil for interior (indices 2 to n-3)
        if n > 4:
            dy_dt[2:-2] = (-y[4:] + 8*y[3:-1] - 8*y[1:-3] + y[:-4]) / (12 * dt)
        
        # 3-point central difference for near-boundary (indices 1 and n-2)
        if n > 2:
            dy_dt[1] = (y[2] - y[0]) / (2 * dt)
            dy_dt[-2] = (y[-1] - y[-3]) / (2 * dt)
        
        # Forward/backward for boundaries
        dy_dt[0] = (-3*y[0] + 4*y[1] - y[2]) / (2 * dt) if n > 2 else (y[1] - y[0]) / dt
        dy_dt[-1] = (3*y[-1] - 4*y[-2] + y[-3]) / (2 * dt) if n > 2 else (y[-1] - y[-2]) / dt
        
        # Residual = dy/dt - f(t, y, y_delayed, ...)
        residual = dy_dt - rhs_values
        
        # Build mask for valid points (skip breaking points)
        valid_mask = np.ones(n, dtype=bool)
        
        # Skip first few points (initial discontinuity)
        skip_start = max(5, int(0.05 * n))
        valid_mask[:skip_start] = False
        valid_mask[-3:] = False  # Skip last few for boundary effects
        
        # Skip points near breaking points (multiples of τ)
        if lags is not None:
            for tau in lags:
                if tau > 0:
                    # Breaking points at k*τ for k = 1, 2, 3, ...
                    k = 1
                    while k * tau < t[-1]:
                        break_time = k * tau
                        # Skip points within 2*dt of breaking point
                        near_break = np.abs(t - break_time) < 2 * dt
                        valid_mask[near_break] = False
                        k += 1
        
        # Compute stats on valid points only
        valid_residual = residual[valid_mask]
        
        if len(valid_residual) > 0:
            mean_res = float(np.mean(np.abs(valid_residual)))
            max_res = float(np.max(np.abs(valid_residual)))
        else:
            mean_res = float(np.mean(np.abs(residual)))
            max_res = float(np.max(np.abs(residual)))
        
        self.results.append(ResidualResult(
            sample_id=sample_id,
            mean_residual=mean_res,
            max_residual=max_res,
            residual_vs_time=np.sqrt(np.sum(residual ** 2, axis=1)),
        ))
    
    def compute_stats(self) -> Dict:
        """Compute summary statistics."""
        if len(self.results) == 0:
            return {"error": "No results"}
        
        mean_residuals = np.array([r.mean_residual for r in self.results])
        max_residuals = np.array([r.max_residual for r in self.results])
        
        return {
            "n_samples": len(self.results),
            "mean_residual": {
                "mean": float(np.mean(mean_residuals)),
                "median": float(np.median(mean_residuals)),
                "p95": float(np.percentile(mean_residuals, 95)),
                "max": float(np.max(mean_residuals)),
            },
            "max_residual": {
                "mean": float(np.mean(max_residuals)),
                "median": float(np.median(max_residuals)),
                "p95": float(np.percentile(max_residuals, 95)),
                "max": float(np.max(max_residuals)),
            },
        }
    
    def save(self, path: Path):
        """Save results to JSON."""
        stats = self.compute_stats()
        stats["family"] = self.family
        
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def compute_rhs_linear2(t, y, y_delayed, params):
    """RHS for linear2 family."""
    a, b1, b2 = params["a"], params["b1"], params["b2"]
    return a * y + b1 * y_delayed["tau1"] + b2 * y_delayed["tau2"]


def compute_rhs_hutch(t, y, y_delayed, params):
    """RHS for Hutchinson family."""
    r, K = params["r"], params["K"]
    x_tau = y_delayed["tau"]
    return r * y * (1.0 - x_tau / K)


def compute_rhs_mackey_glass(t, y, y_delayed, params):
    """RHS for Mackey-Glass family."""
    beta, gamma, n = params["beta"], params["gamma"], params.get("n", 10.0)
    x_tau = np.maximum(y_delayed["tau"], 1e-10)
    return beta * x_tau / (1 + x_tau ** n) - gamma * y


def compute_rhs_vdp(t, y, y_delayed, params):
    """
    RHS for Van der Pol oscillator with delayed feedback.
    
    x'(t) = v(t)
    v'(t) = μ*(1 - x²)*v - x + κ*x(t-τ)
    
    State: [x, v]
    """
    mu, kappa = params["mu"], params["kappa"]
    x_val, v_val = y[0], y[1]
    x_tau = y_delayed["tau"][0]
    
    dx = v_val
    dv = mu * (1.0 - x_val**2) * v_val - x_val + kappa * x_tau
    
    return np.array([dx, dv])


def compute_rhs_dist_uniform(t, y, y_delayed, params):
    """
    RHS for distributed delay - uniform kernel (moving average).
    
    x'(t) = r*x(t)*(1 - m(t)/K)
    m'(t) = (x(t) - x(t-τ))/τ
    
    State: [x, m]
    """
    r, K, tau = params["r"], params["K"], params["tau"]
    x_val, m_val = y[0], y[1]
    x_tau = y_delayed["tau"][0]
    
    dx = r * x_val * (1.0 - m_val / K)
    dm = (x_val - x_tau) / tau
    
    return np.array([dx, dm])


def compute_rhs_dist_exp(t, y, y_delayed, params):
    """
    RHS for distributed delay - finite-window exponential kernel.
    
    x'(t) = r*x(t)*(1 - z(t)/K)
    z'(t) = -λ*z(t) + (x(t) - exp(-λτ)*x(t-τ))/C
    
    where C = (1 - exp(-λτ))/λ
    
    State: [x, z]
    """
    r, K, lam, tau = params["r"], params["K"], params["lam"], params["tau"]
    x_val, z_val = y[0], y[1]
    x_tau = y_delayed["tau"][0]
    
    # Normalization constant
    C = (1.0 - np.exp(-lam * tau)) / lam
    
    dx = r * x_val * (1.0 - z_val / K)
    dz = -lam * z_val + (x_val - np.exp(-lam * tau) * x_tau) / C
    
    return np.array([dx, dz])


def evaluate_residuals_for_sample(
    family_name: str,
    t: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    t_hist: np.ndarray,
    params: Dict,
    lags: List[float],
) -> np.ndarray:
    """
    Evaluate RHS at all time points and return residuals.
    
    Args:
        family_name: Name of DDE family
        t: Output time grid (N,)
        y: Solution (N, d_state)
        phi: History (N_hist, d_hist)
        t_hist: History time grid
        params: Parameter dictionary
        lags: List of delay values
        
    Returns:
        RHS values at each time point (N, d_state)
    """
    from scipy.interpolate import interp1d
    
    # Build full trajectory (history + solution)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)
    
    # Match dimensions
    d_state = y.shape[1]
    d_hist = phi.shape[1]
    
    if d_hist < d_state:
        # Pad history with last values
        phi_padded = np.zeros((len(t_hist), d_state))
        phi_padded[:, :d_hist] = phi
        phi_padded[:, d_hist:] = phi[:, -1:]
        phi = phi_padded
    
    # Concatenate time and values, avoiding duplicates at t=0
    # t_hist ends at ~0, t starts at ~0
    if len(t) > 0 and len(t_hist) > 0 and np.abs(t_hist[-1] - t[0]) < 1e-10:
        # Remove duplicate point
        t_full = np.concatenate([t_hist[:-1], t])
        y_full = np.vstack([phi[:-1, :d_state], y])
    else:
        t_full = np.concatenate([t_hist, t])
        y_full = np.vstack([phi[:, :d_state], y])
    
    # Ensure strictly increasing (remove any remaining duplicates)
    _, unique_idx = np.unique(t_full, return_index=True)
    t_full = t_full[unique_idx]
    y_full = y_full[unique_idx]
    
    # Create interpolator
    interp = interp1d(t_full, y_full, axis=0, kind='cubic', fill_value='extrapolate')
    
    # Evaluate RHS at each output time point
    rhs_values = np.zeros_like(y)
    
    for i, ti in enumerate(t):
        yi = y[i]
        
        # Get delayed values
        y_delayed = {}
        if len(lags) == 1:
            y_delayed["tau"] = interp(ti - lags[0])
        else:
            for j, lag in enumerate(lags):
                y_delayed[f"tau{j+1}"] = interp(ti - lag)
        
        # Evaluate RHS based on family
        if family_name in ["linear", "linear2"]:
            rhs_values[i] = compute_rhs_linear2(ti, yi, y_delayed, params)
        elif family_name in ["hutchinson", "hutch"]:
            rhs_values[i] = compute_rhs_hutch(ti, yi, y_delayed, params)
        elif family_name == "mackey_glass":
            rhs_values[i] = compute_rhs_mackey_glass(ti, yi, y_delayed, params)
        elif family_name == "vdp":
            rhs_values[i] = compute_rhs_vdp(ti, yi, y_delayed, params)
        elif family_name == "dist_uniform":
            rhs_values[i] = compute_rhs_dist_uniform(ti, yi, y_delayed, params)
        elif family_name == "dist_exp":
            rhs_values[i] = compute_rhs_dist_exp(ti, yi, y_delayed, params)
        else:
            # For other families, skip detailed RHS
            rhs_values[i] = np.nan
    
    return rhs_values


def run_residual_benchmark(
    data_dir: Path,
    family: str,
    n_samples: int = 200,
) -> ResidualBenchmark:
    """
    Run residual benchmark on generated data.
    """
    from datasets.sharded_dataset import ShardedDDEDataset
    
    benchmark = ResidualBenchmark(family)
    
    try:
        dataset = ShardedDDEDataset(str(data_dir), family, "train", normalize=False)
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return benchmark
    
    n_check = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_check, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        
        t = sample["t"].numpy()
        target = sample["target"].numpy()
        
        # Get history and solution portions
        n_hist = dataset.n_hist
        phi = target[:n_hist]
        y = target[n_hist:]
        t_hist = t[:n_hist]
        t_out = t[n_hist:]
        
        params_vec = sample["params"].numpy()
        param_names = dataset.param_names
        params = {name: params_vec[i] for i, name in enumerate(param_names)}
        
        # Get lags from params
        lags = []
        for name in param_names:
            if "tau" in name.lower():
                lags.append(params[name])
        
        if len(lags) == 0:
            continue
        
        # Evaluate RHS
        rhs_values = evaluate_residuals_for_sample(
            family, t_out, y, phi, t_hist, params, lags
        )
        
        if np.any(np.isnan(rhs_values)):
            continue
        
        benchmark.add_result(
            sample_id=int(idx),
            t=t_out,
            y=y,
            rhs_values=rhs_values,
            lags=lags,
        )
    
    return benchmark
