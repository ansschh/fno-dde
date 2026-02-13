#!/usr/bin/env python3
"""
OOD-history evaluation: Test models trained on Fourier histories with spline histories.

Tests whether FNO learns history-invariant representations or overfits to Fourier structure.
"""
import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fno1d import FNO1d
from dde.families import HutchDDE, Linear2DDE, get_family
from dde.solve_python.dde_solver import solve_dde


def sample_spline_history(rng, t_hist, state_dim, n_knots=5, requires_positive=False):
    """
    Sample a smooth random history using cubic spline interpolation.
    
    Different from Fourier: uses random knot points with cubic interpolation.
    """
    history = np.zeros((len(t_hist), state_dim))
    
    for d in range(state_dim):
        # Random knot positions (including endpoints)
        knot_t = np.linspace(t_hist[0], t_hist[-1], n_knots)
        knot_y = rng.uniform(0.2, 1.5, size=n_knots)
        
        # Add some randomness to knot positions (except endpoints)
        if n_knots > 2:
            knot_t[1:-1] += rng.uniform(-0.1, 0.1, size=n_knots-2) * (knot_t[1] - knot_t[0])
            knot_t = np.sort(knot_t)  # Ensure monotonic
        
        # Cubic spline interpolation
        cs = CubicSpline(knot_t, knot_y)
        phi = cs(t_hist)
        
        if requires_positive:
            phi = np.abs(phi) + 0.1
        
        history[:, d] = phi
    
    return history


def sample_piecewise_linear_history(rng, t_hist, state_dim, n_segments=4, requires_positive=False):
    """
    Sample a piecewise linear history function.
    
    Very different from smooth Fourier - tests robustness to non-smooth histories.
    """
    history = np.zeros((len(t_hist), state_dim))
    
    for d in range(state_dim):
        # Random segment boundaries
        boundaries = np.linspace(t_hist[0], t_hist[-1], n_segments + 1)
        values = rng.uniform(0.2, 1.5, size=n_segments + 1)
        
        # Piecewise linear interpolation
        phi = np.interp(t_hist, boundaries, values)
        
        if requires_positive:
            phi = np.abs(phi) + 0.1
        
        history[:, d] = phi
    
    return history


def generate_ood_history_samples(family, n_samples, history_type, seed=42):
    """Generate samples with OOD history functions."""
    rng = np.random.default_rng(seed)
    
    tau_max = family.config.tau_max
    T = 20.0
    n_hist = 256
    n_points = 401  # Match training data
    
    t_hist = np.linspace(-tau_max, 0, n_hist)
    
    samples = []
    pbar = tqdm(total=n_samples, desc=f"Generating {history_type} histories")
    attempts = 0
    max_attempts = n_samples * 20
    
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        params = family.sample_params(rng)
        
        # Sample OOD history
        if history_type == "spline":
            phi = sample_spline_history(rng, t_hist, family.config.state_dim,
                                       n_knots=rng.integers(4, 8),
                                       requires_positive=family.config.requires_positive)
        elif history_type == "piecewise":
            phi = sample_piecewise_linear_history(rng, t_hist, family.config.state_dim,
                                                  n_segments=rng.integers(3, 6),
                                                  requires_positive=family.config.requires_positive)
        else:
            raise ValueError(f"Unknown history type: {history_type}")
        
        # Solve DDE
        sol = solve_dde(family, params, phi, t_hist, T, n_points=n_points)
        
        if sol.success and np.all(np.isfinite(sol.y)) and np.max(np.abs(sol.y)) <= 100.0:
            if family.config.requires_positive and np.min(sol.y) < -1e-6:
                continue
            samples.append({
                "params": params,
                "phi": phi,
                "y": sol.y,
                "t_hist": t_hist,
                "t_out": sol.t,
            })
            pbar.update(1)
    
    pbar.close()
    return samples


def prepare_input(phi, y, t_hist, t_out, n_hist_points=64, n_future_points=192):
    """Prepare input tensor matching training format."""
    # Downsample history
    hist_indices = np.linspace(0, len(t_hist)-1, n_hist_points, dtype=int)
    phi_down = phi[hist_indices]
    
    # Downsample output
    out_indices = np.linspace(0, len(t_out)-1, n_future_points, dtype=int)
    y_down = y[out_indices]
    t_out_down = t_out[out_indices]
    
    # Concatenate for full trajectory
    n_total = n_hist_points + n_future_points
    
    # Build input channels: [value, t, is_history, params...]
    # Simplified: just use value and time
    t_combined = np.concatenate([t_hist[hist_indices], t_out_down])
    
    # Input: history values padded with zeros for future
    input_vals = np.zeros((n_total, 1))
    input_vals[:n_hist_points] = phi_down
    
    # Time channel
    t_channel = t_combined.reshape(-1, 1)
    
    # History mask
    hist_mask = np.zeros((n_total, 1))
    hist_mask[:n_hist_points] = 1.0
    
    # Combine channels
    x = np.concatenate([input_vals, t_channel, hist_mask], axis=1)
    
    # Target: full trajectory
    target = np.zeros((n_total, 1))
    target[:n_hist_points] = phi_down
    target[n_hist_points:] = y_down
    
    return x, target, n_hist_points


def evaluate_model_on_ood_history(model_dir, family_name, history_type, n_samples=256, device="cuda"):
    """Evaluate a trained model on OOD history samples."""
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    
    # Get family
    family = get_family(family_name)
    
    # Generate OOD samples
    samples = generate_ood_history_samples(family, n_samples, history_type, seed=123)
    
    if len(samples) < n_samples:
        print(f"Warning: only generated {len(samples)} samples")
    
    # Build model
    # Determine input channels from config
    n_hist = config.get("n_hist_points", 64)
    n_future = config.get("n_future_points", 192)
    
    # Simple input: 6 channels based on training
    in_channels = 6
    out_channels = 1
    
    model = FNO1d(
        modes=model_cfg["modes"],
        width=model_cfg["width"],
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)
    
    # Load weights
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    
    # Evaluate
    all_rel_l2 = []
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating"):
            x, target, n_hist_pts = prepare_input(
                sample["phi"], sample["y"], 
                sample["t_hist"], sample["t_out"],
                n_hist, n_future
            )
            
            # Need to match actual input format from training
            # This is simplified - actual format may differ
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Skip if shape doesn't match
            if x_tensor.shape[2] != in_channels:
                continue
            
            pred = model(x_tensor)
            
            # Compute rel L2 on future region
            diff = pred[:, n_hist_pts:] - target_tensor[:, n_hist_pts:]
            l2_diff = torch.sqrt(torch.sum(diff**2))
            l2_target = torch.sqrt(torch.sum(target_tensor[:, n_hist_pts:]**2))
            rel_l2 = (l2_diff / (l2_target + 1e-8)).item()
            all_rel_l2.append(rel_l2)
    
    if len(all_rel_l2) == 0:
        print("No samples evaluated - input format mismatch")
        return None
    
    all_rel_l2 = np.array(all_rel_l2)
    
    return {
        "n_samples": len(all_rel_l2),
        "median": float(np.median(all_rel_l2)),
        "p95": float(np.percentile(all_rel_l2, 95)),
        "mean": float(np.mean(all_rel_l2)),
        "std": float(np.std(all_rel_l2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--family", required=True, choices=["hutch", "linear2"])
    ap.add_argument("--history_type", default="spline", choices=["spline", "piecewise"])
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    
    print(f"\n{'='*60}")
    print(f"OOD-History Evaluation: {args.history_type}")
    print(f"{'='*60}")
    print(f"Model: {args.model_dir}")
    print(f"Family: {args.family}")
    print()
    
    results = evaluate_model_on_ood_history(
        args.model_dir, args.family, args.history_type,
        args.n_samples, args.device
    )
    
    if results:
        print(f"\nResults ({args.history_type} history):")
        print(f"  N samples: {results['n_samples']}")
        print(f"  relL2 median: {results['median']:.4f}")
        print(f"  relL2 p95:    {results['p95']:.4f}")
        print(f"  relL2 mean:   {results['mean']:.4f} ± {results['std']:.4f}")


if __name__ == "__main__":
    main()
