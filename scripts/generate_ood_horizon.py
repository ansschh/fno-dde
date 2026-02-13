#!/usr/bin/env python3
"""
Generate OOD-horizon datasets (T=40 instead of T=20).

Tests how well models trained on T=20 generalize to longer time horizons.
The model predicts T=20, then we continue from the prediction to T=40.

For evaluation, we can:
1. Evaluate on [0, T=20] segment (should match ID performance)
2. Evaluate on [0, T=40] segment (tests horizon extrapolation)
3. Evaluate on [T=20, T=40] segment only (pure extrapolation region)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

from dde.families import HutchDDE, Linear2DDE
from dde.solve_python.dde_solver import solve_dde


def generate_hutch_horizon_samples(
    n_samples: int,
    T_long: float = 40.0,
    dt: float = 0.05,
    tau_range: tuple = (0.1, 2.0),
    seed: int = 42,
) -> dict:
    """Generate Hutch samples with extended horizon."""
    rng = np.random.default_rng(seed)
    
    family = HutchDDE()
    tau_max = tau_range[1]
    
    # Time grids
    n_hist = int(tau_max / dt)
    n_out = int(T_long / dt) + 1
    t_hist = np.linspace(-tau_max, 0, n_hist, endpoint=False)
    t_out = np.linspace(0, T_long, n_out)
    
    all_params = []
    all_phi = []
    all_y = []
    all_lags = []
    
    pbar = tqdm(range(n_samples * 2), desc="Generating Hutch T=40")
    failed = 0
    
    for _ in pbar:
        if len(all_params) >= n_samples:
            break
            
        # Sample parameters
        params = family.sample_params(rng)
        r, K, tau = params['r'], params['K'], params['tau']
        
        if tau < tau_range[0] or tau > tau_range[1]:
            params['tau'] = rng.uniform(tau_range[0], tau_range[1])
            tau = params['tau']
        
        # Sample history (returns array)
        phi_vals = family.sample_history(rng, t_hist)
        if phi_vals.ndim == 1:
            phi_vals = phi_vals.reshape(-1, 1)
        
        # Solve DDE using the correct API
        try:
            sol = solve_dde(
                family=family,
                params=params,
                history=phi_vals,
                t_hist=t_hist,
                T=T_long,
                n_points=n_out,
            )
            
            if sol is None or not sol.success:
                failed += 1
                continue
            
            y_vals = sol.y
            if y_vals.ndim == 1:
                y_vals = y_vals.reshape(-1, 1)
            
            # Check for blowup
            if np.any(np.abs(y_vals) > 1000) or np.any(np.isnan(y_vals)):
                failed += 1
                continue
            
            all_params.append([r, K, tau])
            all_phi.append(phi_vals)
            all_y.append(y_vals)
            all_lags.append([tau])
            
        except Exception as e:
            failed += 1
            continue
        
        pbar.set_postfix({"failed": failed, "collected": len(all_params)})
    
    return {
        "params": np.array(all_params) if all_params else np.array([]).reshape(0, 3),
        "phi": np.array(all_phi) if all_phi else np.array([]).reshape(0, n_hist, 1),
        "y": np.array(all_y) if all_y else np.array([]).reshape(0, n_out, 1),
        "lags": np.array(all_lags) if all_lags else np.array([]).reshape(0, 1),
        "t_hist": t_hist,
        "t_out": t_out,
        "n_failed": failed,
    }


def generate_linear2_horizon_samples(
    n_samples: int,
    T_long: float = 40.0,
    dt: float = 0.05,
    tau_range: tuple = (0.1, 2.0),
    seed: int = 42,
) -> dict:
    """Generate Linear2 samples with extended horizon."""
    rng = np.random.default_rng(seed + 1000)  # Different seed from Hutch
    
    family = Linear2DDE()
    tau_max = tau_range[1]
    
    # Time grids
    n_hist = int(tau_max / dt)
    n_out = int(T_long / dt) + 1
    t_hist = np.linspace(-tau_max, 0, n_hist, endpoint=False)
    t_out = np.linspace(0, T_long, n_out)
    
    all_params = []
    all_phi = []
    all_y = []
    all_lags = []
    
    pbar = tqdm(range(n_samples * 3), desc="Generating Linear2 T=40")  # Extra for failures
    failed = 0
    
    for _ in pbar:
        if len(all_params) >= n_samples:
            break
            
        # Sample parameters
        params = family.sample_params(rng)
        a = params['a']
        b1, b2 = params['b1'], params['b2']
        tau1, tau2 = params['tau1'], params['tau2']
        
        # Ensure taus in range
        params['tau1'] = np.clip(tau1, tau_range[0], tau_range[1])
        params['tau2'] = np.clip(tau2, tau_range[0], tau_range[1])
        tau1, tau2 = params['tau1'], params['tau2']
        
        # Sample history (returns array)
        phi_vals = family.sample_history(rng, t_hist)
        if phi_vals.ndim == 1:
            phi_vals = phi_vals.reshape(-1, 1)
        
        # Solve DDE using the correct API
        try:
            sol = solve_dde(
                family=family,
                params=params,
                history=phi_vals,
                t_hist=t_hist,
                T=T_long,
                n_points=n_out,
            )
            
            if sol is None or not sol.success:
                failed += 1
                continue
            
            y_vals = sol.y
            if y_vals.ndim == 1:
                y_vals = y_vals.reshape(-1, 1)
            
            # Check for blowup (Linear2 can be unstable at T=40)
            if np.any(np.abs(y_vals) > 100) or np.any(np.isnan(y_vals)):
                failed += 1
                continue
            
            all_params.append([a, b1, b2, tau1, tau2])
            all_phi.append(phi_vals)
            all_y.append(y_vals)
            all_lags.append([tau1, tau2])
            
        except Exception as e:
            failed += 1
            continue
        
        pbar.set_postfix({"failed": failed, "collected": len(all_params)})
    
    return {
        "params": np.array(all_params) if all_params else np.array([]).reshape(0, 5),
        "phi": np.array(all_phi) if all_phi else np.array([]).reshape(0, n_hist, 1),
        "y": np.array(all_y) if all_y else np.array([]).reshape(0, n_out, 1),
        "lags": np.array(all_lags) if all_lags else np.array([]).reshape(0, 2),
        "t_hist": t_hist,
        "t_out": t_out,
        "n_failed": failed,
    }


def save_dataset(data: dict, output_dir: Path, family: str, split: str):
    """Save dataset to npz format."""
    split_dir = output_dir / family / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as single shard
    shard_path = split_dir / "shard_000.npz"
    np.savez(
        shard_path,
        params=data["params"],
        phi=data["phi"],
        y=data["y"],
        lags=data["lags"],
        t_hist=data["t_hist"],
        t_out=data["t_out"],
    )
    
    print(f"Saved {len(data['params'])} samples to {shard_path}")
    return len(data["params"])


def main():
    parser = argparse.ArgumentParser(description="Generate OOD-horizon datasets")
    parser.add_argument("--families", nargs="+", default=["hutch", "linear2"])
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--T_long", type=float, default=40.0)
    parser.add_argument("--output_dir", default="data_ood_horizon")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Generating OOD-horizon datasets (T={args.T_long})")
    print("=" * 70)
    
    for family in args.families:
        print(f"\n{'='*50}")
        print(f"Family: {family}")
        print(f"{'='*50}")
        
        if family == "hutch":
            data = generate_hutch_horizon_samples(
                n_samples=args.n_samples,
                T_long=args.T_long,
                seed=args.seed,
            )
        elif family == "linear2":
            data = generate_linear2_horizon_samples(
                n_samples=args.n_samples,
                T_long=args.T_long,
                seed=args.seed,
            )
        else:
            print(f"Unknown family: {family}")
            continue
        
        n_saved = save_dataset(data, output_dir, family, "test_horizon")
        
        # Create manifest
        manifest = {
            "family": family,
            "n_samples": {"test_horizon": n_saved},
            "T": args.T_long,
            "T_train": 20.0,
            "dt": 0.05,
            "tau_max": 2.0,
            "generated_at": datetime.now().isoformat(),
            "seed": args.seed,
            "state_dim": 1,
            "description": f"OOD-horizon test set with T={args.T_long} (training used T=20)",
        }
        
        manifest_path = output_dir / family / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {manifest_path}")
    
    print("\n" + "=" * 70)
    print("OOD-horizon dataset generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
