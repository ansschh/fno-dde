#!/usr/bin/env python3
"""
Generate OOD-delay datasets for evaluating delay generalization.

For Hutch: train τ∈[0.1,1.3], test OOD τ∈(1.3,2.0]
For Linear2: train max(τ1,τ2)∈[0.1,1.3], test OOD max(τ1,τ2)∈(1.3,2.0]
"""
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dde.families import get_family, DDEFamily, HutchDDE, Linear2DDE
from dde.solve_python.dde_solver import solve_dde


def generate_hutch_ood_dataset(output_dir, n_train=10240, n_test_id=256, n_test_ood=256, seed=42):
    """Generate Hutch OOD-delay dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    # ID range: τ∈[0.1,1.3], OOD range: τ∈(1.3,2.0]
    tau_id_range = (0.1, 1.3)
    tau_ood_range = (1.3, 2.0)
    
    family = HutchDDE()
    tau_max = family.config.tau_max
    T = 20.0
    dt_out = 0.05
    n_hist = 256
    
    def sample_with_tau_range(tau_range):
        """Sample params with tau in specified range."""
        params = family.sample_params(rng)
        params["tau"] = rng.uniform(tau_range[0], tau_range[1])
        return params
    
    def generate_split(n_samples, tau_range, desc):
        samples = []
        pbar = tqdm(total=n_samples, desc=desc)
        attempts = 0
        max_attempts = n_samples * 20
        n_points = int(T / dt_out) + 1
        
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            params = sample_with_tau_range(tau_range)
            
            t_hist = np.linspace(-tau_max, 0, n_hist)
            phi = family.sample_history(rng, t_hist)
            
            sol = solve_dde(family, params, phi, t_hist, T, n_points=n_points)
            
            if sol.success and np.all(np.isfinite(sol.y)) and np.max(np.abs(sol.y)) <= 100.0:
                if family.config.requires_positive and np.min(sol.y) < -1e-6:
                    continue
                samples.append({
                    "params": np.array([params["r"], params["K"], params["tau"]]),
                    "phi": phi,
                    "y": sol.y,
                })
                pbar.update(1)
        
        pbar.close()
        return samples
    
    # Generate splits
    print(f"\n=== Generating Hutch OOD-delay dataset ===")
    print(f"ID tau range: {tau_id_range}")
    print(f"OOD tau range: {tau_ood_range}")
    
    n_val = min(256, n_train // 10)  # 10% for validation
    train_samples = generate_split(n_train, tau_id_range, "Train (ID)")
    val_samples = generate_split(n_val, tau_id_range, "Val (ID)")
    test_id_samples = generate_split(n_test_id, tau_id_range, "Test ID")
    test_ood_samples = generate_split(n_test_ood, tau_ood_range, "Test OOD")
    
    # Time grids
    t_hist_arr = np.linspace(-tau_max, 0, n_hist)
    t_out_arr = np.linspace(0, T, int(T / dt_out) + 1)
    
    # Save as npz files (matching expected format)
    def save_split(samples, split_name):
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        params = np.array([s["params"] for s in samples])
        phi = np.array([s["phi"] for s in samples])
        y = np.array([s["y"] for s in samples])
        # Extract lags (tau) from params - it's the 3rd param for Hutch
        lags = params[:, 2:3]  # Shape (N, 1)
        
        np.savez(
            split_dir / "shard_000.npz",
            params=params,
            phi=phi,
            y=y,
            lags=lags,
            t_hist=t_hist_arr,
            t_out=t_out_arr,
        )
        print(f"Saved {split_name}: {len(samples)} samples")
    
    save_split(train_samples, "train")
    save_split(val_samples, "val")
    save_split(test_id_samples, "test")
    save_split(test_ood_samples, "test_ood")
    
    # Save manifest
    manifest = {
        "family": "hutch",
        "config": {
            "tau_max": tau_max,
            "T": T,
            "dt_out": dt_out,
            "n_hist": n_hist,
            "y_clip": 100.0,
        },
        "param_names": ["r", "K", "tau"],
        "param_ranges": {
            "r": [0.5, 3.0],
            "K": [0.5, 2.0],
            "tau_id": list(tau_id_range),
            "tau_ood": list(tau_ood_range),
        },
        "state_dim": 1,
        "splits": {
            "train": {"n_samples": len(train_samples), "tau_range": list(tau_id_range)},
            "val": {"n_samples": len(val_samples), "tau_range": list(tau_id_range)},
            "test": {"n_samples": len(test_id_samples), "tau_range": list(tau_id_range)},
            "test_ood": {"n_samples": len(test_ood_samples), "tau_range": list(tau_ood_range)},
        },
        "seed": seed,
        "generator": "python_ood",
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDataset saved to: {output_dir}")
    return output_dir


def generate_linear2_ood_dataset(output_dir, n_train=10240, n_test_id=256, n_test_ood=256, seed=42):
    """Generate Linear2 OOD-delay dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    # ID: max(τ1,τ2)∈[0.1,1.3], OOD: max(τ1,τ2)∈(1.3,2.0]
    max_tau_id = 1.3
    max_tau_ood_low = 1.3
    max_tau_ood_high = 2.0
    
    family = Linear2DDE()
    tau_max = family.config.tau_max
    T = 20.0
    dt_out = 0.05
    n_hist = 256
    
    def sample_with_max_tau_range(is_ood=False):
        """Sample params with max(τ1,τ2) in specified range."""
        params = family.sample_params(rng)
        
        if is_ood:
            # Ensure max(τ1,τ2) > 1.3
            max_tau = rng.uniform(max_tau_ood_low, max_tau_ood_high)
            if rng.random() < 0.5:
                params["tau1"] = max_tau
                params["tau2"] = rng.uniform(0.1, max_tau)
            else:
                params["tau2"] = max_tau
                params["tau1"] = rng.uniform(0.1, max_tau)
        else:
            # Ensure max(τ1,τ2) <= 1.3
            params["tau1"] = rng.uniform(0.1, max_tau_id)
            params["tau2"] = rng.uniform(0.1, max_tau_id)
        
        return params
    
    def generate_split(n_samples, is_ood, desc):
        samples = []
        pbar = tqdm(total=n_samples, desc=desc)
        attempts = 0
        max_attempts = n_samples * 20
        n_points = int(T / dt_out) + 1
        
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            params = sample_with_max_tau_range(is_ood)
            
            t_hist = np.linspace(-tau_max, 0, n_hist)
            phi = family.sample_history(rng, t_hist)
            
            sol = solve_dde(family, params, phi, t_hist, T, n_points=n_points)
            
            if sol.success and np.all(np.isfinite(sol.y)) and np.max(np.abs(sol.y)) <= 100.0:
                samples.append({
                    "params": np.array([params["a"], params["b1"], params["b2"], 
                                       params["tau1"], params["tau2"]]),
                    "phi": phi,
                    "y": sol.y,
                })
                pbar.update(1)
        
        pbar.close()
        return samples
    
    # Generate splits
    print(f"\n=== Generating Linear2 OOD-delay dataset ===")
    print(f"ID: max(τ1,τ2) ∈ [0.1, {max_tau_id}]")
    print(f"OOD: max(τ1,τ2) ∈ ({max_tau_ood_low}, {max_tau_ood_high}]")
    
    n_val = min(256, n_train // 10)  # 10% for validation
    train_samples = generate_split(n_train, is_ood=False, desc="Train (ID)")
    val_samples = generate_split(n_val, is_ood=False, desc="Val (ID)")
    test_id_samples = generate_split(n_test_id, is_ood=False, desc="Test ID")
    test_ood_samples = generate_split(n_test_ood, is_ood=True, desc="Test OOD")
    
    # Time grids
    t_hist_arr = np.linspace(-tau_max, 0, n_hist)
    t_out_arr = np.linspace(0, T, int(T / dt_out) + 1)
    
    # Save splits (matching expected format)
    def save_split(samples, split_name):
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        params = np.array([s["params"] for s in samples])
        phi = np.array([s["phi"] for s in samples])
        y = np.array([s["y"] for s in samples])
        # Extract lags (tau1, tau2) from params - they're params 4 and 5 for Linear2
        lags = params[:, 3:5]  # Shape (N, 2)
        
        np.savez(
            split_dir / "shard_000.npz",
            params=params,
            phi=phi,
            y=y,
            lags=lags,
            t_hist=t_hist_arr,
            t_out=t_out_arr,
        )
        print(f"Saved {split_name}: {len(samples)} samples")
    
    save_split(train_samples, "train")
    save_split(val_samples, "val")
    save_split(test_id_samples, "test")
    save_split(test_ood_samples, "test_ood")
    
    # Save manifest
    manifest = {
        "family": "linear2",
        "config": {
            "tau_max": tau_max,
            "T": T,
            "dt_out": dt_out,
            "n_hist": n_hist,
            "y_clip": 100.0,
        },
        "param_names": ["a", "b1", "b2", "tau1", "tau2"],
        "param_ranges": {
            "a": [-1.0, 0.5],
            "b1": [-1.0, 1.0],
            "b2": [-1.0, 1.0],
            "max_tau_id": max_tau_id,
            "max_tau_ood": [max_tau_ood_low, max_tau_ood_high],
        },
        "state_dim": 1,
        "splits": {
            "train": {"n_samples": len(train_samples), "max_tau_range": [0.1, max_tau_id]},
            "val": {"n_samples": len(val_samples), "max_tau_range": [0.1, max_tau_id]},
            "test": {"n_samples": len(test_id_samples), "max_tau_range": [0.1, max_tau_id]},
            "test_ood": {"n_samples": len(test_ood_samples), "max_tau_range": [max_tau_ood_low, max_tau_ood_high]},
        },
        "seed": seed,
        "generator": "python_ood",
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDataset saved to: {output_dir}")
    return output_dir


def main():
    ap = argparse.ArgumentParser(description="Generate OOD-delay datasets")
    ap.add_argument("--family", choices=["hutch", "linear2", "both"], default="both")
    ap.add_argument("--n_train", type=int, default=10240)
    ap.add_argument("--n_test", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_base", default="data_ood_delay")
    args = ap.parse_args()
    
    if args.family in ["hutch", "both"]:
        generate_hutch_ood_dataset(
            output_dir=f"{args.output_base}/hutch",
            n_train=args.n_train,
            n_test_id=args.n_test,
            n_test_ood=args.n_test,
            seed=args.seed,
        )
    
    if args.family in ["linear2", "both"]:
        generate_linear2_ood_dataset(
            output_dir=f"{args.output_base}/linear2",
            n_train=args.n_train,
            n_test_id=args.n_test,
            n_test_ood=args.n_test,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
