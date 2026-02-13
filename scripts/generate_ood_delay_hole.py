#!/usr/bin/env python3
"""
Generate OOD-delay-hole datasets for interpolation benchmark.

This creates datasets where:
- Train: τ ∈ [0.1, 0.9] ∪ [1.1, 2.0] (gap in middle)
- Test (hole): τ ∈ [0.9, 1.1] (the missing band)

This tests whether the model can interpolate in delay space.

Usage:
    python scripts/generate_ood_delay_hole.py --family hutch --n_train 10240 --seed 42
    python scripts/generate_ood_delay_hole.py --family linear2 --n_train 10240 --seed 42
"""
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dde.families import HutchDDE, Linear2DDE, get_family
from dde.solve_python.dde_solver import solve_dde


def sample_delay_with_constraint(rng, tau_min, tau_max, exclude_min=None, exclude_max=None):
    """
    Sample delay uniformly from [tau_min, tau_max] excluding [exclude_min, exclude_max].
    """
    if exclude_min is None or exclude_max is None:
        return rng.uniform(tau_min, tau_max)
    
    # Sample from union of [tau_min, exclude_min] and [exclude_max, tau_max]
    range1 = exclude_min - tau_min
    range2 = tau_max - exclude_max
    total_range = range1 + range2
    
    if total_range <= 0:
        raise ValueError("No valid delay range after exclusion")
    
    u = rng.uniform(0, total_range)
    if u < range1:
        return tau_min + u
    else:
        return exclude_max + (u - range1)


def sample_delay_in_hole(rng, hole_min, hole_max):
    """Sample delay uniformly from the hole region."""
    return rng.uniform(hole_min, hole_max)


def generate_hutch_hole_dataset(args):
    """Generate Hutch OOD-delay-hole dataset."""
    family = HutchDDE()
    tau_max = family.config.tau_max
    T = family.config.T
    dt_out = 0.05
    n_hist = 256
    n_points = int(T / dt_out) + 1
    
    # Hole configuration
    hole_min, hole_max = 0.9, 1.1
    
    output_dir = Path(args.output_dir) / "hutch"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t_hist = np.linspace(-tau_max, 0, n_hist)
    
    def generate_split(n_samples, split_name, delay_mode, seed):
        """Generate a single split."""
        rng = np.random.default_rng(seed)
        samples = []
        attempts = 0
        
        pbar = tqdm(total=n_samples, desc=f"Generating {split_name}")
        while len(samples) < n_samples and attempts < n_samples * 50:
            attempts += 1
            
            # Sample parameters
            r = rng.uniform(0.5, 3.0)
            K = rng.uniform(0.5, 2.0)
            
            # Sample delay based on mode
            if delay_mode == "train":
                tau = sample_delay_with_constraint(rng, 0.1, tau_max, hole_min, hole_max)
            elif delay_mode == "hole":
                tau = sample_delay_in_hole(rng, hole_min, hole_max)
            else:
                tau = rng.uniform(0.1, tau_max)
            
            params = {"r": r, "K": K, "tau": tau}
            phi = family.sample_history(rng, t_hist)
            
            sol = solve_dde(family, params, phi, t_hist, T, n_points=n_points)
            
            if sol.success and np.all(np.isfinite(sol.y)):
                if np.max(np.abs(sol.y)) <= 100.0 and np.min(sol.y) >= -1e-6:
                    samples.append({
                        "params": params,
                        "phi": phi,
                        "y": sol.y,
                        "t": sol.t,
                    })
                    pbar.update(1)
        
        pbar.close()
        return samples
    
    # Generate splits
    print(f"\nGenerating Hutch OOD-delay-hole dataset")
    print(f"  Hole region: τ ∈ [{hole_min}, {hole_max}]")
    print(f"  Train: τ ∈ [0.1, {hole_min}] ∪ [{hole_max}, {tau_max}]")
    
    train_samples = generate_split(args.n_train, "train", "train", args.seed)
    val_samples = generate_split(args.n_val, "val", "train", args.seed + 1)
    test_id_samples = generate_split(args.n_test, "test", "train", args.seed + 2)
    test_hole_samples = generate_split(args.n_test, "test_hole", "hole", args.seed + 3)
    
    # Save datasets
    def save_split(samples, split_name):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        params_arr = np.array([[s["params"]["r"], s["params"]["K"], s["params"]["tau"]] 
                               for s in samples])
        phi_arr = np.array([s["phi"] for s in samples])
        y_arr = np.array([s["y"] for s in samples])
        lags_arr = params_arr[:, 2:3]
        
        np.savez(
            split_dir / "shard_000.npz",
            params=params_arr,
            phi=phi_arr,
            y=y_arr,
            lags=lags_arr,
            t_hist=t_hist,
            t_out=samples[0]["t"],
        )
        return len(samples)
    
    n_train = save_split(train_samples, "train")
    n_val = save_split(val_samples, "val")
    n_test = save_split(test_id_samples, "test")
    n_hole = save_split(test_hole_samples, "test_hole")
    
    # Save manifest
    manifest = {
        "family": "hutch",
        "config": {
            "tau_max": tau_max,
            "T": T,
            "dt_out": dt_out,
            "n_hist": n_hist,
            "y_clip": 100.0,
            "hole_min": hole_min,
            "hole_max": hole_max,
        },
        "param_names": ["r", "K", "tau"],
        "splits": {
            "train": {"n_samples": n_train, "delay_range": f"[0.1,{hole_min}]∪[{hole_max},{tau_max}]"},
            "val": {"n_samples": n_val, "delay_range": f"[0.1,{hole_min}]∪[{hole_max},{tau_max}]"},
            "test": {"n_samples": n_test, "delay_range": f"[0.1,{hole_min}]∪[{hole_max},{tau_max}]"},
            "test_hole": {"n_samples": n_hole, "delay_range": f"[{hole_min},{hole_max}]"},
        },
        "seed": args.seed,
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nSaved to {output_dir}")
    print(f"  train: {n_train}, val: {n_val}, test: {n_test}, test_hole: {n_hole}")


def generate_linear2_hole_dataset(args):
    """Generate Linear2 OOD-delay-hole dataset."""
    family = Linear2DDE()
    tau_max = family.config.tau_max
    T = family.config.T
    dt_out = 0.05
    n_hist = 256
    n_points = int(T / dt_out) + 1
    
    # Hole configuration (based on max delay)
    hole_min, hole_max = 0.9, 1.1
    
    output_dir = Path(args.output_dir) / "linear2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t_hist = np.linspace(-tau_max, 0, n_hist)
    
    def generate_split(n_samples, split_name, delay_mode, seed):
        """Generate a single split."""
        rng = np.random.default_rng(seed)
        samples = []
        attempts = 0
        
        pbar = tqdm(total=n_samples, desc=f"Generating {split_name}")
        while len(samples) < n_samples and attempts < n_samples * 50:
            attempts += 1
            
            # Sample parameters
            a = rng.uniform(-1.0, 1.0)
            b1 = rng.uniform(-1.0, 1.0)
            b2 = rng.uniform(-1.0, 1.0)
            
            # Sample delays based on mode
            if delay_mode == "train":
                # Both delays outside hole
                tau1 = sample_delay_with_constraint(rng, 0.1, tau_max, hole_min, hole_max)
                tau2 = sample_delay_with_constraint(rng, 0.1, tau_max, hole_min, hole_max)
            elif delay_mode == "hole":
                # At least one delay in hole
                if rng.random() < 0.5:
                    tau1 = sample_delay_in_hole(rng, hole_min, hole_max)
                    tau2 = rng.uniform(0.1, tau_max)
                else:
                    tau1 = rng.uniform(0.1, tau_max)
                    tau2 = sample_delay_in_hole(rng, hole_min, hole_max)
            else:
                tau1 = rng.uniform(0.1, tau_max)
                tau2 = rng.uniform(0.1, tau_max)
            
            params = {"a": a, "b1": b1, "b2": b2, "tau1": tau1, "tau2": tau2}
            phi = family.sample_history(rng, t_hist)
            
            sol = solve_dde(family, params, phi, t_hist, T, n_points=n_points)
            
            if sol.success and np.all(np.isfinite(sol.y)):
                if np.max(np.abs(sol.y)) <= 100.0:
                    samples.append({
                        "params": params,
                        "phi": phi,
                        "y": sol.y,
                        "t": sol.t,
                    })
                    pbar.update(1)
        
        pbar.close()
        return samples
    
    print(f"\nGenerating Linear2 OOD-delay-hole dataset")
    print(f"  Hole region: τ ∈ [{hole_min}, {hole_max}]")
    
    train_samples = generate_split(args.n_train, "train", "train", args.seed)
    val_samples = generate_split(args.n_val, "val", "train", args.seed + 1)
    test_id_samples = generate_split(args.n_test, "test", "train", args.seed + 2)
    test_hole_samples = generate_split(args.n_test, "test_hole", "hole", args.seed + 3)
    
    # Save datasets
    def save_split(samples, split_name):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        params_arr = np.array([[s["params"]["a"], s["params"]["b1"], s["params"]["b2"],
                                s["params"]["tau1"], s["params"]["tau2"]] 
                               for s in samples])
        phi_arr = np.array([s["phi"] for s in samples])
        y_arr = np.array([s["y"] for s in samples])
        lags_arr = params_arr[:, 3:5]
        
        np.savez(
            split_dir / "shard_000.npz",
            params=params_arr,
            phi=phi_arr,
            y=y_arr,
            lags=lags_arr,
            t_hist=t_hist,
            t_out=samples[0]["t"],
        )
        return len(samples)
    
    n_train = save_split(train_samples, "train")
    n_val = save_split(val_samples, "val")
    n_test = save_split(test_id_samples, "test")
    n_hole = save_split(test_hole_samples, "test_hole")
    
    # Save manifest
    manifest = {
        "family": "linear2",
        "config": {
            "tau_max": tau_max,
            "T": T,
            "dt_out": dt_out,
            "n_hist": n_hist,
            "y_clip": 100.0,
            "hole_min": hole_min,
            "hole_max": hole_max,
        },
        "param_names": ["a", "b1", "b2", "tau1", "tau2"],
        "splits": {
            "train": {"n_samples": n_train},
            "val": {"n_samples": n_val},
            "test": {"n_samples": n_test},
            "test_hole": {"n_samples": n_hole},
        },
        "seed": args.seed,
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nSaved to {output_dir}")
    print(f"  train: {n_train}, val: {n_val}, test: {n_test}, test_hole: {n_hole}")


def main():
    parser = argparse.ArgumentParser(description="Generate OOD-delay-hole datasets")
    parser.add_argument("--family", required=True, choices=["hutch", "linear2"])
    parser.add_argument("--n_train", type=int, default=10240)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=256)
    parser.add_argument("--output_dir", default="data_ood_delay_hole")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.family == "hutch":
        generate_hutch_hole_dataset(args)
    else:
        generate_linear2_hole_dataset(args)


if __name__ == "__main__":
    main()
