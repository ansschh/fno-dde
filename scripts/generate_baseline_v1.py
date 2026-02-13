#!/usr/bin/env python3
"""
Generate Baseline v1 Datasets

Creates large-scale, reproducible datasets for the baseline FNO evaluation.
This includes:
- ID dataset (full τ range): data_baseline_v1/{family}/
- OOD-delay: data_ood_delay/{family}/
- OOD-delay-hole: data_ood_delay_hole/{family}/
- OOD-history: data_ood_history/{family}/

Usage:
    python scripts/generate_baseline_v1.py --family hutch --n_train 50000
    python scripts/generate_baseline_v1.py --family linear2 --n_train 50000
    python scripts/generate_baseline_v1.py --family hutch --n_train 20000 --quick
"""
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import hashlib
from datetime import datetime
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dde.families import HutchDDE, Linear2DDE, get_family
from dde.solve_python.dde_solver import solve_dde


def get_config_hash(config: dict) -> str:
    """Generate a hash of the config for reproducibility tracking."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def sample_spline_history(rng, t_hist, state_dim, n_knots=5, requires_positive=False):
    """Sample spline-based history (for OOD-history)."""
    history = np.zeros((len(t_hist), state_dim))
    for d in range(state_dim):
        knot_t = np.linspace(t_hist[0], t_hist[-1], n_knots)
        if requires_positive:
            knot_y = rng.uniform(0.2, 1.5, size=n_knots)
        else:
            knot_y = rng.uniform(-1.0, 1.0, size=n_knots)
        cs = CubicSpline(knot_t, knot_y)
        phi = cs(t_hist)
        if requires_positive:
            phi = np.abs(phi) + 0.1
        history[:, d] = phi
    return history


class BaselineDatasetGenerator:
    """Generator for baseline v1 datasets."""
    
    def __init__(self, family_name: str, seed: int = 42):
        self.family_name = family_name
        self.family = get_family(family_name)
        self.seed = seed
        
        # Common config
        self.tau_max = self.family.config.tau_max
        self.T = self.family.config.T
        self.dt_out = 0.05
        self.n_hist = 256
        self.n_points = int(self.T / self.dt_out) + 1
        self.y_clip = 100.0
        
        self.t_hist = np.linspace(-self.tau_max, 0, self.n_hist)
        
    def _sample_params(self, rng, delay_constraint=None):
        """Sample parameters with optional delay constraint."""
        params = self.family.sample_params(rng)
        
        if delay_constraint is not None:
            if self.family_name == "hutch":
                params["tau"] = self._sample_constrained_delay(rng, delay_constraint)
            elif self.family_name == "linear2":
                params["tau1"] = self._sample_constrained_delay(rng, delay_constraint)
                params["tau2"] = self._sample_constrained_delay(rng, delay_constraint)
        
        return params
    
    def _sample_constrained_delay(self, rng, constraint):
        """Sample delay with constraint."""
        if constraint == "full":
            return rng.uniform(0.1, self.tau_max)
        elif constraint == "low":  # [0.1, 1.3] for OOD-delay training
            return rng.uniform(0.1, 1.3)
        elif constraint == "high":  # (1.3, 2.0] for OOD-delay test
            return rng.uniform(1.3, self.tau_max)
        elif constraint == "hole_train":  # [0.1, 0.9] ∪ [1.1, 2.0]
            if rng.random() < 0.4:  # 40% low, 60% high (proportional to range)
                return rng.uniform(0.1, 0.9)
            else:
                return rng.uniform(1.1, self.tau_max)
        elif constraint == "hole_test":  # [0.9, 1.1]
            return rng.uniform(0.9, 1.1)
        else:
            return rng.uniform(0.1, self.tau_max)
    
    def _is_valid_solution(self, sol):
        """Check if solution is valid."""
        if not sol.success:
            return False
        if not np.all(np.isfinite(sol.y)):
            return False
        if np.max(np.abs(sol.y)) > self.y_clip:
            return False
        if self.family.config.requires_positive and np.min(sol.y) < -1e-6:
            return False
        return True
    
    def generate_split(self, n_samples, seed, delay_constraint=None, 
                       history_type="fourier", desc="Generating"):
        """Generate a dataset split."""
        rng = np.random.default_rng(seed)
        samples = []
        attempts = 0
        max_attempts = n_samples * 100
        
        pbar = tqdm(total=n_samples, desc=desc)
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            
            params = self._sample_params(rng, delay_constraint)
            
            if history_type == "fourier":
                phi = self.family.sample_history(rng, self.t_hist)
            elif history_type == "spline":
                phi = sample_spline_history(
                    rng, self.t_hist, self.family.config.state_dim,
                    n_knots=rng.integers(4, 8),
                    requires_positive=self.family.config.requires_positive
                )
            
            sol = solve_dde(self.family, params, phi, self.t_hist, 
                           self.T, n_points=self.n_points)
            
            if self._is_valid_solution(sol):
                samples.append({
                    "params": params,
                    "phi": phi,
                    "y": sol.y,
                    "t": sol.t,
                })
                pbar.update(1)
        
        pbar.close()
        
        if len(samples) < n_samples:
            print(f"Warning: Only generated {len(samples)}/{n_samples} samples")
        
        return samples
    
    def save_split(self, samples, output_dir, split_name, shard_size=512):
        """Save split to sharded npz files."""
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        n_samples = len(samples)
        n_shards = (n_samples + shard_size - 1) // shard_size
        
        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, n_samples)
            shard_samples = samples[start:end]
            
            # Build arrays based on family
            if self.family_name == "hutch":
                params_arr = np.array([
                    [s["params"]["r"], s["params"]["K"], s["params"]["tau"]]
                    for s in shard_samples
                ])
                lags_arr = params_arr[:, 2:3]
            elif self.family_name == "linear2":
                params_arr = np.array([
                    [s["params"]["a"], s["params"]["b1"], s["params"]["b2"],
                     s["params"]["tau1"], s["params"]["tau2"]]
                    for s in shard_samples
                ])
                lags_arr = params_arr[:, 3:5]
            else:
                raise ValueError(f"Unknown family: {self.family_name}")
            
            phi_arr = np.array([s["phi"] for s in shard_samples])
            y_arr = np.array([s["y"] for s in shard_samples])
            
            np.savez(
                split_dir / f"shard_{shard_idx:03d}.npz",
                params=params_arr,
                phi=phi_arr,
                y=y_arr,
                lags=lags_arr,
                t_hist=self.t_hist,
                t_out=shard_samples[0]["t"],
            )
        
        return n_samples, n_shards


def generate_id_dataset(gen: BaselineDatasetGenerator, args):
    """Generate ID dataset (full τ range)."""
    output_dir = Path(f"data_baseline_v1/{gen.family_name}")
    
    print(f"\n{'='*60}")
    print(f"Generating ID dataset: {output_dir}")
    print(f"{'='*60}")
    
    train = gen.generate_split(args.n_train, args.seed, "full", "fourier", "train")
    val = gen.generate_split(args.n_val, args.seed + 1, "full", "fourier", "val")
    test = gen.generate_split(args.n_test, args.seed + 2, "full", "fourier", "test")
    
    n_train, n_shards_train = gen.save_split(train, output_dir, "train")
    n_val, _ = gen.save_split(val, output_dir, "val")
    n_test, _ = gen.save_split(test, output_dir, "test")
    
    manifest = {
        "family": gen.family_name,
        "dataset_type": "baseline_v1_id",
        "config": {
            "tau_max": gen.tau_max,
            "T": gen.T,
            "dt_out": gen.dt_out,
            "n_hist": gen.n_hist,
            "y_clip": gen.y_clip,
        },
        "state_dim": gen.family.config.state_dim,
        "param_names": gen.family.config.param_names,
        "param_ranges": gen.family.config.param_ranges,
        "splits": {
            "train": {"n_samples": n_train, "n_shards": n_shards_train},
            "val": {"n_samples": n_val},
            "test": {"n_samples": n_test},
        },
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved: train={n_train}, val={n_val}, test={n_test}")
    return output_dir


def generate_ood_delay_dataset(gen: BaselineDatasetGenerator, args):
    """Generate OOD-delay dataset (extrapolation)."""
    output_dir = Path(f"data_ood_delay/{gen.family_name}")
    
    print(f"\n{'='*60}")
    print(f"Generating OOD-delay dataset: {output_dir}")
    print(f"Train: τ ∈ [0.1, 1.3], Test OOD: τ ∈ (1.3, 2.0]")
    print(f"{'='*60}")
    
    train = gen.generate_split(args.n_train, args.seed + 100, "low", "fourier", "train")
    val = gen.generate_split(args.n_val, args.seed + 101, "low", "fourier", "val")
    test = gen.generate_split(args.n_test, args.seed + 102, "low", "fourier", "test")
    test_ood = gen.generate_split(args.n_test, args.seed + 103, "high", "fourier", "test_ood")
    
    n_train, n_shards = gen.save_split(train, output_dir, "train")
    n_val, _ = gen.save_split(val, output_dir, "val")
    n_test, _ = gen.save_split(test, output_dir, "test")
    n_ood, _ = gen.save_split(test_ood, output_dir, "test_ood")
    
    manifest = {
        "family": gen.family_name,
        "dataset_type": "ood_delay_extrapolation",
        "config": {
            "tau_max": gen.tau_max,
            "T": gen.T,
            "dt_out": gen.dt_out,
            "n_hist": gen.n_hist,
            "train_tau_range": [0.1, 1.3],
            "test_ood_tau_range": [1.3, 2.0],
        },
        "state_dim": gen.family.config.state_dim,
        "param_names": gen.family.config.param_names,
        "param_ranges": gen.family.config.param_ranges,
        "splits": {
            "train": {"n_samples": n_train, "n_shards": n_shards},
            "val": {"n_samples": n_val},
            "test": {"n_samples": n_test, "tau_range": "[0.1, 1.3]"},
            "test_ood": {"n_samples": n_ood, "tau_range": "(1.3, 2.0]"},
        },
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved: train={n_train}, val={n_val}, test={n_test}, test_ood={n_ood}")
    return output_dir


def generate_ood_delay_hole_dataset(gen: BaselineDatasetGenerator, args):
    """Generate OOD-delay-hole dataset (interpolation)."""
    output_dir = Path(f"data_ood_delay_hole/{gen.family_name}")
    
    print(f"\n{'='*60}")
    print(f"Generating OOD-delay-hole dataset: {output_dir}")
    print(f"Train: τ ∈ [0.1, 0.9] ∪ [1.1, 2.0], Test hole: τ ∈ [0.9, 1.1]")
    print(f"{'='*60}")
    
    train = gen.generate_split(args.n_train, args.seed + 200, "hole_train", "fourier", "train")
    val = gen.generate_split(args.n_val, args.seed + 201, "hole_train", "fourier", "val")
    test = gen.generate_split(args.n_test, args.seed + 202, "hole_train", "fourier", "test")
    test_hole = gen.generate_split(args.n_test, args.seed + 203, "hole_test", "fourier", "test_hole")
    
    n_train, n_shards = gen.save_split(train, output_dir, "train")
    n_val, _ = gen.save_split(val, output_dir, "val")
    n_test, _ = gen.save_split(test, output_dir, "test")
    n_hole, _ = gen.save_split(test_hole, output_dir, "test_hole")
    
    manifest = {
        "family": gen.family_name,
        "dataset_type": "ood_delay_hole_interpolation",
        "config": {
            "tau_max": gen.tau_max,
            "T": gen.T,
            "dt_out": gen.dt_out,
            "n_hist": gen.n_hist,
            "hole_min": 0.9,
            "hole_max": 1.1,
        },
        "state_dim": gen.family.config.state_dim,
        "param_names": gen.family.config.param_names,
        "param_ranges": gen.family.config.param_ranges,
        "splits": {
            "train": {"n_samples": n_train, "n_shards": n_shards, "tau_range": "[0.1,0.9]∪[1.1,2.0]"},
            "val": {"n_samples": n_val},
            "test": {"n_samples": n_test},
            "test_hole": {"n_samples": n_hole, "tau_range": "[0.9, 1.1]"},
        },
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved: train={n_train}, val={n_val}, test={n_test}, test_hole={n_hole}")
    return output_dir


def generate_ood_history_dataset(gen: BaselineDatasetGenerator, args):
    """Generate OOD-history dataset (Fourier → spline)."""
    output_dir = Path(f"data_ood_history/{gen.family_name}")
    
    print(f"\n{'='*60}")
    print(f"Generating OOD-history dataset: {output_dir}")
    print(f"Train: Fourier history, Test OOD: Spline history")
    print(f"{'='*60}")
    
    # Only need test_spline (training uses ID dataset)
    test_spline = gen.generate_split(args.n_test, args.seed + 300, "full", "spline", "test_spline")
    
    n_spline, _ = gen.save_split(test_spline, output_dir, "test_spline")
    
    manifest = {
        "family": gen.family_name,
        "dataset_type": "ood_history",
        "config": {
            "tau_max": gen.tau_max,
            "T": gen.T,
            "dt_out": gen.dt_out,
            "n_hist": gen.n_hist,
            "train_history_type": "fourier",
            "test_history_type": "spline",
        },
        "state_dim": gen.family.config.state_dim,
        "param_names": gen.family.config.param_names,
        "param_ranges": gen.family.config.param_ranges,
        "splits": {
            "test_spline": {"n_samples": n_spline, "history_type": "spline"},
        },
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved: test_spline={n_spline}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate baseline v1 datasets")
    parser.add_argument("--family", required=True, choices=["hutch", "linear2"])
    parser.add_argument("--n_train", type=int, default=50000, help="Training samples")
    parser.add_argument("--n_val", type=int, default=2000, help="Validation samples")
    parser.add_argument("--n_test", type=int, default=2000, help="Test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller sizes")
    parser.add_argument("--only", choices=["id", "ood_delay", "ood_delay_hole", "ood_history"],
                        help="Only generate specific dataset type")
    args = parser.parse_args()
    
    if args.quick:
        args.n_train = min(args.n_train, 10240)
        args.n_val = min(args.n_val, 256)
        args.n_test = min(args.n_test, 256)
    
    print("=" * 70)
    print("Baseline v1 Dataset Generation")
    print("=" * 70)
    print(f"Family: {args.family}")
    print(f"N_train: {args.n_train}, N_val: {args.n_val}, N_test: {args.n_test}")
    print(f"Seed: {args.seed}")
    
    gen = BaselineDatasetGenerator(args.family, args.seed)
    
    generated = []
    
    if args.only is None or args.only == "id":
        generated.append(generate_id_dataset(gen, args))
    
    if args.only is None or args.only == "ood_delay":
        generated.append(generate_ood_delay_dataset(gen, args))
    
    if args.only is None or args.only == "ood_delay_hole":
        generated.append(generate_ood_delay_hole_dataset(gen, args))
    
    if args.only is None or args.only == "ood_history":
        generated.append(generate_ood_history_dataset(gen, args))
    
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
