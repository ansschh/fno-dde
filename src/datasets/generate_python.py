"""
Pure Python Dataset Generator (Fallback)

For users without Julia installed. Uses scipy-based DDE solver.
Slower and less robust than Julia, but works for small datasets.

Note: For production use, the Julia generator is strongly recommended.
"""

import numpy as np
import h5py
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dde.families import get_family, DDEFamily
from dde.solve_python.dde_solver import solve_dde, DDESolution


@dataclass
class ShardConfig:
    """Configuration for shard generation."""
    family: str
    tau_max: float = 2.0
    T: float = 20.0
    dt_out: float = 0.05
    n_hist: int = 256
    n_train: int = 800
    n_val: int = 100
    n_test: int = 100
    shard_size: int = 64
    seed: int = 42
    y_clip: float = 100.0
    max_retries: int = 10
    output_dir: str = "data"


def run_qc(
    y: np.ndarray,
    phi: np.ndarray,
    family: DDEFamily,
    y_clip: float = 100.0,
    cont_tol: float = 1e-4,
) -> Tuple[bool, List[str]]:
    """
    Run quality control on a generated sample.
    
    Returns:
        (passed, fail_reasons)
    """
    fail_reasons = []
    
    # Finite check
    if not np.all(np.isfinite(y)):
        fail_reasons.append("non_finite")
        return False, fail_reasons
    
    # Bounded check
    max_val = np.max(np.abs(y))
    if max_val > y_clip:
        fail_reasons.append(f"amplitude_exceeded:{max_val:.2f}")
        return False, fail_reasons
    
    # Positivity check
    if family.config.requires_positive:
        min_val = np.min(y)
        if min_val < -1e-6:
            fail_reasons.append(f"negative_state:{min_val:.6f}")
            return False, fail_reasons
    
    # Continuity at t=0
    d = min(phi.shape[1], y.shape[1])
    if d > 0:
        cont_err = np.max(np.abs(phi[-1, :d] - y[0, :d]))
        if cont_err > cont_tol:
            fail_reasons.append(f"discontinuity:{cont_err:.6f}")
            return False, fail_reasons
    
    return True, fail_reasons


def generate_sample_python(
    family: DDEFamily,
    rng: np.random.Generator,
    tau_max: float,
    T: float,
    dt_out: float,
    n_hist: int,
) -> Optional[Dict]:
    """
    Generate a single sample using the Python solver.
    
    Returns:
        Dictionary with sample data, or None if generation failed
    """
    config = family.config
    
    # Time grids
    t_hist = np.linspace(-tau_max, 0, n_hist)
    t_out = np.arange(0, T + dt_out, dt_out)
    
    # Sample parameters
    params = family.sample_params(rng)
    
    # Sample history
    history = family.sample_history(rng, t_hist)
    
    # Get delays
    lags = family.get_delays(params)
    
    # Solve
    try:
        sol = solve_dde(
            family=family,
            params=params,
            history=history,
            t_hist=t_hist,
            T=T,
            n_points=len(t_out),
        )
        
        if not sol.success:
            return None
        
        y = sol.y
        
    except Exception as e:
        return None
    
    # Convert params to array
    param_array = np.array([params[name] for name in config.param_names])
    lag_array = np.array(lags)
    
    return {
        "phi": history,
        "y": y,
        "params": param_array,
        "lags": lag_array,
        "t_hist": t_hist,
        "t_out": t_out,
    }


def write_shard_npz(
    path: str,
    samples: List[Dict],
    meta: Dict,
):
    """Write a shard to NPZ file."""
    n = len(samples)
    
    t_hist = samples[0]["t_hist"]
    t_out = samples[0]["t_out"]
    
    phi = np.stack([s["phi"] for s in samples])
    y = np.stack([s["y"] for s in samples])
    params = np.stack([s["params"] for s in samples])
    lags = np.stack([s["lags"] for s in samples])
    
    np.savez(
        path,
        t_hist=t_hist,
        t_out=t_out,
        phi=phi,
        y=y,
        params=params,
        lags=lags,
        attempts=np.ones(n, dtype=np.int32),
        meta_json=json.dumps(meta),
    )


def generate_split(
    config: ShardConfig,
    split: str,
    n_samples: int,
    base_seed: int,
) -> int:
    """Generate one split (train/val/test)."""
    family = get_family(config.family)
    
    split_dir = Path(config.output_dir) / config.family / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    n_shards = (n_samples + config.shard_size - 1) // config.shard_size
    total_generated = 0
    
    fail_counts: Dict[str, int] = {}
    
    for shard_id in range(n_shards):
        shard_path = split_dir / f"shard_{shard_id:03d}.npz"
        
        # Resume: skip existing
        if shard_path.exists():
            print(f"  Shard {shard_id} exists, skipping...")
            total_generated += config.shard_size
            continue
        
        # Determine shard size
        remaining = n_samples - shard_id * config.shard_size
        B = min(config.shard_size, remaining)
        
        rng = np.random.default_rng(base_seed + shard_id * 1000)
        samples = []
        
        pbar = tqdm(total=B, desc=f"  Shard {shard_id}", leave=False)
        retries = 0
        
        while len(samples) < B and retries < B * config.max_retries:
            sample = generate_sample_python(
                family, rng,
                config.tau_max, config.T, config.dt_out, config.n_hist
            )
            
            if sample is None:
                fail_counts["solver_failed"] = fail_counts.get("solver_failed", 0) + 1
                retries += 1
                continue
            
            # QC
            passed, reasons = run_qc(
                sample["y"], sample["phi"], family, config.y_clip
            )
            
            if not passed:
                for r in reasons:
                    fail_counts[r] = fail_counts.get(r, 0) + 1
                retries += 1
                continue
            
            samples.append(sample)
            pbar.update(1)
            retries = 0
        
        pbar.close()
        
        if len(samples) < B:
            print(f"  Warning: Only generated {len(samples)}/{B} samples for shard {shard_id}")
        
        if len(samples) > 0:
            meta = {
                "family": config.family,
                "split": split,
                "shard_id": shard_id,
                "n_samples": len(samples),
                "config": {
                    "tau_max": config.tau_max,
                    "T": config.T,
                    "dt_out": config.dt_out,
                    "n_hist": config.n_hist,
                },
                "seed": base_seed + shard_id * 1000,
                "generator": "python",
            }
            write_shard_npz(str(shard_path), samples, meta)
            total_generated += len(samples)
    
    # Print failure summary
    if fail_counts:
        print(f"  Failures: {fail_counts}")
    
    return total_generated


def generate_dataset_python(config: ShardConfig):
    """Generate full dataset using Python solver."""
    family = get_family(config.family)
    
    print("=" * 60)
    print(f"Generating dataset: {config.family} (Python solver)")
    print("=" * 60)
    print(f"  tau_max={config.tau_max}, T={config.T}, dt_out={config.dt_out}")
    print(f"  train={config.n_train}, val={config.n_val}, test={config.n_test}")
    print()
    
    family_dir = Path(config.output_dir) / config.family
    family_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "family": config.family,
        "config": {
            "tau_max": config.tau_max,
            "T": config.T,
            "dt_out": config.dt_out,
            "n_hist": config.n_hist,
            "y_clip": config.y_clip,
        },
        "param_names": family.config.param_names,
        "param_ranges": {k: list(v) for k, v in family.config.param_ranges.items()},
        "state_dim": family.config.state_dim,
        "splits": {},
        "seed": config.seed,
        "generator": "python",
    }
    
    splits = [
        ("train", config.n_train, config.seed),
        ("val", config.n_val, config.seed + 100_000),
        ("test", config.n_test, config.seed + 200_000),
    ]
    
    for split, n, seed in splits:
        print(f"\nGenerating {split} ({n} samples)...")
        n_gen = generate_split(config, split, n, seed)
        manifest["splits"][split] = {
            "n_samples": n_gen,
            "n_shards": (n + config.shard_size - 1) // config.shard_size,
        }
    
    # Write manifest
    manifest_path = family_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate DDE dataset (Python solver)")
    parser.add_argument("family", type=str,
                        choices=["linear2", "hutch", "mackey_glass", "vdp",
                                "predator_prey", "dist_uniform", "dist_exp"],
                        help="DDE family name")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--shard_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--dt_out", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=2.0)
    
    args = parser.parse_args()
    
    config = ShardConfig(
        family=args.family,
        tau_max=args.tau_max,
        T=args.T,
        dt_out=args.dt_out,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        shard_size=args.shard_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    generate_dataset_python(config)


if __name__ == "__main__":
    main()
