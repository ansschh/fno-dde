#!/usr/bin/env python3
"""
Generate Baseline-All-5 Datasets

Generates ID and all OOD splits for all 5 DDE families using the locked
baseline protocol. This is the master generation script for the baseline-all-5
benchmark.

Usage:
    python scripts/generate_baseline_all5.py --families vdp dist_uniform dist_exp
    python scripts/generate_baseline_all5.py --families all --splits id ood_history
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
import yaml
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import CubicSpline

from dde.families import get_family, DDEFamily
from dde.solve_python.dde_solver import solve_dde


# Load baseline protocol
PROTOCOL_PATH = Path(__file__).parent.parent / "configs" / "baseline_protocol.yaml"
with open(PROTOCOL_PATH) as f:
    PROTOCOL = yaml.safe_load(f)

# Extract protocol values
TAU_MAX = PROTOCOL["data"]["tau_max"]
T = PROTOCOL["data"]["T"]
T_OOD_HORIZON = PROTOCOL["data"]["T_ood_horizon"]
DT = PROTOCOL["data"]["dt_out"]
N_HIST = PROTOCOL["data"]["n_hist"]
N_OUT = PROTOCOL["data"]["n_out"]

N_TRAIN = PROTOCOL["dataset"]["n_train"]
N_VAL = PROTOCOL["dataset"]["n_val"]
N_TEST = PROTOCOL["dataset"]["n_test"]
N_OOD = PROTOCOL["dataset"]["n_ood"]

OOD_DELAY_THRESHOLD = PROTOCOL["ood_splits"]["ood_delay"]["tau_threshold"]
OOD_HOLE_RANGE = PROTOCOL["ood_splits"]["ood_delay_hole"]["hole_range"]

ALL_FAMILIES = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]


def get_tau_key(family_name: str, params: Dict) -> float:
    """Get the tau value used for OOD-delay splits."""
    if family_name == "linear2":
        return max(params["tau1"], params["tau2"])
    else:
        return params["tau"]


def sample_spline_history(rng: np.random.Generator, t_hist: np.ndarray, 
                          state_dim: int, requires_positive: bool) -> np.ndarray:
    """Sample history using cubic spline (OOD-history)."""
    n_knots = rng.integers(4, 8)
    knot_times = np.sort(rng.uniform(t_hist[0], t_hist[-1], n_knots))
    knot_times = np.concatenate([[t_hist[0]], knot_times, [t_hist[-1]]])
    
    history = np.zeros((len(t_hist), state_dim))
    
    for d in range(state_dim):
        knot_values = rng.uniform(-1, 1, len(knot_times))
        if requires_positive:
            knot_values = np.abs(knot_values) + 0.1
        
        spline = CubicSpline(knot_times, knot_values)
        history[:, d] = spline(t_hist)
        
        if requires_positive:
            history[:, d] = np.abs(history[:, d]) + 0.05
    
    return history


def generate_samples(
    family: DDEFamily,
    n_samples: int,
    t_hist: np.ndarray,
    T_solve: float,
    n_out: int,
    rng: np.random.Generator,
    tau_filter: Optional[callable] = None,
    use_spline_history: bool = False,
    max_attempts_multiplier: int = 3,
    blowup_threshold: float = 1000.0,
    desc: str = "Generating",
) -> Dict:
    """Generate samples for a DDE family."""
    all_params = []
    all_phi = []
    all_y = []
    all_lags = []
    
    failed = 0
    filtered = 0
    
    pbar = tqdm(range(n_samples * max_attempts_multiplier), desc=desc)
    
    for _ in pbar:
        if len(all_params) >= n_samples:
            break
        
        # Sample parameters
        params = family.sample_params(rng)
        
        # Apply tau filter if provided
        if tau_filter is not None:
            tau_key = get_tau_key(family.config.name, params)
            if not tau_filter(tau_key):
                filtered += 1
                continue
        
        # Sample history
        if use_spline_history:
            phi_vals = sample_spline_history(
                rng, t_hist, 
                family.config.state_dim,
                family.config.requires_positive
            )
        else:
            phi_vals = family.sample_history(rng, t_hist)
        
        if phi_vals.ndim == 1:
            phi_vals = phi_vals.reshape(-1, 1)
        
        # Solve DDE
        try:
            sol = solve_dde(
                family=family,
                params=params,
                history=phi_vals,
                t_hist=t_hist,
                T=T_solve,
                n_points=n_out,
            )
            
            if sol is None or not sol.success:
                failed += 1
                continue
            
            y_vals = sol.y
            if y_vals.ndim == 1:
                y_vals = y_vals.reshape(-1, 1)
            
            # Check for blowup
            if np.any(np.abs(y_vals) > blowup_threshold) or np.any(np.isnan(y_vals)):
                failed += 1
                continue
            
            # Check positivity constraint
            if family.config.requires_positive and np.any(y_vals[:, 0] < -1e-6):
                failed += 1
                continue
            
            # Store sample
            param_list = [params[name] for name in family.config.param_names]
            delays = family.get_delays(params)
            
            all_params.append(param_list)
            all_phi.append(phi_vals)
            all_y.append(y_vals)
            all_lags.append(delays)
            
        except Exception as e:
            failed += 1
            continue
        
        pbar.set_postfix({
            "collected": len(all_params), 
            "failed": failed,
            "filtered": filtered
        })
    
    if len(all_params) < n_samples:
        print(f"  WARNING: Only collected {len(all_params)}/{n_samples} samples")
    
    return {
        "params": np.array(all_params) if all_params else np.array([]),
        "phi": np.array(all_phi) if all_phi else np.array([]),
        "y": np.array(all_y) if all_y else np.array([]),
        "lags": np.array(all_lags) if all_lags else np.array([]),
        "t_hist": t_hist,
        "t_out": np.linspace(0, T_solve, n_out),
        "n_failed": failed,
        "n_filtered": filtered,
    }


def save_dataset(data: Dict, output_dir: Path, family_name: str, split: str,
                 family: DDEFamily, extra_metadata: Dict = None):
    """Save dataset to npz format with manifest."""
    split_dir = output_dir / family_name / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = len(data["params"])
    
    # Save shard
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
    print(f"  Saved {n_samples} samples to {shard_path}")
    
    # Create manifest
    manifest = {
        "family": family_name,
        "n_samples": {split: n_samples},
        "T": float(data["t_out"][-1]),
        "dt": DT,
        "tau_max": TAU_MAX,
        "state_dim": family.config.state_dim,
        "param_names": family.config.param_names,
        "param_ranges": {k: list(v) for k, v in family.config.param_ranges.items()},
        "generated_at": datetime.now().isoformat(),
        "n_failed": data["n_failed"],
        "n_filtered": data.get("n_filtered", 0),
    }
    
    if extra_metadata:
        manifest.update(extra_metadata)
    
    manifest_path = output_dir / family_name / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest to {manifest_path}")


def generate_id_dataset(family_name: str, seed: int, output_dir: Path):
    """Generate ID dataset (train/val/test) for a family."""
    print(f"\n{'='*60}")
    print(f"Generating ID dataset for {family_name}")
    print(f"{'='*60}")
    
    family = get_family(family_name)
    rng = np.random.default_rng(seed)
    
    t_hist = np.linspace(-TAU_MAX, 0, N_HIST, endpoint=False)
    
    splits = [
        ("train", N_TRAIN),
        ("val", N_VAL),
        ("test", N_TEST),
    ]
    
    for split_name, n_samples in splits:
        print(f"\n  Generating {split_name} ({n_samples} samples)...")
        
        data = generate_samples(
            family=family,
            n_samples=n_samples,
            t_hist=t_hist,
            T_solve=T,
            n_out=N_OUT,
            rng=rng,
            desc=f"{family_name} {split_name}",
        )
        
        save_dataset(data, output_dir, family_name, split_name, family,
                    extra_metadata={"split_type": "id", "seed": seed})


def generate_ood_delay_dataset(family_name: str, seed: int, output_dir: Path):
    """Generate OOD-delay dataset (large tau subset)."""
    print(f"\n{'='*60}")
    print(f"Generating OOD-delay dataset for {family_name}")
    print(f"{'='*60}")
    
    family = get_family(family_name)
    rng = np.random.default_rng(seed + 100)
    
    t_hist = np.linspace(-TAU_MAX, 0, N_HIST, endpoint=False)
    
    # Filter: only keep samples with tau > threshold
    tau_filter = lambda tau: tau > OOD_DELAY_THRESHOLD
    
    print(f"\n  Generating test_ood ({N_OOD} samples, tau > {OOD_DELAY_THRESHOLD})...")
    
    data = generate_samples(
        family=family,
        n_samples=N_OOD,
        t_hist=t_hist,
        T_solve=T,
        n_out=N_OUT,
        rng=rng,
        tau_filter=tau_filter,
        max_attempts_multiplier=10,  # Need more attempts due to filtering
        desc=f"{family_name} ood_delay",
    )
    
    save_dataset(data, output_dir, family_name, "test_ood", family,
                extra_metadata={
                    "split_type": "ood_delay",
                    "tau_threshold": OOD_DELAY_THRESHOLD,
                    "seed": seed + 100,
                })


def generate_ood_delay_hole_dataset(family_name: str, seed: int, output_dir: Path):
    """Generate OOD-delay-hole dataset (tau in interpolation band)."""
    print(f"\n{'='*60}")
    print(f"Generating OOD-delay-hole dataset for {family_name}")
    print(f"{'='*60}")
    
    family = get_family(family_name)
    rng = np.random.default_rng(seed + 200)
    
    t_hist = np.linspace(-TAU_MAX, 0, N_HIST, endpoint=False)
    
    # Filter: only keep samples with tau in hole range
    hole_low, hole_high = OOD_HOLE_RANGE
    tau_filter = lambda tau: hole_low <= tau <= hole_high
    
    print(f"\n  Generating test_hole ({N_OOD} samples, tau in [{hole_low}, {hole_high}])...")
    
    data = generate_samples(
        family=family,
        n_samples=N_OOD,
        t_hist=t_hist,
        T_solve=T,
        n_out=N_OUT,
        rng=rng,
        tau_filter=tau_filter,
        max_attempts_multiplier=20,  # Need many more attempts due to narrow band
        desc=f"{family_name} ood_hole",
    )
    
    save_dataset(data, output_dir, family_name, "test_hole", family,
                extra_metadata={
                    "split_type": "ood_delay_hole",
                    "hole_range": OOD_HOLE_RANGE,
                    "seed": seed + 200,
                })


def generate_ood_history_dataset(family_name: str, seed: int, output_dir: Path):
    """Generate OOD-history dataset (spline histories)."""
    print(f"\n{'='*60}")
    print(f"Generating OOD-history dataset for {family_name}")
    print(f"{'='*60}")
    
    family = get_family(family_name)
    rng = np.random.default_rng(seed + 300)
    
    t_hist = np.linspace(-TAU_MAX, 0, N_HIST, endpoint=False)
    
    print(f"\n  Generating test_spline ({N_OOD} samples, spline histories)...")
    
    data = generate_samples(
        family=family,
        n_samples=N_OOD,
        t_hist=t_hist,
        T_solve=T,
        n_out=N_OUT,
        rng=rng,
        use_spline_history=True,
        desc=f"{family_name} ood_history",
    )
    
    save_dataset(data, output_dir, family_name, "test_spline", family,
                extra_metadata={
                    "split_type": "ood_history",
                    "history_type": "spline",
                    "seed": seed + 300,
                })


def generate_ood_horizon_dataset(family_name: str, seed: int, output_dir: Path):
    """Generate OOD-horizon dataset (T=40)."""
    print(f"\n{'='*60}")
    print(f"Generating OOD-horizon dataset for {family_name}")
    print(f"{'='*60}")
    
    family = get_family(family_name)
    rng = np.random.default_rng(seed + 400)
    
    t_hist = np.linspace(-TAU_MAX, 0, N_HIST, endpoint=False)
    n_out_long = int(T_OOD_HORIZON / DT) + 1
    
    print(f"\n  Generating test_horizon ({N_OOD} samples, T={T_OOD_HORIZON})...")
    
    data = generate_samples(
        family=family,
        n_samples=N_OOD,
        t_hist=t_hist,
        T_solve=T_OOD_HORIZON,
        n_out=n_out_long,
        rng=rng,
        max_attempts_multiplier=5,  # More failures expected at T=40
        desc=f"{family_name} ood_horizon",
    )
    
    save_dataset(data, output_dir, family_name, "test_horizon", family,
                extra_metadata={
                    "split_type": "ood_horizon",
                    "T_train": T,
                    "seed": seed + 400,
                })


def main():
    parser = argparse.ArgumentParser(description="Generate Baseline-All-5 Datasets")
    parser.add_argument("--families", nargs="+", default=["all"],
                       help="Families to generate (or 'all')")
    parser.add_argument("--splits", nargs="+", 
                       default=["id", "ood_delay", "ood_delay_hole", "ood_history", "ood_horizon"],
                       help="Splits to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_base", default="data_baseline_v1",
                       help="Base output directory for ID data")
    args = parser.parse_args()
    
    # Resolve families
    if "all" in args.families:
        families = ALL_FAMILIES
    else:
        families = args.families
    
    # Output directories
    output_dirs = {
        "id": Path(args.output_base),
        "ood_delay": Path("data_ood_delay"),
        "ood_delay_hole": Path("data_ood_delay_hole"),
        "ood_history": Path("data_ood_history"),
        "ood_horizon": Path("data_ood_horizon"),
    }
    
    print("=" * 70)
    print("BASELINE-ALL-5 DATASET GENERATION")
    print("=" * 70)
    print(f"Families: {families}")
    print(f"Splits: {args.splits}")
    print(f"Seed: {args.seed}")
    print(f"Protocol: {PROTOCOL_PATH}")
    
    # Generate datasets
    for family_name in families:
        print(f"\n{'#'*70}")
        print(f"# FAMILY: {family_name.upper()}")
        print(f"{'#'*70}")
        
        if "id" in args.splits:
            generate_id_dataset(family_name, args.seed, output_dirs["id"])
        
        if "ood_delay" in args.splits:
            generate_ood_delay_dataset(family_name, args.seed, output_dirs["ood_delay"])
        
        if "ood_delay_hole" in args.splits:
            generate_ood_delay_hole_dataset(family_name, args.seed, output_dirs["ood_delay_hole"])
        
        if "ood_history" in args.splits:
            generate_ood_history_dataset(family_name, args.seed, output_dirs["ood_history"])
        
        if "ood_horizon" in args.splits:
            generate_ood_horizon_dataset(family_name, args.seed, output_dirs["ood_horizon"])
    
    # Save generation log
    log_dir = Path("reports/data_gen")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    gen_log = {
        "families": families,
        "splits": args.splits,
        "seed": args.seed,
        "protocol": str(PROTOCOL_PATH),
        "generated_at": datetime.now().isoformat(),
        "data_config": PROTOCOL["data"],
        "dataset_config": PROTOCOL["dataset"],
    }
    
    log_path = log_dir / f"gen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(gen_log, f, indent=2)
    print(f"\nGeneration log saved to {log_path}")
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
