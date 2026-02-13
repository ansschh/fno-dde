"""
PDE Data Generation CLI

Generates sharded NPZ datasets for PDE operator learning, following the
same format as the DDE data pipeline. Supports all registered PDE families.

Usage:
    python -m src.datasets.generate_pde burgers --n_train 1000 --n_val 100 --n_test 100
    python -m src.datasets.generate_pde ks --n_train 2000 --output_dir data/pde
    python -m src.datasets.generate_pde helmholtz --n_train 500 --shard_size 32
    python -m src.datasets.generate_pde wave --n_train 1000 --seed 123

Shard NPZ format:
    x_grid: (N_x,) spatial grid
    input_func: (B, N_x, d_in) input function(s)
    solution: (B, N_x, d_out) PDE solution
    params: (B, P) parameter values
    meta_json: str JSON metadata
"""

import numpy as np
import json
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pde.families import get_pde_family, PDEFamily


@dataclass
class PDEShardConfig:
    """Configuration for PDE shard generation."""
    family: str
    n_train: int = 1000
    n_val: int = 100
    n_test: int = 100
    shard_size: int = 64
    seed: int = 42
    output_dir: str = "data/pde"
    sol_clip: float = 100.0
    max_retries: int = 10
    check_spectral: bool = False


def run_pde_qc(
    solution: np.ndarray,
    input_func: np.ndarray,
    x_grid: np.ndarray,
    sol_clip: float = 100.0,
    check_spectral: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Run quality control on a generated PDE sample.

    Checks:
    1. Finite values (no NaN or Inf)
    2. Bounded amplitude (|u| < sol_clip)
    3. Optional: spectral convergence (high-freq energy < threshold)

    Args:
        solution: PDE solution, shape (N_x,) or (N_x, d_out).
        input_func: Input function, shape (N_x,) or (N_x, d_in).
        x_grid: Spatial grid, shape (N_x,).
        sol_clip: Maximum allowed amplitude.
        check_spectral: Whether to check spectral convergence.

    Returns:
        (passed, fail_reasons)
    """
    fail_reasons = []

    # 1. Finite check
    if not np.all(np.isfinite(solution)):
        fail_reasons.append("non_finite_solution")
        return False, fail_reasons

    if not np.all(np.isfinite(input_func)):
        fail_reasons.append("non_finite_input")
        return False, fail_reasons

    # 2. Bounded check
    max_val = np.max(np.abs(solution))
    if max_val > sol_clip:
        fail_reasons.append(f"amplitude_exceeded:{max_val:.2f}")
        return False, fail_reasons

    # 3. Spectral convergence (optional)
    if check_spectral:
        # Check that the solution's high-frequency Fourier energy is small
        # relative to total energy. This catches under-resolved solutions.
        sol_1d = solution if solution.ndim == 1 else solution[:, 0]
        u_hat = np.fft.rfft(sol_1d)
        energy = np.abs(u_hat) ** 2
        total_energy = np.sum(energy)

        if total_energy > 1e-15:
            # Check top 10% of modes
            n_modes = len(u_hat)
            top_start = int(0.9 * n_modes)
            high_freq_energy = np.sum(energy[top_start:])
            ratio = high_freq_energy / total_energy

            if ratio > 0.01:  # More than 1% energy in top 10% modes
                fail_reasons.append(f"spectral_convergence:{ratio:.4f}")
                return False, fail_reasons

    return True, fail_reasons


def generate_pde_sample(
    family: PDEFamily,
    rng: np.random.Generator,
    x_grid: np.ndarray,
) -> Optional[Dict]:
    """
    Generate a single PDE sample.

    Args:
        family: PDE family instance.
        rng: NumPy random generator.
        x_grid: Spatial grid.

    Returns:
        Dictionary with sample data, or None if generation failed.
    """
    config = family.config

    # Sample parameters
    params = family.sample_params(rng)

    # Handle KS special case: domain length varies
    if config.name == "ks":
        L = params["L"]
        x_grid = np.linspace(0, L, config.n_spatial, endpoint=False)

    # Handle wave special case: pass c_contrast to input sampler
    if config.name == "wave":
        c_contrast = params.get("c_contrast", 1.5)
        input_func = family.sample_input_function(rng, x_grid, c_contrast=c_contrast)
    else:
        input_func = family.sample_input_function(rng, x_grid)

    # Solve PDE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = family.solve(input_func, x_grid, params)
    except Exception as e:
        return None

    # Ensure proper shapes
    if solution.ndim == 1:
        solution = solution  # Keep 1D for now, will be reshaped at shard write
    # input_func may already be multi-channel

    # Convert params to array
    param_array = np.array([params[name] for name in config.param_names])

    return {
        "x_grid": x_grid,
        "input_func": input_func,
        "solution": solution,
        "params": param_array,
    }


def write_pde_shard(
    path: str,
    samples: List[Dict],
    meta: Dict,
):
    """
    Write a PDE shard to NPZ file.

    Args:
        path: Output file path.
        samples: List of sample dictionaries.
        meta: Metadata dictionary.
    """
    n = len(samples)

    x_grid = samples[0]["x_grid"]

    # Stack input functions
    inp_list = []
    for s in samples:
        inp = s["input_func"]
        if inp.ndim == 1:
            inp = inp[:, np.newaxis]
        inp_list.append(inp)
    input_func = np.stack(inp_list)  # (B, N_x, d_in)

    # Stack solutions
    sol_list = []
    for s in samples:
        sol = s["solution"]
        if sol.ndim == 1:
            sol = sol[:, np.newaxis]
        sol_list.append(sol)
    solution = np.stack(sol_list)  # (B, N_x, d_out)

    # Stack params
    params = np.stack([s["params"] for s in samples])  # (B, P)

    np.savez(
        path,
        x_grid=x_grid,
        input_func=input_func,
        solution=solution,
        params=params,
        meta_json=json.dumps(meta),
    )


def generate_pde_split(
    config: PDEShardConfig,
    family: PDEFamily,
    split: str,
    n_samples: int,
    base_seed: int,
) -> int:
    """
    Generate one split (train/val/test) for a PDE family.

    Args:
        config: Shard generation configuration.
        family: PDE family instance.
        split: Split name ("train", "val", "test").
        n_samples: Number of samples to generate.
        base_seed: Base random seed for this split.

    Returns:
        Number of samples successfully generated.
    """
    split_dir = Path(config.output_dir) / config.family / split
    split_dir.mkdir(parents=True, exist_ok=True)

    x_grid = family.get_spatial_grid()
    n_shards = (n_samples + config.shard_size - 1) // config.shard_size
    total_generated = 0
    fail_counts: Dict[str, int] = {}

    for shard_id in range(n_shards):
        shard_path = split_dir / f"shard_{shard_id:03d}.npz"

        # Resume: skip existing shards
        if shard_path.exists():
            print(f"  Shard {shard_id} exists, skipping...")
            total_generated += config.shard_size
            continue

        # Determine shard size
        remaining = n_samples - shard_id * config.shard_size
        B = min(config.shard_size, remaining)

        rng = np.random.default_rng(base_seed + shard_id * 1000)
        samples = []
        retries = 0
        max_total_retries = B * config.max_retries

        while len(samples) < B and retries < max_total_retries:
            sample = generate_pde_sample(family, rng, x_grid.copy())

            if sample is None:
                fail_counts["solver_failed"] = fail_counts.get("solver_failed", 0) + 1
                retries += 1
                continue

            # QC
            passed, reasons = run_pde_qc(
                sample["solution"],
                sample["input_func"],
                sample["x_grid"],
                sol_clip=config.sol_clip,
                check_spectral=config.check_spectral,
            )

            if not passed:
                for r in reasons:
                    fail_counts[r] = fail_counts.get(r, 0) + 1
                retries += 1
                continue

            samples.append(sample)
            retries = 0

            # Progress
            if len(samples) % 10 == 0 or len(samples) == B:
                print(f"  Shard {shard_id}: {len(samples)}/{B}", end="\r")

        print(f"  Shard {shard_id}: {len(samples)}/{B} samples generated")

        if len(samples) < B:
            print(f"  WARNING: Only generated {len(samples)}/{B} samples for shard {shard_id}")

        if len(samples) > 0:
            meta = {
                "family": config.family,
                "split": split,
                "shard_id": shard_id,
                "n_samples": len(samples),
                "n_spatial": family.config.n_spatial,
                "domain": list(family.config.domain),
                "T": family.config.T,
                "seed": base_seed + shard_id * 1000,
                "generator": "python_pde",
            }
            write_pde_shard(str(shard_path), samples, meta)
            total_generated += len(samples)

    # Print failure summary
    if fail_counts:
        print(f"  Failures: {fail_counts}")

    return total_generated


def generate_pde_dataset(config: PDEShardConfig):
    """Generate full PDE dataset."""
    family = get_pde_family(config.family)

    print("=" * 60)
    print(f"Generating PDE dataset: {config.family}")
    print("=" * 60)
    print(f"  n_spatial={family.config.n_spatial}, domain={family.config.domain}")
    print(f"  T={family.config.T}, state_dim={family.config.state_dim}")
    print(f"  input_type={family.config.input_type}")
    print(f"  params: {family.config.param_names}")
    print(f"  train={config.n_train}, val={config.n_val}, test={config.n_test}")
    print(f"  shard_size={config.shard_size}, seed={config.seed}")
    print()

    family_dir = Path(config.output_dir) / config.family
    family_dir.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest = {
        "family": config.family,
        "config": {
            "n_spatial": family.config.n_spatial,
            "domain": list(family.config.domain),
            "T": family.config.T,
            "sol_clip": config.sol_clip,
        },
        "param_names": family.config.param_names,
        "param_ranges": {k: list(v) for k, v in family.config.param_ranges.items()},
        "state_dim": family.config.state_dim,
        "input_type": family.config.input_type,
        "splits": {},
        "seed": config.seed,
        "generator": "python_pde",
    }

    splits = [
        ("train", config.n_train, config.seed),
        ("val", config.n_val, config.seed + 100_000),
        ("test", config.n_test, config.seed + 200_000),
    ]

    for split, n, seed in splits:
        print(f"\nGenerating {split} ({n} samples)...")
        n_gen = generate_pde_split(config, family, split, n, seed)
        manifest["splits"][split] = {
            "n_samples": n_gen,
            "n_shards": (n + config.shard_size - 1) // config.shard_size,
        }

    # Write manifest
    manifest_path = family_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("PDE dataset generation complete!")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDE dataset for operator learning"
    )
    parser.add_argument(
        "family",
        type=str,
        choices=["burgers", "ks", "helmholtz", "wave"],
        help="PDE family name",
    )
    parser.add_argument("--output_dir", type=str, default="data/pde",
                        help="Output directory for dataset")
    parser.add_argument("--n_train", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=100,
                        help="Number of validation samples")
    parser.add_argument("--n_test", type=int, default=100,
                        help="Number of test samples")
    parser.add_argument("--shard_size", type=int, default=64,
                        help="Samples per shard")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sol_clip", type=float, default=100.0,
                        help="Maximum allowed solution amplitude")
    parser.add_argument("--check_spectral", action="store_true",
                        help="Enable spectral convergence QC check")

    args = parser.parse_args()

    config = PDEShardConfig(
        family=args.family,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        shard_size=args.shard_size,
        seed=args.seed,
        output_dir=args.output_dir,
        sol_clip=args.sol_clip,
        check_spectral=args.check_spectral,
    )

    generate_pde_dataset(config)


if __name__ == "__main__":
    main()
