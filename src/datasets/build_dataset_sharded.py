"""
Python Wrapper for Sharded Dataset Generation

Calls the Julia generator and handles:
- Julia environment setup
- Batch generation orchestration
- Progress monitoring
- Dataset verification

For direct Julia usage, run:
    julia src/dde/solve_julia/dataset_generator.jl <family> [options]

This Python wrapper provides additional conveniences.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Optional
import argparse
import shutil


def check_julia_available() -> bool:
    """Check if Julia is available on the system."""
    try:
        result = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def setup_julia_environment(julia_dir: Path) -> bool:
    """
    Setup Julia environment with required packages.
    
    Args:
        julia_dir: Path to Julia source directory
        
    Returns:
        True if setup successful
    """
    print("Setting up Julia environment...")
    
    # Check if Project.toml exists
    project_toml = julia_dir / "Project.toml"
    if not project_toml.exists():
        print(f"Error: Project.toml not found at {project_toml}")
        return False
    
    # Instantiate project
    result = subprocess.run(
        ["julia", "--project=" + str(julia_dir), "-e", 
         "using Pkg; Pkg.instantiate()"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error setting up Julia environment:")
        print(result.stderr)
        return False
    
    print("Julia environment ready.")
    return True


def generate_dataset_julia(
    family: str,
    output_dir: str = "data",
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    shard_size: int = 64,
    seed: int = 42,
    T: float = 20.0,
    dt_out: float = 0.05,
    tau_max: float = 2.0,
) -> bool:
    """
    Generate dataset using Julia backend.
    
    Args:
        family: DDE family name
        output_dir: Output directory
        n_train, n_val, n_test: Number of samples per split
        shard_size: Samples per shard file
        seed: Random seed
        T: Solution horizon
        dt_out: Output time step
        tau_max: Maximum delay
        
    Returns:
        True if generation successful
    """
    # Get path to Julia script
    script_dir = Path(__file__).parent.parent / "dde" / "solve_julia"
    generator_script = script_dir / "dataset_generator.jl"
    
    if not generator_script.exists():
        print(f"Error: Generator script not found at {generator_script}")
        return False
    
    # Build command
    cmd = [
        "julia",
        "--project=" + str(script_dir),
        str(generator_script),
        family,
        f"--n_train={n_train}",
        f"--n_val={n_val}",
        f"--n_test={n_test}",
        f"--shard_size={shard_size}",
        f"--seed={seed}",
        f"--T={T}",
        f"--dt_out={dt_out}",
        f"--τmax={tau_max}",
        f"--output_dir={output_dir}",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run Julia generator
    result = subprocess.run(cmd, cwd=str(script_dir))
    
    return result.returncode == 0


def verify_dataset(data_dir: Path, family: str) -> dict:
    """
    Verify generated dataset integrity.
    
    Returns:
        Dictionary with verification results
    """
    import numpy as np
    
    family_dir = data_dir / family
    manifest_path = family_dir / "manifest.json"
    
    results = {
        "valid": True,
        "errors": [],
        "splits": {}
    }
    
    # Check manifest
    if not manifest_path.exists():
        results["valid"] = False
        results["errors"].append(f"Manifest not found: {manifest_path}")
        return results
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Check each split
    for split in ["train", "val", "test"]:
        split_dir = family_dir / split
        if not split_dir.exists():
            results["valid"] = False
            results["errors"].append(f"Split directory not found: {split_dir}")
            continue
        
        shard_files = sorted(split_dir.glob("shard_*.npz"))
        n_shards = len(shard_files)
        n_samples = 0
        
        split_results = {
            "n_shards": n_shards,
            "n_samples": 0,
            "has_nans": False,
            "has_infs": False,
        }
        
        for shard_path in shard_files:
            try:
                data = np.load(shard_path)
                y = data["y"]
                n_samples += y.shape[0]
                
                if np.any(np.isnan(y)):
                    split_results["has_nans"] = True
                if np.any(np.isinf(y)):
                    split_results["has_infs"] = True
                    
            except Exception as e:
                results["errors"].append(f"Error loading {shard_path}: {e}")
        
        split_results["n_samples"] = n_samples
        results["splits"][split] = split_results
        
        if split_results["has_nans"] or split_results["has_infs"]:
            results["valid"] = False
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate DDE dataset (Python wrapper)")
    parser.add_argument("family", type=str,
                        choices=["linear2", "hutch", "mackey_glass", "vdp",
                                "predator_prey", "dist_uniform", "dist_exp"],
                        help="DDE family name")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--n_train", type=int, default=800,
                        help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=100,
                        help="Number of validation samples")
    parser.add_argument("--n_test", type=int, default=100,
                        help="Number of test samples")
    parser.add_argument("--shard_size", type=int, default=64,
                        help="Samples per shard")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--T", type=float, default=20.0,
                        help="Solution horizon")
    parser.add_argument("--dt_out", type=float, default=0.05,
                        help="Output time step")
    parser.add_argument("--tau_max", type=float, default=2.0,
                        help="Maximum delay")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset after generation")
    parser.add_argument("--setup_julia", action="store_true",
                        help="Setup Julia environment before generation")
    
    args = parser.parse_args()
    
    # Check Julia
    if not check_julia_available():
        print("Error: Julia not found. Please install Julia and add it to PATH.")
        print("Download from: https://julialang.org/downloads/")
        sys.exit(1)
    
    # Setup Julia environment if requested
    if args.setup_julia:
        julia_dir = Path(__file__).parent.parent / "dde" / "solve_julia"
        if not setup_julia_environment(julia_dir):
            sys.exit(1)
    
    # Generate dataset
    success = generate_dataset_julia(
        family=args.family,
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        shard_size=args.shard_size,
        seed=args.seed,
        T=args.T,
        dt_out=args.dt_out,
        tau_max=args.tau_max,
    )
    
    if not success:
        print("\nDataset generation failed!")
        sys.exit(1)
    
    # Verify if requested
    if args.verify:
        print("\nVerifying dataset...")
        results = verify_dataset(Path(args.output_dir), args.family)
        
        if results["valid"]:
            print("Dataset verification PASSED")
            for split, info in results["splits"].items():
                print(f"  {split}: {info['n_samples']} samples in {info['n_shards']} shards")
        else:
            print("Dataset verification FAILED")
            for error in results["errors"]:
                print(f"  Error: {error}")
            sys.exit(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
