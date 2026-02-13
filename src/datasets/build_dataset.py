"""
Dataset Generation Script

Generates paired (input, output) data for DDE operator learning.
Input: history function + parameters
Output: solution trajectory
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dde import get_family, DDEFamily
from dde.solve_python.dde_solver import solve_dde


def generate_sample(
    family: DDEFamily,
    rng: np.random.Generator,
    n_hist_points: int = 64,
    n_future_points: int = 192,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Generate a single (input, output) sample for a DDE family.
    
    Returns:
        Dictionary with keys:
        - 'history': shape (n_hist_points, state_dim)
        - 'params': shape (n_params,)
        - 'solution': shape (n_future_points, state_dim)
        - 't_hist': shape (n_hist_points,)
        - 't_future': shape (n_future_points,)
    """
    config = family.config
    
    # Sample parameters
    params = family.sample_params(rng)
    
    # Generate history time grid
    t_hist = np.linspace(-config.tau_max, 0, n_hist_points)
    
    # Sample history function
    history = family.sample_history(rng, t_hist)
    
    # Solve DDE
    try:
        sol = solve_dde(
            family=family,
            params=params,
            history=history,
            t_hist=t_hist,
            T=config.T,
            n_points=n_future_points,
        )
        
        if not sol.success:
            return None
        
        # Check for NaN or Inf
        if np.any(~np.isfinite(sol.y)):
            return None
        
        # For positive-required families, check positivity
        if config.requires_positive and np.any(sol.y < 0):
            return None
        
    except Exception as e:
        print(f"Solver error: {e}")
        return None
    
    # Convert params to array
    param_array = np.array([params[name] for name in config.param_names])
    
    return {
        'history': history,
        'params': param_array,
        'solution': sol.y,
        't_hist': t_hist,
        't_future': sol.t,
    }


def build_dataset(
    family_name: str,
    n_samples: int,
    output_dir: Path,
    seed: int = 42,
    n_hist_points: int = 64,
    n_future_points: int = 192,
):
    """
    Build a dataset for a DDE family.
    
    Saves:
    - {family_name}_train.h5
    - {family_name}_val.h5
    - {family_name}_test.h5
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    family = get_family(family_name)
    rng = np.random.default_rng(seed)
    
    # Collect samples
    samples = []
    pbar = tqdm(total=n_samples, desc=f"Generating {family_name}")
    
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(samples) < n_samples and attempts < max_attempts:
        sample = generate_sample(
            family, rng, n_hist_points, n_future_points
        )
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1
    
    pbar.close()
    
    if len(samples) < n_samples:
        print(f"Warning: Only generated {len(samples)}/{n_samples} samples")
    
    # Split into train/val/test (80/10/10)
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    # Save each split
    for split_name, split_samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples),
    ]:
        if len(split_samples) == 0:
            continue
            
        output_file = output_dir / f"{family_name}_{split_name}.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Stack arrays
            histories = np.stack([s['history'] for s in split_samples])
            params = np.stack([s['params'] for s in split_samples])
            solutions = np.stack([s['solution'] for s in split_samples])
            t_hist = split_samples[0]['t_hist']
            t_future = split_samples[0]['t_future']
            
            f.create_dataset('history', data=histories)
            f.create_dataset('params', data=params)
            f.create_dataset('solution', data=solutions)
            f.create_dataset('t_hist', data=t_hist)
            f.create_dataset('t_future', data=t_future)
            
            # Save metadata
            f.attrs['family'] = family_name
            f.attrs['n_samples'] = len(split_samples)
            f.attrs['state_dim'] = family.config.state_dim
            f.attrs['param_names'] = family.config.param_names
            f.attrs['tau_max'] = family.config.tau_max
            f.attrs['T'] = family.config.T
        
        print(f"Saved {len(split_samples)} samples to {output_file}")
    
    # Save config
    config_file = output_dir / f"{family_name}_config.yaml"
    config_dict = {
        'family': family_name,
        'tau_max': family.config.tau_max,
        'T': family.config.T,
        'state_dim': family.config.state_dim,
        'param_names': family.config.param_names,
        'param_ranges': {k: list(v) for k, v in family.config.param_ranges.items()},
        'n_hist_points': n_hist_points,
        'n_future_points': n_future_points,
        'seed': seed,
        'n_samples': len(samples),
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)


def main():
    parser = argparse.ArgumentParser(description='Generate DDE dataset')
    parser.add_argument('--family', type=str, required=True,
                        choices=['linear', 'hutchinson', 'mackey_glass', 
                                'predator_prey', 'distributed_delay'],
                        help='DDE family name')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_hist', type=int, default=64,
                        help='Number of history grid points')
    parser.add_argument('--n_future', type=int, default=192,
                        help='Number of future grid points')
    
    args = parser.parse_args()
    
    build_dataset(
        family_name=args.family,
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        n_hist_points=args.n_hist,
        n_future_points=args.n_future,
    )


if __name__ == '__main__':
    main()
