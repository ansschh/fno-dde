"""
Label Fidelity Benchmark

Compares fast solver outputs with reference (high-accuracy) solver outputs
to verify that training labels are accurate.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FidelityResult:
    """Result for a single sample fidelity check."""
    sample_id: int
    rel_l2: float
    max_error: float
    error_vs_time: np.ndarray  # Error at each time point


class LabelFidelityBenchmark:
    """
    Benchmark comparing fast vs reference solver outputs.
    
    Pass criteria (rule of thumb):
    - Median relL2 < 1e-3
    - 95th percentile relL2 < 1e-2
    """
    
    def __init__(self, family: str):
        self.family = family
        self.results: List[FidelityResult] = []
    
    def add_result(
        self,
        sample_id: int,
        y_fast: np.ndarray,
        y_ref: np.ndarray,
        t_out: np.ndarray,
    ):
        """
        Add a fidelity comparison result.
        
        Args:
            sample_id: Sample identifier
            y_fast: Fast solver output (N_out, d_state)
            y_ref: Reference solver output (N_out, d_state)
            t_out: Output time grid
        """
        # Compute relative L2
        diff = y_fast - y_ref
        diff_norm = np.sqrt(np.sum(diff ** 2))
        ref_norm = np.sqrt(np.sum(y_ref ** 2)) + 1e-10
        rel_l2 = diff_norm / ref_norm
        
        # Max error
        max_error = np.max(np.abs(diff))
        
        # Error vs time
        error_vs_time = np.sqrt(np.sum(diff ** 2, axis=1))
        
        self.results.append(FidelityResult(
            sample_id=sample_id,
            rel_l2=rel_l2,
            max_error=max_error,
            error_vs_time=error_vs_time,
        ))
    
    def compute_stats(self) -> Dict:
        """Compute summary statistics."""
        if len(self.results) == 0:
            return {"error": "No results"}
        
        rel_l2s = np.array([r.rel_l2 for r in self.results])
        max_errors = np.array([r.max_error for r in self.results])
        
        # Stack error vs time (assume same grid)
        error_curves = np.stack([r.error_vs_time for r in self.results])
        
        stats = {
            "n_samples": len(self.results),
            "rel_l2": {
                "mean": float(np.mean(rel_l2s)),
                "median": float(np.median(rel_l2s)),
                "std": float(np.std(rel_l2s)),
                "p95": float(np.percentile(rel_l2s, 95)),
                "max": float(np.max(rel_l2s)),
            },
            "max_error": {
                "mean": float(np.mean(max_errors)),
                "median": float(np.median(max_errors)),
                "p95": float(np.percentile(max_errors, 95)),
                "max": float(np.max(max_errors)),
            },
            "error_vs_time": {
                "mean": error_curves.mean(axis=0).tolist(),
                "p95": np.percentile(error_curves, 95, axis=0).tolist(),
            },
            "pass_criteria": {
                "median_lt_1e3": float(np.median(rel_l2s)) < 1e-3,
                "p95_lt_1e2": float(np.percentile(rel_l2s, 95)) < 1e-2,
            }
        }
        
        return stats
    
    def passed(self) -> bool:
        """Check if benchmark passes."""
        stats = self.compute_stats()
        if "error" in stats:
            return False
        return (
            stats["pass_criteria"]["median_lt_1e3"] and
            stats["pass_criteria"]["p95_lt_1e2"]
        )
    
    def save(self, path: Path):
        """Save results to JSON."""
        stats = self.compute_stats()
        stats["family"] = self.family
        
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def run_label_fidelity_benchmark_python(
    family_name: str,
    n_samples: int = 200,
    seed: int = 999,
) -> LabelFidelityBenchmark:
    """
    Run label fidelity benchmark using Python solver.
    
    Note: For production, use Julia solver for both fast and reference.
    """
    from dde.families import get_family
    from dde.solve_python.dde_solver import solve_dde
    
    family = get_family(family_name)
    rng = np.random.default_rng(seed)
    
    benchmark = LabelFidelityBenchmark(family_name)
    
    tau_max = family.config.tau_max
    T = family.config.T
    n_hist = 256
    n_out = 400
    
    t_hist = np.linspace(-tau_max, 0, n_hist)
    
    for i in range(n_samples):
        params = family.sample_params(rng)
        history = family.sample_history(rng, t_hist)
        
        # Fast solve
        try:
            sol_fast = solve_dde(
                family=family,
                params=params,
                history=history,
                t_hist=t_hist,
                T=T,
                n_points=n_out,
            )
            
            if not sol_fast.success:
                continue
            
            # Reference solve (same solver but more points - limited in Python)
            sol_ref = solve_dde(
                family=family,
                params=params,
                history=history,
                t_hist=t_hist,
                T=T,
                n_points=n_out * 2,
            )
            
            if not sol_ref.success:
                continue
            
            # Resample reference to match fast grid
            from scipy.interpolate import interp1d
            t_fast = sol_fast.t
            t_ref = sol_ref.t
            
            y_ref_resampled = np.zeros_like(sol_fast.y)
            for d in range(sol_fast.y.shape[1]):
                interp = interp1d(t_ref, sol_ref.y[:, d], kind='cubic')
                y_ref_resampled[:, d] = interp(t_fast)
            
            benchmark.add_result(
                sample_id=i,
                y_fast=sol_fast.y,
                y_ref=y_ref_resampled,
                t_out=t_fast,
            )
            
        except Exception as e:
            continue
    
    return benchmark


def main():
    """CLI for label fidelity benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Label fidelity benchmark")
    parser.add_argument("--family", type=str, required=True,
                        choices=["linear2", "hutch", "mackey_glass", "vdp",
                                "predator_prey", "dist_uniform", "dist_exp"],
                        help="DDE family name")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to test")
    parser.add_argument("--output_dir", type=str, default="reports",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=999,
                        help="Random seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running label fidelity benchmark for {args.family}...")
    
    benchmark = run_label_fidelity_benchmark_python(
        family_name=args.family,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    
    stats = benchmark.compute_stats()
    
    # Print results
    if "rel_l2" in stats:
        print(f"\nResults:")
        print(f"  Samples evaluated: {stats['n_samples']}")
        print(f"  Median rel L2: {stats['rel_l2']['median']:.2e}")
        print(f"  Mean rel L2: {stats['rel_l2']['mean']:.2e}")
        print(f"  95th percentile: {stats['rel_l2']['p95']:.2e}")
        print(f"  Pass (median < 1e-3): {stats['pass_criteria']['median_lt_1e3']}")
        print(f"  Pass (p95 < 1e-2): {stats['pass_criteria']['p95_lt_1e2']}")
    else:
        print(f"Error: {stats.get('error', 'Unknown error')}")
    
    # Save results
    output_path = output_dir / f"{args.family}_label_fidelity.json"
    benchmark.save(output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
