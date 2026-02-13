"""
Comprehensive Benchmark Runner

Runs all benchmarks:
1. Dataset quality benchmarks (solver health, label fidelity, residuals, diversity)
2. Model performance benchmarks (multiple metrics, OOD tests)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.solver_health import SolverHealthLog, compute_solver_stats, aggregate_health_logs
from benchmarks.label_fidelity import LabelFidelityBenchmark, run_label_fidelity_benchmark_python
from benchmarks.residual_check import ResidualBenchmark, run_residual_benchmark
from benchmarks.diversity_metrics import DiversityMetrics, run_diversity_benchmark
from benchmarks.model_metrics import ModelMetrics, compute_all_metrics
from datasets.sharded_dataset import ShardedDDEDataset, create_sharded_dataloaders


def run_dataset_benchmarks(
    data_dir: Path,
    family: str,
    output_dir: Path,
    n_fidelity_samples: int = 200,
    n_residual_samples: int = 200,
):
    """
    Run all dataset quality benchmarks.
    
    Args:
        data_dir: Path to data directory
        family: DDE family name
        output_dir: Where to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Dataset Benchmarks: {family}")
    print('='*60)
    
    # 1. Solver health (load from generation logs if available)
    print("\n1. Solver Health Stats...")
    health_log_path = data_dir / family / "train" / "failures.jsonl"
    if health_log_path.exists():
        # Parse failure log
        failures = []
        with open(health_log_path, "r") as f:
            for line in f:
                try:
                    failures.append(json.loads(line))
                except:
                    pass
        
        results["solver_health"] = {
            "n_failures_logged": len(failures),
            "failure_reasons": {}
        }
        for f in failures:
            reason = f.get("reason", "unknown")
            results["solver_health"]["failure_reasons"][reason] = \
                results["solver_health"]["failure_reasons"].get(reason, 0) + 1
        
        print(f"  Failures logged: {len(failures)}")
    else:
        print("  No solver health log found")
        results["solver_health"] = {"status": "no_log_found"}
    
    # 2. Label fidelity (using Python solver - limited)
    print("\n2. Label Fidelity (Python solver)...")
    try:
        fidelity = run_label_fidelity_benchmark_python(
            family_name=family,
            n_samples=min(n_fidelity_samples, 50),  # Limited for Python
            seed=999,
        )
        fidelity_stats = fidelity.compute_stats()
        results["label_fidelity"] = fidelity_stats
        
        if "rel_l2" in fidelity_stats:
            print(f"  Median rel L2: {fidelity_stats['rel_l2']['median']:.2e}")
            print(f"  95th percentile: {fidelity_stats['rel_l2']['p95']:.2e}")
            print(f"  Pass: {fidelity_stats['pass_criteria']}")
    except Exception as e:
        print(f"  Error: {e}")
        results["label_fidelity"] = {"error": str(e)}
    
    # 3. Residual check
    print("\n3. Residual Benchmark...")
    try:
        residual = run_residual_benchmark(
            data_dir=data_dir,
            family=family,
            n_samples=n_residual_samples,
        )
        residual_stats = residual.compute_stats()
        results["residual_check"] = residual_stats
        
        if "mean_residual" in residual_stats:
            print(f"  Mean residual: {residual_stats['mean_residual']['mean']:.2e}")
            print(f"  Max residual (p95): {residual_stats['max_residual']['p95']:.2e}")
    except Exception as e:
        print(f"  Error: {e}")
        results["residual_check"] = {"error": str(e)}
    
    # 4. Diversity metrics
    print("\n4. Diversity Metrics...")
    try:
        diversity = run_diversity_benchmark(
            data_dir=data_dir,
            family=family,
            split="train",
            max_samples=500,
        )
        diversity_stats = diversity.compute_stats()
        results["diversity"] = diversity_stats
        
        if "dynamics" in diversity_stats:
            dyn = diversity_stats["dynamics"]
            print(f"  Amplitude range: {dyn['amplitude_range']['mean']:.3f} ± {dyn['amplitude_range']['std']:.3f}")
            print(f"  Oscillations (mean): {dyn['n_oscillations']['mean']:.1f}")
            print(f"  Frac with zero oscillations: {dyn['n_oscillations']['frac_zero']:.2%}")
    except Exception as e:
        print(f"  Error: {e}")
        results["diversity"] = {"error": str(e)}
    
    # Save results
    results_path = output_dir / f"{family}_dataset_benchmarks.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def run_model_benchmarks(
    model: torch.nn.Module,
    data_dir: Path,
    family: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    requires_positive: bool = False,
):
    """
    Run all model performance benchmarks.
    
    Args:
        model: Trained model
        data_dir: Path to data directory
        family: DDE family name
        output_dir: Where to save results
        device: Torch device
        batch_size: Batch size for evaluation
        requires_positive: Whether family requires positive solutions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Model Benchmarks: {family}")
    print('='*60)
    
    model.eval()
    
    # Test splits to evaluate
    splits = ["test"]  # Start with ID test
    
    # Check for OOD splits
    family_dir = data_dir / family
    for split_name in ["test_ood_delay", "test_ood_params", "test_horizon"]:
        if (family_dir / split_name).exists():
            splits.append(split_name)
    
    for split in splits:
        print(f"\n--- Split: {split} ---")
        
        try:
            dataset = ShardedDDEDataset(str(data_dir), family, split)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            )
            
            # Compute metrics
            compute_spectral = family in ["vdp", "predator_prey"]
            metrics = compute_all_metrics(
                model, loader, device, family,
                requires_positive=requires_positive,
                compute_spectral=compute_spectral,
            )
            
            stats = metrics.compute_stats()
            results[split] = stats
            
            # Print summary
            if "rel_l2" in stats:
                rl2 = stats["rel_l2"]
                print(f"  Rel L2: {rl2['mean']:.4f} ± {rl2['std']:.4f} (median: {rl2['median']:.4f})")
                print(f"  95th percentile: {rl2['p95']:.4f}")
            
            if "positivity" in stats:
                pos = stats["positivity"]
                print(f"  Frac samples with negative: {pos['frac_samples_with_negative']:.2%}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[split] = {"error": str(e)}
    
    # Save results
    results_path = output_dir / f"{family}_model_benchmarks.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def run_baseline_comparison(
    data_dir: Path,
    family: str,
    output_dir: Path,
    device: torch.device,
    model_checkpoint: Optional[Path] = None,
):
    """
    Compare FNO against baselines.
    """
    from models.baselines import create_baseline, count_parameters
    from models.fno1d import FNO1d
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Baseline Comparison: {family}")
    print('='*60)
    
    # Load test data
    try:
        dataset = ShardedDDEDataset(str(data_dir), family, "test")
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        sample = dataset[0]
        in_channels = sample["input"].shape[-1]
        out_channels = sample["target"].shape[-1]
        seq_length = sample["input"].shape[0]
        n_hist = dataset.n_hist
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}
    
    # Define baselines to test
    baselines = {
        "naive": {"type": "naive"},
        "tcn_small": {"type": "tcn", "hidden_channels": 32, "n_layers": 4},
        "tcn_large": {"type": "tcn", "hidden_channels": 64, "n_layers": 6},
        "mlp": {"type": "mlp", "hidden_dim": 256, "n_layers": 4},
    }
    
    for name, config in baselines.items():
        print(f"\n--- Baseline: {name} ---")
        
        try:
            model = create_baseline(
                baseline_type=config.pop("type"),
                in_channels=in_channels,
                out_channels=out_channels,
                seq_length=seq_length,
                **config,
            ).to(device)
            
            n_params = count_parameters(model)
            print(f"  Parameters: {n_params:,}")
            
            # For naive baseline, no training needed
            if name == "naive":
                metrics = ModelMetrics(family)
                model.eval()
                
                with torch.no_grad():
                    for batch in loader:
                        inputs = batch["input"].to(device)
                        targets = batch["target"].numpy()
                        t = batch["t"].numpy()
                        
                        outputs = model(inputs, n_hist).cpu().numpy()
                        
                        for i in range(outputs.shape[0]):
                            metrics.add_sample(
                                sample_id=i,
                                y_pred=outputs[i],
                                y_true=targets[i],
                                t=t[i],
                                n_hist=n_hist,
                            )
                
                stats = metrics.compute_stats()
                results[name] = {
                    "n_params": n_params,
                    "rel_l2": stats.get("rel_l2", {}),
                }
                
                if "rel_l2" in stats:
                    print(f"  Rel L2: {stats['rel_l2']['mean']:.4f}")
            else:
                # Would need to train - just report parameter count
                results[name] = {
                    "n_params": n_params,
                    "status": "not_trained",
                }
                print(f"  (Not trained - would need training script)")
                
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {"error": str(e)}
    
    # Save results
    results_path = output_dir / f"{family}_baseline_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run DDE-FNO benchmarks")
    parser.add_argument("--family", type=str, required=True,
                        help="DDE family name")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Output directory")
    parser.add_argument("--model_checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--run_dataset", action="store_true",
                        help="Run dataset benchmarks")
    parser.add_argument("--run_model", action="store_true",
                        help="Run model benchmarks (requires checkpoint)")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Run baseline comparison")
    parser.add_argument("--all", action="store_true",
                        help="Run all benchmarks")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.all:
        args.run_dataset = True
        args.run_baselines = True
        if args.model_checkpoint:
            args.run_model = True
    
    # Dataset benchmarks
    if args.run_dataset:
        run_dataset_benchmarks(
            data_dir=data_dir,
            family=args.family,
            output_dir=output_dir,
        )
    
    # Model benchmarks
    if args.run_model and args.model_checkpoint:
        from models.fno1d import create_fno1d
        
        # Load model
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        config = checkpoint.get("config", {})
        
        # Get dimensions from dataset
        dataset = ShardedDDEDataset(str(data_dir), args.family, "test")
        sample = dataset[0]
        in_channels = sample["input"].shape[-1]
        out_channels = sample["target"].shape[-1]
        
        model = create_fno1d(
            in_channels=in_channels,
            out_channels=out_channels,
            config=config.get("model", {}),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        
        run_model_benchmarks(
            model=model,
            data_dir=data_dir,
            family=args.family,
            output_dir=output_dir,
            device=device,
        )
    
    # Baseline comparison
    if args.run_baselines:
        run_baseline_comparison(
            data_dir=data_dir,
            family=args.family,
            output_dir=output_dir,
            device=device,
        )
    
    print("\nBenchmarks complete!")


if __name__ == "__main__":
    main()
