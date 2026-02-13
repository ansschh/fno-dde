"""
Comprehensive Model Evaluation

Evaluates trained models across multiple test splits with all metrics.
Supports:
- ID test
- OOD delay/params/history tests
- Horizon generalization
- Resolution generalization
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import yaml
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.model_metrics import ModelMetrics, compute_all_metrics
from datasets.sharded_dataset import ShardedDDEDataset
from models.fno1d import create_fno1d
from models.baselines import create_baseline


def evaluate_on_split(
    model: torch.nn.Module,
    data_dir: Path,
    family: str,
    split: str,
    device: torch.device,
    batch_size: int = 32,
    requires_positive: bool = False,
    compute_spectral: bool = False,
) -> Dict:
    """Evaluate model on a single split."""
    try:
        dataset = ShardedDDEDataset(str(data_dir), family, split, normalize=True)
    except FileNotFoundError:
        return {"error": f"Split not found: {split}"}
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    metrics = compute_all_metrics(
        model, loader, device, family,
        requires_positive=requires_positive,
        compute_spectral=compute_spectral,
    )
    
    return metrics.compute_stats()


def evaluate_resolution_generalization(
    model: torch.nn.Module,
    data_dir: Path,
    family: str,
    device: torch.device,
    train_dt: float = 0.05,
    test_dt: float = 0.025,
) -> Dict:
    """
    Test resolution generalization.
    
    Resamples predictions and ground truth to common grid for comparison.
    """
    from scipy.interpolate import interp1d
    
    # Load test data
    try:
        dataset = ShardedDDEDataset(str(data_dir), family, "test", normalize=False)
    except FileNotFoundError:
        return {"error": "Test split not found"}
    
    model.eval()
    
    errors = []
    
    with torch.no_grad():
        for idx in range(min(100, len(dataset))):
            sample = dataset[idx]
            
            inputs = sample["input"].unsqueeze(0).to(device)
            targets = sample["target"].numpy()
            t = sample["t"].numpy()
            n_hist = dataset.n_hist
            
            # Get prediction
            outputs = model(inputs).cpu().numpy()[0]
            
            # Future portion
            t_future = t[n_hist:]
            y_pred = outputs[n_hist:]
            y_true = targets[n_hist:]
            
            # Create finer grid
            t_fine = np.arange(t_future[0], t_future[-1], test_dt)
            
            # Interpolate both to fine grid
            if len(t_future) > 3:
                interp_pred = interp1d(t_future, y_pred, axis=0, kind='cubic')
                interp_true = interp1d(t_future, y_true, axis=0, kind='cubic')
                
                y_pred_fine = interp_pred(t_fine)
                y_true_fine = interp_true(t_fine)
                
                # Compute error on fine grid
                diff = y_pred_fine - y_true_fine
                rel_l2 = np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(y_true_fine**2)) + 1e-10)
                errors.append(rel_l2)
    
    if not errors:
        return {"error": "No samples evaluated"}
    
    errors = np.array(errors)
    
    return {
        "train_dt": train_dt,
        "test_dt": test_dt,
        "n_samples": len(errors),
        "rel_l2": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "std": float(np.std(errors)),
            "p95": float(np.percentile(errors, 95)),
        }
    }


def run_comprehensive_evaluation(
    model_checkpoint: Path,
    data_dir: Path,
    family: str,
    output_dir: Path,
    device: torch.device,
    model_config: Optional[Dict] = None,
):
    """
    Run comprehensive evaluation on all available splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint = torch.load(model_checkpoint, map_location=device)
    
    if model_config is None:
        model_config = checkpoint.get("config", {})
    
    # Get dimensions from test data
    try:
        test_dataset = ShardedDDEDataset(str(data_dir), family, "test")
        sample = test_dataset[0]
        in_channels = sample["input"].shape[-1]
        out_channels = sample["target"].shape[-1]
        requires_positive = test_dataset.manifest.get("requires_positive", False)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {}
    
    # Create model
    model = create_fno1d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=model_config.get("model", {}),
        use_residual=model_config.get("use_residual", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    results = {
        "family": family,
        "model_checkpoint": str(model_checkpoint),
        "timestamp": datetime.now().isoformat(),
        "splits": {},
    }
    
    # Standard splits
    standard_splits = ["val", "test"]
    
    # OOD splits (check if they exist)
    family_dir = data_dir / family
    ood_splits = []
    for split_name in ["test_ood_delay", "test_ood_params", "test_ood_history", "test_horizon"]:
        if (family_dir / split_name).exists():
            ood_splits.append(split_name)
    
    all_splits = standard_splits + ood_splits
    
    print(f"\nEvaluating on {len(all_splits)} splits...")
    
    for split in all_splits:
        print(f"  {split}...", end=" ")
        
        stats = evaluate_on_split(
            model, data_dir, family, split, device,
            requires_positive=requires_positive,
            compute_spectral=(family in ["vdp", "predator_prey"]),
        )
        
        results["splits"][split] = stats
        
        if "error" in stats:
            print(f"Error: {stats['error']}")
        elif "rel_l2" in stats:
            print(f"rel_L2 = {stats['rel_l2']['mean']:.4f}")
        else:
            print("Done")
    
    # Resolution generalization
    print("  resolution...", end=" ")
    res_results = evaluate_resolution_generalization(
        model, data_dir, family, device
    )
    results["resolution_generalization"] = res_results
    
    if "error" in res_results:
        print(f"Error: {res_results['error']}")
    elif "rel_l2" in res_results:
        print(f"rel_L2 = {res_results['rel_l2']['mean']:.4f}")
    else:
        print("Done")
    
    # Save results
    output_path = output_dir / f"{family}_comprehensive_eval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Split':<20} {'Rel L2 (mean)':<15} {'Rel L2 (p95)':<15}")
    print("-"*60)
    
    for split, stats in results["splits"].items():
        if "rel_l2" in stats:
            mean = stats["rel_l2"]["mean"]
            p95 = stats["rel_l2"]["p95"]
            print(f"{split:<20} {mean:<15.4f} {p95:<15.4f}")
        elif "error" in stats:
            print(f"{split:<20} {'ERROR':<15} {'':<15}")
    
    return results


def compare_models(
    checkpoints: List[Path],
    data_dir: Path,
    family: str,
    output_dir: Path,
    device: torch.device,
    names: Optional[List[str]] = None,
):
    """Compare multiple models on the same data."""
    if names is None:
        names = [f"model_{i}" for i in range(len(checkpoints))]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        "family": family,
        "models": {},
    }
    
    for name, ckpt in zip(names, checkpoints):
        print(f"\nEvaluating {name}...")
        
        results = run_comprehensive_evaluation(
            ckpt, data_dir, family, output_dir / name, device
        )
        
        # Extract key metrics
        if results and "splits" in results:
            comparison["models"][name] = {
                "checkpoint": str(ckpt),
                "test_rel_l2": results["splits"].get("test", {}).get("rel_l2", {}),
            }
            
            for split in ["test_ood_delay", "test_ood_params", "test_horizon"]:
                if split in results["splits"]:
                    comparison["models"][name][f"{split}_rel_l2"] = \
                        results["splits"][split].get("rel_l2", {})
    
    # Save comparison
    comp_path = output_dir / f"{family}_model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--family", type=str, required=True,
                        help="DDE family name")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    run_comprehensive_evaluation(
        model_checkpoint=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        family=args.family,
        output_dir=Path(args.output_dir),
        device=device,
    )


if __name__ == "__main__":
    main()
