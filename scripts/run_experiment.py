"""
End-to-End Experiment Runner

Orchestrates the full pipeline:
1. Generate dataset (Julia or Python)
2. Train FNO model
3. Evaluate and generate reports

Usage:
    python scripts/run_experiment.py --family hutch --n_train 1000
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import shutil


def run_command(cmd: list, cwd: str = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full DDE-FNO experiment")
    parser.add_argument("--family", type=str, required=True,
                        help="DDE family name")
    parser.add_argument("--n_train", type=int, default=800,
                        help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_python_solver", action="store_true",
                        help="Use Python solver instead of Julia")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip data generation (use existing data)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training (only evaluate)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = output_dir / f"{args.family}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = vars(args)
    config["timestamp"] = timestamp
    with open(experiment_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nExperiment: {args.family}")
    print(f"Output: {experiment_dir}")
    
    # Step 1: Generate data
    if not args.skip_generation:
        print("\n" + "="*60)
        print("STEP 1: Data Generation")
        print("="*60)
        
        if args.use_python_solver:
            gen_cmd = [
                sys.executable,
                str(src_dir / "datasets" / "generate_python.py"),
                args.family,
                f"--n_train={args.n_train}",
                f"--n_val={args.n_val}",
                f"--n_test={args.n_test}",
                f"--seed={args.seed}",
                f"--output_dir={data_dir}",
            ]
        else:
            gen_cmd = [
                sys.executable,
                str(src_dir / "datasets" / "build_dataset_sharded.py"),
                args.family,
                f"--n_train={args.n_train}",
                f"--n_val={args.n_val}",
                f"--n_test={args.n_test}",
                f"--seed={args.seed}",
                f"--output_dir={data_dir}",
                "--verify",
            ]
        
        if not run_command(gen_cmd):
            print("Data generation failed!")
            sys.exit(1)
    else:
        print("\nSkipping data generation (using existing data)")
    
    # Step 2: Create training config
    train_config = {
        "family": args.family,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": 1e-3,
        "lr_min": 1e-6,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "patience": 20,
        "model": {
            "modes": 16,
            "width": 64,
            "n_layers": 4,
            "activation": "gelu",
            "padding": 8,
        },
        "use_residual": False,
    }
    
    config_path = experiment_dir / "train_config.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(train_config, f)
    
    # Step 3: Train model
    if not args.skip_training:
        print("\n" + "="*60)
        print("STEP 2: Training")
        print("="*60)
        
        train_cmd = [
            sys.executable,
            str(src_dir / "train" / "train_fno_sharded.py"),
            f"--config={config_path}",
            f"--data_dir={data_dir}",
            f"--output_dir={experiment_dir}",
            f"--device={args.device}",
        ]
        
        if not run_command(train_cmd):
            print("Training failed!")
            sys.exit(1)
    else:
        print("\nSkipping training")
    
    # Step 4: Generate visualizations
    print("\n" + "="*60)
    print("STEP 3: Visualization")
    print("="*60)
    
    viz_dir = experiment_dir / "plots"
    viz_cmd = [
        sys.executable,
        str(src_dir / "datasets" / "visualize.py"),
        args.family,
        f"--data_dir={data_dir}",
        f"--output_dir={viz_dir}",
    ]
    
    run_command(viz_cmd)  # Don't fail if viz fails
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {experiment_dir}")
    print(f"  - Training config: train_config.yaml")
    print(f"  - Model checkpoints: best_model.pt, final_model.pt")
    print(f"  - Training history: history.json")
    print(f"  - Test results: test_results.json")
    print(f"  - Visualizations: plots/")


if __name__ == "__main__":
    main()
