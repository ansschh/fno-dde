"""
Full Benchmark Pipeline

Runs complete benchmarking:
1. Dataset quality benchmarks
2. Train FNO + baselines
3. Model evaluation on all splits
4. Ablation studies

Usage:
    python scripts/run_full_benchmark.py --family hutch --n_train 1000
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_step(name: str, cmd: list, cwd: str = None) -> bool:
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"Step: {name}")
    print('='*60)
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=cwd)
    
    if result.returncode != 0:
        print(f"FAILED: {name}")
        return False
    
    print(f"PASSED: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline")
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="benchmark_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_data_gen", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--use_python_solver", action="store_true")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    scripts_dir = project_root / "scripts"
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        "family": args.family,
        "timestamp": timestamp,
        "config": vars(args),
        "steps": {},
    }
    
    # Step 1: Data generation
    if not args.skip_data_gen:
        if args.use_python_solver:
            gen_cmd = [
                sys.executable,
                str(src_dir / "datasets" / "generate_python.py"),
                args.family,
                f"--n_train={args.n_train}",
                f"--n_val={args.n_val}",
                f"--n_test={args.n_test}",
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
                f"--output_dir={data_dir}",
            ]
        
        results["steps"]["data_generation"] = run_step("Data Generation", gen_cmd)
    else:
        results["steps"]["data_generation"] = "skipped"
    
    # Step 2: Dataset benchmarks
    bench_cmd = [
        sys.executable,
        str(src_dir / "benchmarks" / "run_benchmarks.py"),
        f"--family={args.family}",
        f"--data_dir={data_dir}",
        f"--output_dir={output_dir / 'dataset_benchmarks'}",
        "--run_dataset",
    ]
    results["steps"]["dataset_benchmarks"] = run_step("Dataset Benchmarks", bench_cmd)
    
    # Step 3: Train FNO
    if not args.skip_training:
        # Create training config
        train_config = {
            "family": args.family,
            "batch_size": 32,
            "epochs": args.epochs,
            "lr": 1e-3,
            "lr_min": 1e-6,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "model": {
                "modes": 16,
                "width": 64,
                "n_layers": 4,
                "activation": "gelu",
                "padding": 8,
            },
        }
        
        config_path = output_dir / "train_config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(train_config, f)
        
        train_cmd = [
            sys.executable,
            str(src_dir / "train" / "train_fno_sharded.py"),
            f"--config={config_path}",
            f"--data_dir={data_dir}",
            f"--output_dir={output_dir / 'fno_model'}",
            f"--device={args.device}",
        ]
        results["steps"]["fno_training"] = run_step("FNO Training", train_cmd)
    else:
        results["steps"]["fno_training"] = "skipped"
    
    # Step 4: Model evaluation
    model_checkpoint = output_dir / "fno_model" / "best_model.pt"
    if model_checkpoint.exists():
        eval_cmd = [
            sys.executable,
            str(src_dir / "benchmarks" / "eval_comprehensive.py"),
            f"--checkpoint={model_checkpoint}",
            f"--family={args.family}",
            f"--data_dir={data_dir}",
            f"--output_dir={output_dir / 'evaluation'}",
            f"--device={args.device}",
        ]
        results["steps"]["model_evaluation"] = run_step("Model Evaluation", eval_cmd)
    else:
        results["steps"]["model_evaluation"] = "skipped (no checkpoint)"
    
    # Step 5: Baseline comparison
    baseline_cmd = [
        sys.executable,
        str(src_dir / "benchmarks" / "run_benchmarks.py"),
        f"--family={args.family}",
        f"--data_dir={data_dir}",
        f"--output_dir={output_dir / 'baselines'}",
        "--run_baselines",
    ]
    results["steps"]["baseline_comparison"] = run_step("Baseline Comparison", baseline_cmd)
    
    # Save overall results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for step, status in results["steps"].items():
        status_str = "PASS" if status == True else ("FAIL" if status == False else status)
        print(f"  {step}: {status_str}")
    
    print(f"\nResults saved to: {results_path}")
    
    # Check for failures
    failures = [s for s, v in results["steps"].items() if v == False]
    if failures:
        print(f"\nFailed steps: {failures}")
        sys.exit(1)
    
    print("\nBenchmark pipeline complete!")


if __name__ == "__main__":
    main()
