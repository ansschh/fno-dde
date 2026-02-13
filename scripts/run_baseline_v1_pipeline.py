#!/usr/bin/env python3
"""
Baseline v1 Pipeline Runner

Runs the complete baseline v1 pipeline:
1. Generate all datasets (ID + OOD splits)
2. Train baseline FNO on ID dataset
3. Evaluate on all OOD benchmarks
4. Generate summary report

Usage:
    # Full pipeline for both families (recommended)
    python scripts/run_baseline_v1_pipeline.py --families hutch linear2 --n_train 50000

    # Quick test run
    python scripts/run_baseline_v1_pipeline.py --families hutch --n_train 10240 --quick

    # Only generate datasets
    python scripts/run_baseline_v1_pipeline.py --families hutch --only_data

    # Only train (assumes data exists)
    python scripts/run_baseline_v1_pipeline.py --families hutch --only_train

    # Only evaluate (assumes model exists)
    python scripts/run_baseline_v1_pipeline.py --families hutch --only_eval --model_dir outputs/baseline_v1/hutch_...
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, desc=None):
    """Run a command and handle errors."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True


def generate_datasets(family: str, n_train: int, n_val: int, n_test: int, seed: int):
    """Generate all baseline v1 datasets for a family."""
    cmd = [
        sys.executable, "scripts/generate_baseline_v1.py",
        "--family", family,
        "--n_train", str(n_train),
        "--n_val", str(n_val),
        "--n_test", str(n_test),
        "--seed", str(seed),
    ]
    return run_cmd(cmd, f"Generating datasets for {family}")


def train_baseline(family: str, data_dir: str, output_dir: str, seed: int, device: str):
    """Train baseline FNO on ID dataset."""
    cmd = [
        sys.executable, "src/train/train_fno_sharded.py",
        "--config", f"configs/train_{family}.yaml",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--device", device,
    ]
    return run_cmd(cmd, f"Training baseline FNO for {family}")


def evaluate_baseline(family: str, model_dir: str, output_dir: str, device: str):
    """Evaluate baseline on all OOD benchmarks."""
    cmd = [
        sys.executable, "scripts/eval_baseline_comprehensive.py",
        "--family", family,
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--device", device,
    ]
    return run_cmd(cmd, f"Evaluating baseline for {family}")


def find_latest_model(output_base: str, family: str) -> str:
    """Find the latest model directory for a family."""
    base = Path(output_base)
    candidates = sorted(base.glob(f"{family}_seed*"))
    if candidates:
        return str(candidates[-1])
    return None


def generate_summary_report(families: list, reports_dir: Path):
    """Generate combined summary report."""
    print(f"\n{'='*60}")
    print("  Generating Summary Report")
    print(f"{'='*60}")
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "families": {},
    }
    
    for family in families:
        family_dir = reports_dir / family
        if not family_dir.exists():
            continue
        
        # Find latest run
        runs = sorted(family_dir.glob("run_*"))
        if not runs:
            continue
        
        latest_run = runs[-1]
        metrics_file = latest_run / "metrics_all.json"
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                summary["families"][family] = json.load(f)
    
    # Save summary
    summary_path = reports_dir / "baseline_v1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate markdown table
    md_lines = ["# Baseline v1 Results", "", "## Summary Table", ""]
    md_lines.append("| Family | Split | N | Median | P95 | Mean±Std |")
    md_lines.append("|--------|-------|---|--------|-----|----------|")
    
    for family, metrics in summary.get("families", {}).items():
        for split, m in metrics.items():
            md_lines.append(
                f"| {family} | {split} | {m['n_samples']} | "
                f"{m['relL2_orig_median']:.4f} | {m['relL2_orig_p95']:.4f} | "
                f"{m['relL2_orig_mean']:.4f}±{m['relL2_orig_std']:.4f} |"
            )
    
    md_lines.extend(["", "## OOD Gaps", ""])
    
    for family, metrics in summary.get("families", {}).items():
        if "id" in metrics:
            id_median = metrics["id"]["relL2_orig_median"]
            md_lines.append(f"### {family}")
            md_lines.append("")
            for split, m in metrics.items():
                if split != "id":
                    gap = m["relL2_orig_median"] / id_median
                    md_lines.append(f"- **{split}**: {gap:.2f}x")
            md_lines.append("")
    
    md_path = reports_dir / "baseline_v1_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"Summary saved to:")
    print(f"  {summary_path}")
    print(f"  {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline v1 pipeline")
    parser.add_argument("--families", nargs="+", default=["hutch", "linear2"],
                        choices=["hutch", "linear2", "vdp"])
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--only_data", action="store_true", help="Only generate data")
    parser.add_argument("--only_train", action="store_true", help="Only train models")
    parser.add_argument("--only_eval", action="store_true", help="Only evaluate")
    parser.add_argument("--model_dir", help="Model directory for evaluation")
    args = parser.parse_args()
    
    if args.quick:
        args.n_train = min(args.n_train, 10240)
        args.n_val = min(args.n_val, 256)
        args.n_test = min(args.n_test, 256)
    
    print("=" * 70)
    print("BASELINE V1 PIPELINE")
    print("=" * 70)
    print(f"Families: {args.families}")
    print(f"N_train: {args.n_train}, N_val: {args.n_val}, N_test: {args.n_test}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    
    reports_dir = Path("reports/baseline_eval")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    for family in args.families:
        print(f"\n{'#'*70}")
        print(f"# Processing family: {family}")
        print(f"{'#'*70}")
        
        # Step 1: Generate datasets
        if not args.only_train and not args.only_eval:
            success = generate_datasets(family, args.n_train, args.n_val, 
                                        args.n_test, args.seed)
            if not success:
                print(f"Dataset generation failed for {family}")
                continue
        
        if args.only_data:
            continue
        
        # Step 2: Train baseline
        # Note: ShardedDDEDataset expects data_dir where manifest is at data_dir/{family}/manifest.json
        if not args.only_eval:
            data_dir = "data_baseline_v1"  # Parent dir, not family-specific
            output_dir = "outputs/baseline_v1"
            
            success = train_baseline(family, data_dir, output_dir, 
                                    args.seed, args.device)
            if not success:
                print(f"Training failed for {family}")
                continue
        
        if args.only_train:
            continue
        
        # Step 3: Evaluate
        if args.model_dir:
            model_dir = args.model_dir
        else:
            model_dir = find_latest_model("outputs/baseline_v1", family)
        
        if model_dir is None:
            print(f"No model found for {family}")
            continue
        
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        eval_output = reports_dir / family / run_id
        
        success = evaluate_baseline(family, model_dir, str(eval_output), args.device)
        if not success:
            print(f"Evaluation failed for {family}")
    
    # Generate summary
    if not args.only_data and not args.only_train:
        generate_summary_report(args.families, reports_dir)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
