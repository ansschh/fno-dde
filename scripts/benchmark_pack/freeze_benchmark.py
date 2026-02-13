#!/usr/bin/env python3
"""
Part F: Freeze Benchmark Pack

Generate comprehensive freeze manifest that includes:
- Dataset paths + hashes
- Model run IDs  
- Evaluation report paths
- Configs used
- Git state

This guarantees apples-to-apples comparison for future architectures.
"""
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
import shutil


def get_git_info():
    """Get current git state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        return {
            "commit": commit,
            "branch": branch,
            "dirty": len(status) > 0,
            "uncommitted_changes": status.split("\n") if status else [],
        }
    except Exception as e:
        return {"error": str(e)}


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def hash_directory(dir_path: Path) -> dict:
    """Compute hashes for all relevant files in a directory."""
    hashes = {}
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.suffix in [".npz", ".json", ".yaml", ".pt"]:
            rel_path = path.relative_to(dir_path)
            hashes[str(rel_path)] = hash_file(path)
    return hashes


def collect_dataset_info(data_dirs: list, families: list) -> dict:
    """Collect dataset information and hashes."""
    datasets = {}
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            continue
        
        dir_name = data_path.name
        datasets[dir_name] = {
            "path": str(data_path),
            "families": {},
        }
        
        for family in families:
            family_path = data_path / family
            if family_path.exists():
                manifest_path = family_path / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    
                    datasets[dir_name]["families"][family] = {
                        "n_samples": sum(s["n_samples"] for s in manifest.get("shards", [])),
                        "shards": len(manifest.get("shards", [])),
                        "manifest_hash": hash_file(manifest_path),
                    }
    
    return datasets


def collect_model_info(baseline_dir: Path, families: list) -> dict:
    """Collect model checkpoint information."""
    models = {}
    
    for family in families:
        # Find model directories for this family
        family_dirs = list(baseline_dir.glob(f"{family}_seed*"))
        
        for model_dir in family_dirs:
            checkpoint = model_dir / "best_model.pt"
            if checkpoint.exists():
                run_id = model_dir.name
                models[run_id] = {
                    "family": family,
                    "path": str(model_dir),
                    "checkpoint_hash": hash_file(checkpoint),
                    "has_config": (model_dir / "config.yaml").exists(),
                    "has_stats": (model_dir / "training_stats.json").exists(),
                }
    
    return models


def collect_evaluation_info(eval_dir: Path, families: list) -> dict:
    """Collect evaluation report information."""
    evaluations = {}
    
    for family in families:
        family_dir = eval_dir / family
        if family_dir.exists():
            evaluations[family] = {
                "path": str(family_dir),
                "reports": [],
            }
            
            for json_file in family_dir.glob("*.json"):
                evaluations[family]["reports"].append({
                    "name": json_file.name,
                    "hash": hash_file(json_file),
                })
    
    return evaluations


def collect_config_info(config_dir: Path) -> dict:
    """Collect configuration file information."""
    configs = {}
    
    for config_file in config_dir.glob("*.yaml"):
        configs[config_file.name] = {
            "path": str(config_file),
            "hash": hash_file(config_file),
        }
    
    return configs


def generate_freeze_manifest(
    output_path: Path,
    tag_name: str = "baseline_all5_benchmarkpack_frozen",
    baseline_dir: Path = Path("outputs/baseline_v1"),
    eval_dir: Path = Path("reports/baseline_eval"),
    quality_dir: Path = Path("reports/data_quality"),
    config_dir: Path = Path("configs"),
):
    """Generate comprehensive freeze manifest."""
    
    print("="*70)
    print("Generating Benchmark Pack Freeze Manifest")
    print("="*70)
    
    families = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]
    
    data_dirs = [
        "data_baseline_v1",
        "data_ood_delay",
        "data_ood_delay_hole", 
        "data_ood_history",
        "data_ood_horizon",
    ]
    
    manifest = {
        "tag": tag_name,
        "frozen_at": datetime.now().isoformat(),
        "description": "Baseline-All-5 Benchmark Pack - Bulletproof baseline for architecture comparison",
        "git": get_git_info(),
        "families": families,
        "datasets": collect_dataset_info(data_dirs, families),
        "models": collect_model_info(baseline_dir, families),
        "evaluations": collect_evaluation_info(eval_dir, families),
        "configs": collect_config_info(config_dir),
    }
    
    # Add quality check info if available
    if quality_dir.exists():
        quality_summary = quality_dir / "quality_summary.json"
        if quality_summary.exists():
            with open(quality_summary) as f:
                manifest["quality_checks"] = json.load(f)
    
    # Summary stats
    manifest["summary"] = {
        "n_families": len(families),
        "n_datasets": sum(
            len(d.get("families", {})) 
            for d in manifest["datasets"].values()
        ),
        "n_models": len(manifest["models"]),
        "n_evaluations": sum(
            len(e.get("reports", []))
            for e in manifest["evaluations"].values()
        ),
    }
    
    # Save manifest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Families: {manifest['summary']['n_families']}")
    print(f"  Dataset splits: {manifest['summary']['n_datasets']}")
    print(f"  Model checkpoints: {manifest['summary']['n_models']}")
    print(f"  Evaluation reports: {manifest['summary']['n_evaluations']}")
    print(f"  Git commit: {manifest['git'].get('commit', 'unknown')[:8]}")
    
    if manifest["git"].get("dirty"):
        print("\n  ⚠️  WARNING: Uncommitted changes detected!")
        print("     Please commit all changes before final freeze.")
    
    return manifest


def create_git_tag(tag_name: str, message: str):
    """Create annotated git tag."""
    try:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", message],
            check=True,
            capture_output=True,
        )
        print(f"\n✓ Git tag '{tag_name}' created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to create git tag: {e.stderr.decode()}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Freeze benchmark pack")
    parser.add_argument("--output", default="outputs/baseline_v1/freeze_manifest_benchmarkpack.json")
    parser.add_argument("--tag", default="baseline_all5_benchmarkpack_frozen")
    parser.add_argument("--create_tag", action="store_true", help="Also create git tag")
    parser.add_argument("--baseline_dir", default="outputs/baseline_v1")
    parser.add_argument("--eval_dir", default="reports/baseline_eval")
    parser.add_argument("--quality_dir", default="reports/data_quality")
    args = parser.parse_args()
    
    manifest = generate_freeze_manifest(
        Path(args.output),
        args.tag,
        Path(args.baseline_dir),
        Path(args.eval_dir),
        Path(args.quality_dir),
    )
    
    if args.create_tag:
        if manifest["git"].get("dirty"):
            print("\n⚠️  Cannot create tag with uncommitted changes.")
            print("   Please commit changes first.")
        else:
            create_git_tag(
                args.tag,
                f"Baseline-All-5 Benchmark Pack frozen at {manifest['frozen_at']}"
            )
    
    print("\n" + "="*70)
    print("FREEZE CHECKLIST")
    print("="*70)
    print("Before final freeze, ensure:")
    print("  [ ] All quality checks pass (Part A-B)")
    print("  [ ] All visualizations generated (Part C)")
    print("  [ ] All model cards created (Part D)")
    print("  [ ] All evaluations complete (Part E)")
    print("  [ ] No uncommitted changes")
    print("  [ ] Git tag created")


if __name__ == "__main__":
    main()
