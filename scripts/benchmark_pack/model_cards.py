#!/usr/bin/env python3
"""
Part D: Model Cards for Baseline FNO

Generate model card artifacts for each trained baseline model:
- Training curves (loss vs epoch)
- Best checkpoint info
- Config snapshot
- Git commit hash
- Final metrics summary
"""
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def get_git_info():
    """Get current git commit hash and status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
        
        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0
        
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def parse_training_log(log_path: Path):
    """Parse training log to extract epoch metrics."""
    epochs = []
    train_losses = []
    val_losses = []
    rel_l2s = []
    lrs = []
    
    with open(log_path) as f:
        for line in f:
            if line.startswith("Epoch"):
                try:
                    # Format: "Epoch  10: train=0.05 val=0.04 rel_L2=0.15 lr=1e-3"
                    parts = line.split()
                    epoch = int(parts[1].rstrip(":"))
                    
                    for part in parts[2:]:
                        if part.startswith("train="):
                            train_losses.append(float(part.split("=")[1]))
                        elif part.startswith("val="):
                            val_losses.append(float(part.split("=")[1]))
                        elif part.startswith("rel_L2="):
                            rel_l2s.append(float(part.split("=")[1]))
                        elif part.startswith("lr="):
                            lrs.append(float(part.split("=")[1]))
                    
                    epochs.append(epoch)
                except Exception:
                    continue
    
    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "rel_l2": rel_l2s,
        "lr": lrs,
    }


def generate_model_card(model_dir: Path, output_dir: Path = None):
    """Generate model card for a trained model."""
    
    if output_dir is None:
        output_dir = model_dir
    
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating model card for: {model_dir.name}")
    
    # Extract family from directory name
    family = model_dir.name.split("_")[0]
    
    # Load config if exists
    config_path = model_dir / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Load training stats if exists
    stats_path = model_dir / "training_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    
    # Check for best model
    best_model_path = model_dir / "best_model.pt"
    has_best_model = best_model_path.exists()
    
    # Parse training log if exists
    log_path = model_dir / "training.log"
    training_curves = None
    if log_path.exists():
        training_curves = parse_training_log(log_path)
    
    # Build model card
    model_card = {
        "family": family,
        "model_dir": str(model_dir),
        "created": datetime.now().isoformat(),
        "git": get_git_info(),
        "has_best_model": has_best_model,
        "config": config,
        "training_stats": stats,
    }
    
    # Add test results if available
    results_path = model_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            model_card["test_results"] = json.load(f)
    
    # Save model card
    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    
    print(f"  Family: {family}")
    print(f"  Has best model: {has_best_model}")
    print(f"  Git: {model_card['git']['commit']}")
    
    # Generate training curve plot if we have data
    if training_curves and len(training_curves["epochs"]) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        epochs = training_curves["epochs"]
        
        # Loss curves
        if training_curves["train_loss"]:
            axes[0].plot(epochs[:len(training_curves["train_loss"])], 
                        training_curves["train_loss"], label="Train")
        if training_curves["val_loss"]:
            axes[0].plot(epochs[:len(training_curves["val_loss"])], 
                        training_curves["val_loss"], label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].set_yscale("log")
        
        # RelL2 curve
        if training_curves["rel_l2"]:
            axes[1].plot(epochs[:len(training_curves["rel_l2"])], 
                        training_curves["rel_l2"])
            best_idx = np.argmin(training_curves["rel_l2"])
            best_val = training_curves["rel_l2"][best_idx]
            axes[1].axhline(best_val, color='r', linestyle='--', alpha=0.5,
                           label=f"Best: {best_val:.4f}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Relative L2")
        axes[1].set_title("Validation RelL2")
        axes[1].legend()
        
        # Learning rate
        if training_curves["lr"]:
            axes[2].plot(epochs[:len(training_curves["lr"])], 
                        training_curves["lr"])
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("LR Schedule")
        axes[2].set_yscale("log")
        
        plt.suptitle(f"{family}: Training Curves")
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()
        
        print(f"  Training curves saved")
    
    return model_card


def generate_all_model_cards(baseline_dir: Path, output_dir: Path):
    """Generate model cards for all baseline models."""
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)
    
    print("="*70)
    print("Generating Model Cards for All Baselines")
    print("="*70)
    
    # Find all model directories
    model_dirs = sorted(baseline_dir.glob("*_seed*"))
    
    cards = {}
    for model_dir in model_dirs:
        if model_dir.is_dir():
            family = model_dir.name.split("_")[0]
            card = generate_model_card(model_dir, output_dir / family)
            if family not in cards:
                cards[family] = []
            cards[family].append(card)
    
    # Summary
    print("\n" + "="*70)
    print("MODEL CARDS SUMMARY")
    print("="*70)
    
    summary = {}
    for family, family_cards in cards.items():
        summary[family] = {
            "n_runs": len(family_cards),
            "has_best_model": all(c["has_best_model"] for c in family_cards),
        }
        print(f"  {family}: {len(family_cards)} run(s)")
    
    with open(output_dir / "model_cards_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return cards


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate model cards")
    parser.add_argument("--baseline_dir", default="outputs/baseline_v1")
    parser.add_argument("--output_dir", default="reports/model_cards")
    parser.add_argument("--model_dir", help="Specific model directory (optional)")
    args = parser.parse_args()
    
    if args.model_dir:
        generate_model_card(Path(args.model_dir))
    else:
        generate_all_model_cards(Path(args.baseline_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
