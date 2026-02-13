#!/usr/bin/env python3
"""
Dataset Visualization Package for Baseline-All-5

Generates trajectory galleries, parameter histograms, and OOD comparison plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from dde.families import get_family


def load_shard(data_dir: Path, family: str, split: str):
    shard_path = data_dir / family / split / "shard_000.npz"
    if not shard_path.exists():
        return None
    data = np.load(shard_path)
    return {k: data[k] for k in data.files}


def plot_trajectory_gallery(data, family_name: str, output_path: Path, n_plots: int = 20):
    """Plot trajectory gallery with history and future."""
    family = get_family(family_name)
    state_dim = family.config.state_dim
    
    n_plots = min(n_plots, len(data["phi"]))
    indices = np.random.choice(len(data["phi"]), n_plots, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    t_hist = data["t_hist"]
    t_out = data["t_out"]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        phi = data["phi"][idx]
        y = data["y"][idx]
        
        # Plot each component
        colors = ['#2E86AB', '#E94F37', '#8E44AD']
        for d in range(state_dim):
            ax.plot(t_hist, phi[:, d], color=colors[d], alpha=0.5, linestyle='--')
            ax.plot(t_out, y[:, d], color=colors[d], linewidth=1.5)
        
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f"Sample {idx}", fontsize=9)
        ax.tick_params(labelsize=7)
    
    plt.suptitle(f"{family_name.upper()} Trajectories (dashed=history, solid=future)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_param_histograms(data, family_name: str, output_path: Path):
    """Plot parameter and delay histograms."""
    family = get_family(family_name)
    params = data["params"]
    
    n_params = len(family.config.param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3))
    if n_params == 1:
        axes = [axes]
    
    for i, name in enumerate(family.config.param_names):
        ax = axes[i]
        ax.hist(params[:, i], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{name} Distribution", fontsize=12)
    
    plt.suptitle(f"{family_name.upper()} Parameter Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ood_comparison(id_data, ood_data, family_name: str, ood_name: str, output_path: Path):
    """Compare ID vs OOD distributions."""
    family = get_family(family_name)
    
    # Find tau index
    tau_idx = None
    for i, name in enumerate(family.config.param_names):
        if 'tau' in name.lower():
            tau_idx = i
            break
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    # Tau distribution
    if tau_idx is not None:
        ax = axes[0]
        ax.hist(id_data["params"][:, tau_idx], bins=30, alpha=0.6, label='ID', edgecolor='black')
        ax.hist(ood_data["params"][:, tau_idx], bins=30, alpha=0.6, label=ood_name, edgecolor='black')
        ax.set_xlabel("tau")
        ax.set_title("Delay Distribution")
        ax.legend()
    
    # y_norm distribution
    ax = axes[1]
    id_norms = np.linalg.norm(id_data["y"].reshape(len(id_data["y"]), -1), axis=1)
    ood_norms = np.linalg.norm(ood_data["y"].reshape(len(ood_data["y"]), -1), axis=1)
    ax.hist(id_norms, bins=30, alpha=0.6, label='ID', edgecolor='black')
    ax.hist(ood_norms, bins=30, alpha=0.6, label=ood_name, edgecolor='black')
    ax.set_xlabel("||y||_2")
    ax.set_title("Target Energy Distribution")
    ax.legend()
    
    # History roughness
    ax = axes[2]
    id_rough = np.mean(np.abs(np.diff(id_data["phi"][:, :, 0], n=2, axis=1)), axis=1)
    ood_rough = np.mean(np.abs(np.diff(ood_data["phi"][:, :, 0], n=2, axis=1)), axis=1)
    ax.hist(id_rough, bins=30, alpha=0.6, label='ID', edgecolor='black')
    ax.hist(ood_rough, bins=30, alpha=0.6, label=ood_name, edgecolor='black')
    ax.set_xlabel("Roughness")
    ax.set_title("History Roughness")
    ax.legend()
    
    plt.suptitle(f"{family_name.upper()}: ID vs {ood_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_vdp_phase(data, output_path: Path):
    """Phase plot for VdP (x vs v)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n_plots = min(20, len(data["y"]))
    indices = np.random.choice(len(data["y"]), n_plots, replace=False)
    
    for idx in indices:
        y = data["y"][idx]
        ax.plot(y[:, 0], y[:, 1], alpha=0.5, linewidth=0.8)
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("v = dx/dt", fontsize=12)
    ax.set_title("VdP Phase Portrait (x vs v)", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--output_dir", default="reports/data_viz")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Visualizing {args.family}")
    print(f"{'='*60}")
    
    # Load ID data
    id_data = load_shard(Path("data_baseline_v1"), args.family, "test")
    if id_data is None:
        id_data = load_shard(Path("data_baseline_v1"), args.family, "train")
    
    if id_data is None:
        print(f"ERROR: No data found for {args.family}")
        return
    
    # Trajectory gallery
    plot_trajectory_gallery(id_data, args.family, output_dir / "trajectories_id.png")
    
    # Parameter histograms
    plot_param_histograms(id_data, args.family, output_dir / "histograms_params.png")
    
    # VdP phase plot
    if args.family == "vdp":
        plot_vdp_phase(id_data, output_dir / "phaseplot_vdp.png")
    
    # OOD comparisons
    ood_configs = [
        ("data_ood_delay", "test_ood", "OOD-delay"),
        ("data_ood_delay_hole", "test_hole", "OOD-hole"),
        ("data_ood_history", "test_spline", "OOD-history"),
    ]
    
    for data_dir, split, ood_name in ood_configs:
        ood_data = load_shard(Path(data_dir), args.family, split)
        if ood_data is not None:
            plot_ood_comparison(
                id_data, ood_data, args.family, ood_name,
                output_dir / f"ood_compare_{ood_name.lower().replace('-', '_')}.png"
            )
    
    # Generate README
    readme = f"""# {args.family.upper()} Dataset Visualization

Generated: {__import__('datetime').datetime.now().isoformat()}

## Files
- `trajectories_id.png` - Gallery of 20 random ID trajectories
- `histograms_params.png` - Parameter distributions
- `ood_compare_*.png` - ID vs OOD distribution comparisons
"""
    if args.family == "vdp":
        readme += "- `phaseplot_vdp.png` - Phase portrait (x vs v)\n"
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    
    print(f"\nVisualization complete: {output_dir}")


if __name__ == "__main__":
    main()
