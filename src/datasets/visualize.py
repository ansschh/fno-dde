"""
Dataset Visualization Utilities

Provides functions to visualize generated DDE data:
- Sample trajectories
- Parameter distributions
- Amplitude histograms
- Delay distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Optional, List
import argparse


def load_shard(shard_path: str) -> dict:
    """Load a single shard."""
    data = np.load(shard_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_all_shards(data_dir: Path, family: str, split: str = "train") -> dict:
    """Load all shards for a split."""
    split_dir = data_dir / family / split
    shard_files = sorted(split_dir.glob("shard_*.npz"))
    
    if len(shard_files) == 0:
        raise FileNotFoundError(f"No shards found in {split_dir}")
    
    phi_list, y_list, params_list, lags_list = [], [], [], []
    t_hist, t_out = None, None
    
    for shard_path in shard_files:
        data = np.load(shard_path, allow_pickle=True)
        phi_list.append(data["phi"])
        y_list.append(data["y"])
        params_list.append(data["params"])
        lags_list.append(data["lags"])
        if t_hist is None:
            t_hist = data["t_hist"]
            t_out = data["t_out"]
    
    return {
        "phi": np.concatenate(phi_list, axis=0),
        "y": np.concatenate(y_list, axis=0),
        "params": np.concatenate(params_list, axis=0),
        "lags": np.concatenate(lags_list, axis=0),
        "t_hist": t_hist,
        "t_out": t_out,
    }


def plot_sample_trajectories(
    data_dir: Path,
    family: str,
    n_samples: int = 5,
    split: str = "train",
    output_path: Optional[Path] = None,
):
    """
    Plot example trajectories for a family.
    """
    data = load_all_shards(data_dir, family, split)
    
    t_hist = data["t_hist"]
    t_out = data["t_out"]
    t_full = np.concatenate([t_hist, t_out])
    
    n_total = len(data["phi"])
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    
    state_dim = data["y"].shape[2]
    
    fig, axes = plt.subplots(n_samples, state_dim, figsize=(5 * state_dim, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if state_dim == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(indices):
        phi = data["phi"][idx]      # (N_hist, d_hist)
        y = data["y"][idx]          # (N_out, d_state)
        params = data["params"][idx]
        lags = data["lags"][idx]
        
        for d in range(state_dim):
            ax = axes[i, d]
            
            # Plot history
            if d < phi.shape[1]:
                ax.plot(t_hist, phi[:, d], "b-", linewidth=2, label="History")
            
            # Plot solution
            ax.plot(t_out, y[:, d], "r-", linewidth=2, label="Solution")
            
            # Mark t=0
            ax.axvline(x=0, color="gray", linestyle=":", alpha=0.7)
            
            # Mark delays
            for lag in lags:
                ax.axvline(x=-lag, color="green", linestyle="--", alpha=0.5)
            
            ax.set_xlabel("Time")
            ax.set_ylabel(f"State {d}")
            ax.grid(True, alpha=0.3)
            
            if i == 0 and d == 0:
                ax.legend()
            
            if d == 0:
                param_str = ", ".join([f"{p:.2f}" for p in params[:3]])
                ax.set_title(f"Sample {idx} | params: [{param_str}...]")
    
    plt.suptitle(f"Family: {family} ({split})", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_amplitude_histogram(
    data_dir: Path,
    family: str,
    split: str = "train",
    output_path: Optional[Path] = None,
):
    """
    Plot histogram of solution amplitudes.
    """
    data = load_all_shards(data_dir, family, split)
    y = data["y"]
    
    max_vals = np.max(np.abs(y), axis=(1, 2))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(max_vals, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Max |y|")
    ax.set_ylabel("Count")
    ax.set_title(f"Amplitude Distribution: {family} ({split})")
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Mean: {np.mean(max_vals):.2f}\nMedian: {np.median(max_vals):.2f}\nMax: {np.max(max_vals):.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_delay_histogram(
    data_dir: Path,
    family: str,
    split: str = "train",
    output_path: Optional[Path] = None,
):
    """
    Plot histogram of sampled delays.
    """
    data = load_all_shards(data_dir, family, split)
    lags = data["lags"]
    
    n_lags = lags.shape[1]
    
    fig, axes = plt.subplots(1, n_lags, figsize=(5 * n_lags, 4))
    if n_lags == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.hist(lags[:, i], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"τ{i+1}")
        ax.set_ylabel("Count")
        ax.set_title(f"Delay {i+1} Distribution")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Delay Distributions: {family} ({split})", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_parameter_distributions(
    data_dir: Path,
    family: str,
    split: str = "train",
    output_path: Optional[Path] = None,
):
    """
    Plot histograms of all parameters.
    """
    # Load manifest for param names
    manifest_path = data_dir / family / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    param_names = manifest["param_names"]
    param_ranges = manifest["param_ranges"]
    
    data = load_all_shards(data_dir, family, split)
    params = data["params"]
    
    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.hist(params[:, i], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        
        # Mark expected range
        if name in param_ranges:
            lo, hi = param_ranges[name]
            ax.axvline(x=lo, color="red", linestyle="--", alpha=0.7, label="range")
            ax.axvline(x=hi, color="red", linestyle="--", alpha=0.7)
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"Parameter Distributions: {family} ({split})", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def generate_all_plots(
    data_dir: Path,
    family: str,
    output_dir: Optional[Path] = None,
):
    """
    Generate all visualization plots for a family.
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots for {family}...")
    
    # Sample trajectories
    plot_sample_trajectories(
        data_dir, family, n_samples=5,
        output_path=output_dir / f"{family}_trajectories.png" if output_dir else None
    )
    
    # Amplitude histogram
    plot_amplitude_histogram(
        data_dir, family,
        output_path=output_dir / f"{family}_amplitudes.png" if output_dir else None
    )
    
    # Delay histogram
    plot_delay_histogram(
        data_dir, family,
        output_path=output_dir / f"{family}_delays.png" if output_dir else None
    )
    
    # Parameter distributions
    plot_parameter_distributions(
        data_dir, family,
        output_path=output_dir / f"{family}_params.png" if output_dir else None
    )
    
    print(f"Done! Plots saved to {output_dir}" if output_dir else "Done!")


def main():
    parser = argparse.ArgumentParser(description="Visualize DDE dataset")
    parser.add_argument("family", type=str, help="DDE family name")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--split", type=str, default="train",
                        help="Data split to visualize")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    generate_all_plots(data_dir, args.family, output_dir)


if __name__ == "__main__":
    main()
