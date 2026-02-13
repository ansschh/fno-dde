#!/usr/bin/env python3
"""
Visualize sample trajectories from all 5 DDE families.
Shows input (history) and target (future) for each family.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from datasets.sharded_dataset import ShardedDDEDataset

FAMILIES = {
    "hutch": ("data_baseline_v1", "Hutchinson (Population)"),
    "linear2": ("data_baseline_v1", "Linear2 (Two Delays)"),
    "vdp": ("data_baseline_v1", "Van der Pol (Oscillator)"),
    "dist_uniform": ("data_baseline_v1", "Distributed Uniform"),
    "dist_exp": ("data_baseline_v2", "Distributed Exponential"),
}


def plot_family_samples(family: str, data_dir: str, title: str, ax, n_samples: int = 5):
    """Plot sample trajectories for a family."""
    dataset = ShardedDDEDataset(data_dir, family, "test")
    
    # Get random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
    
    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        
        # Get data
        x = sample["input"].numpy()  # (T_in, C) - history
        y = sample["target"].numpy()  # (T_out, C) - future target
        target_mean = sample["target_mean"].numpy()
        target_std = sample["target_std"].numpy()
        
        # Denormalize target
        y_orig = y * target_std + target_mean
        
        # Time axes (approximate)
        t_hist = np.linspace(-2, 0, len(x))  # History: t < 0
        t_future = np.linspace(0, 15, len(y))  # Future: t >= 0
        
        # Plot first dimension only
        ax.plot(t_hist, x[:, 0], color=colors[i], alpha=0.7, linewidth=1.5)
        ax.plot(t_future, y_orig[:, 0], color=colors[i], alpha=0.7, linewidth=1.5)
    
    # Mark t=0 transition
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.02, 0.98, 'History → Future', transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', alpha=0.7)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=9)
    ax.set_ylabel('State', fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    print("Loading and plotting trajectories from all 5 families...")
    
    for i, (family, (data_dir, title)) in enumerate(FAMILIES.items()):
        print(f"  {family}...")
        plot_family_samples(family, data_dir, title, axes[i], n_samples=5)
    
    # Hide the 6th subplot
    axes[5].axis('off')
    axes[5].text(0.5, 0.5, 
                 "5 DDE Families\n\n"
                 "• Hutchinson: Population dynamics\n"
                 "• Linear2: Two delay terms\n"
                 "• Van der Pol: Nonlinear oscillator\n"
                 "• DistUniform: Integral delay\n"
                 "• DistExp: Exponential kernel",
                 transform=axes[5].transAxes,
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DDE Trajectory Samples: History (t<0) → Future (t>0)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_dir = Path("reports/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "all5_family_samples.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_dir / 'all5_family_samples.png'}")
    
    plt.show()


if __name__ == "__main__":
    main()
