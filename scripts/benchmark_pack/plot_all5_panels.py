#!/usr/bin/env python3
"""
Generate 10 publication-quality 1×5 panel figures for all 5 DDE families.

Figures generated:
1-4. Error vs time (mean/median + P50-P90 band) for id, ood_delay, ood_history, ood_horizon
5-8. Tail error vs time (P95) for id, ood_delay, ood_history, ood_horizon
9.   Training loss (train + val)
10.  Validation relL2

Output: reports/model_viz/all5_panels/
"""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

FAMILY_ORDER = ["dist_exp", "hutch", "dist_uniform", "vdp", "linear2"]
FAMILY_DISPLAY = {
    "dist_exp": "DistExp",
    "hutch": "Hutch",
    "linear2": "Linear2",
    "vdp": "VdP",
    "dist_uniform": "DistUniform",
}
SPLITS = ["id", "ood_delay", "ood_history", "ood_horizon"]
SPLIT_DISPLAY = {
    "id": "ID Test",
    "ood_delay": "OOD-Delay",
    "ood_history": "OOD-History",
    "ood_horizon": "OOD-Horizon",
}


def load_npz(path: Path) -> dict:
    """Load NPZ file and return as dict."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def make_row_5panels(title: str, families: list, plot_fn, outpath: Path,
                     sharey: bool = True, yscale: str = None,
                     ylabel: str = None, xlabel: str = None):
    """Create a 1×5 panel figure with consistent styling."""
    fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=sharey)
    
    handles, labels = None, None
    
    for idx, (ax, fam) in enumerate(zip(axs, families)):
        result = plot_fn(ax, fam)
        ax.set_title(FAMILY_DISPLAY.get(fam, fam), fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        
        # Collect legend handles from first subplot only
        if idx == 0:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=10)
    
    if yscale is not None:
        for ax in axs:
            ax.set_yscale(yscale)
    
    # Single figure-level legend
    if handles and labels:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95),
                   fontsize=9, framealpha=0.9)
    
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate 10 all-5 panel figures")
    parser.add_argument("--viz_root", default="reports/model_viz",
                        help="Root containing reports/model_viz/{family}/{run_id}/curves/")
    parser.add_argument("--run_map_json", default="reports/model_viz/run_map_baseline_v2.json",
                        help="JSON mapping family -> run_id")
    parser.add_argument("--out_dir", default="reports/model_viz/all5_panels")
    parser.add_argument("--families", nargs="*", default=FAMILY_ORDER)
    args = parser.parse_args()
    
    viz_root = Path(args.viz_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load run map
    run_map_path = Path(args.run_map_json)
    if not run_map_path.exists():
        print(f"Error: Run map not found: {run_map_path}")
        print("Run save_curves.py first to generate it.")
        return
    
    with open(run_map_path) as f:
        run_map = json.load(f)
    
    families = args.families
    
    def curve_path(fam: str, fname: str) -> Path:
        run_id = run_map.get(fam, "")
        return viz_root / fam / run_id / "curves" / fname
    
    print(f"\nGenerating 10 panel figures to: {out_dir}\n")
    
    # ========================================================================
    # Figures 1-4: Error vs Time (mean/median + P50-P90 band) per split
    # ========================================================================
    for split in SPLITS:
        def plot_error_vs_time(ax, fam):
            d = load_npz(curve_path(fam, f"error_vs_time_{split}.npz"))
            if d is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10, color='gray')
                return
            
            t = d["t"]
            ax.plot(t, d["median"], linewidth=2, label="Median", color='#2c7fb8')
            ax.plot(t, d["mean"], linewidth=1.5, linestyle='--', label="Mean", color='#41b6c4')
            ax.fill_between(t, d["p50"], d["p90"], alpha=0.25, color='#2c7fb8', label="P50–P90")
        
        make_row_5panels(
            title=f"Error vs Time — {SPLIT_DISPLAY[split]}",
            families=families,
            plot_fn=plot_error_vs_time,
            outpath=out_dir / f"all5_error_vs_time_{split}.png",
            sharey=True,
            ylabel="Relative Error",
            xlabel="Time",
        )
    
    # ========================================================================
    # Figures 5-8: Tail Error vs Time (P95) per split
    # ========================================================================
    for split in SPLITS:
        def plot_tail_p95(ax, fam):
            d = load_npz(curve_path(fam, f"error_vs_time_{split}.npz"))
            if d is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='gray')
                return
            
            t = d["t"]
            ax.plot(t, d["p95"], linewidth=2, label="P95", color='#e34a33')
        
        make_row_5panels(
            title=f"Tail Error vs Time (P95) — {SPLIT_DISPLAY[split]}",
            families=families,
            plot_fn=plot_tail_p95,
            outpath=out_dir / f"all5_tail_p95_vs_time_{split}.png",
            sharey=True,
            yscale="log",
            ylabel="P95 Relative Error",
            xlabel="Time",
        )
    
    # ========================================================================
    # Figure 9: Training & Validation Loss
    # ========================================================================
    def plot_train_val_loss(ax, fam):
        d = load_npz(curve_path(fam, "training_curves.npz"))
        if d is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=10, color='gray')
            return
        
        e = d["epoch"]
        ax.plot(e, d["train_loss"], linewidth=1.5, label="Train", color='#2c7fb8')
        ax.plot(e, d["val_loss"], linewidth=1.5, label="Val", color='#e34a33')
        if "best_epoch" in d:
            best = int(d["best_epoch"])
            ax.axvline(best, linestyle='--', color='gray', alpha=0.7, label=f"Best ({best})")
    
    make_row_5panels(
        title="Training & Validation Loss",
        families=families,
        plot_fn=plot_train_val_loss,
        outpath=out_dir / "all5_train_val_loss.png",
        sharey=True,
        yscale="log",
        ylabel="Loss (MSE)",
        xlabel="Epoch",
    )
    
    # ========================================================================
    # Figure 10: Validation relL2
    # ========================================================================
    def plot_val_rel_l2(ax, fam):
        d = load_npz(curve_path(fam, "training_curves.npz"))
        if d is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=10, color='gray')
            return
        
        e = d["epoch"]
        ax.plot(e, d["val_rel_l2"], linewidth=2, label="Val relL2", color='#31a354')
        if "best_epoch" in d:
            best = int(d["best_epoch"])
            ax.axvline(best, linestyle='--', color='gray', alpha=0.7, label=f"Best ({best})")
    
    make_row_5panels(
        title="Validation Relative L2 Error",
        families=families,
        plot_fn=plot_val_rel_l2,
        outpath=out_dir / "all5_val_rel_l2.png",
        sharey=True,
        ylabel="Val relL2",
        xlabel="Epoch",
    )
    
    print(f"\n✓ Generated 10 panel figures in: {out_dir}")


if __name__ == "__main__":
    main()
