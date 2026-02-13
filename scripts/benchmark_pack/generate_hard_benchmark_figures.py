#!/usr/bin/env python3
"""
Generate publication-quality figures for the Hard Benchmark Extension.

Figures generated:
1. All-family comparison bar chart (relL2 median for all DDE + PDE families)
2. Frequency-binned RMSE heatmap (family x frequency band)
3. fRMSE grouped bar chart (low/mid/high per family)
4. OOD gap comparison (ID vs OOD performance)
5. Training convergence panel (loss curves, 1 subplot per family)
6. Hardness ranking (sorted relL2, color-coded by DDE/PDE)

Output: reports/hard_benchmark_figures/

Usage:
    python scripts/benchmark_pack/generate_hard_benchmark_figures.py \\
        --results_dir reports/sweep_results \\
        --out_dir reports/hard_benchmark_figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    print("matplotlib required. Install with: pip install matplotlib")
    raise


# ============================================================
# Color schemes and styling
# ============================================================

DDE_FAMILIES = [
    "dist_exp", "hutch", "dist_uniform", "vdp", "linear2",
    "neutral", "chaotic_mg", "forced_duffing", "multi_delay_comb", "stiff_vdp",
]
PDE_FAMILIES = ["burgers", "ks", "helmholtz", "wave"]

ALL_FAMILIES = DDE_FAMILIES + PDE_FAMILIES

DISPLAY_NAMES = {
    "dist_exp": "DistExp", "hutch": "Hutch", "dist_uniform": "DistUniform",
    "vdp": "VdP", "linear2": "Linear2",
    "neutral": "Neutral", "chaotic_mg": "Chaotic MG",
    "forced_duffing": "Forced Duffing", "multi_delay_comb": "Multi-Delay Comb",
    "stiff_vdp": "Stiff VdP",
    "burgers": "Burgers", "ks": "KS", "helmholtz": "Helmholtz", "wave": "Wave",
}

DDE_COLOR = "#2c7fb8"
PDE_COLOR = "#e34a33"
FRMSE_COLORS = {"low": "#2c7fb8", "mid": "#41b6c4", "high": "#e34a33"}


def load_results(results_dir: Path) -> Dict:
    """Load all result JSON files from the results directory."""
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        family = data.get("family", f.stem)
        results[family] = data
    # Also try JSONL files
    for f in sorted(results_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                data = json.loads(line.strip())
                if "family" in data:
                    results[data["family"]] = data
    return results


def get_metric(results: Dict, family: str, key: str, default=None):
    """Safely get a metric value from results."""
    if family not in results:
        return default
    return results[family].get(key, default)


# ============================================================
# Figure 1: All-family relL2 comparison bar chart
# ============================================================

def fig_all_family_bar(results: Dict, out_dir: Path):
    """Bar chart of relL2 median for all families, sorted by difficulty."""
    families = [f for f in ALL_FAMILIES if f in results]
    if not families:
        print("  [Skip] No results found for bar chart")
        return

    rel_l2s = [get_metric(results, f, "test_rel_l2_median",
                          get_metric(results, f, "relL2_orig_median", 0)) for f in families]
    colors = [DDE_COLOR if f in DDE_FAMILIES else PDE_COLOR for f in families]

    # Sort by relL2
    order = np.argsort(rel_l2s)[::-1]
    families = [families[i] for i in order]
    rel_l2s = [rel_l2s[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(families))
    bars = ax.bar(x, rel_l2s, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, rel_l2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(f, f) for f in families],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Median Relative L2 Error", fontsize=12)
    ax.set_title("All-Family Performance Comparison (relL2 median)", fontsize=14)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="10% threshold")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold (hard)")

    # Legend for DDE vs PDE
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=DDE_COLOR, label="DDE"),
                       Patch(facecolor=PDE_COLOR, label="PDE")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "all_family_rel_l2_bar.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "all_family_rel_l2_bar.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: all_family_rel_l2_bar.png/pdf")


# ============================================================
# Figure 2: fRMSE Heatmap
# ============================================================

def fig_frmse_heatmap(results: Dict, out_dir: Path):
    """Heatmap of fRMSE (family x frequency band)."""
    families = [f for f in ALL_FAMILIES if f in results
                and get_metric(results, f, "frmse_low") is not None]
    if not families:
        print("  [Skip] No fRMSE data found for heatmap")
        return

    bands = ["low", "mid", "high"]
    data = np.zeros((len(families), 3))
    for i, f in enumerate(families):
        for j, band in enumerate(bands):
            data[i, j] = get_metric(results, f, f"frmse_{band}", 0)

    fig, ax = plt.subplots(figsize=(8, max(6, len(families) * 0.4)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Low Freq", "Mid Freq", "High Freq"], fontsize=11)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([DISPLAY_NAMES.get(f, f) for f in families], fontsize=10)

    # Annotate cells
    for i in range(len(families)):
        for j in range(3):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("fRMSE", fontsize=11)
    ax.set_title("Frequency-Binned RMSE (fRMSE) by Family", fontsize=13)

    fig.tight_layout()
    fig.savefig(out_dir / "frmse_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "frmse_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: frmse_heatmap.png/pdf")


# ============================================================
# Figure 3: fRMSE grouped bar chart
# ============================================================

def fig_frmse_bars(results: Dict, out_dir: Path):
    """Grouped bar chart of fRMSE low/mid/high per family."""
    families = [f for f in ALL_FAMILIES if f in results
                and get_metric(results, f, "frmse_low") is not None]
    if not families:
        print("  [Skip] No fRMSE data found for bar chart")
        return

    bands = ["low", "mid", "high"]
    x = np.arange(len(families))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, band in enumerate(bands):
        vals = [get_metric(results, f, f"frmse_{band}", 0) for f in families]
        ax.bar(x + i * width, vals, width, label=f"{band.title()} Freq",
               color=FRMSE_COLORS[band], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([DISPLAY_NAMES.get(f, f) for f in families],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("fRMSE", fontsize=12)
    ax.set_title("Frequency-Binned RMSE by Family", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "frmse_grouped_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "frmse_grouped_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: frmse_grouped_bars.png/pdf")


# ============================================================
# Figure 4: OOD gap comparison
# ============================================================

def fig_ood_gap(results: Dict, out_dir: Path):
    """Bar chart comparing ID vs OOD performance per family."""
    families = [f for f in ALL_FAMILIES if f in results]
    families_with_ood = []
    id_vals, ood_vals = [], []

    for f in families:
        id_val = get_metric(results, f, "test_rel_l2_median",
                            get_metric(results, f, "relL2_orig_median"))
        ood_val = get_metric(results, f, "ood_rel_l2_median",
                             get_metric(results, f, "ood_relL2_orig_median"))
        if id_val is not None and ood_val is not None:
            families_with_ood.append(f)
            id_vals.append(id_val)
            ood_vals.append(ood_val)

    if not families_with_ood:
        print("  [Skip] No OOD data found for gap chart")
        return

    x = np.arange(len(families_with_ood))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, id_vals, width, label="ID Test", color="#2c7fb8")
    ax.bar(x + width / 2, ood_vals, width, label="OOD Test", color="#e34a33")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(f, f) for f in families_with_ood],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Median Relative L2 Error", fontsize=12)
    ax.set_title("In-Distribution vs Out-of-Distribution Performance", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "ood_gap_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "ood_gap_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ood_gap_comparison.png/pdf")


# ============================================================
# Figure 5: Training convergence panels
# ============================================================

def fig_training_convergence(results: Dict, out_dir: Path):
    """Panel of training loss curves for all families."""
    families = [f for f in ALL_FAMILIES if f in results
                and "train_loss_history" in results[f]]
    if not families:
        # Try loading from training curve NPZ files
        print("  [Skip] No training history found for convergence panels")
        return

    n = len(families)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                            squeeze=False)

    for idx, fam in enumerate(families):
        row, col = idx // ncols, idx % ncols
        ax = axs[row][col]

        train_loss = results[fam].get("train_loss_history", [])
        val_loss = results[fam].get("val_loss_history", [])
        epochs = list(range(1, len(train_loss) + 1))

        if train_loss:
            ax.plot(epochs, train_loss, label="Train", linewidth=1.2, color="#2c7fb8")
        if val_loss:
            ax.plot(epochs[:len(val_loss)], val_loss, label="Val",
                    linewidth=1.2, color="#e34a33")

        ax.set_yscale("log")
        ax.set_title(DISPLAY_NAMES.get(fam, fam), fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axs[row][col].set_visible(False)

    fig.suptitle("Training Convergence", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "training_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: training_convergence.png")


# ============================================================
# Figure 6: Hardness ranking
# ============================================================

def fig_hardness_ranking(results: Dict, out_dir: Path):
    """Horizontal bar chart sorted by difficulty (relL2), color-coded."""
    families = [f for f in ALL_FAMILIES if f in results]
    if not families:
        print("  [Skip] No results for hardness ranking")
        return

    rel_l2s = []
    for f in families:
        val = get_metric(results, f, "test_rel_l2_median",
                         get_metric(results, f, "relL2_orig_median", 0))
        rel_l2s.append(val if val is not None else 0)

    # Sort ascending (easiest at top)
    order = np.argsort(rel_l2s)
    families = [families[i] for i in order]
    rel_l2s = [rel_l2s[i] for i in order]
    colors = [DDE_COLOR if f in DDE_FAMILIES else PDE_COLOR for f in families]

    fig, ax = plt.subplots(figsize=(10, max(5, len(families) * 0.4)))
    y = np.arange(len(families))
    bars = ax.barh(y, rel_l2s, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, rel_l2s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES.get(f, f) for f in families], fontsize=10)
    ax.set_xlabel("Median Relative L2 Error", fontsize=12)
    ax.set_title("Family Difficulty Ranking", fontsize=14)
    ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DDE_COLOR, label="DDE"),
        Patch(facecolor=PDE_COLOR, label="PDE"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "hardness_ranking.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "hardness_ranking.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: hardness_ranking.png/pdf")


# ============================================================
# Summary table (LaTeX)
# ============================================================

def generate_latex_table(results: Dict, out_dir: Path):
    """Generate a LaTeX table of all results."""
    families = [f for f in ALL_FAMILIES if f in results]
    if not families:
        return

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hard Benchmark Results: All Families}",
        r"\label{tab:hard-benchmark}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Family & Type & relL2 Med & relL2 P95 & fRMSE$_\text{low}$ & fRMSE$_\text{mid}$ & fRMSE$_\text{high}$ \\",
        r"\midrule",
    ]

    for f in families:
        ftype = "DDE" if f in DDE_FAMILIES else "PDE"
        med = get_metric(results, f, "test_rel_l2_median",
                         get_metric(results, f, "relL2_orig_median", "-"))
        p95 = get_metric(results, f, "test_rel_l2_p95",
                         get_metric(results, f, "relL2_orig_p95", "-"))
        fl = get_metric(results, f, "frmse_low", "-")
        fm = get_metric(results, f, "frmse_mid", "-")
        fh = get_metric(results, f, "frmse_high", "-")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

        name = DISPLAY_NAMES.get(f, f)
        lines.append(
            f"  {name} & {ftype} & {fmt(med)} & {fmt(p95)} & "
            f"{fmt(fl)} & {fmt(fm)} & {fmt(fh)} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_text = "\n".join(lines)
    out_path = out_dir / "hard_benchmark_table.tex"
    with open(out_path, "w") as f:
        f.write(table_text)
    print(f"  Saved: {out_path.name}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate hard benchmark paper figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", default="reports/sweep_results",
        help="Directory containing per-family result JSON files.",
    )
    parser.add_argument(
        "--out_dir", default="reports/hard_benchmark_figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--pilot_results", default=None,
        help="Path to pilot_results.json (from run_pilot.py --save-results).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = {}
    if results_dir.exists():
        results = load_results(results_dir)

    # Also load pilot results if provided
    if args.pilot_results and Path(args.pilot_results).exists():
        with open(args.pilot_results) as f:
            pilot = json.load(f)
        for fam, data in pilot.items():
            if fam not in results:
                results[fam] = data

    if not results:
        print("No results found. Run pilots or full sweep first.")
        print(f"  Expected: {results_dir}/*.json or --pilot_results")
        return

    print(f"\nLoaded results for {len(results)} families: {list(results.keys())}")
    print(f"Output: {out_dir}\n")

    # Generate all figures
    fig_all_family_bar(results, out_dir)
    fig_frmse_heatmap(results, out_dir)
    fig_frmse_bars(results, out_dir)
    fig_ood_gap(results, out_dir)
    fig_training_convergence(results, out_dir)
    fig_hardness_ranking(results, out_dir)
    generate_latex_table(results, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
