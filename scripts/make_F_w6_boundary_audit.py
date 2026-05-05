"""W6 boundary audit figure: per_frame.json based boundary vs interior error.

For each cell with per_frame.json, split rel_l2_per_step[t] into:
  - boundary region: t ∈ [0, k) ∪ [T-k, T)   (first/last k frames near wrap)
  - interior region: t ∈ [k, T-k)             (middle)

Computes mean boundary error / mean interior error per cell, aggregated by
(model, family). Plots:
  - Left:  per-frame rel-L2 over t for cyclic vs B5 (causal smoother) on dist_*_rd_2d
  - Right: ratio (boundary mean / interior mean) per family per model

Output: NeurIPS_LEMO/figures/kept/main/F_w6_boundary_audit.{pdf,png}
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401  (sets Times New Roman globally)
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
FIG_PDF = NEURIPS / "figures" / "kept" / "main" / "F_w6_boundary_audit.pdf"
FIG_PNG = NEURIPS / "figures" / "kept" / "png" / "F_w6_boundary_audit.png"
FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
FIG_PNG.parent.mkdir(parents=True, exist_ok=True)

FAM_LABEL = {
    "dist_exp_rd_2d": "Exp",
    "dist_gaussian_rd_2d": "Gauss",
    "dist_gamma_rd_2d": "Gamma",
    "dist_uniform_rd_2d": "Uniform",
    "dist_powerlaw_rd_2d": "Power",
}


def parse_path(pf_path):
    parts = pf_path.parts
    try:
        idx = parts.index("raw")
    except ValueError:
        return None
    if idx + 4 >= len(parts):
        return None
    fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
    if not seed_str.startswith("s"):
        return None
    return fam, reg, mdl, int(seed_str[1:])


def crawl(roots, regime_filter="clean"):
    """Return dict[(model, family)] -> list of {E_t array}."""
    by_mf = defaultdict(list)
    for r in roots:
        rp = Path(r)
        if not rp.is_absolute():
            rp = REPO / rp
        for pf in rp.glob("**/per_frame.json"):
            meta = parse_path(pf)
            if meta is None:
                continue
            fam, reg, mdl, seed = meta
            if reg != regime_filter:
                continue
            try:
                d = json.loads(pf.read_text())
            except Exception:
                continue
            E_t = d.get("rel_l2_per_step") or d.get("rel_l2_mean_per_step")
            if E_t is None:
                continue
            by_mf[(mdl, fam)].append(np.array(E_t, dtype=np.float64))
    return by_mf


def plot(by_mf, k=8):
    fams = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
            "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
    models_to_plot = ["lemo_pc_nd", "causal_smooth_lemo_pc_nd"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: per-frame curves
    colors = {"lemo_pc_nd": "C0", "causal_smooth_lemo_pc_nd": "C1"}
    labels = {"lemo_pc_nd": "Cyclic LEMO-PC", "causal_smooth_lemo_pc_nd": "B5: CausalSmoother"}
    for mdl in models_to_plot:
        all_E = []
        for fam in fams:
            cells = by_mf.get((mdl, fam), [])
            for E_t in cells:
                all_E.append(E_t)
        if not all_E:
            continue
        E_arr = np.stack(all_E)
        E_mean = E_arr.mean(axis=0)
        E_std = E_arr.std(axis=0)
        ts = np.arange(len(E_mean))
        axes[0].plot(ts, E_mean, color=colors[mdl], linewidth=2,
                     label=f"{labels[mdl]} (n={len(all_E)})")
        axes[0].fill_between(ts, E_mean - E_std, E_mean + E_std,
                              color=colors[mdl], alpha=0.15)

    axes[0].axvspan(0, k, color="gray", alpha=0.1, label=f"rollout edge (k={k})")
    if cells and len(E_mean) > k:
        axes[0].axvspan(len(E_mean) - k, len(E_mean), color="gray", alpha=0.1)
    axes[0].set_xlabel("rollout step t")
    axes[0].set_ylabel(r"rel-$L_2$")
    axes[0].set_title("Per-step rollout error")
    axes[0].set_yscale("log")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    # Panel 2: boundary/interior ratio per family per model
    width_bar = 0.35
    x_pos = np.arange(len(fams))
    for i, mdl in enumerate(models_to_plot):
        ratios = []
        for fam in fams:
            cells = by_mf.get((mdl, fam), [])
            if not cells:
                ratios.append(np.nan)
                continue
            E_arr = np.stack(cells)
            T = E_arr.shape[1]
            if T < 2 * k:
                ratios.append(np.nan)
                continue
            boundary = np.concatenate([E_arr[:, :k], E_arr[:, T-k:]], axis=1).mean()
            interior = E_arr[:, k:T-k].mean()
            ratios.append(boundary / interior if interior > 0 else np.nan)
        x_offset = (i - 0.5) * width_bar
        axes[1].bar(x_pos + x_offset, ratios, width_bar,
                    color=colors[mdl], label=labels[mdl], alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="edge = interior")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([FAM_LABEL[f] for f in fams])
    axes[1].set_ylabel("rollout edge / interior error ratio")
    axes[1].set_title(f"Rollout-edge / interior ratio (k={k} edge frames)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=160, bbox_inches="tight")
    print(f"[F_w6] saved {FIG_PDF}")
    print(f"[F_w6] saved {FIG_PNG}")
    print(f"[F_w6] cells per (model, family): "
          f"{ {k: len(v) for k, v in by_mf.items()} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=8, help="Number of edge frames considered boundary")
    args = ap.parse_args()
    by_mf = crawl(args.roots)
    print(f"[F_w6] {sum(len(v) for v in by_mf.values())} cells crawled")
    plot(by_mf, k=args.k)


if __name__ == "__main__":
    main()
