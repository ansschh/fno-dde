"""One-off: generate V01 diff variant with three scaling options for comparison.

Outputs:
  V01_diff_optionA_linear.png
  V01_diff_optionB_sqrt.png
  V01_diff_optionC_symlog_small.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, PowerNorm, LinearSegmentedColormap
from scipy.ndimage import zoom
import matplotlib.patheffects as pe

REPO = Path(__file__).resolve().parent.parent
FIG = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
sys.path.insert(0, str(REPO / "scripts"))
from make_visual_figures import (load_viz, load_viz_fno, FAMS, FAM_LABELS, PASTEL_DIV)

UPSAMPLE = 32
SAMPLE = 0
TARGET_T = -1
halo = pe.withStroke(linewidth=3.5, foreground="white")


def _draw(ax, field, vmax, mode):
    """mode in {'linear', 'sqrt', 'symlog'}."""
    f_hi = zoom(field, UPSAMPLE, order=3)
    H, W = field.shape
    extent = [-0.5, W - 0.5, H - 0.5, -0.5]
    if mode == "linear":
        im = ax.imshow(f_hi, cmap=PASTEL_DIV, vmin=-vmax, vmax=vmax,
                        interpolation="bilinear", extent=extent)
    elif mode == "sqrt":
        # PowerNorm with gamma<1 expands small values; need to handle signs.
        # Use sign(x) * |x|^0.5 as transform via a helper colormap norm.
        # Simpler: pre-apply the transform to the data.
        f_t = np.sign(f_hi) * np.power(np.abs(f_hi) / max(vmax, 1e-9), 0.5)
        im = ax.imshow(f_t, cmap=PASTEL_DIV, vmin=-1, vmax=1,
                        interpolation="bilinear", extent=extent)
    elif mode == "symlog":
        # Smaller linthresh than before so symlog only kicks in for very small
        # values; the rest is roughly linear.
        linthresh = max(1e-2 * vmax, 1e-9)
        im = ax.imshow(f_hi, cmap=PASTEL_DIV,
                        norm=SymLogNorm(linthresh=linthresh,
                                         vmin=-vmax, vmax=vmax, base=10),
                        interpolation="bilinear", extent=extent)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return im


def _overlay_contours(ax, y_gt, vmax):
    H, W = y_gt.shape
    pos = np.linspace(0.15 * vmax, 0.85 * vmax, 4)
    levels = np.concatenate([-pos[::-1], pos])
    cs = ax.contour(np.arange(W), np.arange(H), y_gt, levels=levels,
                     colors="black", linewidths=1.5)
    try:
        cs.set(path_effects=[halo])
    except Exception:
        for c in getattr(cs, "collections", []):
            c.set_path_effects([halo])


def make_variant(mode: str, label: str, suffix: str):
    panels = []
    for fam in FAMS:
        dl = load_viz(fam)
        if dl is None: continue
        df = load_viz_fno(fam)
        y_gt = dl["target"][SAMPLE][TARGET_T, ..., 0]
        y_lemo = dl["pred"][SAMPLE][TARGET_T, ..., 0]
        y_fno = (df["pred"][SAMPLE][TARGET_T, ..., 0]
                  if df is not None else None)
        panels.append((fam, y_gt, y_lemo, y_fno))
    n = len(panels)
    gt_vmax = float(np.max([np.abs(p[1]).max() for p in panels]))
    le_vmax = float(np.max([np.abs(p[2] - p[1]).max() for p in panels]))
    diffs = [np.abs(p[2] - p[1]) - np.abs(p[3] - p[1])
              for p in panels if p[3] is not None]
    diff_vmax = float(np.max([np.abs(d).max() for d in diffs])) if diffs else 1e-9

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10.5),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.07,
                                           "right": 0.92})
    if n == 1: axes = axes.reshape(3, 1)
    for i, lbl in enumerate(["Ground Truth", "LEMO Error", "Error Difference"]):
        axes[i, 0].set_ylabel(lbl, fontsize=14, rotation=0, ha="right",
                                va="center", labelpad=20)
    im_gt = im_le = im_dd = None
    for j, (fam, y_gt, y_lemo, y_fno) in enumerate(panels):
        axes[0, j].set_title(FAM_LABELS[fam], fontsize=14)
        im = _draw(axes[0, j], y_gt, gt_vmax, "linear")
        if j == n - 1: im_gt = im
        _overlay_contours(axes[0, j], y_gt, gt_vmax)
        im = _draw(axes[1, j], y_lemo - y_gt, le_vmax, mode)
        if j == n - 1: im_le = im
        _overlay_contours(axes[1, j], y_gt, le_vmax)
        if y_fno is not None:
            d = np.abs(y_lemo - y_gt) - np.abs(y_fno - y_gt)
            im = _draw(axes[2, j], d, diff_vmax, mode)
            if j == n - 1: im_dd = im
            _overlay_contours(axes[2, j], y_gt, diff_vmax)
        else:
            axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
            for sp in axes[2, j].spines.values():
                sp.set_linewidth(0.6)
            axes[2, j].text(0.5, 0.5, "n/a", ha="center", va="center",
                              transform=axes[2, j].transAxes,
                              color="dimgrey", fontsize=12)
    for ax_row, im_row, lbl in [(axes[0, -1], im_gt, "field"),
                                  (axes[1, -1], im_le, "pred − GT"),
                                  (axes[2, -1], im_dd, "|LEMO err| − |FNO err|")]:
        if im_row is None: continue
        cb = fig.colorbar(im_row, ax=ax_row, fraction=0.05, pad=0.04,
                           shrink=0.95, aspect=18)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(lbl, fontsize=9)
    fig.suptitle(f"Predictions ({label})", fontsize=18, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    out = FIG / f"V01_diff_option{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.name}")


if __name__ == "__main__":
    make_variant("linear", "linear, no headroom", "A_linear")
    make_variant("sqrt",   "sqrt power-norm",     "B_sqrt")
    make_variant("symlog", "symlog small linthresh", "C_symlog")
