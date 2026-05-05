"""M3 — Failure-mode gallery. **DROPPED 2026-05-03** — redundant with
V01_family_triptych (appendix) which already shows GT vs LEMO-PC fields per
family. M3 only added an easiest/hardest-sample axis on top, which doesn't
add narrative beyond V01. Script kept on disk; not invoked by any pipeline.

For each of the 5 dist-kernel families, find the easiest and hardest test
sample (by per-sample relL2 from `residuals.npz`), then render side-by-side:

  Easiest sample:  GT  |  LEMO-PC pred  |  |GT - pred|
  Hardest sample:  GT  |  LEMO-PC pred  |  |GT - pred|

5 families × 2 (easy/hard) × 3 panels = 5 rows × 6 columns figure.

Annotations: per-sample relL2 in each row.

Source: residuals.npz (rel_l2_per_sample) + viz_samples.npz (limited
to the 4 viz samples per cell — so we pick the easy/hard *among the
4 viz samples* rather than the full test set).  This is fine because
viz_samples were saved by capture as a representative slice.

Output: paper/figures/M3_failure_mode_gallery.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG.mkdir(parents=True, exist_ok=True)
BASE = REPO / "extracted" / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}
SEEDS = ["s42", "s123", "s456"]


def _per_sample_rel(viz_d, target_step=-1):
    """For each of the 4 viz samples, compute per-sample relL2 at the
    final rollout step using mask-aware aggregate-then-sqrt."""
    target = viz_d["target"]            # (B, T, *spatial, C)
    pred   = viz_d["pred"]
    spatial_dims = tuple(range(2, target.ndim))
    diff_sq = ((pred - target) ** 2).sum(axis=spatial_dims)    # (B, T)
    tgt_sq  = (target ** 2).sum(axis=spatial_dims) + 1e-12
    rel = np.sqrt(diff_sq / tgt_sq)                              # (B, T)
    return rel[:, target_step]                                   # (B,)


def main():
    # Build {fam: list of (rel_l2, target_field, pred_field)} from
    # all 3 seeds' viz_samples.
    per_fam = {f: [] for f in FAMS}
    for fam in FAMS:
        for seed in SEEDS:
            p = BASE / fam / "clean" / "lemo_pc_nd" / seed / "viz_samples.npz"
            if not p.exists():
                continue
            d = np.load(p)
            target = d["target"]    # (B, T, H, W, C)
            pred = d["pred"]
            rels = _per_sample_rel(d, target_step=-1)
            for i in range(target.shape[0]):
                per_fam[fam].append((float(rels[i]),
                                       target[i, -1, ..., 0],
                                       pred[i, -1, ..., 0]))
    # For each family, pick easiest and hardest by relL2.
    rows = []
    for fam in FAMS:
        if not per_fam[fam]:
            continue
        sorted_samples = sorted(per_fam[fam], key=lambda t: t[0])
        easy = sorted_samples[0]
        hard = sorted_samples[-1]
        rows.append((fam, easy, hard))
    if not rows:
        print("[M3] no viz_samples found")
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(11, 1.95 * n),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.10})
    if n == 1:
        axes = axes.reshape(1, 6)
    col_titles = ["GT", "LEMO-PC", r"$|\hat{u}-u|$",
                  "GT", "LEMO-PC", r"$|\hat{u}-u|$"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)
    # Two block-headers above the columns.
    fig.text(0.27, 0.99, "easiest sample", ha="center", fontsize=11, weight="bold")
    fig.text(0.74, 0.99, "hardest sample", ha="center", fontsize=11, weight="bold")
    for i, (fam, easy, hard) in enumerate(rows):
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
        # easy: cols 0,1,2
        e_rel, e_tgt, e_pred = easy
        h_rel, h_tgt, h_pred = hard
        e_vmax = max(np.abs(e_tgt).max(), np.abs(e_pred).max())
        h_vmax = max(np.abs(h_tgt).max(), np.abs(h_pred).max())
        axes[i, 0].imshow(e_tgt, cmap="RdBu_r", vmin=-e_vmax, vmax=e_vmax)
        axes[i, 1].imshow(e_pred, cmap="RdBu_r", vmin=-e_vmax, vmax=e_vmax)
        e_err = e_pred - e_tgt
        e_err_max = np.abs(e_err).max()
        if e_err_max > 0:
            axes[i, 2].imshow(e_err, cmap="RdBu_r",
                                norm=SymLogNorm(linthresh=1e-3 * e_err_max,
                                                vmin=-e_err_max, vmax=e_err_max))
        else:
            axes[i, 2].imshow(e_err, cmap="RdBu_r")
        # hard: cols 3,4,5
        axes[i, 3].imshow(h_tgt, cmap="RdBu_r", vmin=-h_vmax, vmax=h_vmax)
        axes[i, 4].imshow(h_pred, cmap="RdBu_r", vmin=-h_vmax, vmax=h_vmax)
        h_err = h_pred - h_tgt
        h_err_max = np.abs(h_err).max()
        if h_err_max > 0:
            axes[i, 5].imshow(h_err, cmap="RdBu_r",
                                norm=SymLogNorm(linthresh=1e-3 * h_err_max,
                                                vmin=-h_err_max, vmax=h_err_max))
        else:
            axes[i, 5].imshow(h_err, cmap="RdBu_r")
        # Family label + relL2 annotations.
        axes[i, 0].set_ylabel(FAM_LABELS[fam], rotation=0, ha="right",
                                va="center", fontsize=10)
        axes[i, 2].text(0.02, 0.98, f"$\\ell_2 = {e_rel:.3f}$",
                          transform=axes[i, 2].transAxes,
                          fontsize=8, va="top", ha="left",
                          bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="white", alpha=0.85, edgecolor="none"))
        axes[i, 5].text(0.02, 0.98, f"$\\ell_2 = {h_rel:.3f}$",
                          transform=axes[i, 5].transAxes,
                          fontsize=8, va="top", ha="left",
                          bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="white", alpha=0.85, edgecolor="none"))
        # Subtle vertical separator between easy and hard halves.
        for ax in (axes[i, 3],):
            ax.spines["left"].set_color("black")
            ax.spines["left"].set_linewidth(1.2)
    out = FIG / "M3_failure_mode_gallery.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
