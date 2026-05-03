"""Visual figures (V-series) — actual PDE field visualizations.

Replaces the numbers-only F01/F02/F04 charts (data already in T01/T02
tables).  Builds NO-paper-Figure-4-style field viz from `viz_samples.npz`
and `kernel_snapshot.npz` extracted at
extracted/pod1/outputs/dist_kernel_v2_p1/raw/<fam>/clean/lemo_pc_nd/<seed>/.

Produces (in paper/figures/):
  V01_family_triptych.{pdf,png}    5 fams x [input | target | LEMO-PC pred] (final frame)
  V02_rollout_sequence.{pdf,png}   one family, [target | pred] x t=0,16,32,48,63
  V03_error_maps.{pdf,png}         5 fams x |target - pred|, signed log scale
  V04_spectral_kernel.{pdf,png}    spectral kernel magnitude per (in, out, mode) for one family
  V05_kernel_recovery.{pdf,png}    learned vs ground-truth distributed-delay kernel (5 panels)
  V06_residual_fft.{pdf,png}       residual FFT energy per spectral lag mode

All viz uses RdBu_r diverging colormap with shared per-panel limits where
appropriate, no legends inside panels, tight whitespace.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap, FuncNorm

# Custom pastel-diverging colormap matching the localNO paper aesthetic.
# Endpoints are light salmon / light blue rather than saturated red/blue.
PASTEL_DIV = LinearSegmentedColormap.from_list(
    "pastel_div",
    [(0.00, "#0b3d6b"),  # deep navy blue (strong features pop)
     (0.18, "#3e6fa3"),  # medium blue
     (0.40, "#bccfde"),  # pale blue
     (0.50, "#f4ece8"),  # near-white at zero
     (0.60, "#e8a48f"),  # light salmon
     (0.82, "#bf5a45"),  # medium red
     (1.00, "#7a1f10")]  # deep red (strong features pop)
)
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


def load_viz(fam, regime="clean", seed="s42"):
    p = BASE / fam / regime / "lemo_pc_nd" / seed / "viz_samples.npz"
    if not p.exists():
        return None
    return np.load(p)


def load_kernel(fam, regime="clean", seed="s42"):
    p = BASE / fam / regime / "lemo_pc_nd" / seed / "kernel_snapshot.npz"
    if not p.exists():
        return None
    return np.load(p)


def _sym_lim(arr):
    v = np.nanmax(np.abs(arr))
    return -v, v


FNO_BASE = REPO / "outputs" / "film_ablation_caltech" / "raw"


def load_viz_fno(fam, regime="clean", seed="s42"):
    """Load FNO+FiLM viz_samples.npz from local outputs/."""
    p = FNO_BASE / fam / regime / "fno_film_nd" / seed / "viz_samples.npz"
    if not p.exists():
        return None
    return np.load(p)


def fig_v01_family_triptych(target_step: int = -1, hist_step: int = 0,
                              sample_idx: int = 0):
    """Per-family field viz — generates two variants:

      V01_family_triptych.{pdf,png}      3 rows × N cols
        rows = [GT, LEMO-PC pred, FNO+FiLM pred]; each cell is the field as
        a heatmap with GT iso-contours overlaid (so you can see where each
        prediction tracks the GT structure).

      V01_family_triptych_diff.{pdf,png} 3 rows × N cols
        rows = [LEMO-PC pred, FNO+FiLM pred, signed error |LEMO err| - |FNO err|]
        bottom row = where LEMO beats FNO (negative, blue) vs where FNO beats
        LEMO (positive, red). GT iso-contours overlaid on bottom row.

    Background: high-fidelity RdBu_r diverging colormap (matches the localNO
    aesthetic). 32× cubic-spline upsample of the 64×64 field (2048×2048
    source per panel) for the very-fine-grained look.
    """
    from scipy.ndimage import zoom
    import matplotlib.patheffects as pe
    halo = pe.withStroke(linewidth=3.5, foreground="white")

    UPSAMPLE = 32

    panels = []
    for fam in FAMS:
        dl = load_viz(fam)
        if dl is None:
            continue
        df = load_viz_fno(fam)
        y_gt = dl["target"][sample_idx][target_step, ..., 0]
        y_lemo = dl["pred"][sample_idx][target_step, ..., 0]
        y_fno = (df["pred"][sample_idx][target_step, ..., 0]
                  if df is not None else None)
        panels.append((fam, y_gt, y_lemo, y_fno))
    if not panels:
        return None
    n = len(panels)

    def _draw_heatmap(ax, field, vmax, cmap=PASTEL_DIV, scale="linear",
                        headroom=1.0, return_im=False):
        """Upsampled imshow of `field`. scale in {'linear', 'sqrt', 'symlog'}.
        sqrt = signed sqrt-of-magnitude (FuncNorm) — good middle ground:
        amplifies small errors moderately while keeping the colorbar
        showing real values."""
        f_hi = zoom(field, UPSAMPLE, order=3)
        H, W = field.shape
        v = vmax * headroom
        extent = [-0.5, W - 0.5, H - 0.5, -0.5]
        if scale == "symlog":
            linthresh = max(1e-3 * v, 1e-9)
            im = ax.imshow(f_hi, cmap=cmap,
                            norm=SymLogNorm(linthresh=linthresh,
                                             vmin=-v, vmax=v, base=10),
                            interpolation="bilinear", extent=extent)
        elif scale == "sqrt":
            fwd = lambda x: np.sign(x) * np.sqrt(np.abs(x))
            inv = lambda x: np.sign(x) * (x ** 2)
            im = ax.imshow(f_hi, cmap=cmap,
                            norm=FuncNorm((fwd, inv), vmin=-v, vmax=v),
                            interpolation="bilinear", extent=extent)
        else:  # linear
            im = ax.imshow(f_hi, cmap=cmap, vmin=-v, vmax=v,
                           interpolation="bilinear", extent=extent)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if return_im:
            return im

    def _overlay_gt_contours(ax, y_gt, vmax):
        H, W = y_gt.shape
        pos_levels = np.linspace(0.15 * vmax, 0.85 * vmax, 4)
        levels = np.concatenate([-pos_levels[::-1], pos_levels])
        cs = ax.contour(np.arange(W), np.arange(H), y_gt, levels=levels,
                         colors="black", linewidths=1.5)
        try:
            cs.set(path_effects=[halo])
        except Exception:
            for c in getattr(cs, "collections", []):
                c.set_path_effects([halo])

    # ----- Variant A: GT / LEMO / FNO heatmaps (3 rows × N cols) -----
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10.5),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.07})
    if n == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["Ground Truth", "LEMO-PC", "FNO+FiLM"]
    for i, lbl in enumerate(row_labels):
        axes[i, 0].set_ylabel(lbl, fontsize=14, rotation=0, ha="right",
                                va="center", labelpad=20)
    for j, (fam, y_gt, y_lemo, y_fno) in enumerate(panels):
        all_arrs = [y_gt, y_lemo] + ([y_fno] if y_fno is not None else [])
        vmax = float(np.max([np.abs(v).max() for v in all_arrs]))
        axes[0, j].set_title(FAM_LABELS[fam], fontsize=14)
        _draw_heatmap(axes[0, j], y_gt, vmax)
        _overlay_gt_contours(axes[0, j], y_gt, vmax)
        _draw_heatmap(axes[1, j], y_lemo, vmax)
        _overlay_gt_contours(axes[1, j], y_gt, vmax)
        if y_fno is not None:
            _draw_heatmap(axes[2, j], y_fno, vmax)
            _overlay_gt_contours(axes[2, j], y_gt, vmax)
        else:
            axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
            for sp in axes[2, j].spines.values():
                sp.set_linewidth(0.6)
            axes[2, j].text(0.5, 0.5, "n/a", ha="center", va="center",
                              transform=axes[2, j].transAxes,
                              color="dimgrey", fontsize=12)

    fig.suptitle("Predicted vs GT fields", fontsize=18, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "V01_family_triptych.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----- Variant B: GT / LEMO Error / Err Diff (3 rows × N cols) -----
    # Compute SHARED per-row vmax across all families so colors are comparable
    # within a row, and one colorbar per row is enough.
    gt_vmax = float(np.max([np.abs(v).max() for v in
                              [p[1] for p in panels]]))
    lemo_err_vmax = float(np.max([np.abs(p[2] - p[1]).max() for p in panels]))
    diffs = []
    for p in panels:
        if p[3] is not None:
            diffs.append(np.abs(p[2] - p[1]) - np.abs(p[3] - p[1]))
    diff_vmax = float(np.max([np.abs(d).max() for d in diffs])) if diffs else 1e-9

    fig2, axes2 = plt.subplots(3, n, figsize=(3.5 * n, 10.5),
                                  gridspec_kw={"wspace": 0.04, "hspace": 0.07,
                                               "right": 0.92})
    if n == 1:
        axes2 = axes2.reshape(3, 1)
    row_labels2 = ["Ground Truth", "LEMO Error", "Error Difference"]
    for i, lbl in enumerate(row_labels2):
        axes2[i, 0].set_ylabel(lbl, fontsize=14, rotation=0, ha="right",
                                 va="center", labelpad=20)

    im_gt = im_le = im_dd = None
    for j, (fam, y_gt, y_lemo, y_fno) in enumerate(panels):
        axes2[0, j].set_title(FAM_LABELS[fam], fontsize=14)
        # Row 1: ground truth field — linear, no headroom (full contrast)
        im = _draw_heatmap(axes2[0, j], y_gt, gt_vmax,
                            headroom=1.0, return_im=True)
        if j == n - 1: im_gt = im
        _overlay_gt_contours(axes2[0, j], y_gt, gt_vmax)
        # Row 2: signed LEMO error — sqrt power-norm (moderate amplification)
        lemo_err_signed = y_lemo - y_gt
        im = _draw_heatmap(axes2[1, j], lemo_err_signed, lemo_err_vmax,
                            scale="sqrt", return_im=True)
        if j == n - 1: im_le = im
        _overlay_gt_contours(axes2[1, j], y_gt, lemo_err_vmax)
        # Row 3: |LEMO err| − |FNO err| — sqrt power-norm
        if y_fno is not None:
            diff = np.abs(y_lemo - y_gt) - np.abs(y_fno - y_gt)
            im = _draw_heatmap(axes2[2, j], diff, diff_vmax,
                                scale="sqrt", return_im=True)
            if j == n - 1: im_dd = im
            _overlay_gt_contours(axes2[2, j], y_gt, diff_vmax)
        else:
            axes2[2, j].set_xticks([]); axes2[2, j].set_yticks([])
            for sp in axes2[2, j].spines.values():
                sp.set_linewidth(0.6)
            axes2[2, j].text(0.5, 0.5, "n/a", ha="center", va="center",
                               transform=axes2[2, j].transAxes,
                               color="dimgrey", fontsize=12)
    # Shared colorbars per row, anchored to the rightmost panel
    for ax_row, im_row, lbl in [(axes2[0, -1], im_gt, "field"),
                                  (axes2[1, -1], im_le, "pred − GT"),
                                  (axes2[2, -1], im_dd, "|LEMO err| − |FNO err|")]:
        if im_row is None: continue
        cb = fig2.colorbar(im_row, ax=ax_row, fraction=0.05, pad=0.04,
                            shrink=0.95, aspect=18)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(lbl, fontsize=9)

    fig2.suptitle("Predictions", fontsize=18, y=0.99)
    fig2.tight_layout(rect=[0, 0, 0.92, 0.96])
    out2 = FIG / "V01_family_triptych_diff.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    fig2.savefig(out2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    return out


def fig_v02_rollout_sequence(fam_pick="dist_gaussian_rd_2d",
                                t_steps=(0, 16, 32, 48, 63),
                                sample_idx: int = 0):
    """V02 rollout-sequence figures (two outputs):

      V02_rollout_sequence.{pdf,png}
        Option A — 2 rows × 5 cols on a single family (default Gauss = LEMO's
        hardest family per T01). Top row = GT, bottom row = Error Difference
        (|LEMO err| − |FNO err|). Aesthetic matches V01_diff (RdBu_r, 32×
        cubic-spline upsample, GT iso-contours with white halo).

      V02_rollout_grid.{pdf,png}
        Option B — 5 rows (family) × 5 cols (timestep) of Error Difference.
        Covers all families in one shot; no cherry-picking. Each cell is a
        compact heatmap; consistent diff_vmax across cells of the same family.
    """
    from scipy.ndimage import zoom
    import matplotlib.patheffects as pe
    halo = pe.withStroke(linewidth=3.5, foreground="white")
    UPSAMPLE = 32

    # Pre-load all 5 families (LEMO + FNO+FiLM) so both A and B have data.
    fam_data = {}
    for fam in FAMS:
        dl = load_viz(fam)
        df = load_viz_fno(fam)
        if dl is None:
            continue
        y = dl["target"][sample_idx]
        yhat_l = dl["pred"][sample_idx]
        yhat_f = df["pred"][sample_idx] if df is not None else None
        T_full = y.shape[0]
        n_hist = T_full // 2
        # Absolute time indices for the requested forecast t_steps
        t_abs = [n_hist + t for t in t_steps if (n_hist + t) < T_full]
        t_lbls = [t for t in t_steps if (n_hist + t) < T_full]
        fam_data[fam] = (y, yhat_l, yhat_f, t_abs, t_lbls)
    if not fam_data:
        return None

    def _draw_heatmap(ax, field, vmax, cmap=PASTEL_DIV, scale="linear",
                       headroom=1.0, return_im=False):
        f_hi = zoom(field, UPSAMPLE, order=3)
        H, W = field.shape
        v = vmax * headroom
        extent = [-0.5, W - 0.5, H - 0.5, -0.5]
        if scale == "symlog":
            linthresh = max(1e-3 * v, 1e-9)
            im = ax.imshow(f_hi, cmap=cmap,
                            norm=SymLogNorm(linthresh=linthresh,
                                             vmin=-v, vmax=v, base=10),
                            interpolation="bilinear", extent=extent)
        elif scale == "sqrt":
            fwd = lambda x: np.sign(x) * np.sqrt(np.abs(x))
            inv = lambda x: np.sign(x) * (x ** 2)
            im = ax.imshow(f_hi, cmap=cmap,
                            norm=FuncNorm((fwd, inv), vmin=-v, vmax=v),
                            interpolation="bilinear", extent=extent)
        else:
            im = ax.imshow(f_hi, cmap=cmap, vmin=-v, vmax=v,
                           interpolation="bilinear", extent=extent)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if return_im:
            return im

    def _overlay_gt_contours(ax, y_gt, vmax):
        H, W = y_gt.shape
        pos_levels = np.linspace(0.15 * vmax, 0.85 * vmax, 4)
        levels = np.concatenate([-pos_levels[::-1], pos_levels])
        cs = ax.contour(np.arange(W), np.arange(H), y_gt, levels=levels,
                         colors="black", linewidths=1.4)
        try:
            cs.set(path_effects=[halo])
        except Exception:
            for c in getattr(cs, "collections", []):
                c.set_path_effects([halo])

    # ====================================================================
    # Option A: single family (Gauss by default), 2 rows × 5 cols
    # ====================================================================
    if fam_pick in fam_data:
        y, yhat_l, yhat_f, t_abs, t_lbls = fam_data[fam_pick]
        n_t = len(t_abs)
        if yhat_f is not None and n_t > 0:
            figA, axA = plt.subplots(2, n_t, figsize=(3.2 * n_t, 6.4),
                                       gridspec_kw={"wspace": 0.04,
                                                    "hspace": 0.07})
            if n_t == 1:
                axA = axA.reshape(2, 1)
            row_labels = ["Ground Truth", "Error Difference"]
            for i, lbl in enumerate(row_labels):
                axA[i, 0].set_ylabel(lbl, fontsize=14, rotation=0,
                                       ha="right", va="center", labelpad=20)
            for j, (t, lbl) in enumerate(zip(t_abs, t_lbls)):
                y_gt = y[t, ..., 0]
                y_l = yhat_l[t, ..., 0]
                y_f = yhat_f[t, ..., 0]
                vmax = float(np.max([np.abs(v).max()
                                       for v in (y_gt, y_l, y_f)]))
                axA[0, j].set_title(f"t={lbl}", fontsize=13)
                _draw_heatmap(axA[0, j], y_gt, vmax, headroom=1.0)
                _overlay_gt_contours(axA[0, j], y_gt, vmax)
                err_l = np.abs(y_l - y_gt)
                err_f = np.abs(y_f - y_gt)
                diff = err_l - err_f
                diff_vmax = float(max(np.max(np.abs(diff)), 1e-9))
                _draw_heatmap(axA[1, j], diff, diff_vmax, scale="sqrt")
                _overlay_gt_contours(axA[1, j], y_gt, diff_vmax)
            figA.suptitle(f"Rollout: {FAM_LABELS[fam_pick]}",
                          fontsize=18, y=0.99)
            figA.tight_layout(rect=[0, 0, 1, 0.96])
            out = FIG / "V02_rollout_sequence.pdf"
            figA.savefig(out, bbox_inches="tight")
            figA.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(figA)

    # ====================================================================
    # Option B: 5 families × 5 timesteps grid of Error Difference only
    # ====================================================================
    fams_with_both = [f for f in FAMS
                       if f in fam_data and fam_data[f][2] is not None
                       and len(fam_data[f][3]) > 0]
    if fams_with_both:
        # Use the timesteps from the first family (all should match).
        _, _, _, ref_t_abs, ref_t_lbls = fam_data[fams_with_both[0]]
        n_t = len(ref_t_abs)
        n_f = len(fams_with_both)
        figB, axB = plt.subplots(n_f, n_t,
                                   figsize=(2.2 * n_t, 2.2 * n_f),
                                   gridspec_kw={"wspace": 0.04,
                                                "hspace": 0.06})
        if n_f == 1:
            axB = axB.reshape(1, n_t)
        for i, fam in enumerate(fams_with_both):
            y, yhat_l, yhat_f, t_abs, t_lbls = fam_data[fam]
            # diff_vmax shared across this family's timesteps for fairness
            diffs = []
            for t in t_abs:
                yg = y[t, ..., 0]
                el = np.abs(yhat_l[t, ..., 0] - yg)
                ef = np.abs(yhat_f[t, ..., 0] - yg)
                diffs.append(el - ef)
            diff_vmax = float(max(np.max(np.abs(np.array(diffs))), 1e-9))
            for j, (t, lbl, diff) in enumerate(zip(t_abs, t_lbls, diffs)):
                if i == 0:
                    axB[i, j].set_title(f"t={lbl}", fontsize=12)
                _draw_heatmap(axB[i, j], diff, diff_vmax, scale="sqrt")
                _overlay_gt_contours(axB[i, j], y[t, ..., 0],
                                       diff_vmax)
            axB[i, 0].set_ylabel(FAM_LABELS[fam], fontsize=12, rotation=0,
                                   ha="right", va="center", labelpad=12)
        figB.suptitle("Error Difference grid (rollout × family)",
                       fontsize=16, y=0.995)
        figB.tight_layout(rect=[0, 0, 1, 0.97])
        outB = FIG / "V02_rollout_grid.pdf"
        figB.savefig(outB, bbox_inches="tight")
        figB.savefig(outB.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(figB)

    return FIG / "V02_rollout_sequence.pdf"


def fig_v03_error_maps(target_step: int = -1):
    """5 fams: |target - pred| at last frame (signed-log scale to reveal small errors)."""
    rows = []
    for fam in FAMS:
        d = load_viz(fam)
        if d is None:
            continue
        y = d["target"][0, target_step, ..., 0]
        yh = d["pred"][0, target_step, ..., 0]
        err = yh - y
        rows.append((fam, err))
    if not rows:
        return None
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6),
                              gridspec_kw={"wspace": 0.05})
    if n == 1:
        axes = [axes]
    vmax = max(np.abs(e).max() for _, e in rows)
    for ax, (fam, err) in zip(axes, rows):
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(err, cmap="RdBu_r",
                        norm=SymLogNorm(linthresh=1e-3 * vmax, vmin=-vmax, vmax=vmax))
        ax.set_title(FAM_LABELS[fam], fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85,
                         orientation="vertical")
    cbar.set_label(r"$\hat{u} - u$ (signed log)", fontsize=9)
    fig.suptitle(f"Per-family prediction error at final rollout step (t={target_step})",
                 fontsize=11)
    out = FIG / "V03_error_maps.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_v04_spectral_kernel(fam_pick="dist_exp_rd_2d"):
    """Spectral kernel magnitude per (in, out, mode) for one family.

    Shows the LEMO-PC learned spectral lag kernel.  Two heatmaps:
      LEFT:   per-mode operator norm sigma_max(K[:,:,m])  vs m  (line plot)
      RIGHT:  |K|_F per mode m, broken down by in→out channel (small heatmap)
    """
    d = load_kernel(fam_pick)
    if d is None:
        return None
    # Kernel is stored as cfloat → split into 'weights__re' and 'weights__im'.
    keys = list(d.keys())
    re_keys = [k for k in keys if k.endswith("__re") and "weights" in k and "film" not in k]
    if not re_keys:
        return None
    # Take first such kernel (layer 0).
    re = d[re_keys[0]]
    im_key = re_keys[0].replace("__re", "__im")
    if im_key not in d:
        return None
    im = d[im_key]
    K = re + 1j * im     # shape (in, out, modes)
    in_ch, out_ch, M = K.shape
    # Per-mode op norm = max singular value of K[:,:,m]
    op_norms = np.zeros(M)
    for m in range(M):
        op_norms[m] = np.linalg.norm(K[:, :, m], ord=2)
    # |K|_F per (in, out): sqrt(sum_m |K[i,o,m]|^2)
    fro = np.sqrt((np.abs(K) ** 2).sum(axis=-1))   # (in, out)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2),
                                     gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(np.arange(M), op_norms, marker="o", markersize=3, color="#d62728", lw=1.4)
    ax1.set_xlabel(r"spectral mode $m$")
    ax1.set_ylabel(r"$\sigma_{\max}(K[:,:,m])$")
    ax1.set_title(f"Per-mode op-norm ({FAM_LABELS[fam_pick]})")
    ax1.grid(linestyle="--", alpha=0.4)
    im_obj = ax2.imshow(fro, cmap="viridis", aspect="auto")
    ax2.set_title(r"$\|K_{i,o,\cdot}\|_F$")
    ax2.set_xlabel("output ch")
    ax2.set_ylabel("input ch")
    cbar = fig.colorbar(im_obj, ax=ax2, fraction=0.04, pad=0.04)
    fig.tight_layout()
    out = FIG / "V04_spectral_kernel.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig_v05_kernel_recovery():
    """Time-domain learned kernel vs ground-truth distributed-delay kernel (5 panels)."""
    rows = []
    for fam in FAMS:
        d = load_kernel(fam)
        if d is None:
            continue
        # Find spectral kernel
        keys = list(d.keys())
        re_keys = [k for k in keys if k.endswith("__re") and "weights" in k and "film" not in k]
        if not re_keys:
            continue
        re = d[re_keys[0]]
        im_key = re_keys[0].replace("__re", "__im")
        if im_key not in d:
            continue
        K = re + 1j * d[im_key]
        in_ch, out_ch, M = K.shape
        # K is the TRUNCATED spectral kernel: first M modes of an L_time-long
        # signal, NOT the full spectrum of a length-(2M-2) signal. To recover
        # the time-domain kernel, pad K to the full rfft length L_time//2+1
        # and irfft to length L_time. n_total=128 matches training (n_hist+n_out
        # at 64+64); kernel only depends on lag axis length, not spatial.
        L_time = 128
        n_modes_full = L_time // 2 + 1
        K_full = np.zeros((in_ch, out_ch, n_modes_full), dtype=K.dtype)
        n_keep = min(M, n_modes_full)
        K_full[..., :n_keep] = K[..., :n_keep]
        K_t = np.fft.irfft(K_full, n=L_time, axis=-1)  # (in, out, L_time)
        K_amp = np.abs(K_t).mean(axis=(0, 1))           # (L_time,)
        L = L_time   # downstream uses L for the GT-sampling axis
        # Ground truth shape (normalized).
        t = np.arange(L) / max(L - 1, 1)
        if fam.startswith("dist_exp"):
            gt = np.exp(-3 * t)
        elif fam.startswith("dist_gaussian"):
            gt = np.exp(-((t - 0.3) ** 2) / 0.05)
        elif fam.startswith("dist_gamma"):
            gt = (t ** 1.5) * np.exp(-3 * t)
        elif fam.startswith("dist_uniform"):
            gt = (t < 0.5).astype(np.float32)
        elif fam.startswith("dist_powerlaw"):
            gt = (t + 0.05) ** (-1.2)
        else:
            gt = np.zeros_like(t)
        if gt.sum() > 0:
            gt = gt / gt.sum()
        K_amp_n = K_amp / (K_amp.max() + 1e-12)
        gt_n = gt / (gt.max() + 1e-12)
        a = K_amp_n / (np.linalg.norm(K_amp_n) + 1e-12)
        b = gt_n / (np.linalg.norm(gt_n) + 1e-12)
        cos = float((a * b).sum())
        rows.append((fam, t, K_amp_n, gt_n, cos))
    if not rows:
        return None
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6),
                              gridspec_kw={"wspace": 0.25}, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (fam, t, K_amp, gt, cos) in zip(axes, rows):
        ax.plot(t, K_amp, color="#d62728", lw=1.5, label="learned")
        ax.plot(t, gt, color="black", lw=1.0, linestyle="--", label="ground truth")
        ax.set_title(f"{FAM_LABELS[fam]}\nCosSim = {cos:.2f}", fontsize=9)
        ax.set_xlabel(r"normalized lag $t$")
        ax.grid(linestyle="--", alpha=0.4)
    axes[0].set_ylabel("normalized |kernel|")
    axes[-1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=False)
    out = FIG / "V05_kernel_recovery.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_v06_residual_fft():
    """Residual FFT energy per spectral lag mode, computed directly from
    viz_samples (target - pred) across families and seeds."""
    series = {}
    for fam in FAMS:
        for seed in SEEDS:
            d = load_viz(fam, "clean", seed)
            if d is None:
                continue
            target = d["target"]    # (B, T, *spatial, C)
            pred = d["pred"]
            r = pred - target
            # FFT along time axis: move T to last
            perm = [0] + list(range(2, r.ndim)) + [1]
            r_p = np.transpose(r, axes=perm)        # (..., T)
            R = np.fft.rfft(r_p, axis=-1)
            energy = np.mean(np.abs(R) ** 2, axis=tuple(range(R.ndim - 1)))
            series.setdefault(fam, []).append(energy)
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for fam, es in series.items():
        e = np.array(es)
        m = e.mean(axis=0)
        s = e.std(axis=0)
        modes = np.arange(len(m))
        ax.plot(modes, m, lw=1.5, label=FAM_LABELS[fam])
        ax.fill_between(modes, m - s, m + s, alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("spectral lag mode $m$")
    ax.set_ylabel(r"$\mathbb{E}\,|\hat{r}_m|^2$")
    ax.set_title("Residual FFT energy per spectral lag mode (mean $\\pm$ std over seeds)")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=False)
    ax.grid(linestyle="--", alpha=0.4)
    out = FIG / "V06_residual_fft.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("[viz-figs] generating V01-V06 from extracted/pod1/dist_kernel_v2_p1/")
    n_viz = sum(1 for fam in FAMS for s in SEEDS
                if (BASE / fam / "clean" / "lemo_pc_nd" / s / "viz_samples.npz").exists())
    n_kern = sum(1 for fam in FAMS for s in SEEDS
                 if (BASE / fam / "clean" / "lemo_pc_nd" / s / "kernel_snapshot.npz").exists())
    print(f"  viz_samples.npz cells found:    {n_viz}/15")
    print(f"  kernel_snapshot.npz cells found: {n_kern}/15")
    out_files = []
    for name, fn in [
        ("V01 family triptych",     fig_v01_family_triptych),
        ("V02 rollout sequence",    fig_v02_rollout_sequence),
        # V03 dropped: signed LEMO error is now row 2 of V01_family_triptych_diff.
        ("V04 spectral kernel",     fig_v04_spectral_kernel),
        ("V05 kernel recovery",     fig_v05_kernel_recovery),
        ("V06 residual FFT",        fig_v06_residual_fft),
    ]:
        try:
            out = fn()
        except Exception as e:
            print(f"  {name:<24}: FAIL ({type(e).__name__}: {e})")
            continue
        if out is None:
            print(f"  {name:<24}: skip (data missing)")
        else:
            out_files.append(out)
            print(f"  {name:<24}: -> {out.name}")
    print(f"\n[viz-figs] generated {len(out_files)} figures in {FIG}")


if __name__ == "__main__":
    main()
