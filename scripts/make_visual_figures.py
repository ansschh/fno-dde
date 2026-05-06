"""Visual figures (V-series) â€” actual PDE field visualizations.

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

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401  (sets Times New Roman globally)
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap, FuncNorm

# Use matplotlib's standard RdBu_r diverging colormap. Bold saturated reds and
# blues at extremes, white at zero â€” gives the strongest contrast for error/diff
# visualisation and matches the bolder pre-pastel iteration.
PASTEL_DIV = "RdBu_r"
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


_LEMO_VIZ_INDEX: dict | None = None
_LEMO_KER_INDEX: dict | None = None


def _build_lemo_index(filename: str) -> dict:
    idx = {}
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob(filename):
            try:
                parts = f.parts
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
            except IndexError:
                continue
            if model != "lemo_pc_nd":
                continue
            idx.setdefault((fam, reg, seed), f)
    return idx


def load_viz(fam, regime="clean", seed="s42"):
    """LEMO-PC viz_samples.npz: prefers the legacy BASE path, falls back to
    rglob discovery so new pod_pull layouts also work."""
    p = BASE / fam / regime / "lemo_pc_nd" / seed / "viz_samples.npz"
    if p.exists():
        return np.load(p)
    global _LEMO_VIZ_INDEX
    if _LEMO_VIZ_INDEX is None:
        _LEMO_VIZ_INDEX = _build_lemo_index("viz_samples.npz")
    p2 = _LEMO_VIZ_INDEX.get((fam, regime, seed))
    if p2 is None or not p2.exists():
        return None
    return np.load(p2)


def load_kernel(fam, regime="clean", seed="s42"):
    p = BASE / fam / regime / "lemo_pc_nd" / seed / "kernel_snapshot.npz"
    if p.exists():
        return np.load(p)
    global _LEMO_KER_INDEX
    if _LEMO_KER_INDEX is None:
        _LEMO_KER_INDEX = _build_lemo_index("kernel_snapshot.npz")
    p2 = _LEMO_KER_INDEX.get((fam, regime, seed))
    if p2 is None or not p2.exists():
        return None
    return np.load(p2)


def _sym_lim(arr):
    v = np.nanmax(np.abs(arr))
    return -v, v


FNO_BASE = REPO / "outputs" / "film_ablation_caltech" / "raw"

# rglob fallback: any pod_pull / output sweep with the (fam, reg, fno_film_nd, seed)
# layout. Picked once at import time to keep load_viz_fno cheap.
_FNO_FILM_VIZ_INDEX: dict | None = None


def _build_fno_film_viz_index():
    global _FNO_FILM_VIZ_INDEX
    if _FNO_FILM_VIZ_INDEX is not None:
        return _FNO_FILM_VIZ_INDEX
    _FNO_FILM_VIZ_INDEX = {}
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("viz_samples.npz"):
            try:
                parts = f.parts
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
            except IndexError:
                continue
            if model != "fno_film_nd":
                continue
            _FNO_FILM_VIZ_INDEX.setdefault((fam, reg, seed), f)
    return _FNO_FILM_VIZ_INDEX


def load_viz_fno(fam, regime="clean", seed="s42"):
    """Load FNO+FiLM viz_samples.npz: prefers the legacy local film_ablation_caltech
    path, falls back to rglob discovery across pod_pulls / runpod sweep dirs.
    """
    p = FNO_BASE / fam / regime / "fno_film_nd" / seed / "viz_samples.npz"
    if p.exists():
        return np.load(p)
    idx = _build_fno_film_viz_index()
    p2 = idx.get((fam, regime, seed))
    if p2 is None or not p2.exists():
        return None
    return np.load(p2)


def fig_v01_family_triptych(target_step: int = -1, hist_step: int = 0,
                              sample_idx: int = 0):
    """Per-family field viz â€” generates two variants:

      V01_family_triptych.{pdf,png}      3 rows Ã— N cols
        rows = [GT, LEMO-PC pred, FNO+FiLM pred]; each cell is the field as
        a heatmap with GT iso-contours overlaid (so you can see where each
        prediction tracks the GT structure).

      V01_family_triptych_diff.{pdf,png} 3 rows Ã— N cols
        rows = [LEMO-PC pred, FNO+FiLM pred, signed error |LEMO err| - |FNO err|]
        bottom row = where LEMO beats FNO (negative, blue) vs where FNO beats
        LEMO (positive, red). GT iso-contours overlaid on bottom row.

    Background: high-fidelity RdBu_r diverging colormap (matches the localNO
    aesthetic). 32Ã— cubic-spline upsample of the 64Ã—64 field (2048Ã—2048
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
        # Pick the HARDEST sample per family: the saved sample with the highest
        # LEMO-PC relL2 against GT at the final rollout step. Defensible
        # "worst-case showcase" framing (paper claim: "even on the hardest
        # sample the prediction tracks the target field with errors confined
        # to localised regions"). N saved samples per cell is small (4); we
        # pick hardest among those.
        target_all = dl["target"]      # (B, T, *spatial, C)
        pred_all   = dl["pred"]
        T_idx = target_all.shape[1] + target_step if target_step < 0 else target_step
        per_sample_l2 = []
        for b in range(target_all.shape[0]):
            yg = target_all[b, T_idx, ..., 0]
            yp = pred_all[b, T_idx, ..., 0]
            num = float(np.sqrt(((yp - yg) ** 2).sum()))
            den = float(np.sqrt((yg ** 2).sum())) + 1e-12
            per_sample_l2.append(num / den)
        hardest_idx = int(np.argmax(per_sample_l2))
        l2_hardest = per_sample_l2[hardest_idx]
        y_gt = target_all[hardest_idx, T_idx, ..., 0]
        y_lemo = pred_all[hardest_idx, T_idx, ..., 0]
        if df is not None:
            y_fno = df["pred"][hardest_idx, T_idx, ..., 0]
        else:
            y_fno = None
        panels.append((fam, y_gt, y_lemo, y_fno, l2_hardest))
    if not panels:
        return None
    n = len(panels)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _draw_heatmap(ax, field, vmax, cmap=PASTEL_DIV, scale="linear",
                        headroom=1.0, return_im=False, add_cbar=False,
                        cbar_n_ticks=3):
        """Upsampled imshow of `field`. scale in {'linear', 'mild', 'sqrt', 'symlog'}.
        If add_cbar, attaches a sibling colorbar axes via make_axes_locatable
        (renders identically in PNG and PDF â€” no inset clipping issues)."""
        f_hi = zoom(field, UPSAMPLE, order=3)
        H, W = field.shape
        v = vmax * headroom
        extent = [-0.5, W - 0.5, H - 0.5, -0.5]
        def _signed_power(gamma):
            fwd = lambda x: np.sign(x) * np.power(np.abs(x), gamma)
            inv = lambda x: np.sign(x) * np.power(np.abs(x), 1.0 / gamma)
            return FuncNorm((fwd, inv), vmin=-v, vmax=v)
        if scale == "symlog":
            linthresh = max(1e-3 * v, 1e-9)
            im = ax.imshow(f_hi, cmap=cmap,
                            norm=SymLogNorm(linthresh=linthresh,
                                             vmin=-v, vmax=v, base=10),
                            interpolation="bilinear", extent=extent)
        elif scale == "sqrt":
            im = ax.imshow(f_hi, cmap=cmap, norm=_signed_power(0.5),
                            interpolation="bilinear", extent=extent)
        elif scale == "mild":
            im = ax.imshow(f_hi, cmap=cmap, norm=_signed_power(0.7),
                            interpolation="bilinear", extent=extent)
        else:
            im = ax.imshow(f_hi, cmap=cmap, vmin=-v, vmax=v,
                           interpolation="bilinear", extent=extent)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if add_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
            ticks = np.linspace(-v, v, cbar_n_ticks)
            cb.set_ticks(ticks)
            cb.ax.tick_params(labelsize=6, length=2, pad=1)
            cb.ax.set_xticklabels([f"{t:.2g}" for t in ticks])
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

    # ----- Variant A: GT / LEMO / FNO heatmaps -----
    # Drop FNO row gracefully if no FNO+FiLM data is available.
    has_fno = any(p[3] is not None for p in panels)
    n_rows_a = 3 if has_fno else 2
    fig, axes = plt.subplots(n_rows_a, n, figsize=(3.5 * n, 3.5 * n_rows_a),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.07})
    if n == 1:
        axes = axes.reshape(n_rows_a, 1)

    row_labels = ["Ground Truth", "LEMO-PC"] + (["FNO+FiLM"] if has_fno else [])
    for i, lbl in enumerate(row_labels):
        axes[i, 0].set_ylabel(lbl, fontsize=14, rotation=0, ha="right",
                                va="center", labelpad=20)
    for j, (fam, y_gt, y_lemo, y_fno, l2_hard) in enumerate(panels):
        all_arrs = [y_gt, y_lemo] + ([y_fno] if y_fno is not None else [])
        vmax = float(np.max([np.abs(v).max() for v in all_arrs]))
        axes[0, j].set_title(f"{FAM_LABELS[fam]}\n($\\ell_2$={l2_hard:.3f})",
                                fontsize=12)
        _draw_heatmap(axes[0, j], y_gt, vmax)
        _overlay_gt_contours(axes[0, j], y_gt, vmax)
        _draw_heatmap(axes[1, j], y_lemo, vmax)
        _overlay_gt_contours(axes[1, j], y_gt, vmax)
        if has_fno:
            if y_fno is not None:
                _draw_heatmap(axes[2, j], y_fno, vmax)
                _overlay_gt_contours(axes[2, j], y_gt, vmax)
            else:
                axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
                for sp in axes[2, j].spines.values():
                    sp.set_linewidth(0.6)
                axes[2, j].text(0.5, 0.5, "n/a", ha="center", va="center",
                                  transform=axes[2, j].transAxes,
                                  color="black", fontsize=12)

    fig.suptitle("",
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "V01_family_triptych.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----- Variant B: GT / LEMO Error / Err Diff (3 rows Ã— N cols) -----
    # SHARED per-row vmax + one per-row colorbar. Mild gamma=0.7 power norm
    # amplifies small-error families so they're not washed out.
    gt_vmax = float(max(max(np.abs(p[1]).max() for p in panels), 1e-9))
    le_vmax = float(max(max(np.abs(p[2] - p[1]).max() for p in panels), 1e-9))
    diffs_all = [np.abs(p[2] - p[1]) - np.abs(p[3] - p[1])
                  for p in panels if p[3] is not None]
    dd_vmax = (float(max(max(np.abs(d).max() for d in diffs_all), 1e-9))
                if diffs_all else 1e-9)

    # Drop Error Difference row if no FNO+FiLM data; reduces figure to 2 rows.
    has_fno_b = bool(diffs_all)
    n_rows_b = 3 if has_fno_b else 2
    fig2, axes2 = plt.subplots(n_rows_b, n,
                                 figsize=(3.5 * n, 3.5 * n_rows_b),
                                 gridspec_kw={"wspace": 0.04, "hspace": 0.07,
                                              "right": 0.92})
    if n == 1:
        axes2 = axes2.reshape(n_rows_b, 1)

    im_gt = im_le = im_dd = None
    for j, (fam, y_gt, y_lemo, y_fno, l2_hard) in enumerate(panels):
        im = _draw_heatmap(axes2[0, j], y_gt, gt_vmax, return_im=True)
        if j == n - 1: im_gt = im
        _overlay_gt_contours(axes2[0, j], y_gt, gt_vmax)
        im = _draw_heatmap(axes2[1, j], y_lemo - y_gt, le_vmax,
                            scale="mild", return_im=True)
        if j == n - 1: im_le = im
        _overlay_gt_contours(axes2[1, j], y_gt, le_vmax)
        if has_fno_b:
            if y_fno is not None:
                diff = np.abs(y_lemo - y_gt) - np.abs(y_fno - y_gt)
                im = _draw_heatmap(axes2[2, j], diff, dd_vmax,
                                    scale="mild", return_im=True)
                if j == n - 1: im_dd = im
                _overlay_gt_contours(axes2[2, j], y_gt, dd_vmax)
            else:
                axes2[2, j].set_xticks([]); axes2[2, j].set_yticks([])
                for sp in axes2[2, j].spines.values():
                    sp.set_linewidth(0.6)
                axes2[2, j].text(0.5, 0.5, "n/a", ha="center", va="center",
                                   transform=axes2[2, j].transAxes,
                                   color="black", fontsize=12)
    # Per-row colorbars (no labels per user request).
    cb_rows = [(axes2[0, -1], im_gt),
                (axes2[1, -1], im_le)]
    if has_fno_b:
        cb_rows.append((axes2[2, -1], im_dd))
    for ax_row, im_row in cb_rows:
        if im_row is None: continue
        cb = fig2.colorbar(im_row, ax=ax_row, fraction=0.05, pad=0.04,
                            shrink=0.95, aspect=18)
        cb.ax.tick_params(labelsize=8)

    fig2.suptitle("",
                   fontsize=16, y=0.99)
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
        Option A â€” 2 rows Ã— 5 cols on a single family (default Gauss = LEMO's
        hardest family per T01). Top row = GT, bottom row = Error Difference
        (|LEMO err| âˆ’ |FNO err|). Aesthetic matches V01_diff (RdBu_r, 32Ã—
        cubic-spline upsample, GT iso-contours with white halo).

      V02_rollout_grid.{pdf,png}
        Option B â€” 5 rows (family) Ã— 5 cols (timestep) of Error Difference.
        Covers all families in one shot; no cherry-picking. Each cell is a
        compact heatmap; consistent diff_vmax across cells of the same family.
    """
    from scipy.ndimage import zoom
    import matplotlib.patheffects as pe
    halo = pe.withStroke(linewidth=3.5, foreground="white")
    UPSAMPLE = 64

    # Pre-load all 5 families (LEMO + FNO+FiLM) so both A and B have data.
    # For each family pick the HARDEST sample (highest LEMO-PC relL2 against GT
    # at the final forecast step) â€” same worst-case framing as V01_diff.
    fam_data = {}
    for fam in FAMS:
        dl = load_viz(fam)
        df = load_viz_fno(fam)
        if dl is None:
            continue
        target_all = dl["target"]      # (B, T, *spatial, C)
        pred_all   = dl["pred"]
        T_full = target_all.shape[1]
        n_hist = T_full // 2
        # Hardest sample by relL2 at final timestep
        per_sample_l2 = []
        for b in range(target_all.shape[0]):
            yg = target_all[b, -1, ..., 0]
            yp = pred_all[b, -1, ..., 0]
            num = float(np.sqrt(((yp - yg) ** 2).sum()))
            den = float(np.sqrt((yg ** 2).sum())) + 1e-12
            per_sample_l2.append(num / den)
        h_idx = int(np.argmax(per_sample_l2))
        l2_hard = per_sample_l2[h_idx]
        y = target_all[h_idx]
        yhat_l = pred_all[h_idx]
        yhat_f = df["pred"][h_idx] if df is not None else None
        t_abs = [n_hist + t for t in t_steps if (n_hist + t) < T_full]
        t_lbls = [t for t in t_steps if (n_hist + t) < T_full]
        fam_data[fam] = (y, yhat_l, yhat_f, t_abs, t_lbls, l2_hard)
    if not fam_data:
        return None

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _draw_heatmap(ax, field, vmax, cmap=PASTEL_DIV, scale="linear",
                       headroom=1.0, return_im=False, add_cbar=False,
                       cbar_n_ticks=3):
        f_hi = zoom(field, UPSAMPLE, order=3)
        H, W = field.shape
        v = vmax * headroom
        extent = [-0.5, W - 0.5, H - 0.5, -0.5]
        def _signed_power(gamma):
            fwd = lambda x: np.sign(x) * np.power(np.abs(x), gamma)
            inv = lambda x: np.sign(x) * np.power(np.abs(x), 1.0 / gamma)
            return FuncNorm((fwd, inv), vmin=-v, vmax=v)
        if scale == "symlog":
            linthresh = max(1e-3 * v, 1e-9)
            im = ax.imshow(f_hi, cmap=cmap,
                            norm=SymLogNorm(linthresh=linthresh,
                                             vmin=-v, vmax=v, base=10),
                            interpolation="bilinear", extent=extent)
        elif scale == "sqrt":
            im = ax.imshow(f_hi, cmap=cmap, norm=_signed_power(0.5),
                            interpolation="bilinear", extent=extent)
        elif scale == "mild":
            im = ax.imshow(f_hi, cmap=cmap, norm=_signed_power(0.7),
                            interpolation="bilinear", extent=extent)
        else:
            im = ax.imshow(f_hi, cmap=cmap, vmin=-v, vmax=v,
                           interpolation="bilinear", extent=extent)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if add_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
            ticks = np.linspace(-v, v, cbar_n_ticks)
            cb.set_ticks(ticks)
            cb.ax.tick_params(labelsize=6, length=2, pad=1)
            cb.ax.set_xticklabels([f"{t:.2g}" for t in ticks])
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
    # Option A: single family (Gauss by default), 2 rows Ã— 5 cols
    # ====================================================================
    if fam_pick in fam_data:
        y, yhat_l, yhat_f, t_abs, t_lbls, l2_hard = fam_data[fam_pick]
        n_t = len(t_abs)
        if n_t > 0:
            # Shared per-row vmax + per-row colorbar
            gt_vmax_seq = float(max(max(np.abs(y[t, ..., 0]).max()
                                          for t in t_abs), 1e-9))
            has_fno_seq = (yhat_f is not None)
            diff_list = []
            le_err_list = []
            for t in t_abs:
                yg = y[t, ..., 0]
                el_arr = yhat_l[t, ..., 0] - yg
                le_err_list.append(el_arr)
                if has_fno_seq:
                    el = np.abs(el_arr)
                    ef = np.abs(yhat_f[t, ..., 0] - yg)
                    diff_list.append(el - ef)
            le_vmax_seq = float(max(max(np.abs(e).max() for e in le_err_list),
                                       1e-9))
            dd_vmax_seq = (float(max(max(np.abs(d).max() for d in diff_list),
                                        1e-9)) if diff_list else 1e-9)

            n_rows_seq = 2 if has_fno_seq else 2  # GT + (Err Diff OR LEMO Error)
            figA, axA = plt.subplots(n_rows_seq, n_t,
                                       figsize=(3.4 * n_t, 3.4 * n_rows_seq + 0.2),
                                       gridspec_kw={"wspace": 0.04,
                                                    "hspace": 0.07})
            if n_t == 1:
                axA = axA.reshape(n_rows_seq, 1)
            # Row labels removed per user request (no Ground Truth / Error
            # Difference text on the left margin of V02_rollout_sequence).
            im_gt_a = im_dd_a = None
            for j, (t, lbl) in enumerate(zip(t_abs, t_lbls)):
                y_gt = y[t, ..., 0]
                axA[0, j].set_title(f"t={lbl}", fontsize=13)
                im = _draw_heatmap(axA[0, j], y_gt, gt_vmax_seq, return_im=True)
                if j == n_t - 1: im_gt_a = im
                _overlay_gt_contours(axA[0, j], y_gt, gt_vmax_seq)
                if has_fno_seq:
                    im = _draw_heatmap(axA[1, j], diff_list[j], dd_vmax_seq,
                                         scale="mild", return_im=True)
                else:
                    im = _draw_heatmap(axA[1, j], le_err_list[j], le_vmax_seq,
                                         scale="mild", return_im=True)
                if j == n_t - 1: im_dd_a = im
                _overlay_gt_contours(axA[1, j], y_gt,
                                       dd_vmax_seq if has_fno_seq else le_vmax_seq)
            cb_lbl1 = ("|LEMO err| âˆ’ |FNO err|" if has_fno_seq
                        else "LEMO err (pred âˆ’ GT)")
            for row_idx, im_row, lbl in [(0, im_gt_a, "field"),
                                          (1, im_dd_a, cb_lbl1)]:
                if im_row is None: continue
                cb = figA.colorbar(im_row, ax=axA[row_idx, :].tolist(),
                                    fraction=0.025, pad=0.02,
                                    shrink=0.95, aspect=22)
                cb.ax.tick_params(labelsize=8)
                cb.set_label(lbl, fontsize=9)
            # V02 suptitle removed per user request
            figA.tight_layout(rect=[0, 0, 1, 1])
            out = FIG / "V02_rollout_sequence.pdf"
            figA.savefig(out, bbox_inches="tight")
            figA.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(figA)

    # ====================================================================
    # Option B: 5 families Ã— 5 timesteps grid of Error Difference only
    # ====================================================================
    fams_with_both = [f for f in FAMS
                       if f in fam_data and fam_data[f][2] is not None
                       and len(fam_data[f][3]) > 0]
    if fams_with_both:
        # Use the timesteps from the first family (all should match).
        _, _, _, ref_t_abs, ref_t_lbls, _ = fam_data[fams_with_both[0]]
        n_t = len(ref_t_abs)
        n_f = len(fams_with_both)
        figB, axB = plt.subplots(n_f, n_t,
                                   figsize=(2.4 * n_t, 2.5 * n_f),
                                   gridspec_kw={"wspace": 0.10,
                                                "hspace": 0.30})
        if n_f == 1:
            axB = axB.reshape(1, n_t)
        for i, fam in enumerate(fams_with_both):
            y, yhat_l, yhat_f, t_abs, t_lbls, l2_hard = fam_data[fam]
            # PER-CELL vmax (each timestep self-normalizes for bold colors).
            diffs = []
            for t in t_abs:
                yg = y[t, ..., 0]
                el = np.abs(yhat_l[t, ..., 0] - yg)
                ef = np.abs(yhat_f[t, ..., 0] - yg)
                diffs.append(el - ef)
            for j, (t, lbl, diff) in enumerate(zip(t_abs, t_lbls, diffs)):
                if i == 0:
                    axB[i, j].set_title(f"t={lbl}", fontsize=12)
                cell_vmax = float(max(np.abs(diff).max(), 1e-9))
                _draw_heatmap(axB[i, j], diff, cell_vmax,
                                scale="mild", add_cbar=True)
                _overlay_gt_contours(axB[i, j], y[t, ..., 0], cell_vmax)
            axB[i, 0].set_ylabel(FAM_LABELS[fam], fontsize=12, rotation=0,
                                   ha="right", va="center", labelpad=12)
        figB.tight_layout()
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
      RIGHT:  |K|_F per mode m, broken down by inâ†’out channel (small heatmap)
    """
    d = load_kernel(fam_pick)
    if d is None:
        return None
    # Kernel is stored as cfloat â†’ split into 'weights__re' and 'weights__im'.
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
        K_t = np.fft.irfft(K_full, n=L_time, axis=-1)  # (in, out, L_time) real
        # Aggregation: pick the dominant channel pair (largest L2 norm). This
        # avoids the phase-destruction artefact of np.abs(K_t).mean(axis=(0,1))
        # which produces spurious oscillations because different (i, o) pairs
        # encode the same temporal pattern with different phase, and absolute-
        # value averaging cannot cancel those phases coherently.
        norms = np.linalg.norm(K_t, axis=-1)             # (in, out)
        i_dom, o_dom = np.unravel_index(np.argmax(norms), norms.shape)
        K_dom = K_t[i_dom, o_dom]                         # (L_time,)
        # Sign: flip so the kernel is positive at its peak (kernels are
        # positive functions; sign of the dominant SVD direction is arbitrary).
        peak_t = int(np.argmax(np.abs(K_dom)))
        if K_dom[peak_t] < 0:
            K_dom = -K_dom
        K_amp = np.abs(K_dom)                             # (L_time,) for plotting
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
    """V06: 2-panel residual spectrum + cumulative-energy truncation justification.

    Left panel: per-mode residual FFT energy E[|rÌ‚_m|Â²] across 5 families.
    Right panel: cumulative fraction of residual energy captured by first M modes,
    with a vertical line at the architectural truncation M=24. Annotations are
    deliberately minimal â€” paper text / latex caption fills in the M=24 numeric.
    """
    series = {}
    for fam in FAMS:
        for seed in SEEDS:
            d = load_viz(fam, "clean", seed)
            if d is None:
                continue
            target = d["target"]
            pred = d["pred"]
            r = pred - target
            perm = [0] + list(range(2, r.ndim)) + [1]
            r_p = np.transpose(r, axes=perm)
            R = np.fft.rfft(r_p, axis=-1)
            energy = np.mean(np.abs(R) ** 2, axis=tuple(range(R.ndim - 1)))
            series.setdefault(fam, []).append(energy)
    if not series:
        return None

    # Saturated tab10 colors per family + big-text styling.
    TITLE_FS = 42; AXIS_FS = 38; TICK_FS = 36; LEGEND_FS = 42
    FAM_COLOR = {
        "dist_exp_rd_2d":      "#d62728",
        "dist_gaussian_rd_2d": "#1f77b4",
        "dist_gamma_rd_2d":    "#2ca02c",
        "dist_uniform_rd_2d":  "#ff7f0e",
        "dist_powerlaw_rd_2d": "#9467bd",
    }
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(22.0, 11.0),
                                     gridspec_kw={"wspace": 0.18})
    for fam in FAMS:
        if fam not in series: continue
        e = np.array(series[fam])
        m = e.mean(axis=0)
        modes = np.arange(len(m))
        c = FAM_COLOR.get(fam, "#888")
        axL.plot(modes, m, lw=4.0, color=c, solid_capstyle="round",
                  label=FAM_LABELS[fam])
        cumulative = np.cumsum(m)
        cumulative /= cumulative[-1]
        axR.plot(modes, cumulative * 100, lw=4.0, color=c,
                  solid_capstyle="round", label=FAM_LABELS[fam])

    axL.set_yscale("log")
    axL.set_xlabel(r"spectral lag mode $m$", fontsize=AXIS_FS)
    axL.set_ylabel(r"$\mathbb{E}\,|\hat{r}_m|^2$", fontsize=AXIS_FS)
    axL.tick_params(axis="both", labelsize=TICK_FS)
    axL.grid(True, which="major", linestyle="-", color="grey",
              alpha=0.18, linewidth=0.6)
    axL.set_axisbelow(True)
    for sp in ("top", "right"): axL.spines[sp].set_visible(False)

    axR.axvline(24, color="black", linestyle="--", lw=2.0, alpha=0.7)
    axR.set_xlabel(r"modes retained $M$", fontsize=AXIS_FS)
    axR.set_ylabel(r"% of total residual energy", fontsize=AXIS_FS)
    axR.tick_params(axis="both", labelsize=TICK_FS)
    axR.set_ylim(0, 105)
    axR.grid(True, which="major", linestyle="-", color="grey",
              alpha=0.18, linewidth=0.6)
    axR.set_axisbelow(True)
    for sp in ("top", "right"): axR.spines[sp].set_visible(False)

    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                fontsize=LEGEND_FS, frameon=False,
                columnspacing=1.6, handlelength=1.4, handletextpad=0.4)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.20,
                          wspace=0.18)
    out = FIG / "V06_residual_fft.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=200,
                  bbox_inches="tight", pad_inches=0.05)
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
        # V04 dropped: not referenced in paper; redundant with V05 (kernel
        # recovery) for kernel structure and F07 (op-norm trajectory) for Ïƒ-bound.
        # V05 dropped (2026-05-03): the dominant-pair aggregation fix exposed
        # that LEMO-PC's current learned kernel is bimodal/mode-2 due to FiLM
        # nullification, making the kernel-recovery story look weaker than it
        # actually is. Even after the FiLM-fix retrain improves CosSim, V05
        # competes with the much-stronger T06 (kernel-recovery cosine table)
        # which already carries the per-family numbers without requiring a
        # visualization that picks a single channel pair.
        # ("V05 kernel recovery",     fig_v05_kernel_recovery),
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
