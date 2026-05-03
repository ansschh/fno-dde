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
    """Per-family contour-overlay panels, 1 row × N cols (one per family with
    both LEMO-PC AND FNO+FiLM viz data available — no N/A cells).

    Each panel: GT colormap background + 3 contour sets:
      - GT level sets               (black solid, lw=1.6)
      - LEMO-PC predicted levels    (red solid, lw=1.6)
      - FNO+FiLM predicted levels   (cyan dashed, lw=1.6)

    Where red lies on black → LEMO got it right. Where cyan deviates from
    black → FNO+FiLM missed. Architectural contrast visible at a glance.
    """
    panels = []
    for fam in FAMS:
        dl = load_viz(fam)
        if dl is None:
            continue
        df = load_viz_fno(fam)  # may be None if FNO+FiLM viz unavailable
        y = dl["target"][sample_idx]
        yhat_l = dl["pred"][sample_idx]
        y_l = y[target_step, ..., 0]
        yhL = yhat_l[target_step, ..., 0]
        if df is not None:
            yhF = df["pred"][sample_idx][target_step, ..., 0]
        else:
            yhF = None  # plot panel with GT+LEMO only, omit FNO contour
        panels.append((fam, y_l, yhL, yhF))
    if not panels:
        return None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.6),
                              gridspec_kw={"wspace": 0.06})
    if n == 1:
        axes = [axes]

    import matplotlib.patheffects as pe
    halo = pe.withStroke(linewidth=4, foreground="white")

    for ax, (fam, y_l, yhL, yhF) in zip(axes, panels):
        all_arrs = [y_l, yhL] + ([yhF] if yhF is not None else [])
        vmax = float(np.max([np.abs(v).max() for v in all_arrs]))
        H, W = y_l.shape
        pos_levels = np.linspace(0.15 * vmax, 0.85 * vmax, 4)
        levels = np.concatenate([-pos_levels[::-1], pos_levels])
        xs, ys = np.arange(W), np.arange(H)

        # High-res viridis background — perceptually uniform; no conflict with
        # black/red/cyan contour colours.
        ax.imshow(y_l, cmap="viridis", vmin=-vmax, vmax=vmax,
                  interpolation="bicubic")

        # Plot contours with path-effect halos so each line stays visible
        # even when overlapping. Width-descending order so each line still
        # peeks out from underneath the next.
        cs_gt = ax.contour(xs, ys, y_l, levels=levels,
                            colors="black", linewidths=3.0)
        cs_lemo = ax.contour(xs, ys, yhL, levels=levels,
                              colors="#ff2a2a", linewidths=2.0)
        for cs in (cs_gt, cs_lemo):
            try:
                cs.set(path_effects=[halo])
            except Exception:
                # matplotlib <3.8 fallback: iterate collections
                for c in getattr(cs, "collections", []):
                    c.set_path_effects([halo])
        if yhF is not None:
            cs_fno = ax.contour(xs, ys, yhF, levels=levels,
                                  colors="#00d8ff", linewidths=1.5,
                                  linestyles="dashed")
            try:
                cs_fno.set(path_effects=[halo])
            except Exception:
                for c in getattr(cs_fno, "collections", []):
                    c.set_path_effects([halo])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        ax.set_title(FAM_LABELS[fam], fontsize=12)

    # Single shared legend at the bottom
    handles = [
        plt.Line2D([0], [0], color="black", lw=3.0, label="ground truth"),
        plt.Line2D([0], [0], color="#ff2a2a", lw=2.0, label="LEMO-PC pred"),
        plt.Line2D([0], [0], color="#00d8ff", lw=1.5, ls="--",
                    label="FNO+FiLM pred"),
    ]
    fig.legend(handles=handles, loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=3, frameon=False, fontsize=10)

    fig.suptitle("Final-frame contour overlay, per family "
                 f"(sample idx = {sample_idx}, deterministic)",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = FIG / "V01_family_triptych.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_v02_rollout_sequence(fam_pick="dist_exp_rd_2d", t_steps=(0, 16, 32, 48, 63)):
    d = load_viz(fam_pick)
    if d is None:
        return None
    y = d["target"][0]      # (T_total, *spatial, C) where T_total = n_hist + n_out (typically 128 = 64+64)
    yhat = d["pred"][0]
    T = y.shape[0]
    # Plot FORECAST frames, not history. Pred is meaningful only on the future
    # half (loss is masked to future during training); the history half of pred
    # is unconstrained model output and should not be displayed.
    n_hist = T // 2
    t_abs = [n_hist + t for t in t_steps if (n_hist + t) < T]
    t_labels = [t for t in t_steps if (n_hist + t) < T]
    if not t_abs:
        return None
    if not t_steps:
        return None
    fig, axes = plt.subplots(2, len(t_abs),
                              figsize=(2.0 * len(t_abs), 4.2),
                              gridspec_kw={"wspace": 0.05, "hspace": 0.1})
    if len(t_abs) == 1:
        axes = axes.reshape(2, 1)
    vmax = max(np.abs(y[t, ..., 0]).max() for t in t_abs)
    vmax = max(vmax, max(np.abs(yhat[t, ..., 0]).max() for t in t_abs))
    for j, (t, lbl) in enumerate(zip(t_abs, t_labels)):
        for i in range(2):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[0, j].imshow(y[t, ..., 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, j].imshow(yhat[t, ..., 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, j].set_title(f"t={lbl}", fontsize=10)
    axes[0, 0].set_ylabel("ground truth", rotation=0, ha="right", va="center", fontsize=10)
    axes[1, 0].set_ylabel("LEMO-PC pred", rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(f"Rollout sequence: {FAM_LABELS[fam_pick]} family", fontsize=11)
    out = FIG / "V02_rollout_sequence.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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
        ("V03 error maps",          fig_v03_error_maps),
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
