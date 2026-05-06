"""F_norm_rollout — long-horizon prediction-norm trajectories per architecture.

For every model with a `long_rollout.npz` artefact under extracted/ or
outputs/, plot ‖û(t)‖_2 averaged over chains/seeds/families against
rollout step t (0..127, the saved horizon). Training horizon T=64 is
marked with a vertical guide. The grey band shows the empirical
ground-truth norm range (10th–90th percentile of `target` norms across
the 5 dist_*_rd_2d viz cells we have GT for); contracting models stay
inside the band, divergent ones (e.g. Non-equiv +FiLM) blow past it.

This figure is the contraction-vs-divergence visual companion to F08
(equivariance) and F_boundary (cyclic FFT). All data is read from
existing artefacts — no retraining or fresh inference needed.

Output: NeurIPS_LEMO/figures/F_norm_rollout.{pdf,png}
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401  (sets Times New Roman globally)
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ["clean", "lowres", "noisy"]
SEEDS = ["s42", "s123", "s456"]
TRAIN_T = 64

MODEL_LABELS = {
    "lemo_pc_nd":                 "LEMO-PC",
    "causal_smooth_lemo_pc_nd":   "LEMO-PC (causal)",
    "lemo_bcorrect_nd":           "LEMO (b-correct, no σ)",
    "fno_nd":                     "FNO",
    "fno_film_nd":                "FNO+FiLM",
    "noneq_film_nd":              "Non-equiv +FiLM",
    "markov_fno_nd":              "Markov-FNO",
    "windowed_fno_nd":            "Window-FNO",
    "memno_nd":                   "MemNO",
    "ffno_nd":                    "F-FNO",
    "s4_nd":                      "S4",
    "nide_nd":                    "NIDE",
    "ndde_nd":                    "NDDE",
    "unet_nd":                    "UNet",
}
MODEL_ORDER = ["lemo_pc_nd", "lemo_bcorrect_nd", "fno_film_nd",
               "noneq_film_nd", "memno_nd", "ffno_nd",
               "s4_nd", "nide_nd", "ndde_nd"]
MODEL_COLOR = {
    "lemo_pc_nd":                 "#d62728",
    "lemo_bcorrect_nd":           "#bcbd22",
    "causal_smooth_lemo_pc_nd":   "#ff7f0e",
    "fno_film_nd":                "#17becf",
    "noneq_film_nd":              "#9467bd",
    "memno_nd":                   "#e377c2",
    "ffno_nd":                    "#8c564b",
    "s4_nd":                      "#bcbd22",
    "nide_nd":                    "#008080",
    "ndde_nd":                    "#2ca02c",
}


def discover_long_rollouts() -> dict:
    """{model: list of pred_norm_per_step arrays of shape (n_chain, T)}."""
    out = defaultdict(list)
    seen = set()
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("long_rollout.npz"):
            try:
                parts = f.parts
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
            except IndexError:
                continue
            if fam not in FAMS or reg != "clean" or seed not in SEEDS:
                continue
            if model not in MODEL_LABELS:
                continue
            key = (model, fam, seed)
            if key in seen:
                continue
            try:
                data = np.load(f)
                arr = np.asarray(data["pred_norm_per_step"], dtype=np.float64)
            except Exception:
                continue
            if arr.ndim != 2:
                continue
            seen.add(key)
            out[model].append(arr)
    return dict(out)


def gt_norm_band(percentiles=(10, 90)) -> tuple[float, float] | None:
    """Estimate the GT attractor norm band from existing viz_samples.npz."""
    norms = []
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
            if fam not in FAMS or reg != "clean":
                continue
            if model != "lemo_pc_nd":
                continue
            try:
                d = np.load(f)
                t = d["target"]   # (B, T, *spatial, C)
                # Norm per (b, t): flatten spatial+channel
                flat = t.reshape(t.shape[0], t.shape[1], -1)
                ns = np.linalg.norm(flat, axis=-1)
                norms.append(ns.flatten())
            except Exception:
                continue
            if len(norms) > 8:
                break
        if len(norms) > 8:
            break
    if not norms:
        return None
    arr = np.concatenate(norms)
    return float(np.percentile(arr, percentiles[0])), float(np.percentile(arr, percentiles[1]))


def main():
    rollouts_by_model = discover_long_rollouts()
    if not rollouts_by_model:
        print("[F_norm_rollout] no long_rollout.npz found")
        return
    # Match F11_robustness_appendix sizing.
    TITLE_FS = 42; AXIS_FS = 38; TICK_FS = 36; LEGEND_FS = 42
    # Smaller figsize so text-to-figure ratio matches F07 visual density.
    fig, ax = plt.subplots(figsize=(24.0, 13.0))

    handles, labels = [], []
    plotted_T = 128
    for model in MODEL_ORDER:
        if model not in rollouts_by_model:
            continue
        # Stack across cells: each is (n_chain, T) — we treat all chains × cells as
        # independent rollouts and aggregate to median + IQR.
        stacks = rollouts_by_model[model]
        T = min(s.shape[1] for s in stacks)
        plotted_T = min(plotted_T, T)
        flat = np.concatenate([s[:, :T].reshape(-1, T) for s in stacks], axis=0)
        median = np.median(flat, axis=0)
        q25 = np.percentile(flat, 25, axis=0)
        q75 = np.percentile(flat, 75, axis=0)
        steps = np.arange(T)
        color = MODEL_COLOR.get(model, "#888")
        line, = ax.plot(steps, median, color=color, lw=4.0,
                         solid_capstyle="round",
                         label=MODEL_LABELS[model])
        handles.append(line); labels.append(MODEL_LABELS[model])

    band = gt_norm_band(percentiles=(25, 75))
    if band is not None:
        lo, hi = band
        ax.axhspan(lo, hi, color="grey", alpha=0.22, linewidth=0)
        h_band = plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.22, ec="none")
        handles.append(h_band)
        labels.append(f"GT IQR band [{lo:.0f}, {hi:.0f}]")

    ax.axvline(TRAIN_T, color="black", lw=0.8, linestyle="--", alpha=0.6)

    ax.set_xlim(0, plotted_T - 1)
    ax.set_ylim(2.0, 200.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"rollout step $t$", fontsize=AXIS_FS)
    ax.set_ylabel(r"$\|\hat u(t)\|_2$", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, which="major", linestyle="-", color="grey",
             alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    n = len(handles)
    ncol = 5 if n >= 8 else (4 if n >= 5 else max(1, n))
    fig.legend(handles=handles, labels=labels, loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=ncol, frameon=False, fontsize=LEGEND_FS,
                columnspacing=1.4, handlelength=1.4, handletextpad=0.4)
    rows = int(np.ceil(n / ncol))
    bot = 0.08 + 0.09 * rows
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=bot)
    out = FIG_DIR / "F_norm_rollout.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=150,
                  bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"-> {out.name}")
    n_models = len(labels)
    print(f"   models plotted: {n_models}")
    print(f"   horizon: {plotted_T} steps (training T={TRAIN_T})")
    if band is not None:
        print(f"   GT attractor norm band: {band[0]:.2f} - {band[1]:.2f}")


if __name__ == "__main__":
    main()
