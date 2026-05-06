"""Paper figure generation â€” all 2D dist-kernel paper-blocking figures.

Reads:
  - paper/stats/paired_permutation.json         (headline stats)
  - extracted/pod1/.../dist_kernel_v2_p1/raw/.. (LEMO/LEMO_PC test_results.json)
  - extracted/pod2/.../dist_kernel_v2_p2/logs/  (FNO/MarkovFNO/WindFNO from logs)
  - extracted/pod3/outputs/final_baselines/raw/ (MemNO/F-FNO when available)
  - paper/figures/sigma_sweep/                  (Ïƒ-sweep when Caltech done)

Produces (in paper/figures/):
  - F01_headline_bar.{pdf,png}        6-baseline % improvement with bootstrap CI + p
  - F02_perfamily_heatmap.{pdf,png}   5 fams x N models heatmap of relL2
  - F03_perregime_box.{pdf,png}       clean/lowres/noisy box+strip per model
  - F04_effect_size.{pdf,png}         Hedges g forest plot per comparison
  - F05_training_curves.{pdf,png}     val_rel_l2 vs epoch, mean+/-std over seeds
  - F06_perframe_rollout.{pdf,png}    per-rollout-step rel_l2 (if capture present)
  - F07_op_norm_trajectory.{pdf,png}  op_norm_max vs epoch (if logging in history)
  - F08_equivariance_test.{pdf,png}   T1-shift errors per shift size (if capture)

Usage:
    python3 scripts/make_paper_figures.py

Idempotent: skips figures whose data is missing, prints inventory at end.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
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
EXT = REPO / "extracted"
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = REPO / "paper" / "stats" / "paired_permutation.json"

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": r"Exp", "dist_gaussian_rd_2d": r"Gauss",
              "dist_gamma_rd_2d": r"Gamma", "dist_uniform_rd_2d": r"Uniform",
              "dist_powerlaw_rd_2d": r"Power"}
REGIMES = ["clean", "lowres", "noisy"]
SEEDS = ["s42", "s123", "s456"]

MODEL_LABELS = {
    "lemo_pc_nd":                 "LEMO-PC",
    "causal_smooth_lemo_pc_nd":   "LEMO-PC (causal)",
    "lemo_bcorrect_nd":           "LEMO-PC (b-correct)",
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
MODEL_ORDER = [
    "lemo_pc_nd", "causal_smooth_lemo_pc_nd",
    "fno_film_nd", "noneq_film_nd",
    "fno_nd", "markov_fno_nd", "windowed_fno_nd",
    "memno_nd", "ffno_nd", "s4_nd", "nide_nd", "ndde_nd",
    "unet_nd",
]
MODEL_COLOR = {
    "lemo_pc_nd":                 "#d62728",  # red
    "causal_smooth_lemo_pc_nd":   "#c49c94",  # taupe
    "lemo_bcorrect_nd":           "#bcbd22",  # olive (rarely shown)
    "fno_nd":                     "#1f77b4",  # blue
    "fno_film_nd":                "#17becf",  # cyan
    "noneq_film_nd":              "#c5b0d5",  # lavender
    "markov_fno_nd":              "#2ca02c",  # green
    "windowed_fno_nd":            "#9467bd",  # purple
    "memno_nd":                   "#e377c2",  # pink
    "ffno_nd":                    "#8c564b",  # brown
    "s4_nd":                      "#bcbd22",  # olive
    "nide_nd":                    "#aec7e8",  # light blue
    "ndde_nd":                    "#98df8a",  # light green
    "unet_nd":                    "#7f7f7f",  # gray
}


# -------------------- data loading --------------------

def _try_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _discover_jsons(filename: str) -> dict:
    """rglob across `extracted/` and `outputs/` for files named `<filename>`,
    interpreting paths as `<...>/<fam>/<reg>/<model>/<seed>/<filename>` (the
    layout used uniformly across pod_pulls_2026_05_03_final/*, pod1/v2_p1,
    film_ablation_caltech, memno_ffno_runpod, memory_aware_runpod, etc.).

    Returns {model: {(fam, reg, seed): json_data}}. Dedupes on
    (model, fam, reg, seed) so duplicates across multiple roots don't
    double-count the same cell.
    """
    out = defaultdict(dict)
    seen = set()
    roots = [REPO / "extracted", REPO / "outputs"]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob(filename))
    # Priority: prefer paths from the post-A-fix h100 pull (cyclic_shift_full
    # equivariance, FP32-floor LEMO-PC) over older pulls that used
    # cyclic_shift_state_only and pre-A-fix LEMO-PC.
    def _priority(p):
        s = str(p).replace("\\", "/")
        if "a_fix_runpod" in s or "h100_pull_2026_05_05" in s:
            return 0
        return 1
    candidates.sort(key=_priority)
    for f in candidates:
        try:
            parts = f.parts
            seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
        except IndexError:
            continue
        if fam not in FAMS or reg not in REGIMES or seed not in SEEDS:
            continue
        if model not in MODEL_LABELS:
            continue
        key = (model, fam, reg, seed)
        if key in seen:
            continue
        data = _try_json(f)
        if data is None:
            continue
        seen.add(key)
        out[model][(fam, reg, seed)] = data
    return dict(out)


def gather_all_models() -> dict:
    """Returns {model: {(fam, reg, seed): test_rel_l2 float}}.

    rglob discovery picks up every test_results.json under extracted/ or
    outputs/ for any model in MODEL_LABELS. Keeps the first match per cell;
    cells without test_rel_l2_mean are dropped.
    """
    discovered = _discover_jsons("test_results.json")
    out = {}
    for model, cells in discovered.items():
        per_cell = {}
        for k, d in cells.items():
            v = d.get("test_rel_l2_mean", d.get("test_rel_l2"))
            if v is None:
                continue
            try:
                per_cell[k] = float(v)
            except (TypeError, ValueError):
                continue
        if per_cell:
            out[model] = per_cell
    return out


def load_history(model: str = "lemo_pc_nd"):
    """Returns {(fam, reg, seed): history_dict} for the requested model,
    via the same rglob discoverer used by gather_all_models()."""
    discovered = _discover_jsons("history.json")
    return discovered.get(model, {})


_LOG_PAT = re.compile(r"=== FINAL test relL2 = ([0-9.]+) ===")  # legacy, kept for safety


# -------------------- statistics --------------------

def bootstrap_ci(arr, n_boot=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# -------------------- figures --------------------

def fig01_headline_bar(stats: dict):
    """Bar chart: % improvement of LEMO_PC vs each baseline with 95% CI + p."""
    if not stats:
        return None
    keys = [k for k in ("FNO", "MarkovFNO", "WindFNO", "UNet", "LEMO_ND_ablation")
            if k in stats]
    if not keys:
        return None
    means = [stats[k]["aggregate"]["improvement_ratio_mean_pct"] for k in keys]
    lo = [stats[k]["aggregate"]["improvement_95ci_pct"][0] for k in keys]
    hi = [stats[k]["aggregate"]["improvement_95ci_pct"][1] for k in keys]
    pvals = [stats[k]["aggregate"]["paired_permutation_p"] for k in keys]
    label_map = {"FNO": "vs FNO", "MarkovFNO": "vs Markov-FNO",
                 "WindFNO": "vs Window-FNO", "UNet": "vs UNet",
                 "LEMO_ND_ablation": "vs LEMO (no FiLM)"}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    err = np.array([np.array(means) - np.array(lo), np.array(hi) - np.array(means)])
    bars = ax.bar(x, means, yerr=err, capsize=4, color="#d62728", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[k] for k in keys], rotation=15)
    ax.set_ylabel(r"Improvement in test relL2 (\%)")
    ax.set_ylim(0, 100)
    ax.axhline(0, color="black", linewidth=0.5)
    for xi, m, p in zip(x, means, pvals):
        annotation = f"p<10$^{{-4}}$" if p < 1e-4 else f"p={p:.1e}"
        ax.text(xi, m + 2, f"{m:.1f}%\n{annotation}", ha="center", va="bottom",
                fontsize=8)
    ax.set_title("LEMO-PC: paired-permutation improvement (n=45 paired cells)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / "F01_headline_bar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig02_perfamily_heatmap(data: dict):
    """5 fams x N models heatmap of relL2 (clean regime)."""
    models = [m for m in MODEL_ORDER if m in data]
    if not models or not FAMS:
        return None
    M = np.full((len(FAMS), len(models)), np.nan)
    for i, fam in enumerate(FAMS):
        for j, mdl in enumerate(models):
            vals = [data[mdl].get((fam, "clean", s), np.nan) for s in SEEDS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                M[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(1.5 + len(models) * 1.0, 1 + len(FAMS) * 0.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=30, ha="right")
    ax.set_yticks(range(len(FAMS)))
    ax.set_yticklabels([FAM_LABELS[f] for f in FAMS])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Test rel$L_2$")
    for i in range(len(FAMS)):
        for j in range(len(models)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > 0.05 else "black", fontsize=8)
    ax.set_title(r"Per-family test rel$L_2$ (clean regime, mean over 3 seeds)")
    fig.tight_layout()
    out = FIG_DIR / "F02_perfamily_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig03_perregime_box(data: dict):
    """clean/lowres/noisy box+strip per model."""
    models = [m for m in MODEL_ORDER if m in data]
    if not models:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, reg in zip(axes, REGIMES):
        per_model = []
        labels = []
        colors = []
        for mdl in models:
            vals = [data[mdl].get((fam, reg, s), np.nan)
                    for fam in FAMS for s in SEEDS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                per_model.append(vals)
                labels.append(MODEL_LABELS[mdl])
                colors.append(MODEL_COLOR[mdl])
        if not per_model:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(per_model, labels=labels, patch_artist=True,
                         showfliers=False, medianprops={"color": "black"})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for j, vals in enumerate(per_model, start=1):
            xs = np.full(len(vals), j) + np.random.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(xs, vals, color="black", s=12, alpha=0.6, zorder=3)
        ax.set_title(f"regime: {reg}")
        ax.set_yscale("log")
        ax.set_ylabel(r"Test rel$L_2$")
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Per-regime breakdown across 5 dist-kernel families x 3 seeds")
    fig.tight_layout()
    out = FIG_DIR / "F03_perregime_box.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig04_effect_size(stats: dict):
    """Forest plot: Hedges g per comparison."""
    if not stats:
        return None
    keys = [k for k in ("FNO", "MarkovFNO", "WindFNO", "UNet", "LEMO_ND_ablation")
            if k in stats and "hedges_g" in stats[k]["aggregate"]]
    if not keys:
        return None
    g_values = [stats[k]["aggregate"]["hedges_g"] for k in keys]
    label_map = {"FNO": "vs FNO", "MarkovFNO": "vs Markov-FNO",
                 "WindFNO": "vs Window-FNO", "UNet": "vs UNet",
                 "LEMO_ND_ablation": "vs LEMO (no FiLM)"}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(keys))
    ax.barh(y, g_values, color="#d62728", alpha=0.85, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels([label_map[k] for k in keys])
    for i, g in enumerate(g_values):
        ax.text(g + 0.05, i, f"g={g:.2f}", va="center", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.8, color="grey", linewidth=0.5, linestyle="--")
    ax.text(0.8, len(keys) - 0.4, "  large (g=0.8)", color="grey", fontsize=8)
    ax.set_xlabel("Hedges g")
    ax.set_title("Effect size of LEMO-PC's improvement (Hedges g)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, max(g_values) * 1.2)
    fig.tight_layout()
    out = FIG_DIR / "F04_effect_size.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig05_training_curves(history_by_model: dict):
    """val_rel_l2 vs epoch, one panel per family, one curve per model
    (mean over seeds with shaded std band).

    `history_by_model` is {model_name: {(fam, regime, seed): history_dict}}.
    Plots every model present in MODEL_ORDER for which we have history data,
    so reviewers can compare convergence behaviour across all benchmarked
    architectures rather than just LEMO-PC alone.
    """
    if not history_by_model:
        return None
    fig, axes = plt.subplots(1, len(FAMS), figsize=(4.2 * len(FAMS), 6.4),
                              sharey=True)
    if len(FAMS) == 1:
        axes = [axes]
    handles, labels = [], []
    plotted_any = False
    for ax, fam in zip(axes, FAMS):
        plotted_in_panel = False
        for model in MODEL_ORDER:
            history = history_by_model.get(model, {})
            if not history:
                continue
            curves = []
            for seed in SEEDS:
                d = history.get((fam, "clean", seed))
                if d is None:
                    continue
                v = d.get("val_rel_l2") or d.get("val_relL2") or []
                if v:
                    curves.append(np.array(v))
            if not curves:
                continue
            L = min(len(c) for c in curves)
            curves = np.stack([c[:L] for c in curves], axis=0)
            epochs = np.arange(1, L + 1)
            color = MODEL_COLOR.get(model, "#888888")
            line, = ax.plot(epochs, curves.mean(axis=0), color=color, lw=1.4,
                             label=MODEL_LABELS.get(model, model))
            if curves.shape[0] > 1:
                ax.fill_between(epochs,
                                 curves.mean(axis=0) - curves.std(axis=0),
                                 curves.mean(axis=0) + curves.std(axis=0),
                                 color=color, alpha=0.18, linewidth=0)
            if MODEL_LABELS.get(model, model) not in labels:
                handles.append(line)
                labels.append(MODEL_LABELS.get(model, model))
            plotted_in_panel = True
            plotted_any = True
        if not plotted_in_panel:
            ax.set_visible(False)
            continue
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(FAM_LABELS[fam], color="black", pad=8)
        ax.grid(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if not plotted_any:
        plt.close(fig)
        return None
    axes[0].set_ylabel(r"Validation rel-$L_2$", fontweight="bold")
    # User constraint: clip x-axis to first 100 epochs.
    for ax in axes:
        if ax.get_visible():
            ax.set_xlim(left=1, right=100)
    if handles:
        n = len(handles)
        ncol = 5 if n >= 8 else (4 if n >= 5 else max(1, n))
        fig.legend(handles, labels, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False,
                    columnspacing=1.6, handlelength=1.6, handletextpad=0.5)
        rows = int(np.ceil(n / ncol))
        bot = 0.16 + 0.06 * rows
    else:
        bot = 0.16
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=bot, wspace=0.10)
    out = FIG_DIR / "F05_training_curves.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig06_perframe_rollout():
    """Per-rollout-step rel_l2 across all models with per_frame.json data.

    Discovers per_frame.json files via rglob across extracted/ and outputs/
    so every memory-aware baseline (s4_nd, nide_nd, ndde_nd, memno_nd,
    ffno_nd, fno_film_nd, noneq_film_nd, causal_smooth_lemo_pc_nd, etc.)
    is auto-picked-up alongside the original cyclic-FFT models.
    """
    perframe = _discover_jsons("per_frame.json")
    if not perframe:
        return None
    fig, axes = plt.subplots(1, len(FAMS), figsize=(4.2 * len(FAMS), 6.4),
                              sharey=True)
    if len(FAMS) == 1:
        axes = [axes]
    # Strip leading "history" zeros: rel_l2_per_step pads the history portion
    # with zeros for compatibility with the per-frame plot. Find the first
    # non-zero index across ALL curves and offset the x-axis to "future
    # rollout step" = step - first_nonzero so t=0 is the first prediction.
    def _first_nonzero(arr: np.ndarray, eps: float = 1e-6) -> int:
        nz = np.nonzero(arr > eps)[0]
        return int(nz[0]) if len(nz) else 0

    handles, labels_seen = [], []
    plotted_any = False
    for ax, fam in zip(axes, FAMS):
        plotted_in_panel = False
        # Per-model curves (clean regime). Truncate every curve at first non-zero.
        for model in MODEL_ORDER:
            cells = perframe.get(model, {})
            curves = []
            for seed in SEEDS:
                d = cells.get((fam, "clean", seed))
                if d is None:
                    continue
                r = d.get("rel_l2_per_step", [])
                if r:
                    arr = np.asarray(r, dtype=float)
                    cut = _first_nonzero(arr)
                    curves.append(arr[cut:])
            if not curves:
                continue
            L = min(len(c) for c in curves)
            c_arr = np.stack([c[:L] for c in curves], axis=0)
            steps = np.arange(L)
            color = MODEL_COLOR.get(model, "#888888")
            line, = ax.plot(steps, c_arr.mean(axis=0), color=color, lw=1.4,
                             label=MODEL_LABELS.get(model, model))
            if c_arr.shape[0] > 1:
                ax.fill_between(steps,
                                 c_arr.mean(axis=0) - c_arr.std(axis=0),
                                 c_arr.mean(axis=0) + c_arr.std(axis=0),
                                 color=color, alpha=0.18, linewidth=0)
            if MODEL_LABELS.get(model, model) not in labels_seen:
                handles.append(line)
                labels_seen.append(MODEL_LABELS.get(model, model))
            plotted_in_panel = True
            plotted_any = True
        if not plotted_in_panel:
            ax.set_visible(False)
            continue
        ax.set_yscale("log")
        ax.set_xlabel("future rollout step $t$")
        ax.set_title(FAM_LABELS[fam], pad=8)
        ax.grid(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if not plotted_any:
        plt.close(fig)
        return None
    axes[0].set_ylabel(r"per-step rel-$L_2$", fontweight="bold")
    if handles:
        n = len(handles)
        ncol = 5 if n >= 8 else (4 if n >= 5 else max(1, n))
        fig.legend(handles, labels_seen, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False,
                    columnspacing=1.6, handlelength=1.6, handletextpad=0.5)
        rows = int(np.ceil(n / ncol))
        bot = 0.16 + 0.06 * rows
    else:
        bot = 0.16
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=bot, wspace=0.10)
    out = FIG_DIR / "F06_perframe_rollout.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig07_op_norm_trajectory(history: dict):
    """op_norm_max vs epoch (proves Ïƒ-projection binding once Ïƒ-sweep cells exist)."""
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for fam in FAMS:
        for seed in SEEDS:
            d = history.get((fam, "clean", seed))
            if d is None:
                continue
            v = d.get("op_norm_max", [])
            if v:
                epochs = np.arange(1, len(v) + 1)
                ax.plot(epochs, v, alpha=0.6,
                        color=MODEL_COLOR["lemo_pc_nd"],
                        label=f"{FAM_LABELS[fam]}/{seed}" if not plotted else None)
                plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\max_m\,\sigma_{\max}(K[:,:,m])$")
    ax.set_title("Raw spectral kernel op-norm trajectory (proves Ïƒ-projection binding)")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / "F07_op_norm_trajectory.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig08_equivariance_test():
    """T1 cyclic-shift equivariance error per shift size k, multi-model.

    rglob discovery picks up `equivariance.json` (sparse k) and
    `equivariance_dense.json` (dense k grid) for every model, so the figure
    is multi-architecture: LEMO-PC near the FP32 FFT floor, the FNO/FiLM
    family well above it. Each curve is mean Â± std across (fam, reg, seed)
    triples that have a value for that shift size.
    """
    sparse = _discover_jsons("equivariance.json")
    dense = _discover_jsons("equivariance_dense.json")

    def _rows_for(model: str):
        rows = []  # (shift, err)
        # Dense format: {'shifts': [...], 'e_orbit': {'<k>': {'mean': ..., 'std': ...}, ...}}
        for d in dense.get(model, {}).values():
            e = d.get("e_orbit", {})
            for k_str, stats in e.items():
                if not isinstance(stats, dict):
                    continue
                v = stats.get("mean")
                if v is None:
                    continue
                try:
                    rows.append((int(k_str), float(v)))
                except (TypeError, ValueError):
                    continue
        # Legacy sparse format: flat keys "equiv_shift_<k>_mean".
        for d in sparse.get(model, {}).values():
            for key, v in d.items():
                m = re.match(r"equiv_shift_(\d+)_mean", key)
                if not m or v is None:
                    continue
                try:
                    rows.append((int(m.group(1)), float(v)))
                except (TypeError, ValueError):
                    continue
        return rows

    plotted_any = False
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    handles = []
    for model in MODEL_ORDER:
        rows = _rows_for(model)
        if not rows:
            continue
        shifts = sorted(set(r[0] for r in rows))
        errs_by_shift = {s: [r[1] for r in rows if r[0] == s] for s in shifts}
        means = np.array([np.mean(errs_by_shift[s]) for s in shifts])
        stds  = np.array([np.std(errs_by_shift[s])  for s in shifts])
        color = MODEL_COLOR.get(model, "#888")
        line = ax.errorbar(shifts, means, yerr=stds, fmt="o-", color=color,
                            capsize=3, lw=1.3, ms=4,
                            label=MODEL_LABELS.get(model, model))
        handles.append(line)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return None
    # Cyclic-FFT FP32 round-off floor: relative reconstruction error of a
    # round-trip cyclic shift on these problem sizes is empirically in the
    # 1e-3 to 1e-2 range. Annotated as a band, not a "pass/fail" line.
    ax.axhspan(1e-3, 1e-2, color="grey", alpha=0.12, linewidth=0)
    ax.text(0.98, 0.02, "FP32 FFT floor",
             color="black", fontsize=8, ha="right", va="bottom",
             style="italic", transform=ax.transAxes)
    ax.set_yscale("log")
    ax.set_xlabel("cyclic shift size $k$")
    ax.set_ylabel(r"$\| f(\rho_k x) - \rho_k f(x) \|_2 / \|f(x)\|_2$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=len(handles), frameon=False, fontsize=7.5,
               columnspacing=1.0, handlelength=1.4, handletextpad=0.4)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "F08_equivariance_test.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


# -------------------- main --------------------

def main():
    print(f"[paper-figs] working dir: {REPO}")
    stats = _try_json(STATS_PATH) or {}
    if not stats:
        warnings.warn(f"missing stats: {STATS_PATH}")
    data = gather_all_models()
    history_by_model = {m: load_history(m) for m in MODEL_ORDER}

    print("\n[paper-figs] data inventory")
    print(f"  paired_permutation.json: {'OK' if stats else 'MISSING'}")
    for m in MODEL_ORDER:
        n_cells = len(data.get(m, {}))
        n_hist = len(history_by_model.get(m, {}))
        print(f"  {MODEL_LABELS.get(m, m):<14}: {n_cells:>4} cells, {n_hist:>4} history.json")

    figs = []
    for name, fn, args in [
        # F01 dropped (2026-05-03) â€” same 4 paired-permutation improvement
        # numbers as tables/T01_headline_per_baseline.tex; bar chart adds
        # nothing the table doesn't, plus the title overlap was broken.
        # ("F01 headline bar",        fig01_headline_bar,       (stats,)),
        # F02 dropped (2026-05-03) â€” same data as T02_perfamily_relL2.tex.
        # Heatmap had a colormap-saturation problem: the LEMO no-FiLM column
        # (broken checkpoints, ~0.43) pushed vmax so high that the FNO vs
        # MarkovFNO difference (0.07 vs 0.11) was visually compressed.
        # T02 carries the per-family numbers without this artifact.
        # ("F02 per-family heatmap",  fig02_perfamily_heatmap,  (data,)),
        # F03 dropped (2026-05-03) â€” replaced by tables/T03_perregime_aggregated.tex.
        # Per-regime box plot was redundant with the per-regime breakdown table
        # once F02 was dropped; ranking is regime-stable across clean/lowres/
        # noisy (per C20 finding too) so the visual axis carries no signal.
        # ("F03 per-regime box",      fig03_perregime_box,      (data,)),
        # F04 dropped (2026-05-03) â€” Hedges-g bar chart had two visual bugs:
        # (i) the "large (g=0.8)" annotation collides with the title, and
        # (ii) the vs-LEMO-no-FiLM bar at g=23.43 (from broken checkpoints)
        # distorts the x-axis so the genuine g=5-6 bars look small. Folded
        # into tables/T01_headline_per_baseline.tex as a Hedges-g column.
        # ("F04 effect size",         fig04_effect_size,        (stats,)),
        ("F05 training curves",     fig05_training_curves,    (history_by_model,)),
        ("F06 per-frame rollout",   fig06_perframe_rollout,   ()),
        ("F07 op-norm trajectory",  fig07_op_norm_trajectory, (history_by_model.get("lemo_pc_nd", {}),)),
        ("F08 equivariance test",   fig08_equivariance_test,  ()),
    ]:
        try:
            out = fn(*args)
        except Exception as e:
            out = None
            print(f"  {name:<28}: FAIL ({type(e).__name__}: {e})")
            continue
        if out is None:
            print(f"  {name:<28}: skip (data missing)")
        else:
            figs.append(out)
            print(f"  {name:<28}: -> {out.name}")
    print(f"\n[paper-figs] generated {len(figs)} figures in {FIG_DIR}")
    return figs


if __name__ == "__main__":
    main()
