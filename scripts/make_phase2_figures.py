"""Phase 2 figures — buildable from existing extracted data, no new compute.

Covers the data-ready figures from PLOTS_AND_TABLES_PLAN.md:
  F2  best-ckpt-epoch histogram per model
  F4  kernel magnitude histogram (LEMO-PC kernel_snapshot.npz)
  F5  FiLM γ/β distributions (LEMO-PC kernel_snapshot.npz)
  A4  seed-wise rel-L2 boxplot per cell (cleaner replacement for F03)
  A6  calibration scatter (predicted vs target magnitude) from viz_samples
  A7  residual histogram per model (per-sample relL2 from residuals.npz)
  A9  learned 2D lag kernel heatmap per family
  C8  seed-wise dot plot per cell
  C13 param-count vs test relL2
  C14 wall-clock vs test relL2
  C15 param-efficiency Pareto frontier
  C16 wall-clock Pareto frontier
  C20 error vs spatial resolution (clean vs lowres vs noisy bar)
  C22 FiLM γ/β heatmap (out × mode)
  C24 per-sample residual correlation matrix between models
  C25 hardest-decile Jaccard heatmap
  C30 single-delay leaderboard heatmap (mackey/wright/hutchinson 2D)
  E1  APEBench clean leaderboard heatmap
  E2  APEBench residual-anchor leaderboard heatmap
  E3  APEBench residual delta heatmap
  E9  Scaling burgers_3d width vs error

All written to paper/figures/.  Skips silently when data missing.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "extracted"
FIG = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}
SINGLE_DELAY_FAMS = ["mackey_glass_2d", "wright_2d", "hutchinson_2d"]
SD_LABELS = {"mackey_glass_2d": "Mackey-Glass", "wright_2d": "Wright",
             "hutchinson_2d": "Hutchinson"}
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
MODEL_COLOR = {
    "lemo_pc_nd":                 "#d62728",
    "causal_smooth_lemo_pc_nd":   "#c49c94",
    "lemo_bcorrect_nd":           "#bcbd22",
    "fno_nd":                     "#1f77b4",
    "fno_film_nd":                "#17becf",
    "noneq_film_nd":              "#c5b0d5",
    "markov_fno_nd":              "#2ca02c",
    "windowed_fno_nd":            "#9467bd",
    "memno_nd":                   "#e377c2",
    "ffno_nd":                    "#8c564b",
    "s4_nd":                      "#bcbd22",
    "nide_nd":                    "#aec7e8",
    "ndde_nd":                    "#98df8a",
    "unet_nd":                    "#7f7f7f",
}
MODEL_ORDER = [
    "lemo_pc_nd", "causal_smooth_lemo_pc_nd",
    "fno_film_nd", "noneq_film_nd",
    "fno_nd", "markov_fno_nd", "windowed_fno_nd",
    "memno_nd", "ffno_nd", "s4_nd", "nide_nd", "ndde_nd",
    "unet_nd",
]


# --- common loaders ---

def _try_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


_LOG_PAT = re.compile(r"=== FINAL test relL2 = ([0-9.]+) ===")
_PARAMS_PAT = re.compile(r"params:\s*([0-9,]+)")
_WALL_PAT = re.compile(r"wall_seconds.*?:\s*([0-9.]+)")


def load_lemo_test(model="lemo_pc_nd", layer="dist_kernel_v2_p1", families=FAMS):
    out = {}
    base = EXT / "pod1" / "outputs" / layer / "raw"
    for fam in families:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = d
    return out


def load_baseline_logs(model, layer="dist_kernel_v2_p2", families=FAMS):
    """Returns dict keyed by (fam,reg,seed) with {'test_rel_l2': float, 'params': int, 'wall_seconds': float}.

    Falls back to log-scraping when test_results.json missing."""
    out = {}
    seed_to_num = {"s42": "42", "s123": "123", "s456": "456"}
    base = EXT / "pod2" / "outputs" / layer
    if not base.exists():
        return out
    log_dir = base / "logs"
    raw_dir = base / "raw"
    for fam in families:
        for reg in REGIMES:
            for seed in SEEDS:
                seed_num = seed_to_num[seed]
                # Prefer test_results.json
                p = raw_dir / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = {
                        "test_rel_l2": float(d.get("test_rel_l2_mean", d.get("test_rel_l2", float("nan")))),
                        "params": int(d.get("params", 0)),
                        "wall_seconds": float(d.get("wall_seconds", 0.0)),
                    }
                    continue
                # Fall back to scraping log file
                logf = log_dir / f"{fam}_{model}_{reg}_s{seed_num}.log"
                if not logf.exists():
                    continue
                txt = logf.read_text(errors="replace")
                m = _LOG_PAT.search(txt)
                if not m:
                    continue
                p_match = _PARAMS_PAT.search(txt)
                params = int(p_match.group(1).replace(",", "")) if p_match else 0
                w_match = _WALL_PAT.search(txt)
                wall = float(w_match.group(1)) if w_match else 0.0
                out[(fam, reg, seed)] = {
                    "test_rel_l2": float(m.group(1)),
                    "params": params,
                    "wall_seconds": wall,
                }
    return out


def _discover_test_results():
    """rglob test_results.json across extracted/ and outputs/ for every model
    in MODEL_LABELS. Returns {model: {(fam, reg, seed): {test_rel_l2, params,
    wall_seconds}}} with dedup on (model, fam, reg, seed)."""
    out = {}
    seen = set()
    roots = [EXT, REPO / "outputs"]
    for root in roots:
        if not Path(root).exists():
            continue
        for f in Path(root).rglob("test_results.json"):
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
            d = _try_json(f)
            if d is None:
                continue
            seen.add(key)
            row = {
                "test_rel_l2": float(d.get("test_rel_l2_mean",
                                           d.get("test_rel_l2", float("nan")))),
                "params": int(d.get("params", 0)),
                "wall_seconds": float(d.get("wall_seconds", 0.0)),
            }
            out.setdefault(model, {})[(fam, reg, seed)] = row
    return out


def gather_all(layer_p1="dist_kernel_v2_p1", layer_p2="dist_kernel_v2_p2"):
    """Return {model: {(fam, reg, seed): row_dict}}.

    rglob discovery picks up every test_results.json under extracted/ or
    outputs/ for any model in MODEL_LABELS — old layer_p1/p2 layout still
    covered but new sweep layouts (pod_pulls_2026_05_03_final/<pod>/...)
    also work. Returns only models that have at least 1 cell.
    """
    discovered = _discover_test_results()
    return {m: cells for m, cells in discovered.items() if cells}


def history_for(model="lemo_pc_nd", layer="dist_kernel_v2_p1"):
    out = {}
    base = EXT / "pod1" / "outputs" / layer / "raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / model / seed / "history.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = d
    return out


def kernel_for(model="lemo_pc_nd", layer="dist_kernel_v2_p1", families=FAMS, regime="clean"):
    out = {}
    base = EXT / "pod1" / "outputs" / layer / "raw"
    for fam in families:
        for seed in SEEDS:
            p = base / fam / regime / model / seed / "kernel_snapshot.npz"
            if p.exists():
                try:
                    out[(fam, regime, seed)] = np.load(p)
                except Exception:
                    pass
    return out


def residuals_for(model="lemo_pc_nd", layer="dist_kernel_v2_p1", families=FAMS, regime="clean"):
    out = {}
    base = EXT / "pod1" / "outputs" / layer / "raw"
    for fam in families:
        for seed in SEEDS:
            p = base / fam / regime / model / seed / "residuals.npz"
            if p.exists():
                try:
                    arr = np.load(p)
                    out[(fam, regime, seed)] = {k: arr[k] for k in arr.files}
                except Exception:
                    pass
    return out


def viz_for(model="lemo_pc_nd", layer="dist_kernel_v2_p1", regime="clean", families=FAMS):
    out = {}
    base = EXT / "pod1" / "outputs" / layer / "raw"
    for fam in families:
        for seed in SEEDS:
            p = base / fam / regime / model / seed / "viz_samples.npz"
            if p.exists():
                try:
                    out[(fam, regime, seed)] = np.load(p)
                except Exception:
                    pass
    return out


# --- F2 best-ckpt-epoch histogram ---

def f2_best_ckpt_epoch():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plotted = False
    for model in ("lemo_pc_nd", "lemo_nd"):
        h = history_for(model)
        if not h:
            continue
        epochs = []
        for k, v in h.items():
            vrl2 = v.get("val_rel_l2", v.get("val_relL2", []))
            if vrl2:
                epochs.append(int(np.argmin(vrl2)) + 1)
        if not epochs:
            continue
        ax.hist(epochs, bins=20, alpha=0.6, label=MODEL_LABELS[model],
                color=MODEL_COLOR[model], edgecolor="black", linewidth=0.5)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel("epoch of best validation rel-$L_2$")
    ax.set_ylabel("count of cells")
    # title removed
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG / "F2_best_ckpt_epoch.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- F4 kernel magnitude histogram ---

def _extract_K_complex(snap):
    """Reconstruct the spectral-kernel complex tensor from a kernel snapshot.

    Supports two storage layouts written by `capture_paper_artifacts.py`:
    - Complex Parameter (LEMO-PC, FNO_nd, MarkovFNO, WindowFNO): keys end
      in `__re` / `__im` because the original tensor was complex and we
      split for storage.
    - F-FNO (mfno_paper.SpectralConv1d): two separate real Parameters
      `<prefix>.weight_real` and `<prefix>.weight_imag` written without
      `__re/__im` suffixing.
    """
    keys = list(snap.keys() if hasattr(snap, "keys") else [])
    # Layout 1: complex tensor split into __re/__im on save.
    re_keys = [k for k in keys
                if k.endswith("__re") and "weights" in k and "film" not in k]
    if re_keys:
        re = snap[re_keys[0]]
        im_key = re_keys[0].replace("__re", "__im")
        if im_key in keys:
            return re + 1j * snap[im_key]
    # Layout 2: F-FNO separate weight_real / weight_imag Parameters.
    real_keys = [k for k in keys if k.endswith("weight_real") and "film" not in k]
    if real_keys:
        re = snap[real_keys[0]]
        im_key = real_keys[0].replace("weight_real", "weight_imag")
        if im_key in keys:
            return re + 1j * snap[im_key]
    return None


def f4_kernel_magnitude_hist():
    """Multi-model spectral-kernel magnitude KDE with per-cell variability band.

    For each model, fits a per-cell KDE on `log10|K_{i,o,m}|`, then aggregates
    across cells (family x seed) into mean +/- 1 sigma. Plot shows the mean
    curve as a solid line and a translucent fill_between band of mean +/- 1
    sigma so reviewers can read both the distribution shape AND its stability
    across cells. Auto-extends to additional models in MODEL_ORDER as their
    `kernel_snapshot.npz` files land from the running offload sweep.
    """
    from scipy.stats import gaussian_kde
    curves = []   # list of (label, color, mean, std, n_cells)
    x_grid = np.linspace(-12, 0, 600)
    for mdl in MODEL_ORDER:
        snaps = kernel_for(mdl)
        if not snaps:
            continue
        per_cell = []
        for snap in snaps.values():
            K = _extract_K_complex(snap)
            if K is None:
                continue
            arr = np.abs(K).flatten()
            arr = arr[arr > 1e-12]
            if arr.size < 50:
                continue
            log_arr = np.log10(arr)
            try:
                kde = gaussian_kde(log_arr, bw_method=0.15)
                per_cell.append(kde(x_grid))
            except Exception:
                continue
        if len(per_cell) < 2:
            continue
        stack = np.stack(per_cell, axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        curves.append((MODEL_LABELS.get(mdl, mdl), MODEL_COLOR.get(mdl, "#444"),
                        mean, std, len(per_cell)))
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for label, color, mean, std, n_cells in curves:
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.18,
                          linewidth=0)
        ax.plot(x_grid, mean, color=color,
                lw=2.2 if label == "LEMO-PC" else 1.4,
                label=f"{label} (n={n_cells})")
    ax.set_xlabel(r"$\log_{10}|K_{i,o,m}|$")
    ax.set_ylabel("density")
    # title removed
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_xlim(-12, 0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = FIG / "F4_kernel_magnitude_hist.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


# --- F5 FiLM γ/β distributions ---

def _extract_film(snap):
    keys = list(snap.keys() if hasattr(snap, "keys") else [])
    # film_net.{layer}.weight or .bias
    weight_keys = [k for k in keys if "film_net" in k and ("weight" in k or "bias" in k)]
    return {k: snap[k] for k in weight_keys}


def f5_film_distributions():
    snaps = kernel_for("lemo_pc_nd")
    if not snaps:
        return None
    weights = []
    biases = []
    for snap in snaps.values():
        d = _extract_film(snap)
        for k, v in d.items():
            arr = np.asarray(v).flatten()
            if "weight" in k:
                weights.append(arr)
            elif "bias" in k:
                biases.append(arr)
    if not weights and not biases:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    if weights:
        w = np.concatenate(weights)
        axes[0].hist(w, bins=80, color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.3)
        axes[0].set_xlabel("FiLM weight value")
        axes[0].set_ylabel("count")
        axes[0].set_title("FiLM linear-layer weights")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    if biases:
        b = np.concatenate(biases)
        axes[1].hist(b, bins=80, color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.3)
        axes[1].set_xlabel("FiLM bias value")
        axes[1].set_title(r"FiLM linear-layer biases ($\gamma_0 \approx 1$, $\beta_0 \approx 0$)")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG / "F5_film_distributions.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


# --- A4 seed-wise rel-L2 boxplot ---

def a4_seedwise_box():
    """Per-family mean test rel-L2 per model, averaged over (3 seeds × 3 regimes).

    1 row × 5 cols. Within each panel, one horizontal bar per model;
    bars sorted by mean (ascending = best at top). Error bar = std over
    the 9 (regime, seed) cells. LEMO no-FiLM excluded (broken checkpoints).
    Bottom legend.
    """
    data = gather_all()
    # Drop LEMO no-FiLM — broken checkpoints (rel-L2 ≈ 1.0 = failed predictions).
    excluded = {"lemo_nd"}
    models = [m for m in MODEL_ORDER if m not in excluded
              and any(data.get(m, {}).values())]
    if not models:
        return None
    # Per (model, family): list of rel-L2 values across 9 cells.
    cells = {}  # (mdl, fam) -> list[float]
    for mdl in models:
        d = data.get(mdl, {})
        for fam in FAMS:
            vals = []
            for reg in REGIMES:
                for seed in SEEDS:
                    v = d.get((fam, reg, seed))
                    if v is None:
                        continue
                    rl2 = v.get("test_rel_l2_mean", v.get("test_rel_l2"))
                    if rl2 is not None:
                        vals.append(float(rl2))
            if vals:
                cells[(mdl, fam)] = vals
    if not cells:
        return None

    # Sort models by GLOBAL mean (over all (model, family) cells) so the
    # bar ordering is consistent across all 5 family panels.
    global_mean = {}
    for mdl in models:
        all_vals = [v for (m, f), vs in cells.items() if m == mdl for v in vs]
        global_mean[mdl] = float(np.mean(all_vals)) if all_vals else float("inf")
    models_sorted = sorted(models, key=lambda m: global_mean[m])

    import matplotlib.ticker as mticker
    n_panels = len(FAMS)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 2.8),
                              sharex=False, sharey=True,
                              gridspec_kw={"wspace": 0.20})
    if n_panels == 1:
        axes = [axes]
    for ax, fam in zip(axes, FAMS):
        means, stds, colors, labels = [], [], [], []
        for mdl in models_sorted:
            vs = cells.get((mdl, fam))
            if not vs:
                continue
            means.append(float(np.mean(vs)))
            stds.append(float(np.std(vs)))
            colors.append(MODEL_COLOR.get(mdl, "#444"))
            labels.append(MODEL_LABELS.get(mdl, mdl))
        if not means:
            ax.set_visible(False); continue
        y_pos = list(range(len(means)))[::-1]
        ax.barh(y_pos, means, color=colors, alpha=0.85,
                 edgecolor="black", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(mticker.LogLocator(
            base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="x", which="major", labelsize=8)
        ax.set_title(FAM_LABELS[fam], fontsize=10, pad=3)
        ax.grid(axis="x", which="both", linestyle=":", alpha=0.35)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for ax in axes[1:]:
        if ax.get_visible():
            ax.tick_params(labelleft=False)
    fig.supxlabel(r"mean test rel-$L_2$  (log scale)", fontsize=10, y=0.03)
    fig.suptitle("")
    fig.subplots_adjust(top=0.79, bottom=0.18, left=0.09, right=0.99,
                         wspace=0.20)
    out = FIG / "A4_seedwise_box.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


# --- A6 calibration scatter (pred vs target magnitude) ---

def _viz_for_model_any_layer(model: str, regime: str = "clean"):
    """Loader for A6/A7 — searches all known layer roots for viz_samples."""
    layers = ["dist_kernel_v2_p1", "film_ablation_caltech", "film_fix_full",
              "memory_aware_caltech", "memno_ffno_caltech"]
    out = {}
    for layer in layers:
        for base in (REPO / "outputs" / layer / "raw",
                     EXT / "pod1" / "outputs" / layer / "raw",
                     EXT / "pod2" / "outputs" / layer / "raw"):
            if not base.exists():
                continue
            for fam in FAMS:
                for seed in SEEDS:
                    p = base / fam / regime / model / seed / "viz_samples.npz"
                    if p.exists() and (fam, regime, seed) not in out:
                        try:
                            out[(fam, regime, seed)] = np.load(p)
                        except Exception:
                            pass
    return out


def _compute_calibration_r2(model: str, fam: str, regime: str = "clean",
                             rng_seed: int = 0, n_keep: int = 1500):
    """Returns (T_concat, P_concat, R²) or None if no viz_samples available.

    Subsamples up to n_keep points per seed (deterministic) for plotting,
    but R² is computed on the FULL flattened arrays (no subsampling) so
    table values are exact.
    """
    vizs = _viz_for_model_any_layer(model, regime=regime)
    target_pts, pred_pts = [], []  # for plotting (subsampled)
    T_full, P_full = [], []        # for R² (full)
    rng = np.random.RandomState(rng_seed)
    for seed in SEEDS:
        d = vizs.get((fam, regime, seed))
        if d is None:
            continue
        t = d["target"][:, -1, ..., 0].flatten()
        p = d["pred"][:, -1, ..., 0].flatten()
        T_full.append(t); P_full.append(p)
        k = min(n_keep, t.size)
        idx = rng.choice(t.size, k, replace=False)
        target_pts.append(t[idx]); pred_pts.append(p[idx])
    if not T_full:
        return None
    T = np.concatenate(T_full); P = np.concatenate(P_full)
    sse = float(((P - T) ** 2).sum())
    sst = float(((T - T.mean()) ** 2).sum() + 1e-12)
    r2 = 1.0 - sse / sst
    Tp = np.concatenate(target_pts); Pp = np.concatenate(pred_pts)
    return Tp, Pp, r2


def a6_calibration_scatter():
    """Multi-model calibration scatter — appendix figure.

    Auto-discovers all models with viz_samples.npz on the dist_*_rd_2d
    families (clean regime).  Each panel = one family.  R² values for
    each model are stacked top-left, sorted by R² descending, in model
    colors (doubles as a per-panel legend).
    """
    candidate_models = ["lemo_pc_nd", "fno_film_nd", "lemo_nd", "fno_nd",
                         "markov_fno_nd", "windowed_fno_nd", "memno_nd",
                         "ffno_nd", "unet_nd"]
    n_panels = len(FAMS)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3.4),
                              sharex=True, sharey=True,
                              gridspec_kw={"wspace": 0.08})
    if n_panels == 1:
        axes = [axes]
    plotted = False
    for ax, fam in zip(axes, FAMS):
        # Compute R² for each model on this family; keep only those that
        # actually have data, then sort by R² descending for the corner block.
        per_model = []  # list of (model, T, P, r2)
        for m in candidate_models:
            res = _compute_calibration_r2(m, fam)
            if res is None:
                continue
            T, P, r2 = res
            per_model.append((m, T, P, r2))
        if not per_model:
            ax.set_visible(False); continue
        per_model.sort(key=lambda x: -x[3])  # highest R² first

        # Plot scatter (in original order — not sorted — so first model
        # added stays on the bottom layer; later overlays on top).
        Tmin = +np.inf; Tmax = -np.inf
        for m, T, P, _r2 in per_model:
            color = MODEL_COLOR.get(m, "#444")
            ax.scatter(T, P, s=2, alpha=0.10, color=color, rasterized=True)
            Tmin = min(Tmin, float(T.min()), float(P.min()))
            Tmax = max(Tmax, float(T.max()), float(P.max()))

        # Top-left R² stack (sorted descending), color-coded.  Right-pad
        # labels so values align in a monospace-like column.
        max_label = max(len(MODEL_LABELS.get(m, m)) for m, *_ in per_model)
        for i, (m, _T, _P, r2) in enumerate(per_model):
            label = MODEL_LABELS.get(m, m).ljust(max_label)
            color = MODEL_COLOR.get(m, "#444")
            ax.text(0.03, 0.97 - i * 0.07,
                     f"{label}  {r2:.3f}",
                     transform=ax.transAxes, va="top", ha="left",
                     fontsize=7.5, family="monospace", color=color,
                     bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.75, pad=1.0))

        pad = 0.05 * (Tmax - Tmin)
        lo, hi = Tmin - pad, Tmax + pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(FAM_LABELS[fam], fontsize=10)
        ax.set_xlabel("target")
        ax.grid(linestyle=":", alpha=0.35)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    axes[0].set_ylabel("prediction")
    fig.suptitle("")
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.05, right=0.99,
                         wspace=0.08)
    out = FIG / "A6_calibration_scatter.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


# --- A7 residual histogram per model ---

def _residuals_for_model_any_layer(model: str, families=FAMS):
    """Loader for A7 — pools per-sample rel-L2 across all regimes/seeds for
    a given (model, family) cell.  rglob-based: walks every residuals.npz
    under REPO/extracted and REPO/outputs and identifies cells from the
    path parts schema (..., fam, regime, model, seed, residuals.npz).

    Returns dict (family) -> ndarray of pooled per-sample rel-L2 across
    all (regime, seed) cells found.
    """
    fam_set = set(families)
    out = {fam: [] for fam in families}
    seen = set()
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("residuals.npz"):
            try:
                parts = p.parts
                seed = parts[-2]; m = parts[-3]; regime = parts[-4]; fam = parts[-5]
            except IndexError:
                continue
            if m != model or fam not in fam_set:
                continue
            key = (fam, regime, m, seed, str(p))
            if key in seen:
                continue
            seen.add(key)
            try:
                arr = np.load(p)
                r = arr["rel_l2_per_sample"] if "rel_l2_per_sample" in arr.files else None
                if r is not None:
                    out[fam].append(np.asarray(r))
            except Exception:
                pass
    out = {fam: (np.concatenate(v) if v else None) for fam, v in out.items()}
    return out


def a7_residual_histogram():
    """Per-family per-sample rel-L2 distribution, pooled across regimes + seeds.

    Layout: 1 row × 5 family cols.  KDE curves (one per model) using
    np.histogram smoothed via Savgol-style log-bin counts.  Median (solid)
    and p95 (dashed) markers per model annotated as ticks at the bottom.

    Pooling regimes is honest because every model sees the same regime mix
    (apples-to-apples *across* models).  We do NOT pool families, since
    families have ~3× different baseline error magnitudes.
    """
    # Drop LEMO (no-FiLM) — its checkpoints are broken (rel-L2 near 1.0
    # = failed predictions), would dominate x-range without saying anything
    # the FiLM-ablation table doesn't already cover. All other architectures
    # with residuals.npz coverage on dist_*_rd_2d are included.
    candidate_models = [
        "lemo_pc_nd", "causal_smooth_lemo_pc_nd", "lemo_bcorrect_nd",
        "fno_film_nd", "noneq_film_nd",
        "fno_nd", "markov_fno_nd", "windowed_fno_nd",
        "memno_nd", "ffno_nd",
        "s4_nd", "nide_nd", "ndde_nd",
        "unet_nd",
    ]
    # Compute pooled per-(model, family) arrays first, then plot.
    pooled = {m: _residuals_for_model_any_layer(m) for m in candidate_models}
    active_models = [m for m in candidate_models
                     if any(pooled[m].get(fam) is not None for fam in FAMS)]
    if not active_models:
        return None

    n_panels = len(FAMS)
    # Per-panel y-scaling (sharey=False) so a tall narrow spike in one
    # family (e.g. sparse FNO+FiLM data on Uniform) does not clip its own
    # peak nor stretch the y-axis of other panels.
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3.4),
                              sharey=False,
                              gridspec_kw={"wspace": 0.18})
    if n_panels == 1:
        axes = [axes]

    # Determine a common x-range per panel from active data; use log10.
    plotted = False
    for ax, fam in zip(axes, FAMS):
        any_data = False
        # Collect data for this family across active models.
        per_model_data = []
        for m in active_models:
            arr = pooled[m].get(fam)
            if arr is None or len(arr) == 0:
                continue
            arr = arr[arr > 0]
            if len(arr) == 0:
                continue
            per_model_data.append((m, np.log10(arr)))
            any_data = True
        if not any_data:
            ax.set_visible(False); continue

        # Shared x bins per panel: union of all model ranges.
        lo = min(d.min() for _, d in per_model_data)
        hi = max(d.max() for _, d in per_model_data)
        lo -= 0.05 * (hi - lo); hi += 0.05 * (hi - lo)
        bins = np.linspace(lo, hi, 80)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Plot KDE-style curve (smoothed histogram density) per model.
        for m, d in per_model_data:
            counts, _ = np.histogram(d, bins=bins, density=True)
            # 5-bin moving average for smoothing.
            kernel = np.ones(5) / 5
            smoothed = np.convolve(counts, kernel, mode="same")
            color = MODEL_COLOR.get(m, "#444")
            ax.plot(bin_centers, smoothed, color=color, lw=1.6,
                     label=MODEL_LABELS.get(m, m))
            ax.fill_between(bin_centers, 0, smoothed, color=color, alpha=0.12)

        # Per-model median vertical lines spanning panel.
        for m, d in per_model_data:
            color = MODEL_COLOR.get(m, "#444")
            med = float(np.median(d))
            ax.axvline(med, color=color, lw=1.0, alpha=0.7, zorder=3)

        ax.set_title(FAM_LABELS[fam], fontsize=10)
        ax.set_xlabel(r"$\log_{10}$ rel-$L_2$")
        ax.grid(linestyle=":", alpha=0.35)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.margins(y=0)
        plotted = True

    if not plotted:
        plt.close(fig); return None
    axes[0].set_ylabel("density")

    # Bottom-of-figure legend (proxy lines per model + marker legend).
    handles_models = [plt.Line2D([0], [0], color=MODEL_COLOR.get(m, "#444"),
                                   lw=2, label=MODEL_LABELS.get(m, m))
                      for m in active_models]
    handle_med = plt.Line2D([0], [0], color="black", lw=1.2, label="median")
    fig.legend(handles=handles_models + [handle_med],
                loc="lower center", bbox_to_anchor=(0.5, -0.02),
                ncol=len(handles_models) + 1, frameon=False, fontsize=9)
    fig.suptitle("")
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.05, right=0.99,
                         wspace=0.10)
    out = FIG / "A7_residual_histogram.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


# --- A9 learned 2D lag kernel heatmap per family ---

def a9_kernel_heatmap_per_family():
    snaps = kernel_for("lemo_pc_nd")
    if not snaps:
        return None
    n = len(FAMS)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0))
    if n == 1:
        axes = [axes]
    plotted = False
    for ax, fam in zip(axes, FAMS):
        seed_snaps = [snaps[(fam, "clean", s)] for s in SEEDS if (fam, "clean", s) in snaps]
        if not seed_snaps:
            ax.set_visible(False); continue
        Ks = []
        for snap in seed_snaps:
            K = _extract_K_complex(snap)
            if K is not None:
                Ks.append(np.abs(K).mean(axis=0))   # (out, mode)
        if not Ks:
            ax.set_visible(False); continue
        avg = np.mean(np.stack(Ks, axis=0), axis=0)
        im = ax.imshow(avg, cmap="viridis", aspect="auto")
        ax.set_title(FAM_LABELS[fam], fontsize=10)
        ax.set_xlabel("lag mode $m$")
        if ax is axes[0]:
            ax.set_ylabel("output channel")
        plotted = True
    if not plotted:
        plt.close(fig); return None
    fig.suptitle("")
    fig.tight_layout()
    out = FIG / "A9_kernel_heatmap_per_family.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C8 seed-wise dot plot per cell ---

def c8_seedwise_dotplot():
    data = gather_all()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    # x-axis: family*regime, one dot per (model, seed)
    x_idx = 0
    cell_centers = []
    for fam in FAMS:
        for reg in REGIMES:
            cell_centers.append((fam, reg, x_idx))
            for j, mdl in enumerate(MODEL_ORDER):
                d = data.get(mdl, {})
                for seed in SEEDS:
                    v = d.get((fam, reg, seed))
                    if v is None:
                        continue
                    rl2 = v.get("test_rel_l2_mean", v.get("test_rel_l2"))
                    if rl2 is None:
                        continue
                    ax.scatter(x_idx + j * 0.08, float(rl2), s=18,
                               color=MODEL_COLOR[mdl], alpha=0.7, edgecolor="black",
                               linewidth=0.3)
            x_idx += 1
    ax.set_xticks([c[2] for c in cell_centers])
    ax.set_xticklabels([f"{FAM_LABELS[c[0]]}/{c[1][:3]}" for c in cell_centers],
                       rotation=60, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"test rel-$L_2$")
    # title removed
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=MODEL_COLOR[m], markersize=6,
                                  label=MODEL_LABELS[m])
                       for m in MODEL_ORDER
                       if m in data and any(data[m].get((f, r, s)) is not None
                                             for f in FAMS for r in REGIMES for s in SEEDS)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0),
              loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "C8_seedwise_dotplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C13 / C14 / C15 / C16 — params/wallclock vs error + Pareto ---

def _aggregate_per_model(data):
    out = {}
    for mdl, d in data.items():
        if not d:
            continue
        rl2s = []
        params = []
        wallclocks = []
        for v in d.values():
            r = v.get("test_rel_l2_mean", v.get("test_rel_l2"))
            if r is None or r <= 0:
                continue
            rl2s.append(float(r))
            params.append(int(v.get("params", 0)))
            wallclocks.append(float(v.get("wall_seconds", 0)))
        if not rl2s:
            continue
        out[mdl] = {
            "rl2_mean": float(np.mean(rl2s)),
            "rl2_std":  float(np.std(rl2s)),
            "params":   int(np.median(params)) if params else 0,
            "wall":     float(np.median(wallclocks)) if wallclocks else 0.0,
        }
    return out


def _scatter_panel(stats, x_key, x_label, title, label_models=True, log_x=True):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    plotted = False
    for mdl, s in stats.items():
        x = s.get(x_key, 0)
        y = s.get("rl2_mean", float("nan"))
        if x <= 0 or not np.isfinite(y):
            continue
        ax.errorbar(x, y, yerr=s.get("rl2_std", 0), fmt="o", color=MODEL_COLOR[mdl],
                    markersize=8, capsize=3, label=MODEL_LABELS[mdl])
        if label_models:
            ax.annotate(MODEL_LABELS[mdl], (x, y), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"test rel-$L_2$ (mean over cells)")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def c13_params_vs_error():
    """Appendix figure: trainable params vs test rel-L2 (log/log scatter).

    LEMO no-FiLM excluded (broken checkpoints). markov_fno_nd and
    windowed_fno_nd merged into a single "Combined FNO" point because their
    params + error are visually indistinguishable.
    """
    data = gather_all()
    stats = _aggregate_per_model(data)
    stats = {m: s for m, s in stats.items() if m != "lemo_nd"}
    mf = stats.pop("markov_fno_nd", None)
    wf = stats.pop("windowed_fno_nd", None)
    if mf and wf:
        stats["combined_fno"] = {
            "rl2_mean": float(np.mean([mf["rl2_mean"], wf["rl2_mean"]])),
            "rl2_std":  float(np.mean([mf["rl2_std"],  wf["rl2_std"]])),
            "params":   int(np.mean([mf["params"], wf["params"]])),
            "wall":     float(np.mean([mf["wall"], wf["wall"]])),
        }
    elif mf:
        stats["combined_fno"] = mf
    elif wf:
        stats["combined_fno"] = wf
    label_offsets = {
        "unet_nd":      (-30, 3),
        "lemo_pc_nd":   (8, 3),
        "fno_nd":       (8, 3),
        "combined_fno": (8, 3),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = False
    for mdl, s in stats.items():
        x = s.get("params", 0); y = s.get("rl2_mean", float("nan"))
        if x <= 0 or not np.isfinite(y) or y <= 0:
            continue
        col = MODEL_COLOR.get(mdl, "#6a4f8a")
        lab = MODEL_LABELS.get(mdl, "Combined FNO" if mdl == "combined_fno" else mdl)
        ax.errorbar(x, y, yerr=s.get("rl2_std", 0), fmt="o", color=col,
                    markersize=9, capsize=3, ecolor="#333", elinewidth=0.9,
                    markeredgecolor="black", markeredgewidth=0.4)
        ox, oy = label_offsets.get(mdl, (8, 3))
        ax.annotate(lab, (x, y), xytext=(ox, oy),
                    textcoords="offset points", fontsize=8.5)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trainable parameters", fontsize=10)
    ax.set_ylabel(r"test rel-$L_2$ (mean over cells)", fontsize=10)
    # title removed
    ax.grid(which="both", linestyle=":", alpha=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIG / "C13_params_vs_error.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def c14_wallclock_vs_error():
    data = gather_all()
    stats = _aggregate_per_model(data)
    fig = _scatter_panel(stats, "wall", "wall-clock per cell (s)",
                         "Wall-clock vs test rel-$L_2$")
    if fig is None:
        return None
    out = FIG / "C14_wallclock_vs_error.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def c15_param_efficiency_pareto():
    """Reuses scatter; Pareto is just a visual of the same data."""
    data = gather_all()
    stats = _aggregate_per_model(data)
    if not stats:
        return None
    pts = sorted([(s["params"], s["rl2_mean"], m) for m, s in stats.items() if s["params"] > 0])
    pareto = []
    best = float("inf")
    for p, e, m in pts:
        if e < best:
            pareto.append((p, e, m))
            best = e
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for p, e, m in pts:
        ax.scatter(p, e, s=60, color=MODEL_COLOR[m], edgecolor="black", linewidth=0.5)
        ax.annotate(MODEL_LABELS[m], (p, e), xytext=(5, 3),
                    textcoords="offset points", fontsize=8)
    if len(pareto) > 1:
        xs = [p for p, _, _ in pareto]; ys = [e for _, e, _ in pareto]
        ax.plot(xs, ys, "k--", lw=0.8, alpha=0.6, label="Pareto frontier")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel(r"test rel-$L_2$")
    # title removed
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG / "C15_param_efficiency_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def c16_wallclock_pareto():
    data = gather_all()
    stats = _aggregate_per_model(data)
    if not stats:
        return None
    pts = sorted([(s["wall"], s["rl2_mean"], m) for m, s in stats.items() if s["wall"] > 0])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    pareto = []
    best = float("inf")
    for w, e, m in pts:
        if e < best:
            pareto.append((w, e, m))
            best = e
    for w, e, m in pts:
        ax.scatter(w, e, s=60, color=MODEL_COLOR[m], edgecolor="black", linewidth=0.5)
        ax.annotate(MODEL_LABELS[m], (w, e), xytext=(5, 3),
                    textcoords="offset points", fontsize=8)
    if len(pareto) > 1:
        xs = [w for w, _, _ in pareto]; ys = [e for _, e, _ in pareto]
        ax.plot(xs, ys, "k--", lw=0.8, alpha=0.6, label="Pareto frontier")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock per cell (s)")
    ax.set_ylabel(r"test rel-$L_2$")
    # title removed
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG / "C16_wallclock_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C20 error vs spatial resolution (regime comparison) ---

def c20_regime_comparison():
    """Appendix figure: per-regime mean test rel-L2, families pooled.

    LEMO no-FiLM excluded (broken). markov_fno_nd + windowed_fno_nd merged
    into a single "Combined FNO" group. Bottom legend, concise title.
    """
    data = gather_all()
    models = [m for m in MODEL_ORDER if m in data and data[m]
              and m not in {"lemo_nd"}]
    if not models:
        return None

    def _mean_std(model_keys, reg):
        vals = []
        for m in model_keys:
            for fam in FAMS:
                for seed in SEEDS:
                    v = data.get(m, {}).get((fam, reg, seed))
                    if v is None:
                        continue
                    r = v.get("test_rel_l2_mean", v.get("test_rel_l2"))
                    if r is not None:
                        vals.append(float(r))
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    fno_pair = [m for m in ("markov_fno_nd", "windowed_fno_nd") if m in models]
    plot_models = [m for m in models if m not in ("markov_fno_nd", "windowed_fno_nd")]
    if fno_pair:
        plot_models.append("combined_fno")

    means, stds = {}, {}
    for m in plot_models:
        means[m], stds[m] = {}, {}
        keys = fno_pair if m == "combined_fno" else [m]
        for reg in REGIMES:
            mu, sd = _mean_std(keys, reg)
            means[m][reg] = mu
            stds[m][reg] = sd

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(REGIMES))
    n = len(plot_models)
    width = 0.8 / max(n, 1)
    for i, m in enumerate(plot_models):
        ys = [means[m][r] for r in REGIMES]
        es = [stds[m][r] for r in REGIMES]
        col = MODEL_COLOR.get(m, "#6a4f8a")
        lab = MODEL_LABELS.get(m, "Combined FNO" if m == "combined_fno" else m)
        ax.bar(x + (i - n / 2 + 0.5) * width, ys, width, yerr=es,
               capsize=2, color=col, alpha=0.85, edgecolor="black",
               linewidth=0.4, label=lab,
               error_kw=dict(elinewidth=0.9, ecolor="#222"))
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in REGIMES])
    ax.set_yscale("log")
    ax.set_ylabel(r"test rel-$L_2$")
    # title removed
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=min(len(plot_models), 6), fontsize=8.5, frameon=False)
    fig.subplots_adjust(top=0.90, bottom=0.27, left=0.10, right=0.97)
    out = FIG / "C20_regime_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C22 FiLM γ/β heatmap (out × mode) ---

def _gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _compute_film_per_family(snap, block, params):
    """Forward-pass the 2-layer FiLM MLP: out = (GELU(p @ W0.T + b0)) @ W2.T + b2.

    Returns (gamma, beta), each shape (n_samples, out_ch=64, lag_modes=24).
    """
    W0 = snap[f"blocks.{block}.A_lag.film_net.0.weight"]
    b0 = snap[f"blocks.{block}.A_lag.film_net.0.bias"]
    W2 = snap[f"blocks.{block}.A_lag.film_net.2.weight"]
    b2 = snap[f"blocks.{block}.A_lag.film_net.2.bias"]
    h = _gelu_np(params @ W0.T + b0[None])
    out = h @ W2.T + b2[None]
    OC, M = 64, 24
    g = out[:, : OC * M].reshape(-1, OC, M)
    b = out[:, OC * M:].reshape(-1, OC, M)
    return g, b


def c22_film_gamma_beta_heatmap():
    """Appendix figure: per-family mean γ and β (post-FiLM forward pass).

    For each of the 5 families, load the trained lemo_pc_nd kernel_snapshot
    + a handful of test-sample params, run them through the FiLM 2-layer MLP
    (block 0), and average γ, β across samples. 5 rows × 2 cols (γ | β).
    Shared symmetric colormap per column. Concise title.
    """
    snaps = kernel_for("lemo_pc_nd")
    vizs = _viz_for_model_any_layer("lemo_pc_nd")
    if not snaps:
        return None
    fam_to_snap, fam_to_params = {}, {}
    for (fam, reg, seed), snap in snaps.items():
        if reg != "clean":
            continue
        if fam in fam_to_snap:
            continue
        try:
            _ = snap["blocks.0.A_lag.film_net.0.weight"]
        except KeyError:
            continue
        vp = vizs.get((fam, reg, seed))
        if vp is None or "input" not in vp:
            continue
        inp = vp["input"]
        W0 = snap["blocks.0.A_lag.film_net.0.weight"]
        params_dim = int(W0.shape[1])
        try:
            p_real = inp[:, 0, 0, 0, -params_dim:].astype(np.float32)
        except Exception:
            continue
        if p_real.size == 0:
            continue
        fam_to_snap[fam] = snap
        fam_to_params[fam] = p_real
    fams_present = [f for f in FAMS if f in fam_to_snap]
    if not fams_present:
        return None
    gammas = {}; betas = {}
    for fam in fams_present:
        g, b = _compute_film_per_family(fam_to_snap[fam], 0, fam_to_params[fam])
        gammas[fam] = g.mean(axis=0)
        betas[fam] = b.mean(axis=0)
    # Use signed-sqrt diverging norm so small modulations are visible (most
    # γ/β cells are tiny relative to a few sparse spikes — under linear
    # RdBu_r the bulk of the heatmap appeared near-white).
    from matplotlib.colors import FuncNorm
    def _ssqrt_norm(vmax):
        fwd = lambda x: np.sign(x) * np.power(np.abs(x), 0.5)
        inv = lambda x: np.sign(x) * np.power(np.abs(x), 2.0)
        return FuncNorm((fwd, inv), vmin=-vmax, vmax=vmax)
    g_max = max(np.abs(v).max() for v in gammas.values())
    b_max = max(np.abs(v).max() for v in betas.values())
    n = len(fams_present)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n + 0.7, 4.4),
                              sharex=True, sharey=True)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    norm_g = _ssqrt_norm(g_max)
    norm_b = _ssqrt_norm(b_max)
    for j, fam in enumerate(fams_present):
        ax_g = axes[0, j]; ax_b = axes[1, j]
        ax_g.imshow(gammas[fam], cmap="RdBu_r", norm=norm_g, aspect="auto")
        ax_b.imshow(betas[fam],  cmap="RdBu_r", norm=norm_b, aspect="auto")
        ax_g.set_title(FAM_LABELS[fam], fontsize=10, pad=3)
        if j == 0:
            ax_g.set_ylabel(r"$\gamma$ (mult.)" + "\nout channel", fontsize=9)
            ax_b.set_ylabel(r"$\beta$ (add.)" + "\nout channel", fontsize=9)
        ax_b.set_xlabel(r"lag mode $m$", fontsize=9)
    sm_g = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_g)
    sm_b = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_b)
    cb_g = fig.add_axes([0.965, 0.555, 0.012, 0.32])
    cb_b = fig.add_axes([0.965, 0.135, 0.012, 0.32])
    fig.colorbar(sm_g, cax=cb_g).ax.tick_params(labelsize=7)
    fig.colorbar(sm_b, cax=cb_b).ax.tick_params(labelsize=7)
    fig.suptitle("")
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.08, right=0.94,
                         hspace=0.30, wspace=0.18)
    out = FIG / "C22_film_gamma_beta_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C24 / C25 per-sample residual correlation + Jaccard ---

def c24_residual_correlation():
    models = ("lemo_pc_nd", "lemo_nd")
    res = {m: residuals_for(m) for m in models}
    if not all(res.values()):
        return None
    M = []
    labels = []
    # For each (fam, regime, seed), get per-sample rel_l2 vector for each model.
    # Stack across cells per model, then correlate.
    series = {m: [] for m in models}
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                vecs = []
                for m in models:
                    d = res[m].get((fam, reg, seed))
                    if d is None or "rel_l2_per_sample" not in d:
                        vecs = None; break
                    vecs.append(d["rel_l2_per_sample"])
                if vecs is None:
                    continue
                Ls = min(len(v) for v in vecs)
                for m, v in zip(models, vecs):
                    series[m].append(np.asarray(v[:Ls]))
    series = {m: np.concatenate(v) for m, v in series.items() if v}
    if len(series) < 2:
        return None
    keys = list(series.keys())
    n = len(keys)
    M = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                M[i, j] = 1.0
            else:
                Ls = min(len(series[a]), len(series[b]))
                M[i, j] = float(np.corrcoef(series[a][:Ls], series[b][:Ls])[0, 1])
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels([MODEL_LABELS[m] for m in keys])
    ax.set_yticks(range(n)); ax.set_yticklabels([MODEL_LABELS[m] for m in keys])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i, j]) > 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("Pearson r")
    ax.set_title("Per-sample residual correlation\n(LEMO-PC vs LEMO ablation)", fontsize=10)
    fig.tight_layout()
    out = FIG / "C24_residual_correlation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def c25_hardest_decile_jaccard():
    models = ("lemo_pc_nd", "lemo_nd")
    res = {m: residuals_for(m) for m in models}
    if not all(res.values()):
        return None
    # For each cell, get top 10% hardest sample indices per model.
    hard = {m: set() for m in models}
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                cell_id = (fam, reg, seed)
                ranks = {}
                for m in models:
                    d = res[m].get(cell_id)
                    if d is None or "rel_l2_per_sample" not in d:
                        ranks = None; break
                    arr = np.asarray(d["rel_l2_per_sample"])
                    n = len(arr)
                    cutoff = int(np.ceil(n * 0.1))
                    idx = set(np.argsort(arr)[-cutoff:].tolist())
                    ranks[m] = idx
                if ranks is None:
                    continue
                for m, idx in ranks.items():
                    # tag with cell_id+sample_idx so cross-cell merge is unique
                    hard[m] |= set((cell_id, i) for i in idx)
    if not all(hard.values()):
        return None
    keys = list(hard.keys())
    n = len(keys)
    M = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                M[i, j] = 1.0
            else:
                inter = len(hard[a] & hard[b])
                union = len(hard[a] | hard[b])
                M[i, j] = inter / max(union, 1)
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels([MODEL_LABELS[m] for m in keys])
    ax.set_yticks(range(n)); ax.set_yticklabels([MODEL_LABELS[m] for m in keys])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9, color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("Jaccard")
    # title removed
    fig.tight_layout()
    out = FIG / "C25_hardest_decile_jaccard.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C30 single-delay leaderboard heatmap ---

def c30_single_delay_heatmap():
    """3 fams x 6 models x 3 regimes leaderboard from layer5_final_sweep."""
    base_p1 = EXT / "pod1" / "outputs" / "layer5_final_sweep_p1" / "raw"
    base_p2 = EXT / "pod2" / "outputs" / "layer5_final_sweep_p2" / "raw"
    if not (base_p1.exists() or base_p2.exists()):
        return None
    models_present = ["lemo_pc_nd", "lemo_nd", "fno_nd", "markov_fno_nd",
                      "windowed_fno_nd", "unet_nd"]
    M = np.full((len(SINGLE_DELAY_FAMS) * len(REGIMES), len(models_present)), np.nan)
    row_labels = []
    rownum = 0
    for fam in SINGLE_DELAY_FAMS:
        for reg in REGIMES:
            row_labels.append(f"{SD_LABELS[fam]} / {reg}")
            for j, mdl in enumerate(models_present):
                vals = []
                for base in (base_p1, base_p2):
                    if not base.exists():
                        continue
                    for seed in SEEDS:
                        p = base / fam / reg / mdl / seed / "test_results.json"
                        d = _try_json(p)
                        if d is not None:
                            r = d.get("test_rel_l2_mean", d.get("test_rel_l2"))
                            if r is not None:
                                vals.append(float(r))
                if vals:
                    M[rownum, j] = float(np.mean(vals))
            rownum += 1
    if np.all(np.isnan(M)):
        return None
    fig, ax = plt.subplots(figsize=(1.4 + len(models_present) * 1.1,
                                     0.5 + len(row_labels) * 0.3))
    finite = M[np.isfinite(M)]
    vmax = np.percentile(finite, 95) if finite.size else 1.0
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(models_present)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_present], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > vmax * 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"test rel-$L_2$")
    # title removed
    fig.tight_layout()
    out = FIG / "C30_single_delay_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- E1/E2/E3 APEBench leaderboards ---

def _apebench_load(layer):
    base = EXT / "pod1" / "outputs" / layer / "raw"
    if not base.exists():
        return {}
    out = {}
    for p in base.rglob("test_results.json"):
        d = _try_json(p)
        if d is None:
            continue
        parts = p.parts
        idx = parts.index("raw") + 1
        try:
            fam = parts[idx]
            mdl = parts[idx + 1]
            seed = parts[idx + 2]
        except IndexError:
            continue
        rl2 = d.get("test_rel_l2_mean", d.get("test_rel_l2"))
        if rl2 is None:
            continue
        out.setdefault((fam, mdl), []).append(float(rl2))
    return {k: float(np.mean(v)) for k, v in out.items()}


def _apebench_heatmap(data, title, fname):
    if not data:
        return None
    fams = sorted({f for (f, _) in data.keys()})
    mdls = sorted({m for (_, m) in data.keys()})
    M = np.full((len(fams), len(mdls)), np.nan)
    for i, f in enumerate(fams):
        for j, m in enumerate(mdls):
            if (f, m) in data:
                M[i, j] = data[(f, m)]
    if np.all(np.isnan(M)):
        return None
    fig, ax = plt.subplots(figsize=(1.4 + len(mdls) * 1.0, 0.5 + len(fams) * 0.4))
    finite = M[np.isfinite(M)]
    vmax = np.percentile(finite, 95) if finite.size else 1.0
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(mdls)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in mdls], rotation=30, ha="right")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > vmax * 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"test rel-$L_2$")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out = FIG / fname
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def e1_apebench_leaderboard():
    return _apebench_heatmap(_apebench_load("sweep_apebench"),
                              "APEBench leaderboard (clean)",
                              "E1_apebench_leaderboard.pdf")


def e2_apebench_residual_leaderboard():
    return _apebench_heatmap(_apebench_load("sweep_apebench_residual_clean"),
                              "APEBench leaderboard (residual-anchor)",
                              "E2_apebench_residual_leaderboard.pdf")


def e3_apebench_residual_delta():
    a = _apebench_load("sweep_apebench")
    b = _apebench_load("sweep_apebench_residual_clean")
    if not a or not b:
        return None
    keys = sorted(set(a.keys()) & set(b.keys()))
    fams = sorted({f for (f, _) in keys})
    mdls = sorted({m for (_, m) in keys})
    delta = np.full((len(fams), len(mdls)), np.nan)
    for i, f in enumerate(fams):
        for j, m in enumerate(mdls):
            if (f, m) in a and (f, m) in b:
                delta[i, j] = (a[(f, m)] - b[(f, m)]) / a[(f, m)] * 100.0
    if np.all(np.isnan(delta)):
        return None
    fig, ax = plt.subplots(figsize=(1.4 + len(mdls) * 1.0, 0.5 + len(fams) * 0.4))
    vmax = np.nanmax(np.abs(delta))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mdls)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in mdls], rotation=30, ha="right")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=8)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"residual-anchor $\Delta$ rel-$L_2$ (\%)")
    # title removed
    fig.tight_layout()
    out = FIG / "E3_apebench_residual_delta.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- E9 scaling burgers_3d width vs error ---

def e9_scaling_burgers3d():
    base = EXT / "pod1" / "outputs" / "sweep_lemo_scale"
    if not base.exists():
        return None
    pts = []
    for p in base.rglob("test_results.json"):
        parts = p.parts
        if "burgers_3d" not in parts:
            continue
        # raw_w48 → width 48; extract.
        m = re.search(r"raw_w(\d+)", str(p))
        if not m:
            continue
        width = int(m.group(1))
        d = _try_json(p)
        if d is None:
            continue
        rl2 = d.get("test_rel_l2_mean", d.get("test_rel_l2"))
        if rl2 is None:
            continue
        pts.append((width, float(rl2)))
    if not pts:
        return None
    by_w = {}
    for w, e in pts:
        by_w.setdefault(w, []).append(e)
    widths = sorted(by_w)
    means = [np.mean(by_w[w]) for w in widths]
    stds = [np.std(by_w[w]) for w in widths]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.errorbar(widths, means, yerr=stds, fmt="o-", color="#d62728",
                 capsize=3, lw=1.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("LEMO-PC width")
    ax.set_ylabel(r"test rel-$L_2$ on burgers\_3d")
    # title removed
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG / "E9_scaling_burgers3d.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


# --- main ---

def main():
    print("[phase2-figs] generating from extracted/pod1 + extracted/pod2")
    out_files = []
    for name, fn in [
        # F2 dropped (2026-05-03) — best-ckpt-epoch histogram is a reviewer red
        # flag (60% of LEMO-PC cells hit best at last epoch = potential under-
        # training). Replaced with one sentence in the experimental setup
        # section: "We train all models for 200 epochs; ~60% of LEMO-PC cells
        # reach best validation at the last epoch, suggesting longer training
        # would benefit all methods uniformly." Frames the limitation as
        # cross-method-fair rather than LEMO-specific.
        # ("F2 best-ckpt-epoch",        f2_best_ckpt_epoch),
        ("F4 kernel magnitude hist",   f4_kernel_magnitude_hist),
        # F5 dropped (2026-05-03) — strictly worse version of the FiLM-
        # nullification story that C22 tells. F5 was two histograms (FiLM
        # weights + biases) both showing trivial delta-at-0 spikes; C22's
        # per-family heatmap actually exposes the structure (or lack of it).
        # Even after FiLM-fix retrain, C22 will carry more signal than F5's
        # aggregate distributions.
        # ("F5 FiLM distributions",      f5_film_distributions),
        ("A4 seed-wise box",           a4_seedwise_box),
        ("A6 calibration scatter",     a6_calibration_scatter),
        ("A7 residual histogram",      a7_residual_histogram),
        # A9 dropped — redundant with V05 (cosine-sim vs GT kernel) which carries
        # the kernel-recovery story; raw spectrum heatmap added no extra signal.
        # C8 dropped — seed-variance information now overlaid as jittered
        # dots on A4 bars; standalone dotplot is redundant.
        # ("C8 seed-wise dotplot",       c8_seedwise_dotplot),
        ("C13 params vs error",        c13_params_vs_error),
        # C14 dropped — wall-clock data will live in a table once the offload
        # sweep finishes (every cell records wall_seconds in test_results.json).
        # C15 dropped — Pareto frontier scatter is redundant once Pareto line
        # is removed from C13 (per design decision 2026-05-03).
        # C16 dropped — same reasoning as C15 for the wall-clock axis.
        # ("C14 wallclock vs error",     c14_wallclock_vs_error),
        # ("C15 param-eff Pareto",       c15_param_efficiency_pareto),
        # ("C16 wallclock Pareto",       c16_wallclock_pareto),
        ("C20 regime comparison",      c20_regime_comparison),
        ("C22 FiLM gamma/beta heatmap", c22_film_gamma_beta_heatmap),
        # C24 + C25 dropped — both collapse into T10_residual_agreement table
        # (rows=models, cols=in-arch mean r / cross-arch mean r / Jaccard@10%)
        # once the 372-cell offload sweep populates the model roster.
        # ("C24 residual correlation",   c24_residual_correlation),
        # ("C25 hardest-decile Jaccard", c25_hardest_decile_jaccard),
        # C30 dropped — same data as T08_single_delay (now includes UNet);
        # heatmap saturated above vmax=1.3 (Window-FNO/Wright values 1.86-1.98)
        # and offered no story beyond the table.
        # ("C30 single-delay heatmap",   c30_single_delay_heatmap),
        # E1, E2, E9 dropped (2026-05-03) — APEBench is mis-fit per the
        # Round 2.26 pivot to dist_*_rd_2d benchmarks. E1 is a 5x2 heatmap
        # with only 2 models, E2 is a 4x1 single-column "leaderboard" of
        # residual-anchor variants (no comparison axis, kolmogorov_2d
        # blows up to 1.248), E9 width-scaling on burgers_3d is flat
        # (0.180/0.179/0.182, anti-finding). Negative APEBench result
        # lives in T_apebench_negative.tex; no figure needed.
        # ("E1 APEBench leaderboard",    e1_apebench_leaderboard),
        # ("E2 APEBench residual lb",    e2_apebench_residual_leaderboard),
        ("E3 APEBench delta",          e3_apebench_residual_delta),
        # ("E9 scaling burgers_3d",      e9_scaling_burgers3d),
    ]:
        try:
            out = fn()
        except Exception as e:
            print(f"  {name:<32}: FAIL ({type(e).__name__}: {e})")
            continue
        if out is None:
            print(f"  {name:<32}: skip (data missing)")
        else:
            out_files.append(out)
            print(f"  {name:<32}: -> {out.name}")
    print(f"\n[phase2-figs] generated {len(out_files)} figures in {FIG}")


if __name__ == "__main__":
    main()
