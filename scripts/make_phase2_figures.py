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
FIG = REPO / "paper" / "figures"
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

MODEL_LABELS = {"lemo_pc_nd": "LEMO-PC", "lemo_nd": "LEMO",
                "fno_nd": "FNO", "markov_fno_nd": "Markov-FNO",
                "windowed_fno_nd": "Window-FNO", "memno_nd": "MemNO",
                "ffno_nd": "F-FNO", "unet_nd": "UNet"}
MODEL_COLOR = {"lemo_pc_nd": "#d62728", "lemo_nd": "#ff7f0e",
               "fno_nd": "#1f77b4", "markov_fno_nd": "#2ca02c",
               "windowed_fno_nd": "#9467bd", "memno_nd": "#8c564b",
               "ffno_nd": "#e377c2", "unet_nd": "#7f7f7f"}
MODEL_ORDER = ["lemo_pc_nd", "lemo_nd", "fno_nd", "markov_fno_nd",
               "windowed_fno_nd", "memno_nd", "ffno_nd", "unet_nd"]


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


def gather_all(layer_p1="dist_kernel_v2_p1", layer_p2="dist_kernel_v2_p2"):
    return {
        "lemo_pc_nd":     load_lemo_test("lemo_pc_nd", layer=layer_p1),
        "lemo_nd":        load_lemo_test("lemo_nd",     layer=layer_p1),
        "fno_nd":         load_baseline_logs("fno_nd",         layer=layer_p2),
        "markov_fno_nd":  load_baseline_logs("markov_fno_nd",  layer=layer_p2),
        "windowed_fno_nd": load_baseline_logs("windowed_fno_nd", layer=layer_p2),
        "unet_nd":        load_baseline_logs("unet_nd",        layer=layer_p2),
    }


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
    ax.set_title("Best-checkpoint epoch distribution (45 cells per model)")
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
    keys = list(snap.keys() if hasattr(snap, "keys") else [])
    re_keys = [k for k in keys if k.endswith("__re") and "weights" in k and "film" not in k]
    if not re_keys:
        return None
    re = snap[re_keys[0]]
    im_key = re_keys[0].replace("__re", "__im")
    if im_key not in keys:
        return None
    return re + 1j * snap[im_key]


def f4_kernel_magnitude_hist():
    snaps = kernel_for("lemo_pc_nd")
    if not snaps:
        return None
    mags = []
    for snap in snaps.values():
        K = _extract_K_complex(snap)
        if K is None:
            continue
        mags.append(np.abs(K).flatten())
    if not mags:
        return None
    arr = np.concatenate(mags)
    arr = arr[arr > 1e-12]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(np.log10(arr), bins=80, color="#d62728", alpha=0.85, edgecolor="black",
            linewidth=0.3)
    ax.set_xlabel(r"$\log_{10}|K_{i,o,m}|$")
    ax.set_ylabel("count")
    ax.set_title("LEMO-PC spectral-kernel magnitude distribution (15 cells)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
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
    data = gather_all()
    rows = []
    for fam in FAMS:
        for mdl in MODEL_ORDER:
            d = data.get(mdl, {})
            for reg in REGIMES:
                for seed in SEEDS:
                    v = d.get((fam, reg, seed))
                    if v is None:
                        continue
                    rl2 = v.get("test_rel_l2_mean", v.get("test_rel_l2"))
                    if rl2 is None:
                        continue
                    rows.append((fam, mdl, reg, seed, float(rl2)))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.2))
    # Group per (fam, model) - one box per pair, all regimes pooled.
    fams = FAMS
    models = [m for m in MODEL_ORDER if any(r[1] == m for r in rows)]
    box_data = []
    box_labels = []
    box_colors = []
    x_centers = []
    cx = 0
    fam_centers = []
    for fam in fams:
        fam_start = cx
        for mdl in models:
            vals = [r[4] for r in rows if r[0] == fam and r[1] == mdl]
            if vals:
                box_data.append(vals)
                box_labels.append("")
                box_colors.append(MODEL_COLOR[mdl])
                x_centers.append(cx)
                cx += 1
        if cx > fam_start:
            fam_centers.append((fam, (fam_start + cx - 1) / 2))
            cx += 1   # gap between families
    if not box_data:
        plt.close(fig); return None
    bp = ax.boxplot(box_data, positions=x_centers, widths=0.7, patch_artist=True,
                     showfliers=False, medianprops={"color": "black"},
                     tick_labels=[""] * len(x_centers))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for j, vals in enumerate(box_data):
        ax.scatter([x_centers[j]] * len(vals), vals, color="black", s=10, alpha=0.5, zorder=3)
    ax.set_xticks([c for _, c in fam_centers])
    ax.set_xticklabels([FAM_LABELS[f] for f, _ in fam_centers], fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel(r"test rel-$L_2$ (log scale)")
    ax.set_title("Per-family seed-wise rel-$L_2$ distribution (all 3 regimes pooled)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Build legend OUTSIDE plot.
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m], alpha=0.7)
                       for m in models]
    ax.legend(legend_handles, [MODEL_LABELS[m] for m in models],
              bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "A4_seedwise_box.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- A6 calibration scatter (pred vs target magnitude) ---

def a6_calibration_scatter():
    vizs = viz_for("lemo_pc_nd")
    if not vizs:
        return None
    n_panels = len(FAMS)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.8 * n_panels, 3.0), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    plotted = False
    for ax, fam in zip(axes, FAMS):
        target_pts = []
        pred_pts = []
        for seed in SEEDS:
            d = vizs.get((fam, "clean", seed))
            if d is None:
                continue
            t = d["target"][:, -1, ..., 0].flatten()
            p = d["pred"][:, -1, ..., 0].flatten()
            n_keep = min(2000, t.size)
            idx = np.random.RandomState(0).choice(t.size, n_keep, replace=False)
            target_pts.append(t[idx]); pred_pts.append(p[idx])
        if not target_pts:
            ax.set_visible(False); continue
        T = np.concatenate(target_pts); P = np.concatenate(pred_pts)
        ax.scatter(T, P, s=2, alpha=0.25, color=MODEL_COLOR["lemo_pc_nd"])
        # y=x reference line.
        lim = max(np.abs(T).max(), np.abs(P).max())
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.6)
        ax.set_title(FAM_LABELS[fam], fontsize=10)
        ax.set_xlabel("target")
        ax.grid(linestyle="--", alpha=0.4)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    axes[0].set_ylabel("LEMO-PC pred")
    fig.suptitle("Calibration: target vs prediction at final rollout step", fontsize=11)
    out = FIG / "A6_calibration_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- A7 residual histogram per model ---

def a7_residual_histogram():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plotted = False
    for mdl in ("lemo_pc_nd", "lemo_nd"):
        res = residuals_for(mdl)
        if not res:
            continue
        all_rel = []
        for d in res.values():
            r = d.get("rel_l2_per_sample")
            if r is not None:
                all_rel.append(r)
        if not all_rel:
            continue
        arr = np.concatenate(all_rel)
        arr = arr[arr > 0]
        ax.hist(np.log10(arr + 1e-12), bins=60, alpha=0.55, label=MODEL_LABELS[mdl],
                color=MODEL_COLOR[mdl], edgecolor="black", linewidth=0.3)
        plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel(r"$\log_{10}$(per-sample test rel-$L_2$)")
    ax.set_ylabel("count")
    ax.set_title("Per-sample test rel-$L_2$ distribution (all cells, clean regime)")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG / "A7_residual_histogram.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
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
    fig.suptitle(r"LEMO-PC learned spectral kernel $|K_{:,o,m}|$ averaged over input channels + seeds",
                 fontsize=11)
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
    ax.set_title("Per-cell seed-wise rel-$L_2$ (3 dots per (model, cell))")
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
    data = gather_all()
    stats = _aggregate_per_model(data)
    fig = _scatter_panel(stats, "params", "trainable parameters",
                         "Parameter count vs test rel-$L_2$")
    if fig is None:
        return None
    out = FIG / "C13_params_vs_error.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
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
    ax.set_title("Parameter-efficiency Pareto frontier")
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
    ax.set_title("Wall-clock Pareto frontier")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG / "C16_wallclock_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C20 error vs spatial resolution (regime comparison) ---

def c20_regime_comparison():
    data = gather_all()
    models = [m for m in MODEL_ORDER if m in data and data[m]]
    if not models:
        return None
    means = {m: {} for m in models}
    stds = {m: {} for m in models}
    for m in models:
        for reg in REGIMES:
            vals = []
            for fam in FAMS:
                for seed in SEEDS:
                    v = data[m].get((fam, reg, seed))
                    if v is None:
                        continue
                    r = v.get("test_rel_l2_mean", v.get("test_rel_l2"))
                    if r is not None:
                        vals.append(float(r))
            means[m][reg] = float(np.mean(vals)) if vals else float("nan")
            stds[m][reg] = float(np.std(vals)) if vals else float("nan")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(REGIMES))
    n = len(models)
    width = 0.8 / n
    for i, m in enumerate(models):
        ys = [means[m][r] for r in REGIMES]
        es = [stds[m][r] for r in REGIMES]
        ax.bar(x + (i - n / 2 + 0.5) * width, ys, width, yerr=es, capsize=2,
               color=MODEL_COLOR[m], alpha=0.85, edgecolor="black", linewidth=0.4,
               label=MODEL_LABELS[m])
    ax.set_xticks(x); ax.set_xticklabels(REGIMES)
    ax.set_yscale("log")
    ax.set_ylabel(r"test rel-$L_2$")
    ax.set_title("Mean test rel-$L_2$ per regime, all 5 dist-kernel families pooled")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "C20_regime_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --- C22 FiLM γ/β heatmap (out × mode) ---

def c22_film_gamma_beta_heatmap():
    snaps = kernel_for("lemo_pc_nd")
    if not snaps:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    plotted = False
    for snap in snaps.values():
        # Last film_net layer's bias contains (gamma | beta) packed.  Pick the
        # largest bias key under any film_net.
        keys = list(snap.keys())
        bias_keys = [k for k in keys if "film_net" in k and k.endswith(".bias")]
        if not bias_keys:
            continue
        # Pick the bias with the largest size (final FiLM head).
        chosen = max(bias_keys, key=lambda k: np.asarray(snap[k]).size)
        bias = np.asarray(snap[chosen])
        size = bias.size
        # Expected: size = 2 * out_ch * lag_modes.  out_ch = 64 from cfg.
        # We know the model has out_ch = lag width (64 here) and lag_modes = 24.
        # 2 * 64 * 24 = 3072 should match.
        out_ch = 64
        if size % (2 * out_ch) != 0:
            continue
        lag_modes = size // (2 * out_ch)
        if lag_modes <= 0:
            continue
        half = size // 2
        gamma = bias[:half].reshape(out_ch, lag_modes)
        beta  = bias[half:].reshape(out_ch, lag_modes)
        im0 = axes[0].imshow(gamma, cmap="RdBu_r",
                              vmin=-np.abs(gamma).max(), vmax=np.abs(gamma).max(), aspect="auto")
        axes[0].set_title(r"FiLM $\gamma$ bias (out $\times$ mode)")
        axes[0].set_xlabel("lag mode $m$")
        axes[0].set_ylabel("output channel")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(beta, cmap="RdBu_r",
                              vmin=-np.abs(beta).max(), vmax=np.abs(beta).max(), aspect="auto")
        axes[1].set_title(r"FiLM $\beta$ bias (out $\times$ mode)")
        axes[1].set_xlabel("lag mode $m$")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        plotted = True
        break
    if not plotted:
        plt.close(fig); return None
    fig.suptitle("LEMO-PC FiLM modulation parameters (one representative cell)", fontsize=11)
    fig.tight_layout()
    out = FIG / "C22_film_gamma_beta_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
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
    ax.set_title("Hardest-decile Jaccard between models", fontsize=10)
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
    ax.set_title("Single-delay 2D leaderboard (mackey-glass / wright / hutchinson)",
                 fontsize=11)
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
    ax.set_title("APEBench: residual-anchor vs clean improvement", fontsize=10)
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
    ax.set_title("Width-scaling curve (LEMO-PC on burgers_3d)")
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
        ("F2 best-ckpt-epoch",        f2_best_ckpt_epoch),
        ("F4 kernel magnitude hist",   f4_kernel_magnitude_hist),
        ("F5 FiLM distributions",      f5_film_distributions),
        ("A4 seed-wise box",           a4_seedwise_box),
        ("A6 calibration scatter",     a6_calibration_scatter),
        ("A7 residual histogram",      a7_residual_histogram),
        ("A9 kernel heatmap",          a9_kernel_heatmap_per_family),
        ("C8 seed-wise dotplot",       c8_seedwise_dotplot),
        ("C13 params vs error",        c13_params_vs_error),
        ("C14 wallclock vs error",     c14_wallclock_vs_error),
        ("C15 param-eff Pareto",       c15_param_efficiency_pareto),
        ("C16 wallclock Pareto",       c16_wallclock_pareto),
        ("C20 regime comparison",      c20_regime_comparison),
        ("C22 FiLM gamma/beta heatmap", c22_film_gamma_beta_heatmap),
        ("C24 residual correlation",   c24_residual_correlation),
        ("C25 hardest-decile Jaccard", c25_hardest_decile_jaccard),
        ("C30 single-delay heatmap",   c30_single_delay_heatmap),
        ("E1 APEBench leaderboard",    e1_apebench_leaderboard),
        ("E2 APEBench residual lb",    e2_apebench_residual_leaderboard),
        ("E3 APEBench delta",          e3_apebench_residual_delta),
        ("E9 scaling burgers_3d",      e9_scaling_burgers3d),
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
