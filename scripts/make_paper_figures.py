"""Paper figure generation — all 2D dist-kernel paper-blocking figures.

Reads:
  - paper/stats/paired_permutation.json         (headline stats)
  - extracted/pod1/.../dist_kernel_v2_p1/raw/.. (LEMO/LEMO_PC test_results.json)
  - extracted/pod2/.../dist_kernel_v2_p2/logs/  (FNO/MarkovFNO/WindFNO from logs)
  - extracted/pod3/outputs/final_baselines/raw/ (MemNO/F-FNO when available)
  - paper/figures/sigma_sweep/                  (σ-sweep when Caltech done)

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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    "lemo_pc_nd":     "LEMO-PC",
    "lemo_nd":        "LEMO",
    "fno_nd":         "FNO",
    "markov_fno_nd":  "Markov-FNO",
    "windowed_fno_nd": "Window-FNO",
    "memno_nd":       "MemNO",
    "ffno_nd":        "F-FNO",
    "unet_nd":        "UNet",
}
MODEL_ORDER = ["lemo_pc_nd", "lemo_nd", "fno_nd", "markov_fno_nd",
               "windowed_fno_nd", "memno_nd", "ffno_nd"]
MODEL_COLOR = {
    "lemo_pc_nd":     "#d62728",  # red
    "lemo_nd":        "#ff7f0e",  # orange
    "fno_nd":         "#1f77b4",  # blue
    "markov_fno_nd":  "#2ca02c",  # green
    "windowed_fno_nd": "#9467bd", # purple
    "memno_nd":       "#8c564b",  # brown
    "ffno_nd":        "#e377c2",  # pink
    "unet_nd":        "#7f7f7f",  # gray
}


# -------------------- data loading --------------------

def _try_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_lemo_test(model: str = "lemo_pc_nd") -> dict:
    """LEMO_PC / LEMO numbers from extracted/pod1/.../dist_kernel_v2_p1/raw/."""
    out = {}
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = float(d.get("test_rel_l2_mean",
                                                        d.get("test_rel_l2", float("nan"))))
    return out


_LOG_PAT = re.compile(r"=== FINAL test relL2 = ([0-9.]+) ===")


def load_baseline_from_log(model: str) -> dict:
    """Read final relL2 from extracted/pod2/.../dist_kernel_v2_p2/logs/*.log."""
    out = {}
    log_dir = EXT / "pod2" / "outputs" / "dist_kernel_v2_p2" / "logs"
    if not log_dir.exists():
        return out
    seed_to_num = {"s42": "42", "s123": "123", "s456": "456"}
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                seed_num = seed_to_num[seed]
                logf = log_dir / f"{fam}_{model}_{reg}_s{seed_num}.log"
                if not logf.exists():
                    continue
                m = _LOG_PAT.search(logf.read_text(errors="replace"))
                if m:
                    out[(fam, reg, seed)] = float(m.group(1))
    return out


def load_pod3_baseline(model: str) -> dict:
    """MemNO / F-FNO from pod3 final_baselines (after Phase A bundle pulled)."""
    out = {}
    pod3 = EXT / "pod3" / "outputs" / "final_baselines" / "raw"
    if not pod3.exists():
        # fallback to alt extraction location:
        for alt in (EXT / "pod3", EXT / "pod3_phaseA"):
            cand = alt / "outputs" / "final_baselines" / "raw"
            if cand.exists():
                pod3 = cand
                break
    if not pod3.exists():
        return out
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = pod3 / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = float(d.get("test_rel_l2_mean",
                                                        d.get("test_rel_l2", float("nan"))))
    return out


def gather_all_models() -> dict:
    """Returns {model: {(fam, reg, seed): relL2}}.  Skips models with no data."""
    data = {
        "lemo_pc_nd": load_lemo_test("lemo_pc_nd"),
        "lemo_nd":    load_lemo_test("lemo_nd"),
        "fno_nd":         load_baseline_from_log("fno_nd"),
        "markov_fno_nd":  load_baseline_from_log("markov_fno_nd"),
        "windowed_fno_nd": load_baseline_from_log("windowed_fno_nd"),
        "memno_nd":       load_pod3_baseline("memno_nd"),
        "ffno_nd":        load_pod3_baseline("ffno_nd"),
    }
    # Drop empty.
    return {m: d for m, d in data.items() if len(d) >= 1}


def load_history(model: str = "lemo_pc_nd"):
    """Returns {(fam, reg, seed): history_dict} from history.json files (if present)."""
    out = {}
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / model / seed / "history.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = d
    return out


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
    fig, axes = plt.subplots(1, len(FAMS), figsize=(3.5 * len(FAMS), 3.8),
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
        ax.set_title(FAM_LABELS[fam])
        ax.grid(linestyle="--", alpha=0.4)
    if not plotted_any:
        plt.close(fig)
        return None
    axes[0].set_ylabel(r"Validation rel$L_2$")
    if handles:
        fig.legend(handles, labels, loc="lower center",
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=min(len(handles), 7), frameon=False, fontsize=9)
    fig.suptitle("Validation rel$L_2$ vs epoch (mean over seeds, clean regime, all baselines)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIG_DIR / "F05_training_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig06_perframe_rollout():
    """Per-rollout-step rel_l2 from per_frame.json files (if present)."""
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    files = list(base.glob("*/clean/lemo_pc_nd/*/per_frame.json"))
    if not files:
        return None
    fig, axes = plt.subplots(1, len(FAMS), figsize=(3.2 * len(FAMS), 3.5),
                              sharey=True)
    if len(FAMS) == 1:
        axes = [axes]
    plotted = False
    for ax, fam in zip(axes, FAMS):
        seeds = list((base / fam / "clean" / "lemo_pc_nd").glob("s*"))
        if not seeds:
            ax.set_visible(False)
            continue
        curves = []
        naive_curves = []
        for sd in seeds:
            d = _try_json(sd / "per_frame.json")
            if d is None:
                continue
            r = d.get("rel_l2_per_step", [])
            n = d.get("naive_rel_l2_per_step", [])
            if r:
                curves.append(np.array(r))
            if n:
                naive_curves.append(np.array(n))
        if not curves:
            ax.set_visible(False)
            continue
        L = min(len(c) for c in curves)
        c_arr = np.stack([c[:L] for c in curves], axis=0)
        steps = np.arange(L)
        ax.plot(steps, c_arr.mean(axis=0), color=MODEL_COLOR["lemo_pc_nd"], lw=1.5,
                label="LEMO-PC")
        ax.fill_between(steps, c_arr.mean(axis=0) - c_arr.std(axis=0),
                         c_arr.mean(axis=0) + c_arr.std(axis=0),
                         color=MODEL_COLOR["lemo_pc_nd"], alpha=0.25)
        if naive_curves:
            n_arr = np.stack([c[:L] for c in naive_curves], axis=0)
            ax.plot(steps, n_arr.mean(axis=0), color="grey", lw=1.0, linestyle="--",
                    label="naive copy")
        ax.set_yscale("log")
        ax.set_xlabel("rollout step t")
        ax.set_title(FAM_LABELS[fam])
        ax.grid(linestyle="--", alpha=0.4)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_ylabel(r"per-step rel$L_2$")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-rollout-step rel$L_2$ (LEMO-PC, clean, mean +/- std over seeds)")
    fig.tight_layout()
    out = FIG_DIR / "F06_perframe_rollout.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig07_op_norm_trajectory(history: dict):
    """op_norm_max vs epoch (proves σ-projection binding once σ-sweep cells exist)."""
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
    ax.set_title("Raw spectral kernel op-norm trajectory (proves σ-projection binding)")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / "F07_op_norm_trajectory.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def fig08_equivariance_test():
    """T1-shift errors per shift size (from equivariance.json files)."""
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    files = list(base.glob("*/clean/lemo_pc_nd/*/equivariance.json"))
    if not files:
        return None
    rows = []  # (shift, err)
    for f in files:
        d = _try_json(f)
        if d is None:
            continue
        for k, v in d.items():
            m = re.match(r"equiv_shift_(\d+)_mean", k)
            if m:
                rows.append((int(m.group(1)), v))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    shifts = sorted(set(r[0] for r in rows))
    errs_by_shift = {s: [r[1] for r in rows if r[0] == s] for s in shifts}
    means = [np.mean(errs_by_shift[s]) for s in shifts]
    stds = [np.std(errs_by_shift[s]) for s in shifts]
    ax.errorbar(shifts, means, yerr=stds, fmt="o-", color="#d62728",
                 capsize=4, lw=1.5)
    ax.axhline(1e-3, color="grey", linewidth=0.5, linestyle="--")
    ax.text(shifts[0], 1.5e-3, "T1 pass threshold (1e-3)", color="grey", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("cyclic shift size k")
    ax.set_ylabel(r"$\| \mathrm{LEMO}(\rho_k x) - \rho_k \mathrm{LEMO}(x) \|_2 / \|\mathrm{LEMO}(x)\|_2$")
    ax.set_title("Cyclic-shift equivariance at deployment (T1 in the wild)")
    ax.grid(linestyle="--", alpha=0.4)
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
        ("F01 headline bar",        fig01_headline_bar,       (stats,)),
        ("F02 per-family heatmap",  fig02_perfamily_heatmap,  (data,)),
        ("F03 per-regime box",      fig03_perregime_box,      (data,)),
        ("F04 effect size",         fig04_effect_size,        (stats,)),
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
