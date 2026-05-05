"""F11 robustness — combined FGSM + Gaussian-noise figure.

Generates two outputs:
  - F11_robustness.{pdf,png}           (main paper, 1x2 — regimes collapsed
                                        into mean ± min/max band)
  - F11_robustness_appendix.{pdf,png}  (appendix, 2x3 — regimes broken out
                                        as separate columns for transparency)

The main-paper version uses a thin band as visual proof that regimes carry no
signal; the appendix version shows the full per-regime panels for reviewers
who want to verify that claim cell-by-cell.

Reads BOTH old format (`adv_fgsm.json` / `noise_sweep.json` from post_hoc) and
new dense format (`adversarial_dense.json` / `noise_dense.json` from Caltech eval).
Multi-model curves appear automatically as new data lands.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401  (sets Times New Roman globally)

REPO = Path(r"A:\dde research\dde-fno")
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ["clean", "lowres", "noisy"]
REGIME_LABEL = {"clean": "Clean", "lowres": "Low-res", "noisy": "Noisy"}

MODEL_COLOR = {
    "lemo_pc_nd":                 "#d62728",
    "causal_smooth_lemo_pc_nd":   "#c49c94",
    "fno_nd":                     "#1f77b4",
    "fno_film_nd":                "#17becf",
    "noneq_film_nd":              "#c5b0d5",
    "markov_fno_nd":              "#2ca02c",
    "windowed_fno_nd":            "#9467bd",
    "ffno_nd":                    "#8c564b",
    "memno_nd":                   "#e377c2",
    "s4_nd":                      "#bcbd22",
    "nide_nd":                    "#aec7e8",
    "ndde_nd":                    "#98df8a",
    "unet_nd":                    "#7f7f7f",
}
MODEL_LABEL = {
    "lemo_pc_nd":                 "LEMO-PC",
    "causal_smooth_lemo_pc_nd":   "LEMO-PC (causal)",
    "fno_nd":                     "FNO",
    "fno_film_nd":                "FNO+FiLM",
    "noneq_film_nd":              "Non-equiv +FiLM",
    "markov_fno_nd":              "Markov-FNO",
    "windowed_fno_nd":            "Window-FNO",
    "ffno_nd":                    "F-FNO",
    "memno_nd":                   "MemNO",
    "s4_nd":                      "S4",
    "nide_nd":                    "NIDE",
    "ndde_nd":                    "NDDE",
    "unet_nd":                    "UNet",
}


def _add_old(by_mr: dict, file: Path, key_x: str, key_y: str):
    try:
        parts = file.parts
        seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
        if reg not in REGIMES: return
        if fam not in FAMS: return
        j = json.loads(file.read_text())
        xs = j.get(key_x, []); ys = j.get(key_y, [])
        for x, y in zip(xs, ys):
            if y is None: continue
            by_mr[model][reg][float(x)].append(float(y))
    except Exception:
        pass


def _add_new(by_mr: dict, file: Path, key_root: str):
    """new dense format: file has key_root -> {x_str: {mean, std, n}}."""
    try:
        parts = file.parts
        seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
        if reg not in REGIMES: return
        if fam not in FAMS: return
        j = json.loads(file.read_text())
        d = j.get(key_root, {})
        for x_str, stats in d.items():
            m = stats.get("mean")
            if m is None or not np.isfinite(m): continue
            by_mr[model][reg][float(x_str)].append(float(m))
    except Exception:
        pass


def collect(kind: str):
    """Return {model -> {regime -> {x_value -> [rel_l2 ...]}}}.

    kind: "fgsm" or "noise".
    """
    by_mr = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        if kind == "fgsm":
            for f in root.rglob("adv_fgsm.json"):
                _add_old(by_mr, f, "epsilons", "rel_l2_per_eps")
            for f in root.rglob("adversarial_dense.json"):
                _add_new(by_mr, f, "fgsm")
        else:  # noise
            for f in root.rglob("noise_sweep.json"):
                _add_old(by_mr, f, "noise_levels", "rel_l2_per_noise")
            for f in root.rglob("noise_dense.json"):
                _add_new(by_mr, f, "noise")
    return by_mr


def _model_order_present(data_fgsm, data_noise):
    present = set(data_fgsm.keys()) | set(data_noise.keys())
    return [m for m in MODEL_COLOR if m in present]


def _regime_aggregate(per_regime_per_x):
    """Given {regime -> {x -> [vals]}}, return (xs, mean_over_regimes,
    min_over_regimes, max_over_regimes). Each regime first averaged across
    seeds, then mean/min/max taken across regimes per x."""
    all_xs = sorted({x for d in per_regime_per_x.values() for x in d.keys()})
    if not all_xs:
        return None, None, None, None
    regime_curves = {}  # regime -> list aligned with all_xs (NaN if missing)
    for reg, per_x in per_regime_per_x.items():
        regime_curves[reg] = np.array(
            [np.mean(per_x[x]) if x in per_x and per_x[x] else np.nan
             for x in all_xs])
    if not regime_curves:
        return None, None, None, None
    arr = np.stack(list(regime_curves.values()), axis=0)  # [n_regimes, n_x]
    with np.errstate(invalid="ignore"):
        mn = np.nanmean(arr, axis=0)
        lo = np.nanmin(arr, axis=0)
        hi = np.nanmax(arr, axis=0)
    return np.array(all_xs), mn, lo, hi


def _plot_panel_band(ax, data, x_label, title):
    """Option A: mean ± min/max over regimes per model."""
    handles = []
    for model in _model_order_present(data, data):
        if model not in data: continue
        xs, mn, lo, hi = _regime_aggregate(data[model])
        if xs is None: continue
        color = MODEL_COLOR.get(model, "#888")
        line, = ax.plot(xs, mn, "o-", color=color, lw=1.6, ms=5,
                        label=MODEL_LABEL.get(model, model))
        # Only band if we have >1 regime (otherwise lo==hi==mn).
        if not np.allclose(lo, hi, equal_nan=True):
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
        handles.append(line)
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"test rel$L_2$")
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return handles


def _plot_panel_single_regime(ax, data, regime, x_label, title):
    """Option B: one panel per regime, mean ± std over seeds."""
    handles = []
    for model in _model_order_present(data, data):
        per_x = data[model].get(regime, {})
        if not per_x: continue
        xs = sorted(per_x.keys())
        means = np.array([np.mean(per_x[x]) for x in xs])
        stds = np.array([np.std(per_x[x]) for x in xs])
        color = MODEL_COLOR.get(model, "#888")
        # Smooth solid line, no markers (cleaner look per user feedback).
        line, = ax.plot(xs, means, "-", color=color, lw=2.0,
                        label=MODEL_LABEL.get(model, model))
        ax.fill_between(xs, means - stds, means + stds,
                         color=color, alpha=0.10, linewidth=0)
        handles.append(line)
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    if title:
        ax.set_title(title, color="dimgrey", pad=8)
    ax.grid(False)
    # Drop minor ticks and reduce major-tick density.
    ax.minorticks_off()
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", which="both", right=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return handles


def make_main(data_fgsm, data_noise):
    """1×2 main-paper version: clean regime only, mean ± std over seeds.
    Regime-independence is established empirically and shown in the appendix
    figure; the main figure focuses on the FGSM-vs-Gaussian contrast.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 6.4))
    hL = _plot_panel_single_regime(axL, data_fgsm, "clean",
                                    r"FGSM perturbation $\varepsilon$", "")
    hR = _plot_panel_single_regime(axR, data_noise, "clean",
                                    r"Gaussian noise std $\sigma$", "")
    axL.set_ylabel(r"test rel-$L_2$")
    axR.set_ylabel(r"test rel-$L_2$")
    seen = set(); shared = []
    for h in hL + hR:
        if h.get_label() not in seen:
            seen.add(h.get_label()); shared.append(h)
    n = len(shared)
    ncol = 5 if n >= 8 else (4 if n >= 5 else max(1, n))
    fig.legend(handles=shared, loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, frameon=False,
               columnspacing=1.6, handlelength=1.6, handletextpad=0.5)
    rows = int(np.ceil(n / ncol))
    bot = 0.10 + 0.05 * rows
    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=bot, wspace=0.22)
    out = FIG_DIR / "F11_robustness.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def make_appendix(data_fgsm, data_noise):
    """2×3 appendix version: regimes as columns, perturbation type as rows."""
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.6),
                              sharey="row", sharex="row")
    handles_all = {}
    for j, reg in enumerate(REGIMES):
        h1 = _plot_panel_single_regime(axes[0, j], data_fgsm, reg,
                                        "",
                                        REGIME_LABEL[reg])
        h2 = _plot_panel_single_regime(axes[1, j], data_noise, reg,
                                        "",
                                        "")
        for h in h1 + h2:
            handles_all.setdefault(h.get_label(), h)
    axes[0, 0].set_ylabel("Worst-case (FGSM)\n" + r"test rel-$L_2$")
    axes[1, 0].set_ylabel("Average-case (Gaussian)\n" + r"test rel-$L_2$")
    for j in range(3):
        axes[0, j].set_xlabel("")
        axes[1, j].set_xlabel("")
    axes[0, 1].set_xlabel(r"FGSM perturbation $\varepsilon$")
    axes[1, 1].set_xlabel(r"Gaussian noise std $\sigma$")
    handles_list = list(handles_all.values())
    n = len(handles_list)
    ncol = 5 if n >= 8 else (4 if n >= 5 else max(1, n))
    fig.legend(handles=handles_list,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, frameon=False,
               columnspacing=1.6, handlelength=1.6, handletextpad=0.5)
    rows = int(np.ceil(n / ncol))
    bot = 0.05 + 0.045 * rows
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=bot,
                          hspace=0.30, wspace=0.10)
    out = FIG_DIR / "F11_robustness_appendix.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main():
    data_fgsm = collect("fgsm")
    data_noise = collect("noise")
    print(f"FGSM models: {sorted(data_fgsm.keys())}")
    for m, by_reg in data_fgsm.items():
        print(f"  {m}: regimes = {sorted(by_reg.keys())}")
    print(f"Noise models: {sorted(data_noise.keys())}")
    for m, by_reg in data_noise.items():
        print(f"  {m}: regimes = {sorted(by_reg.keys())}")

    main_out = make_main(data_fgsm, data_noise)
    app_out  = make_appendix(data_fgsm, data_noise)
    print(f"-> main:     {main_out.name}")
    print(f"-> appendix: {app_out.name}")


if __name__ == "__main__":
    main()
