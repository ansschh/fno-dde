"""F_long_horizon — rollout error vs horizon, extended past training T=64.

Two-panel figure:
  Left:  per-step rel-L2 over t in [0, T_short=64] using `per_frame.json`
         (the standard short-horizon eval against ground truth)
  Right: peak prediction-norm at horizons h in {64, 128, 256, 512} using
         `long_horizon.json` (the autoregressive-extrapolation eval). The
         GT attractor norm band is annotated as a shaded grey region; bars
         above the band signal divergence; bars below signal collapse.

This figure addresses the user's "F06 extension to long horizon" ask:
short-horizon error growth + long-horizon stability in one figure.

Output: NeurIPS_LEMO/figures/F_long_horizon.{pdf,png}
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ["clean", "lowres", "noisy"]
SEEDS = ["s42", "s123", "s456"]

MODEL_LABELS = {
    "lemo_pc_nd":                 "LEMO-PC",
    "lemo_nd":                    "LEMO",
    "causal_smooth_lemo_pc_nd":   "LEMO-PC (causal)",
    "lemo_bcorrect_nd":           "LEMO (no σ)",
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
    "lemo_pc_nd", "lemo_nd", "causal_smooth_lemo_pc_nd",
    "lemo_bcorrect_nd",
    "fno_film_nd", "noneq_film_nd",
    "fno_nd", "markov_fno_nd", "windowed_fno_nd",
    "memno_nd", "ffno_nd", "s4_nd", "nide_nd", "ndde_nd",
    "unet_nd",
]
MODEL_COLOR = {
    "lemo_pc_nd":                 "#d62728",
    "lemo_nd":                    "#ff7f0e",
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
TRAIN_T = 64
HORIZONS = [64, 128, 256, 512]


def _try_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _discover_jsons(filename: str) -> dict:
    """Returns {model: {(fam, reg, seed): json_data}}."""
    out = defaultdict(dict)
    seen = set()
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
            if fam not in FAMS or reg != "clean" or seed not in SEEDS:
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


def _first_nonzero(arr: np.ndarray, eps: float = 1e-6) -> int:
    nz = np.nonzero(arr > eps)[0]
    return int(nz[0]) if len(nz) else 0


def main():
    perframe = _discover_jsons("per_frame.json")
    longhz = _discover_jsons("long_horizon.json")
    if not perframe and not longhz:
        print("[F_long_horizon] no data; skipping")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                    gridspec_kw={"width_ratios": [1.2, 1.0]})

    # ---- LEFT: short-horizon per-step rel-L2 ----
    handles, labels = [], []
    for model in MODEL_ORDER:
        cells = perframe.get(model, {})
        all_curves = []
        for (fam, reg, seed), d in cells.items():
            r = d.get("rel_l2_per_step", [])
            if not r:
                continue
            arr = np.asarray(r, dtype=float)
            cut = _first_nonzero(arr)
            all_curves.append(arr[cut:])
        if not all_curves:
            continue
        L = min(len(c) for c in all_curves)
        if L < 2:
            continue
        c_arr = np.stack([c[:L] for c in all_curves], axis=0)
        steps = np.arange(L)
        color = MODEL_COLOR.get(model, "#888")
        line, = axL.plot(steps, c_arr.mean(axis=0), color=color, lw=1.4,
                          label=MODEL_LABELS[model])
        if c_arr.shape[0] > 1:
            axL.fill_between(steps,
                              c_arr.mean(axis=0) - c_arr.std(axis=0),
                              c_arr.mean(axis=0) + c_arr.std(axis=0),
                              color=color, alpha=0.13, linewidth=0)
        handles.append(line); labels.append(MODEL_LABELS[model])
    axL.set_yscale("log")
    axL.set_xlabel(r"future rollout step $t$ (training T=64)")
    axL.set_ylabel(r"per-step rel-$L_2$ vs. GT")
    axL.grid(linestyle=":", alpha=0.4, which="both")
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    axL.text(0.5, 0.97, r"Short horizon: $t \in [0, 64]$",
              transform=axL.transAxes, ha="center", va="top",
              fontsize=10, color="dimgrey")

    # ---- RIGHT: long-horizon norm at h = 64, 128, 256, 512 ----
    # long_horizon.json schema is per-cell; collect peak norm per (model, h).
    norm_by_model_h = defaultdict(lambda: defaultdict(list))
    for model, cells in longhz.items():
        for d in cells.values():
            for h in HORIZONS:
                k = f"h_{h}"
                if k not in d:
                    continue
                arr = d[k].get("norm_per_step")
                if not arr:
                    continue
                arr = np.asarray(arr, dtype=float)
                if not arr.size:
                    continue
                norm_by_model_h[model][h].append(float(arr.max()))

    plotted_models = [m for m in MODEL_ORDER if m in norm_by_model_h]
    if plotted_models:
        n_models = len(plotted_models)
        x_h = np.arange(len(HORIZONS))
        bar_w = 0.8 / n_models
        for i, model in enumerate(plotted_models):
            xs = []
            ys = []
            errs = []
            for j, h in enumerate(HORIZONS):
                vals = norm_by_model_h[model].get(h, [])
                if not vals:
                    continue
                xs.append(j + (i - n_models / 2 + 0.5) * bar_w)
                ys.append(np.mean(vals))
                errs.append(np.std(vals))
            if not xs:
                continue
            color = MODEL_COLOR.get(model, "#888")
            axR.bar(xs, ys, width=bar_w, color=color,
                     yerr=errs if any(errs) else None,
                     capsize=2, edgecolor="black", linewidth=0.4,
                     label=MODEL_LABELS[model])
        axR.set_xticks(x_h)
        axR.set_xticklabels([f"h={h}" for h in HORIZONS])
        # GT attractor band
        try:
            from make_F_norm_rollout import gt_norm_band
            band = gt_norm_band(percentiles=(25, 75))
        except Exception:
            band = (16.0, 59.0)  # fallback
        if band:
            axR.axhspan(band[0], band[1], color="grey", alpha=0.15,
                         linewidth=0, label=f"GT attractor IQR [{band[0]:.0f}, {band[1]:.0f}]")
        axR.set_ylabel(r"peak $\|\hat u(t)\|_2$ at horizon")
        axR.text(0.5, 0.97, r"Long horizon: peak $\|\hat u\|$ at $h$",
                  transform=axR.transAxes, ha="center", va="top",
                  fontsize=10, color="dimgrey")
    else:
        axR.text(0.5, 0.5, "(no long_horizon.json data)",
                  transform=axR.transAxes, ha="center", va="center",
                  fontsize=12, color="dimgrey")
    axR.grid(axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)

    # Single-row legend across both panels.
    if handles:
        n = len(handles)
        ncol = max(1, (n + 1) // 2)
        fig.legend(handles, labels, loc="lower center",
                    bbox_to_anchor=(0.5, 0.01),
                    ncol=ncol, frameon=False, fontsize=8.5,
                    columnspacing=1.6, handlelength=1.8, handletextpad=0.6)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.20, wspace=0.18)
    out = FIG_DIR / "F_long_horizon.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"-> {out.name}")
    print(f"   short-horizon models: {len(handles)}")
    print(f"   long-horizon models: {len(plotted_models)}")


if __name__ == "__main__":
    main()
