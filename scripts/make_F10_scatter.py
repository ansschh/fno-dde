"""F10 alternate layout (Option A): in-dist vs OOD scatter.

For each (model, train_family) cell, compute:
  - in_dist relL2  = rel_l2[train_family]   (the diagonal)
  - mean OOD relL2 = mean of rel_l2 over the OTHER 4 families

Plot scatter: x = in_dist, y = mean OOD. Each point is one (model, train_family),
aggregated over seeds with std error bars. Diagonal y=x line for reference;
"perfect generalist" sits on the line, "specialist" sits well above it.

Currently only LEMO-PC has cross_family_relL2.json data on dist_*_rd_2d (clean regime).
Once Caltech sweeps land + we run post_hoc_analyses on those checkpoints, the same
script will pick up the new data and the scatter will fill in for FNO+FiLM, FNO,
ffno, memno, etc. â€” model is encoded as color, family as marker.
"""
from __future__ import annotations
import json, glob
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
FAM_LABEL = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
             "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
             "dist_powerlaw_rd_2d": "Power"}
FAM_LETTER = {"dist_exp_rd_2d": "E", "dist_gaussian_rd_2d": "G",
              "dist_gamma_rd_2d": "M", "dist_uniform_rd_2d": "U",
              "dist_powerlaw_rd_2d": "P"}

MODEL_COLOR = {
    "lemo_pc_nd":                 "#d62728",
    "causal_smooth_lemo_pc_nd":   "#ff7f0e",
    "lemo_bcorrect_nd":           "#bcbd22",
    "fno_nd":                     "#1f77b4",
    "fno_film_nd":                "#17becf",
    "noneq_film_nd":              "#9467bd",
    "markov_fno_nd":              "#2ca02c",
    "windowed_fno_nd":            "#9467bd",
    "ffno_nd":                    "#8c564b",
    "memno_nd":                   "#e377c2",
    "s4_nd":                      "#bcbd22",
    "nide_nd":                    "#008080",
    "ndde_nd":                    "#2ca02c",
    "unet_nd":                    "#7f7f7f",
}
MODEL_LABEL = {
    "lemo_pc_nd":                 "LEMO-PC",
    "causal_smooth_lemo_pc_nd":   "LEMO-PC (causal)",
    "lemo_bcorrect_nd":           "LEMO (b-correct)",
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


def collect(regime: str):
    """Returns nested dict: out[model][train_family] = list of (in_dist, ood_mean) tuples.

    rglob discovery across the broader extracted/ + outputs/ trees so every
    sweep layout (pod_pulls_2026_05_03_final/<pod>/outputs/<sweep>/raw/...,
    film_ablation_caltech/raw, dist_kernel_v2_p1/raw, etc.) is covered.
    Dedupes on (model, fam, regime, seed) to avoid double-count when the
    same cell appears under both extracted/ and outputs/.
    """
    out = defaultdict(lambda: defaultdict(list))
    seen = set()
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("cross_family_relL2.json"):
            parts = f.parts
            try:
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; ck_fam = parts[-5]
            except IndexError:
                continue
            if reg != regime:
                continue
            if ck_fam not in FAMS:
                continue
            if model not in MODEL_LABEL:
                continue
            key = (model, ck_fam, reg, seed)
            if key in seen:
                continue
            seen.add(key)
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            rl = j.get("rel_l2", {})
            if not rl or ck_fam not in rl:
                continue
            in_dist = rl[ck_fam]
            ood = [rl[ff] for ff in FAMS if ff != ck_fam and ff in rl]
            if len(ood) < 4:
                continue
            ood_mean = float(np.mean(ood))
            out[model][ck_fam].append((float(in_dist), ood_mean))
    return out


def main():
    regimes = [("clean", "Clean"), ("lowres", "Low-res"), ("noisy", "Noisy")]
    data_by_regime = {reg: collect(reg) for reg, _ in regimes}

    # Shared axis limits across all 3 panels for fair regime comparison.
    all_x, all_y = [], []
    for data in data_by_regime.values():
        for by_fam in data.values():
            for cells in by_fam.values():
                for x, y in cells:
                    all_x.append(x); all_y.append(y)
    if not all_x:
        print("no data found")
        return
    lo = max(min(all_x + all_y) * 0.7, 1e-3)
    hi = max(all_x + all_y) * 1.6

    # Big-text styling consistent with F08/F11/F06.
    TITLE_FS = 42; AXIS_FS = 38; TICK_FS = 36; LEGEND_FS = 42
    fig, axes = plt.subplots(1, 3, figsize=(28.0, 11.0),
                              sharex=True, sharey=True)

    all_models = set()
    for data in data_by_regime.values():
        all_models.update(data.keys())

    for ax, (reg, reg_label) in zip(axes, regimes):
        data = data_by_regime[reg]
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1.4, zorder=0)
        for model, by_fam in data.items():
            color = MODEL_COLOR.get(model, "#888")
            for fam in FAMS:
                cells = by_fam.get(fam, [])
                if not cells:
                    continue
                ind = np.array([c[0] for c in cells])
                ood = np.array([c[1] for c in cells])
                x_mean, y_mean = ind.mean(), ood.mean()
                ax.scatter(x_mean, y_mean, color=color, edgecolor="black",
                            s=400, linewidth=1.0, alpha=0.95, zorder=3)
                ax.text(x_mean, y_mean, FAM_LETTER[fam],
                        ha="center", va="center",
                        fontsize=18, fontweight="bold", color="white",
                        zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"in-distribution rel-$L_2$", fontsize=AXIS_FS)
        ax.set_title(reg_label, fontsize=TITLE_FS, pad=10)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, which="major", linestyle="-", color="grey",
                 alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"mean OOD rel-$L_2$", fontsize=AXIS_FS)

    model_handles = [plt.Line2D([0],[0], marker="o", color="w",
                                  mfc=MODEL_COLOR[m], mec="black", mew=0.8,
                                  ms=18, label=MODEL_LABEL[m])
                     for m in MODEL_COLOR if m in all_models]
    n = len(model_handles)
    ncol = 4 if n >= 8 else (3 if n >= 5 else max(1, n))
    fig.legend(handles=model_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               fontsize=LEGEND_FS, frameon=False,
               ncol=ncol,
               columnspacing=1.6, handlelength=1.4, handletextpad=0.4)
    rows = int(np.ceil(n / ncol))
    bot = 0.06 + 0.07 * rows
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=bot,
                          wspace=0.10)

    out = FIG_DIR / "F10_cross_family_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"-> {out.name}")
    print(f"   models present: {sorted(all_models)}")


if __name__ == "__main__":
    main()
