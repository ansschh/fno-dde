"""F10 bar chart redesign: single panel pooled across regimes.

  F10_bars.{pdf,png}
    Single panel. x-axis = train family (E/G/M/U/P). Within each group,
    one bar per model. Bar height = mean OOD rel-L2 pooled over
    {regimes × seeds × 4 unseen test families}. Error bar = std across
    those same {regime × seed × OOD test family} samples.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401
import matplotlib.pyplot as plt

REPO = Path(r"A:\dde research\dde-fno")
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LETTER = {"dist_exp_rd_2d": "E", "dist_gaussian_rd_2d": "G",
              "dist_gamma_rd_2d": "M", "dist_uniform_rd_2d": "U",
              "dist_powerlaw_rd_2d": "P"}

MODEL_COLOR = {
    "lemo_pc_nd": "#d62728",
    "causal_smooth_lemo_pc_nd": "#c49c94",
    "lemo_bcorrect_nd": "#bcbd22",
    "fno_nd": "#1f77b4",
    "fno_film_nd": "#17becf",
    "noneq_film_nd": "#c5b0d5",
    "ffno_nd": "#8c564b",
    "memno_nd": "#e377c2",
    "s4_nd": "#9bba2c",
    "nide_nd": "#aec7e8",
    "ndde_nd": "#98df8a",
}
MODEL_LABEL = {
    "lemo_pc_nd": "LEMO-PC",
    "causal_smooth_lemo_pc_nd": "LEMO-PC (causal)",
    "lemo_bcorrect_nd": "LEMO (b-correct)",
    "fno_nd": "FNO",
    "fno_film_nd": "FNO+FiLM",
    "noneq_film_nd": "Non-equiv +FiLM",
    "ffno_nd": "F-FNO",
    "memno_nd": "MemNO",
    "s4_nd": "S4",
    "nide_nd": "NIDE",
    "ndde_nd": "NDDE",
}
MODEL_ORDER = list(MODEL_LABEL.keys())


def collect_pooled():
    """Pool across regimes × seeds × OOD test families.

    Returns: out[model][train_family] = flat list of per-(regime×seed×ood-fam)
    rel-L2 values. Mean over this list is the bar height; std is the error.
    """
    out = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in (REPO / "extracted", REPO / "outputs"):
        if not r.exists():
            continue
        for f in r.rglob("cross_family_relL2.json"):
            parts = f.parts
            if len(parts) < 5:
                continue
            seed = parts[-2]; model = parts[-3]; reg = parts[-4]; ck_fam = parts[-5]
            if ck_fam not in FAMS or model not in MODEL_LABEL:
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
            for ff in FAMS:
                if ff == ck_fam or ff not in rl:
                    continue
                out[model][ck_fam].append(float(rl[ff]))
    return out


def main():
    data = collect_pooled()
    models_present = [m for m in MODEL_ORDER if data.get(m)]
    if not models_present:
        print("[F10_bars] no data found")
        return
    n_models = len(models_present)
    n_fams = len(FAMS)
    group_w = 0.86
    bar_w = group_w / n_models
    x_centers = np.arange(n_fams)

    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    handles, labels = [], []
    for i, m in enumerate(models_present):
        means = []
        stds = []
        for fam in FAMS:
            cells = data[m].get(fam, [])
            if not cells:
                means.append(np.nan); stds.append(0.0)
            else:
                means.append(float(np.mean(cells)))
                stds.append(float(np.std(cells)) if len(cells) > 1 else 0.0)
        means = np.asarray(means)
        stds = np.asarray(stds)
        offsets = (i - (n_models - 1) / 2.0) * bar_w
        xs = x_centers + offsets
        color = MODEL_COLOR.get(m, "#888")
        bars = ax.bar(xs, means, width=bar_w * 0.95, color=color,
                       edgecolor="none", label=MODEL_LABEL.get(m, m))
        ax.errorbar(xs, means, yerr=stds, fmt="none",
                     ecolor="black", elinewidth=0.8, capsize=2.0, alpha=0.6)
        handles.append(bars[0])
        labels.append(MODEL_LABEL.get(m, m))

    ax.set_xticks(x_centers)
    ax.set_xticklabels([FAM_LETTER[f] for f in FAMS])
    ax.set_yscale("log")
    ax.set_xlabel("trained on family")
    ax.set_ylabel(r"mean OOD rel-$L_2$")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    n = len(handles)
    ncol = 5  # 2 rows x 5 cols for 10 models per user spec
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=ncol, frameon=False,
                columnspacing=1.6, handlelength=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.24)
    out = FIG_DIR / "F10_bars.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  -> {out.name}  ({n} models, {sum(len(v) for d in data.values() for v in d.values())} samples)")


if __name__ == "__main__":
    main()
