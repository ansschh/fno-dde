"""F10 curve chart with shaded variance bands.

  F10_curves_band.{pdf,png}
    3 panels by regime (Clean/Low-res/Noisy). Within each panel, x-axis is
    train family (E/G/M/U/P) ordered. Y = mean OOD rel-L2 averaged across
    seeds and OOD test families. Shaded band = ±std across {seeds × OOD
    test families} at low opacity. One curve per model.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")

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


def collect(regime):
    """out[model][train_family] = flat list of per-(seed × ood-fam) rel-L2."""
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
            if reg != regime or ck_fam not in FAMS or model not in MODEL_LABEL:
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
            if not rl:
                continue
            for ff in FAMS:
                if ff == ck_fam or ff not in rl:
                    continue
                out[model][ck_fam].append(float(rl[ff]))
    return out


def _plot_panel(ax, data, title, models_present):
    handles, labels = [], []
    for m in models_present:
        if m not in data:
            continue
        xs, ys, lo, hi = [], [], [], []
        for fam in FAMS:
            cells = data[m].get(fam, [])
            if not cells:
                continue
            arr = np.asarray(cells, dtype=float)
            mu = float(np.mean(arr))
            sd = float(np.std(arr)) if arr.size > 1 else 0.0
            xs.append(FAM_LETTER[fam])
            ys.append(mu)
            lo.append(max(mu - sd, 1e-6))
            hi.append(mu + sd)
        if not xs:
            continue
        color = MODEL_COLOR.get(m, "#888")
        line, = ax.plot(xs, ys, "o-", color=color, lw=2.0, ms=6.5,
                          label=MODEL_LABEL.get(m, m))
        ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0)
        handles.append(line)
        labels.append(MODEL_LABEL.get(m, m))
    if title:
        ax.set_title(title, color="dimgrey", pad=10)
    ax.set_yscale("log")
    ax.set_xlabel("trained on family")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return handles, labels


def main():
    regimes = [("clean", "Clean"), ("lowres", "Low-res"), ("noisy", "Noisy")]
    data_by_regime = {r: collect(r) for r, _ in regimes}
    models_present = [m for m in MODEL_ORDER
                       if any(data_by_regime[r].get(m) for r, _ in regimes)]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 7.2), sharey=True)
    handles_acc, labels_acc = [], []
    for ax, (r, title) in zip(axes, regimes):
        h, l = _plot_panel(ax, data_by_regime[r], title, models_present)
        for hh, ll in zip(h, l):
            if ll not in labels_acc:
                handles_acc.append(hh); labels_acc.append(ll)
    axes[0].set_ylabel(r"mean OOD rel-$L_2$")
    cur_lo, cur_hi = axes[0].get_ylim()
    axes[0].set_ylim(cur_lo, cur_hi * 1.5)
    if handles_acc:
        ncol = 5
        fig.legend(handles_acc, labels_acc, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False,
                    columnspacing=1.6, handlelength=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.26, wspace=0.10)
    out = FIG_DIR / "F10_curves_band.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
