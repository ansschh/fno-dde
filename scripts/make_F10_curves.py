"""F10 curve redesigns:

  F10a_curves.{pdf,png}
    Option 1: per-model curve over train families. X = train family (E/G/M/U/P
    ordered), Y = mean OOD relL2 averaged across the 4 unseen test families.
    One line per model. 3 panels by regime (clean/lowres/noisy).

  F10b_gap_curves.{pdf,png}
    Option 2: same x-axis, Y = OOD/ID ratio (generalization gap, log scale).
    Bigger ratio = larger generalization gap.
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


def collect(regime):
    """nested dict: out[model][train_family] = list of (in_dist, ood_mean) over seeds."""
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
            if not rl or ck_fam not in rl:
                continue
            in_dist = rl[ck_fam]
            ood = [rl[ff] for ff in FAMS if ff != ck_fam and ff in rl]
            if len(ood) < 4:
                continue
            ood_mean = float(np.mean(ood))
            out[model][ck_fam].append((float(in_dist), ood_mean))
    return out


def _plot_panel(ax, data, mode, title):
    """mode: 'ood_abs' for option 1, 'ood_over_id' for option 2."""
    handles, labels = [], []
    plotted = False
    for m in MODEL_ORDER:
        if m not in data:
            continue
        ys, xs = [], []
        for fam in FAMS:
            cells = data[m].get(fam, [])
            if not cells:
                continue
            in_d = np.mean([c[0] for c in cells])
            ood = np.mean([c[1] for c in cells])
            xs.append(FAM_LETTER[fam])
            if mode == "ood_abs":
                ys.append(ood)
            else:
                ys.append(ood / max(in_d, 1e-12))
        if not xs:
            continue
        color = MODEL_COLOR.get(m, "#888")
        line, = ax.plot(xs, ys, "o-", color=color, lw=1.8, ms=6,
                          label=MODEL_LABEL.get(m, m))
        handles.append(line)
        labels.append(MODEL_LABEL.get(m, m))
        plotted = True
    if title:
        ax.text(0.5, 0.97, title, transform=ax.transAxes,
                ha="center", va="top", fontsize=18, color="dimgrey")
    ax.set_yscale("log")
    ax.set_xlabel("trained on family")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return handles, labels, plotted


def _make_figure(mode: str, ylabel: str, fname: str):
    regimes = [("clean", "Clean"), ("lowres", "Low-res"), ("noisy", "Noisy")]
    data_by_regime = {r: collect(r) for r, _ in regimes}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), sharey=True)
    handles_acc, labels_acc = [], []
    plotted_any = False
    for ax, (r, title) in zip(axes, regimes):
        h, l, p = _plot_panel(ax, data_by_regime[r], mode, title)
        plotted_any = plotted_any or p
        for hh, ll in zip(h, l):
            if ll not in labels_acc:
                handles_acc.append(hh); labels_acc.append(ll)
    if not plotted_any:
        plt.close(fig)
        return None
    axes[0].set_ylabel(ylabel)
    if handles_acc:
        n = len(handles_acc)
        ncol = max(1, min(n, 8))
        fig.legend(handles_acc, labels_acc, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False, fontsize=14,
                    columnspacing=1.2, handlelength=1.6, handletextpad=0.4)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.22, wspace=0.10)
    out = FIG_DIR / fname
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out


def main():
    a = _make_figure("ood_abs",
                      r"mean OOD rel-$L_2$",
                      "F10a_curves.pdf")
    print(f"  -> {a.name if a else 'F10a SKIPPED'}")
    b = _make_figure("ood_over_id",
                      r"OOD/ID rel-$L_2$ ratio (gen-gap)",
                      "F10b_gap_curves.pdf")
    print(f"  -> {b.name if b else 'F10b SKIPPED'}")


if __name__ == "__main__":
    main()
