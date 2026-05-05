"""F_rollout_curves — single-panel per-step rel-L2 vs rollout step t.

Collapses the 5 train families into one per-model curve by averaging
per-step rel-L2 across (regimes x seeds x train-families). Each curve
has a thin solid line (mean) plus a low-opacity shaded ±std band.

X = rollout step t (full 128 steps; the first 64 are history with
loss-mask=0 so error is exactly zero there; the future segment t=64..127
is where models diverge).
Y = mean per-step rel-L2.

Output: NeurIPS_LEMO/figures/F_rollout_curves.{pdf,png}
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

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ("clean", "lowres", "noisy")
SEEDS = ("s42", "s123", "s456")

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


def collect_per_step():
    """out[model] = (B, T) array stacked over all (fam, reg, seed) cells."""
    out = defaultdict(list)
    seen = set()
    for r in (REPO / "extracted", REPO / "outputs"):
        if not r.exists():
            continue
        for f in r.rglob("per_frame.json"):
            parts = f.parts
            try:
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
            except IndexError:
                continue
            if fam not in FAMS or reg not in REGIMES or seed not in SEEDS:
                continue
            if model not in MODEL_LABEL:
                continue
            key = (model, fam, reg, seed)
            if key in seen:
                continue
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            arr = j.get("rel_l2_per_step")
            if not arr:
                continue
            seen.add(key)
            out[model].append(np.asarray(arr, dtype=float))
    final = {}
    for m, arrs in out.items():
        L = min(a.shape[0] for a in arrs)
        if L < 8:
            continue
        stk = np.stack([a[:L] for a in arrs], axis=0)
        final[m] = stk
    return final


def main():
    data = collect_per_step()
    if not data:
        print("[F_rollout_curves] no per_frame data found")
        return
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    handles, labels = [], []
    for m in MODEL_ORDER:
        if m not in data:
            continue
        stk = data[m]                        # (B, T)
        # First non-zero step for the mean curve (skip masked history segment).
        mean_curve = stk.mean(axis=0)
        nz = np.where(mean_curve > 0)[0]
        cut = int(nz[0]) if nz.size else 0
        steps = np.arange(cut, mean_curve.shape[0])
        future = stk[:, cut:]
        mu = future.mean(axis=0)
        sd = future.std(axis=0) if future.shape[0] > 1 else np.zeros_like(mu)
        color = MODEL_COLOR.get(m, "#888")
        line, = ax.plot(steps, mu, color=color, lw=2.0,
                          label=MODEL_LABEL.get(m, m))
        ax.fill_between(steps,
                          np.maximum(mu - sd, 1e-6),
                          mu + sd,
                          color=color, alpha=0.15, linewidth=0)
        handles.append(line)
        labels.append(MODEL_LABEL.get(m, m))

    ax.set_yscale("log")
    ax.set_xlabel(r"rollout step $t$")
    ax.set_ylabel(r"per-step rel-$L_2$", fontweight="bold")
    # Clip y-min so std bands that touch the floor (1e-6) don't dominate.
    ax.set_ylim(3e-3, 2.0)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    n = len(handles)
    if n:
        ncol = 5 if n >= 8 else (4 if n >= 5 else max(1, n))
        fig.legend(handles, labels, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False,
                    columnspacing=1.6, handlelength=1.6, handletextpad=0.5)
        rows = int(np.ceil(n / ncol))
        bot = 0.10 + 0.06 * rows
        fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=bot)
    else:
        fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.14)

    out = FIG_DIR / "F_rollout_curves.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"-> {out.name}  ({n} models)")


if __name__ == "__main__":
    main()
