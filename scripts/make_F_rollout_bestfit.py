"""F_rollout_bestfit — single-panel per-step rel-L2 vs rollout step t with
per-model exponential best-fit lines.

X = rollout step t (future segment, t = 64..127). Y = mean per-step rel-L2,
averaged over (family, seed) pairs. Each model gets a thin raw curve
(low opacity) and a thick best-fit `y = a * exp(b * (t - t0))` line on top
(annotated with the fit slope b in /step). This quantifies which models
contract (b ~ 0) vs which diverge (b > 0).

Data already exists: per_frame.json with `rel_l2_per_step` arrays under
extracted/.../<fam>/clean/<model>/<seed>/.

Output: NeurIPS_LEMO/figures/F_rollout_bestfit.{pdf,png}
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
SEEDS = ["s42", "s123", "s456"]

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
    """out[model] = (B, T) array stacked over (fam, seed) cells."""
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
            if fam not in FAMS or reg != "clean" or seed not in SEEDS:
                continue
            if model not in MODEL_LABEL:
                continue
            key = (model, fam, seed)
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


def first_nonzero_idx(arr):
    """Return index of first strictly positive sample (skip leading zeros)."""
    nz = np.where(arr > 0)[0]
    return int(nz[0]) if nz.size else 0


def _exp_fit(steps, vals):
    """Fit log(y) ~ a + b*t over the strictly positive segment.
    Returns (a, b, fitted_y_array_aligned_to_steps).
    """
    mask = vals > 0
    if mask.sum() < 4:
        return None
    log_y = np.log(vals[mask])
    t = steps[mask].astype(float)
    A = np.vstack([np.ones_like(t), t]).T
    sol, *_ = np.linalg.lstsq(A, log_y, rcond=None)
    a, b = sol
    fit = np.exp(a + b * steps.astype(float))
    return a, b, fit


def main():
    data = collect_per_step()
    if not data:
        print("[F_rollout_bestfit] no per_frame data found")
        return
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    handles, labels = [], []
    for m in MODEL_ORDER:
        if m not in data:
            continue
        stk = data[m]            # (B, T)
        mean_curve = stk.mean(axis=0)
        # Determine truncation point: trim leading zeros (which are masked
        # history segment) and find the future segment where we measure
        # error growth.
        cut = first_nonzero_idx(mean_curve)
        steps = np.arange(cut, mean_curve.shape[0])
        future_y = mean_curve[cut:]
        color = MODEL_COLOR.get(m, "#888")
        # Raw mean curve, low opacity.
        ax.plot(steps, future_y, color=color, lw=0.9, alpha=0.30)
        # Best-fit exponential.
        result = _exp_fit(steps, future_y)
        if result is None:
            continue
        a, b, fit = result
        line, = ax.plot(steps, fit, color=color, lw=2.2,
                          label=f"{MODEL_LABEL.get(m, m)}  ($b{{=}}{b:+.3f}$/step)")
        handles.append(line)
        labels.append(line.get_label())

    ax.set_yscale("log")
    ax.set_xlabel(r"rollout step $t$")
    ax.set_ylabel(r"per-step rel-$L_2$")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if handles:
        n = len(handles)
        ncol = 2  # 2 cols so the slope annotations don't truncate
        fig.legend(handles, labels, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=ncol, frameon=False,
                    columnspacing=2.5, handlelength=1.8, handletextpad=0.6)
        rows = int(np.ceil(n / ncol))
        bot = 0.10 + 0.06 * rows
        fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=bot)
    else:
        fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.14)
    out = FIG_DIR / "F_rollout_bestfit.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"-> {out.name}  ({len(handles)} models)")


if __name__ == "__main__":
    main()
