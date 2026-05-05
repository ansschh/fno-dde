"""Per-position boundary-error audit, multi-model.

For each checkpoint, compute the spatial-mean rel-L2 at each rollout step t and
split into "boundary" (first B + last B frames) vs "interior" (rest). The
boundary/interior ratio quantifies cyclic wrap-around contamination.

For LEMO-PC the cyclic-FFT structure means we expect ratio ≈ 1.0; for non-cyclic
baselines (vanilla FNO, UNet) we expect ratio > 1.

Reads per_frame.json files from all sweep roots locally and from
boundary_dense.json (new dense eval) when those land.
Writes one aggregated stats blob to NeurIPS_LEMO/stats/boundary_error_audit.json
and a multi-model figure to NeurIPS_LEMO/figures/F_boundary_error.{pdf,png}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(r"A:\dde research\dde-fno")
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
STATS_DIR = (REPO.parent / "NeurIPS_LEMO" / "stats").resolve()
OUT_FIG = FIG_DIR / "F_boundary_error.pdf"
OUT_JSON = STATS_DIR / "boundary_error_audit.json"

# Boundary window size (frames at each end of the rollout)
B = 4

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]

MODEL_COLOR = {
    "lemo_pc_nd":                 "#d62728",
    "lemo_nd":                    "#ff7f0e",
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
    "lemo_nd":                    "LEMO",
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


def split_rollout(arr: np.ndarray) -> np.ndarray:
    """Strip leading history zeros, return rollout-only."""
    nz = np.nonzero(arr > 0)[0]
    return arr[int(nz[0]):].copy() if len(nz) else arr.copy()


def collect_rollouts():
    """Returns {model -> [rollout_array, ...]} from per_frame.json across all roots.

    Uses (family, seed) as a dedupe key so we don't double-count the same cell
    when it appears under both extracted/ and outputs/.
    """
    out = defaultdict(list)
    seen = set()  # (model, family, seed) tuples
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        for f in root.rglob("per_frame.json"):
            try:
                parts = f.parts
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
            except IndexError:
                continue
            if reg != "clean":
                continue
            if fam not in FAMS:
                continue
            key = (model, fam, seed)
            if key in seen:
                continue
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            r = j.get("rel_l2_per_step", [])
            if not r:
                continue
            rollout = split_rollout(np.asarray(r, dtype=float))
            if rollout.size < 2 * B + 1:
                continue
            seen.add(key)
            out[model].append(rollout)
    return out


def boundary_stats(rollout: np.ndarray, b: int = B) -> dict:
    T = len(rollout)
    bnd_idx = np.concatenate([np.arange(b), np.arange(T - b, T)])
    int_idx = np.arange(b, T - b)
    bm = float(rollout[bnd_idx].mean())
    im = float(rollout[int_idx].mean())
    return {"T": T, "boundary_mean": bm, "interior_mean": im,
            "ratio": float(bm / im) if im > 0 else float("inf")}


def main():
    rollouts_by_model = collect_rollouts()
    if not rollouts_by_model:
        print("no per_frame.json data found")
        return

    # Aggregate per model: stack rollouts and compute per-step mean.
    agg = {}
    for model, rolls in rollouts_by_model.items():
        T = min(len(r) for r in rolls)
        stack = np.stack([r[:T] for r in rolls], axis=0)
        mean_t = stack.mean(axis=0)
        std_t  = stack.std(axis=0)
        s = boundary_stats(mean_t)
        agg[model] = {
            "n_cells": len(rolls), "T": T,
            "mean_per_t": mean_t.tolist(),
            "std_per_t":  std_t.tolist(),
            **s,
        }

    # Save stats blob
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump({"config": {"boundary_width": B},
                    "models": agg}, fp, indent=2)
    print(f"Wrote {OUT_JSON}")

    # ------------------------------------------------------------------
    # Figure: 1x2 zoom on the rollout edges (first ZOOM steps + last ZOOM
    # steps). The interior 87% of the rollout (t in [ZOOM..T-ZOOM]) carries
    # no architectural signal — both LEMO-PC and FNO+FiLM grow smoothly
    # there — so we skip it and zoom on the edges where the cyclic-FFT
    # vs. non-cyclic-FFT difference is visible. The b/i ratio is in the
    # legend so reviewers can read the architectural contrast at a glance.
    # ------------------------------------------------------------------
    BROKEN_THRESHOLD = 0.5
    T = max(d["T"] for d in agg.values())
    ZOOM = 10

    plotted = []
    for model in MODEL_COLOR:
        if model not in agg:
            continue
        if agg[model]["interior_mean"] > BROKEN_THRESHOLD:
            print(f"  excluding {model} from figure: interior_mean="
                  f"{agg[model]['interior_mean']:.3e} (> {BROKEN_THRESHOLD})")
            continue
        d = agg[model]
        plotted.append({
            "model": model,
            "color": MODEL_COLOR[model],
            "label": f"{MODEL_LABEL[model]} (b/i = {d['ratio']:.2f})",
            "mean":  np.array(d["mean_per_t"]),
            "std":   np.array(d["std_per_t"]),
        })

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=True,
                              gridspec_kw={"wspace": 0.06})
    ax_left, ax_right = axes
    ts_full = np.arange(T)
    panels = [
        (ax_left,  slice(0, ZOOM),         (-0.5, ZOOM - 0.5),       "first {} steps".format(ZOOM)),
        (ax_right, slice(T - ZOOM, T),     (T - ZOOM - 0.5, T - 0.5), "last {} steps".format(ZOOM)),
    ]
    for ax, sl, xlim, _name in panels:
        ts = ts_full[sl]
        for p in plotted:
            ax.plot(ts, p["mean"][sl], color=p["color"], lw=1.8, label=p["label"])
            ax.fill_between(ts,
                             np.maximum(p["mean"][sl] - p["std"][sl], 1e-5),
                             p["mean"][sl] + p["std"][sl],
                             color=p["color"], alpha=0.15, linewidth=0)
        ax.set_xlim(*xlim)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step $t$")
        ax.grid(linestyle=":", alpha=0.4, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Boundary shading on each panel (only the band that falls within view).
    ax_left.axvspan(-0.5, B - 0.5, color="grey", alpha=0.13, lw=0,
                     label=f"boundary (t<{B} or t≥T-{B})")
    ax_right.axvspan(T - B - 0.5, T - 0.5, color="grey", alpha=0.13, lw=0)

    ax_left.set_ylabel(r"rel-$L_2$ (mean over cells)")
    fig.suptitle("Boundary audit: rollout edges (interior skipped)",
                  fontsize=11, y=0.99)

    # Single shared legend below the panels with explicit room reserved.
    handles, labels = ax_left.get_legend_handles_labels()
    n_models = len([h for h in labels if "boundary" not in h])
    ncol = min(n_models + 1, 4)
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, 0.005), ncol=ncol,
                fontsize=9, frameon=False)
    fig.subplots_adjust(top=0.92, bottom=0.20, left=0.10, right=0.98,
                         wspace=0.06)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(str(OUT_FIG).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")

    print("\n=== Boundary/interior ratios ===")
    for p in plotted:
        d = agg[p["model"]]
        print(f"  {MODEL_LABEL[p['model']]:<12s}  n={d['n_cells']:3d}  "
              f"bnd={d['boundary_mean']:.4e}  int={d['interior_mean']:.4e}  "
              f"ratio={d['ratio']:.3f}")


if __name__ == "__main__":
    main()
