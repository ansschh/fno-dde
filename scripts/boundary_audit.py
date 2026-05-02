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
    "lemo_pc_nd":   "#d62728",
    "lemo_nd":      "#ff7f0e",
    "fno_nd":       "#1f77b4",
    "fno_film_nd":  "#17becf",
    "markov_fno_nd": "#2ca02c",
    "windowed_fno_nd": "#9467bd",
    "ffno_nd":      "#8c564b",
    "memno_nd":     "#e377c2",
    "unet_nd":      "#7f7f7f",
}
MODEL_LABEL = {
    "lemo_pc_nd":   "LEMO-PC", "lemo_nd": "LEMO",
    "fno_nd":       "FNO",     "fno_film_nd": "FNO+FiLM",
    "markov_fno_nd": "Markov-FNO", "windowed_fno_nd": "Window-FNO",
    "ffno_nd":      "F-FNO",   "memno_nd": "MemNO",
    "unet_nd":      "UNet",
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
    # Figure: aggregate per-position rel-L2 for each model + boundary shading
    # + ratio annotation per model.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    T = max(d["T"] for d in agg.values())

    # Skip "broken" models (interior_mean above naive-copy floor of ~1.0).
    BROKEN_THRESHOLD = 0.5
    plotted = []
    for model in MODEL_COLOR:
        if model not in agg: continue
        if agg[model]["interior_mean"] > BROKEN_THRESHOLD:
            print(f"  excluding {model} from figure: interior_mean="
                  f"{agg[model]['interior_mean']:.3e} (> {BROKEN_THRESHOLD})")
            continue
        d = agg[model]
        color = MODEL_COLOR[model]
        m = np.array(d["mean_per_t"])
        s = np.array(d["std_per_t"])
        ts = np.arange(len(m))
        label = f"{MODEL_LABEL[model]} (b/i = {d['ratio']:.2f})"
        ax.plot(ts, m, color=color, lw=1.8, label=label)
        ax.fill_between(ts, np.maximum(m - s, 1e-5), m + s,
                         color=color, alpha=0.15, linewidth=0)
        plotted.append(model)

    # Boundary shading on the single axis
    ax.axvspan(-0.5, B - 0.5, color="grey", alpha=0.13, lw=0,
                label=f"boundary (t<{B} or t≥T-{B})")
    ax.axvspan(T - B - 0.5, T - 0.5, color="grey", alpha=0.13, lw=0)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel("rollout step $t$")
    ax.set_ylabel(r"rel-$L_2$ (mean over cells)")
    ax.set_yscale("log")
    ax.set_title("Per-position boundary audit", fontsize=11)
    ax.grid(linestyle=":", alpha=0.4, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, frameon=False)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(str(OUT_FIG).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")

    print("\n=== Boundary/interior ratios ===")
    for model in plotted:
        d = agg[model]
        print(f"  {MODEL_LABEL[model]:<12s}  n={d['n_cells']:3d}  "
              f"bnd={d['boundary_mean']:.4e}  int={d['interior_mean']:.4e}  "
              f"ratio={d['ratio']:.3f}")


if __name__ == "__main__":
    main()
