"""
Per-position boundary-error audit on LEMO-PC checkpoints.

For each cell, compute the spatial-mean rel-L2 at each rollout step t in [0, T-1]
and split into "boundary" (first 4 + last 4 frames) vs "interior" (rest). If the
boundary is significantly worse than the interior, that quantifies cyclic
wrap-around contamination.

Inputs:
    A:/dde research/dde-fno/extracted/pod1/outputs/dist_kernel_v2_p1/raw/
        <fam>/clean/lemo_pc_nd/<seed>/per_frame.json
    Each JSON contains 'rel_l2_per_step' of length T_total = 128, where the
    first 64 entries are zeros (history window) and the last 64 are the
    rollout predictions.  We define the rollout window of length T = 64
    (the rollout horizon) and extract the boundary / interior split on
    the rollout window.

Outputs:
    A:/dde research/NeurIPS_LEMO/stats/boundary_error_audit.json
    A:/dde research/NeurIPS_LEMO/figures/F_boundary_error.pdf
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("A:/dde research/dde-fno/extracted/pod1/outputs/dist_kernel_v2_p1/raw")
OUT_JSON = Path("A:/dde research/NeurIPS_LEMO/stats/boundary_error_audit.json")
OUT_FIG = Path("A:/dde research/NeurIPS_LEMO/figures/F_boundary_error.pdf")

# Boundary window size (frames at each end of the rollout)
B = 4


def collect_cells():
    pattern = str(ROOT / "*" / "clean" / "lemo_pc_nd" / "*" / "per_frame.json")
    files = sorted(glob.glob(pattern))
    cells = []
    for f in files:
        parts = Path(f).parts
        # .../raw/<fam>/clean/lemo_pc_nd/<seed>/per_frame.json
        seed = parts[-2]
        fam = parts[-5]
        with open(f, "r") as fp:
            d = json.load(fp)
        arr = np.array(d["rel_l2_per_step"], dtype=np.float64)
        cells.append({"fam": fam, "seed": seed, "path": f, "arr": arr})
    return cells


def split_rollout(arr: np.ndarray) -> np.ndarray:
    """Strip the leading history window of zeros, return the rollout-only array.

    The exporter writes rel_l2_per_step of length T_total = 128 with the first
    64 entries equal to 0 (history) and the remaining 64 the actual rollout
    rel-L2.  We keep only the rollout window so that t=0 is the first
    predicted frame and t=T-1 is the last.
    """
    # Find the first index where the array is non-zero; everything before that
    # is the history window of length L (here L=64).
    nonzero = np.nonzero(arr > 0)[0]
    if len(nonzero) == 0:
        # all zeros -> degenerate, treat the whole thing as rollout
        return arr.copy()
    start = int(nonzero[0])
    return arr[start:].copy()


def boundary_stats(rollout: np.ndarray, b: int = B) -> dict:
    T = len(rollout)
    if T < 2 * b + 1:
        raise ValueError(f"Rollout length {T} too short for boundary width {b}.")
    boundary_idx = np.concatenate([np.arange(b), np.arange(T - b, T)])
    interior_idx = np.arange(b, T - b)
    boundary = rollout[boundary_idx]
    interior = rollout[interior_idx]
    bm = float(boundary.mean())
    im = float(interior.mean())
    ratio = float(bm / im) if im > 0 else float("inf")
    return {
        "T": T,
        "boundary_mean": bm,
        "interior_mean": im,
        "ratio": ratio,
        "boundary_indices": boundary_idx.tolist(),
        "interior_indices": interior_idx.tolist(),
    }


def main() -> None:
    cells = collect_cells()
    print(f"Found {len(cells)} cells")

    # Stack per-cell rollouts (assume same length T)
    rollouts = [split_rollout(c["arr"]) for c in cells]
    T = len(rollouts[0])
    assert all(len(r) == T for r in rollouts), "rollout lengths differ"

    rollouts_np = np.stack(rollouts, axis=0)  # (N_cells, T)

    # Per-cell boundary stats
    per_cell_records = []
    for c, roll in zip(cells, rollouts):
        s = boundary_stats(roll)
        per_cell_records.append(
            {
                "fam": c["fam"],
                "seed": c["seed"],
                "T": s["T"],
                "boundary_mean": s["boundary_mean"],
                "interior_mean": s["interior_mean"],
                "ratio": s["ratio"],
            }
        )

    # Aggregate per-position across all cells
    mean_per_t = rollouts_np.mean(axis=0)  # length T
    std_per_t = rollouts_np.std(axis=0)

    # Per-family aggregate
    fams = sorted({c["fam"] for c in cells})
    per_family = {}
    for fam in fams:
        idx = [i for i, c in enumerate(cells) if c["fam"] == fam]
        fam_roll = rollouts_np[idx]  # (n_seeds, T)
        fam_mean_t = fam_roll.mean(axis=0)
        # boundary/interior on the family-mean curve
        s = boundary_stats(fam_mean_t)
        per_family[fam] = {
            "n_seeds": len(idx),
            "mean_per_t": fam_mean_t.tolist(),
            "boundary_mean": s["boundary_mean"],
            "interior_mean": s["interior_mean"],
            "ratio": s["ratio"],
        }

    # Aggregate stats on the cross-cell mean curve
    agg = boundary_stats(mean_per_t)

    out = {
        "config": {
            "root": str(ROOT),
            "boundary_width": B,
            "T_rollout": T,
            "n_cells": len(cells),
        },
        "aggregate": {
            "boundary_mean": agg["boundary_mean"],
            "interior_mean": agg["interior_mean"],
            "ratio": agg["ratio"],
            "mean_per_t": mean_per_t.tolist(),
            "std_per_t": std_per_t.tolist(),
            "boundary_indices": agg["boundary_indices"],
            "interior_indices": agg["interior_indices"],
        },
        "per_family": per_family,
        "per_cell": per_cell_records,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"Wrote {OUT_JSON}")

    # ------------------------------------------------------------------
    # Figure: per-position rel-L2, with boundary regions shaded
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ts = np.arange(T)
    # per-family curves (light)
    for fam in fams:
        ax.plot(ts, per_family[fam]["mean_per_t"], lw=0.8, alpha=0.55,
                label=fam.replace("dist_", "").replace("_rd_2d", ""))
    # aggregate curve (bold black)
    ax.plot(ts, mean_per_t, color="black", lw=1.6, label="aggregate")
    # boundary shading
    ax.axvspan(0, B - 0.5, color="tab:red", alpha=0.12, label=f"boundary ($t<{B}$)")
    ax.axvspan(T - B - 0.5, T - 1, color="tab:red", alpha=0.12,
               label=f"boundary ($t\\geq T-{B}$)")
    ax.set_xlabel(r"rollout step $t$")
    ax.set_ylabel(r"rel-$L_2$ (mean over cells)")
    ratio_str = f"boundary/interior = {agg['ratio']:.3f}"
    ax.set_title(f"LEMO-PC per-position boundary audit  ({ratio_str})")
    ax.set_xlim(0, T - 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    fig.savefig(str(OUT_FIG).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("")
    print("=== Per-family boundary/interior ratios ===")
    for fam, rec in per_family.items():
        print(f"  {fam:32s}  bnd={rec['boundary_mean']:.4e}  "
              f"int={rec['interior_mean']:.4e}  ratio={rec['ratio']:.3f}")
    print("")
    print(f"=== Aggregate (n={len(cells)} cells, T={T}) ===")
    print(f"  boundary_mean = {agg['boundary_mean']:.4e}")
    print(f"  interior_mean = {agg['interior_mean']:.4e}")
    print(f"  ratio         = {agg['ratio']:.3f}")
    if agg["ratio"] <= 2.0:
        bucket = "MILD (<=2x)"
    elif agg["ratio"] <= 5.0:
        bucket = "WORTH FLAGGING (2-5x)"
    else:
        bucket = "LOAD-BEARING (>5x)"
    print(f"  bucket        = {bucket}")


if __name__ == "__main__":
    main()
