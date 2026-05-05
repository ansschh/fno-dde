"""F_w4 supplement: η(θ) decomposition by component, per σ value.

For each cell with per_block_lipschitz.json:
  - Read per-layer C_lag,ℓ, C_sp,ℓ, C_film,ℓ, C_out, C_pool, C_act, eta_total
  - Group by σ (None, 0.5, 0.7, 0.9, 0.99) and family

Stacked-bar figure: x = σ value, y = log(C_*) per component, per layer ℓ.

Output: NeurIPS_LEMO/figures/kept/main/F_w4_eta_breakdown.{pdf,png}
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
FIG_PDF = NEURIPS / "figures" / "kept" / "main" / "F_w4_eta_breakdown.pdf"
FIG_PNG = NEURIPS / "figures" / "kept" / "png" / "F_w4_eta_breakdown.png"
FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
FIG_PNG.parent.mkdir(parents=True, exist_ok=True)


def crawl(roots):
    by_sigma = defaultdict(list)
    for root in roots:
        rp = Path(root)
        if not rp.is_absolute():
            rp = REPO / rp
        for pb in rp.glob("**/per_block_lipschitz.json"):
            try:
                d = json.loads(pb.read_text())
            except Exception:
                continue
            tr = pb.parent / "test_results.json"
            sigma = None
            if tr.exists():
                try:
                    tr_data = json.loads(tr.read_text())
                    sigma = tr_data.get("sigma") or (tr_data.get("config", {}).get("model", {}) or {}).get("sigma")
                except Exception:
                    pass
            sigma_key = float(sigma) if sigma is not None else "unconstrained"
            by_sigma[sigma_key].append(d)
    return by_sigma


def plot(by_sigma):
    if not by_sigma:
        print("[F_w4-bd] no per-block data found")
        return
    sigmas = sorted(by_sigma.keys(), key=lambda x: -1 if x == "unconstrained" else float(x))
    component_names = ["C_lag", "C_sp", "C_film", "C_act", "C_pool", "C_out"]
    colors = {"C_lag": "C0", "C_sp": "C1", "C_film": "C2",
              "C_act": "C3", "C_pool": "C4", "C_out": "C5"}

    # Compute the mean contribution (in linear space, product over layers
    # already aggregated). Plot as grouped bars showing each component value
    # directly (NOT stacked log) — much clearer.
    fig, ax = plt.subplots(figsize=(11, 5))
    x_pos = np.arange(len(sigmas))
    n_comp = len(component_names)
    bar_width = 0.8 / n_comp

    for i, comp in enumerate(component_names):
        vals_per_sigma = []
        for s in sigmas:
            cells = by_sigma[s]
            cell_vals = []
            for c in cells:
                if comp == "C_pool":
                    v = c.get("C_pool")
                elif comp == "C_out":
                    v = c.get("C_out")
                elif comp == "C_act":
                    v = c.get("C_act") if "C_act" in c else 1.0
                    n_layers = c.get("n_layers", 3)
                    v = v ** n_layers
                else:
                    v = 1.0
                    for blk in c.get("per_layer", []):
                        bv = blk.get(comp)
                        if bv is not None:
                            v *= float(bv)
                if v is None or v <= 0:
                    continue
                cell_vals.append(float(v))
            if cell_vals:
                vals_per_sigma.append(np.mean(cell_vals))
            else:
                vals_per_sigma.append(np.nan)
        vals = np.array(vals_per_sigma)
        offsets = (i - (n_comp - 1) / 2) * bar_width
        ax.bar(x_pos + offsets, vals, bar_width,
                color=colors[comp], label=comp, alpha=0.9,
                edgecolor="white", linewidth=0.6)

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1.2,
                label=r"unit Lipschitz ($C_*=1$)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"σ={s}" if s != "unconstrained" else "unconstrained" for s in sigmas])
    ax.set_ylabel(r"$C_*$ component value (product over layers)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                ncol=n_comp + 1, fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=160, bbox_inches="tight")
    print(f"[F_w4-bd] saved {FIG_PDF}")
    print(f"[F_w4-bd] saved {FIG_PNG}")
    print(f"[F_w4-bd] σ groups (n_cells): {[(s, len(by_sigma[s])) for s in sigmas]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    args = ap.parse_args()
    by_sigma = crawl(args.roots)
    plot(by_sigma)


if __name__ == "__main__":
    main()
