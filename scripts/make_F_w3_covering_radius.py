"""F_w3: covering-radius obstruction figure (validates revised Cor 5.9).

Predicted by the bad-completion theorem:
  - Non-equivariant baselines (per_lag_mlp_nd) error ≈ C·r(A) ≈ C·L/(2m)
    → straight 1/m line on log-log
  - Equivariant LEMO-PC error flat at the asymptote (no augmentation
    ambiguity)

Reads B3 orbit OOD data from outputs/orbit_*_runpod/ — m ∈ {1, 2, 4, 8, 16, 32}
× 3 seeds × 2 models (lemo_pc_nd, per_lag_mlp_nd).

Output: NeurIPS_LEMO/figures/kept/{main,png}/F_w3_covering_radius.{pdf,png}
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401  (sets Times New Roman globally)
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
FIG_PDF = NEURIPS / "figures" / "kept" / "main" / "F_w3_covering_radius.pdf"
FIG_PNG = NEURIPS / "figures" / "kept" / "png" / "F_w3_covering_radius.png"
FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
FIG_PNG.parent.mkdir(parents=True, exist_ok=True)


def crawl_orbit_results(roots):
    """Return dict[model][m] = list of test_rel_l2.

    Handles three layouts:
      - outputs/orbit_*/raw/m{N}/dist_exp_rd_2d_orbit/clean/<model>/s<seed>/test_results.json
      - outputs/orbit_h100/<model>_m{N}/raw/dist_exp_rd_2d_orbit/clean/<model>/s<seed>/test_results.json
      - outputs/orbit_*_h100/lemo_pc_nd_m{N}/raw/dist_exp_rd_2d_orbit/clean/<model>/s<seed>/test_results.json
    """
    import re
    m_pattern = re.compile(r"(?:^|_)m(\d+)$")
    out = defaultdict(lambda: defaultdict(list))
    for root in roots:
        rp = Path(root)
        if not rp.is_absolute():
            rp = REPO / rp
        if not rp.exists():
            continue
        for tr in rp.glob("**/test_results.json"):
            parts = tr.parts
            if "raw" not in parts:
                continue
            # Find m segment in any part of the path:
            # Could be 'm8' or 'per_lag_mlp_nd_m8' or '<sweep>_m32' etc.
            m_seg = None
            for seg in parts:
                m = m_pattern.search(seg)
                if m:
                    m_seg = int(m.group(1))
                    break
            if m_seg is None:
                continue
            mdl = None
            for known in ("lemo_pc_nd", "per_lag_mlp_nd",
                          "fno_film_nd", "fno_nd", "ndde_nd", "memno_nd",
                          "ffno_nd", "s4_nd"):
                if known in parts:
                    mdl = known
                    break
            if mdl is None:
                continue
            try:
                data = json.loads(tr.read_text())
            except Exception:
                continue
            err = data.get("test_rel_l2_mean") or data.get("test_rel_l2")
            if err is None:
                continue
            out[mdl][m_seg].append(float(err))
    return out


def plot(data, n_lag=64):
    """Plot rel-L2 vs 1/m (covering radius proxy) on log-log."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {
        "per_lag_mlp_nd": "C1",
        "lemo_pc_nd": "C0",
        "fno_film_nd": "C2",
        "fno_nd": "C3",
        "ndde_nd": "C4",
        "memno_nd": "C5",
    }
    labels = {
        "per_lag_mlp_nd": "per-lag MLP (non-equivariant)",
        "lemo_pc_nd": r"LEMO-PC (exactly equivariant, $\mathbb{Z}/n\mathbb{Z}$)",
        "fno_film_nd": "FNO + FiLM",
        "fno_nd": "FNO (vanilla)",
        "ndde_nd": "NDDE",
        "memno_nd": "MemNO",
    }
    markers = {
        "per_lag_mlp_nd": "o",
        "lemo_pc_nd": "s",
        "fno_film_nd": "^",
        "fno_nd": "v",
        "ndde_nd": "D",
        "memno_nd": "P",
    }

    for mdl, m_to_errs in sorted(data.items()):
        if not m_to_errs:
            continue
        ms = sorted(m_to_errs.keys())
        means = [np.mean(m_to_errs[m]) for m in ms]
        stds = [np.std(m_to_errs[m]) for m in ms]
        n_seeds = [len(m_to_errs[m]) for m in ms]
        # x = covering radius proxy = L/(2m). Use L=2π ≈ 6.28 (cyclic shift orbit length)
        # scaled — what matters is the 1/m shape.
        x = 1.0 / np.array(ms)
        ax.errorbar(x, means, yerr=stds,
                    color=colors.get(mdl, "gray"),
                    marker=markers.get(mdl, "x"),
                    linewidth=2, markersize=8,
                    label=labels.get(mdl, mdl) +
                          f" (n={','.join(str(n) for n in n_seeds)})")

        # Per-cell scatter
        for m, errs in m_to_errs.items():
            ax.scatter([1.0 / m] * len(errs), errs,
                       color=colors.get(mdl, "gray"),
                       alpha=0.3, s=20, zorder=2)

    # Add 1/m reference line
    if "per_lag_mlp_nd" in data and data["per_lag_mlp_nd"]:
        ms = sorted(data["per_lag_mlp_nd"].keys())
        if len(ms) >= 2:
            err_at_largest_m = np.mean(data["per_lag_mlp_nd"][ms[-1]])
            ref_x = np.array([1.0 / ms[-1], 1.0 / ms[0]])
            # err = C * 1/m → if at m=ms[-1] err = e_last, C·1/ms[-1] = e_last → C = e_last·ms[-1]
            C = err_at_largest_m * ms[-1]
            ref_y = C * ref_x
            ax.plot(ref_x, ref_y, "k--", alpha=0.4, linewidth=1.2,
                    label=r"$\propto C \cdot r(A)$ prediction")

    ax.set_xlabel(r"$1/m$ (covering-radius proxy, $r(A) \approx L/(2m)$)")
    ax.set_ylabel(r"Test rel-$L_2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=160, bbox_inches="tight")
    print(f"[F_w3] saved {FIG_PDF}")
    print(f"[F_w3] saved {FIG_PNG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    args = ap.parse_args()
    data = crawl_orbit_results(args.roots)
    print(f"[F_w3] models found: {sorted(data.keys())}")
    for mdl, m_to_errs in sorted(data.items()):
        ms = sorted(m_to_errs.keys())
        for m in ms:
            errs = m_to_errs[m]
            print(f"  {mdl} m={m}: n={len(errs)} mean={np.mean(errs):.4f}")
    if not data:
        print("[F_w3] no orbit data found — skipping figure")
        return
    plot(data)


if __name__ == "__main__":
    main()
