"""F_w4: empirical rollout error E_t vs Cor 5.14 corrected envelope.

Per advisor's CHANGE_1C, the corrected rollout envelope is:
    E_t ≤ (1 - ρ^t) / (1 - ρ) · ε
where ρ = Lipschitz of *learned* U_{Φ̂} (from B4 σ-sweep) and
ε = teacher-forced one-step defect = first-frame rel-L2 from per_frame.json.

For each σ-sweep cell:
  - Read per_frame.json: rel_l2_per_step[t] for t = 0..T-1
  - Read empirical_lipschitz.json: get L_emp_p95 as a proxy for ρ
  - Read test_results.json: get sigma target

Plot:
  Left panel:  empirical E_t vs t for each σ ∈ {0.5, 0.7, 0.9, 0.99}, mean over fam+seed
  Right panel: ratio E_t / envelope_t — should be < 1 if certified, else > 1
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
FIG_PDF = NEURIPS / "figures" / "kept" / "main" / "F_w4_rollout_envelope.pdf"
FIG_PNG = NEURIPS / "figures" / "kept" / "png" / "F_w4_rollout_envelope.png"
FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
FIG_PNG.parent.mkdir(parents=True, exist_ok=True)


def crawl_sigma_cells(roots):
    """Yield per-cell records: σ, D, n_lag, ε, E_t, L_emp."""
    for root in roots:
        rp = Path(root)
        if not rp.is_absolute():
            rp = REPO / rp
        for tr in rp.glob("**/test_results.json"):
            cell_dir = tr.parent
            try:
                d = json.loads(tr.read_text())
            except Exception:
                continue
            sigma = (d.get("sigma") or
                     (d.get("config", {}).get("model", {}) or {}).get("sigma"))
            if sigma is None:
                continue
            sigma = float(sigma)
            n_layers = int(d.get("config", {}).get("model", {}).get("n_layers",
                          d.get("config", {}).get("n_layers", 3)))
            n_lag = int(d.get("config", {}).get("model", {}).get("length",
                        d.get("config", {}).get("length", 64)))
            pf_path = cell_dir / "per_frame.json"
            if not pf_path.exists():
                continue
            try:
                pf = json.loads(pf_path.read_text())
            except Exception:
                continue
            E_t = pf.get("rel_l2_per_step") or pf.get("rel_l2_mean_per_step")
            if not E_t:
                continue
            E_t = np.array(E_t, dtype=np.float64)
            # Empirical Lipschitz if available
            lip = None
            lip_path = cell_dir / "empirical_lipschitz.json"
            if lip_path.exists():
                try:
                    lip_data = json.loads(lip_path.read_text())
                    lip = float(lip_data.get("L_emp_p95"))
                except Exception:
                    pass
            yield {
                "sigma": sigma,
                "D": n_layers,
                "n_lag": n_lag,
                "E_t": E_t,
                "L_emp_p95": lip,
            }


def plot(cells):
    """Plot empirical vs envelope per σ value."""
    by_sigma = defaultdict(list)
    for c in cells:
        by_sigma[c["sigma"]].append(c)

    if not by_sigma:
        print("[F_w4] no σ-cells found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sigmas = sorted(by_sigma.keys())
    cmap = plt.cm.viridis
    colors = {s: cmap(i / max(len(sigmas) - 1, 1)) for i, s in enumerate(sigmas)}

    # Panel 1: empirical E_t vs t
    for sigma in sigmas:
        cs = by_sigma[sigma]
        if not cs:
            continue
        E_arr = np.stack([c["E_t"] for c in cs])  # (n_cells, T)
        E_mean = E_arr.mean(axis=0)
        E_std = E_arr.std(axis=0)
        ts = np.arange(1, len(E_mean) + 1)
        axes[0].plot(ts, E_mean, color=colors[sigma], linewidth=2,
                     label=fr"$\sigma$={sigma} (n={len(cs)} cells)")
        axes[0].fill_between(ts, E_mean - E_std, E_mean + E_std,
                              color=colors[sigma], alpha=0.15)

    axes[0].set_xlabel("rollout step t")
    axes[0].set_ylabel(r"empirical rel-$L_2$ $E_t$")
    axes[0].set_title(r"Per-step empirical rollout error")
    axes[0].set_yscale("log")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    # Panel 2: ratio E_t / envelope (using empirical Lipschitz as ρ when available)
    threshold_satisfied = []
    for sigma in sigmas:
        cs = by_sigma[sigma]
        if not cs:
            continue
        D = cs[0]["D"]
        n_lag = cs[0]["n_lag"]
        eta_sigmadaa = sigma ** (D + 1)  # certified Euclidean Lipschitz
        threshold = 1.0 / math.sqrt(n_lag)
        is_certified = eta_sigmadaa < threshold
        threshold_satisfied.append((sigma, is_certified))

        # Compute the envelope using the *empirical* Lipschitz (more honest)
        # Plus the certified one for comparison
        avg_lip = np.mean([c["L_emp_p95"] for c in cs if c["L_emp_p95"] is not None])
        if not np.isnan(avg_lip) and avg_lip > 0 and avg_lip < 1:
            rho_emp = float(avg_lip)
        else:
            rho_emp = None

        E_arr = np.stack([c["E_t"] for c in cs])
        E_mean = E_arr.mean(axis=0)
        T = len(E_mean)
        # Find the first prediction frame (residual_anchor zeros the history;
        # prediction starts where E exceeds 1e-4).
        pred_idx = np.where(E_mean > 1e-4)[0]
        if len(pred_idx) == 0:
            continue
        t0 = int(pred_idx[0])
        E_pred = E_mean[t0:]
        ts_pred = np.arange(1, len(E_pred) + 1)
        # ε = error at first prediction frame (proper one-step defect)
        eps = float(E_pred[0])
        if eps < 1e-12:
            continue

        if rho_emp is not None:
            envelope_emp = (1 - rho_emp ** ts_pred) / (1 - rho_emp) * eps
            ratio = E_pred / envelope_emp
            axes[1].plot(ts_pred, ratio, color=colors[sigma], linewidth=2,
                         label=fr"$\sigma$={sigma}, "
                               fr"$\rho_{{\rm emp}}$={rho_emp:.3f}"
                               + (" (cert)" if is_certified else " (NOT cert)"))

    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5,
                    label="envelope attained")
    axes[1].set_xlabel("rollout step from first prediction frame")
    axes[1].set_ylabel(r"$E_t \,/\,$ certified envelope")
    axes[1].set_title("Empirical-to-envelope ratio")
    axes[1].set_yscale("log")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=160, bbox_inches="tight")
    print(f"[F_w4] saved {FIG_PDF}")
    print(f"[F_w4] saved {FIG_PNG}")
    print(f"[F_w4] threshold check (n={cells[0]['n_lag'] if cells else 64}):")
    for s, c in threshold_satisfied:
        print(f"    σ={s}: {'certified' if c else 'NOT certified'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    args = ap.parse_args()
    cells = list(crawl_sigma_cells(args.roots))
    print(f"[F_w4] {len(cells)} σ-projected cells with per_frame.json")
    if cells:
        plot(cells)


if __name__ == "__main__":
    main()
