"""W1 figure: σ-stability frontier with shaded certified region.

Reads:
  - Per-cell `rollout_certified.json` (from eval_w1_rollout_certified.py)
  - Per-cell `empirical_lipschitz.json` (from eval_w1_empirical_lipschitz.py)
  - Per-cell `test_results.json` for σ-target + final relL2

Produces three panels in a single figure (saved to NeurIPS_LEMO/figures/):

  Left:  σ-stability frontier — final-rollout rel-L2 vs σ, with shaded
         certified region σ < n^{-1/(2(D+1))}.  Each (family, seed) cell
         a dot; per-σ mean ± 95% CI.

  Mid:   Empirical Lipschitz vs certified bound — for each cell, plot
         L_emp_p95 / σ^{D+1} (tightness ratio).  Hline at 1.0 = bound is
         attained.

  Right: Rollout envelope tracking — for each *certified* cell, plot
         empirical E_t (mean over seeds) and the certified envelope.

Outputs `F_w1_certified_region.{pdf,png}` in NeurIPS_LEMO/figures/main/.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
FIG_DIR = NEURIPS / "figures" / "kept" / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = NEURIPS / "figures" / "kept" / "png"
PNG_DIR.mkdir(parents=True, exist_ok=True)

FAM_LABEL = {
    "dist_exp_rd_2d": "Exp",
    "dist_gaussian_rd_2d": "Gauss",
    "dist_gamma_rd_2d": "Gamma",
    "dist_uniform_rd_2d": "Uniform",
    "dist_powerlaw_rd_2d": "Power",
}


def crawl_cells(roots):
    """Yield per-cell records pulled from each root."""
    for root in roots:
        root = Path(root)
        if not root.is_absolute():
            root = REPO / root
        if not root.exists():
            continue
        for tr in root.glob("**/test_results.json"):
            parts = tr.parts
            try:
                idx = parts.index("raw")
            except ValueError:
                continue
            if idx + 4 >= len(parts):
                continue
            fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
            if not seed_str.startswith("s"):
                continue
            seed = int(seed_str[1:])
            cell_dir = tr.parent
            try:
                tr_data = json.loads(tr.read_text())
            except Exception:
                continue
            cfg = tr_data.get("config", {})
            sigma = (cfg.get("model", {}).get("sigma")
                     if cfg.get("model") else None) or cfg.get("sigma")
            sigma = float(sigma) if sigma is not None else None
            n_layers = int(cfg.get("model", {}).get("n_layers",
                          cfg.get("n_layers", 3)))
            n_lag = int(cfg.get("model", {}).get("length",
                       cfg.get("length", 64)))
            rec = {
                "family": fam, "regime": reg, "model": mdl, "seed": seed,
                "sigma": sigma, "D": n_layers, "n_lag": n_lag,
                "test_rel_l2": tr_data.get("test_rel_l2_mean",
                                            tr_data.get("test_rel_l2", float("nan"))),
                "cell_dir": cell_dir,
            }
            for fname in ("empirical_lipschitz.json", "rollout_certified.json",
                          "per_frame.json"):
                p = cell_dir / fname
                rec[fname] = json.loads(p.read_text()) if p.exists() else None
            yield rec


def panel_left(ax, cells, n_default=128, D_default=3):
    """Final-rollout error vs σ, shaded certified region."""
    sigma_to_errs = defaultdict(list)
    for c in cells:
        if c["sigma"] is None:
            continue
        if c["model"] != "lemo_pc_nd":  # focus on LEMO-PC for cert claim
            continue
        if c["regime"] != "clean":
            continue
        sigma_to_errs[c["sigma"]].append(c["test_rel_l2"])

    if not sigma_to_errs:
        ax.text(0.5, 0.5, "no σ-sweep data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("σ target")
        ax.set_ylabel("test rel-L₂")
        return

    sigmas = sorted(sigma_to_errs.keys())
    means = [np.mean(sigma_to_errs[s]) for s in sigmas]
    stds = [np.std(sigma_to_errs[s]) for s in sigmas]

    # Plot certified region (left of threshold)
    sigma_threshold = n_default ** (-1.0 / (2 * (D_default + 1)))
    ax.axvspan(0.0, sigma_threshold, color="lightgreen", alpha=0.25,
               label=f"Certified σ < {sigma_threshold:.3f}")
    ax.axvline(sigma_threshold, color="green", linestyle="--", linewidth=1.0)

    # Per-cell dots
    for s, errs in sigma_to_errs.items():
        ax.scatter([s] * len(errs), errs, color="C0", alpha=0.4, s=20)
    # Mean ± std
    ax.errorbar(sigmas, means, yerr=stds, color="C0", marker="o",
                linewidth=2, label="mean ± 1σ")

    ax.set_xlabel(r"$\sigma$ target (per-mode SVD projection)")
    ax.set_ylabel("Test rel-$L_2$ (final rollout)")
    ax.set_yscale("log")
    ax.set_title(rf"$\sigma$-stability frontier (n={n_default}, D={D_default})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def panel_mid(ax, cells):
    """Empirical L / certified bound (tightness ratio)."""
    sigma_ratios = defaultdict(list)
    for c in cells:
        emp = c["empirical_lipschitz.json"]
        if emp is None or c["sigma"] is None:
            continue
        if c["model"] != "lemo_pc_nd":
            continue
        eta = c["sigma"] ** (c["D"] + 1)
        if eta <= 0:
            continue
        sigma_ratios[c["sigma"]].append(emp["L_emp_p95"] / eta)
    if not sigma_ratios:
        ax.text(0.5, 0.5, "no L_emp data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(r"$\sigma$ target")
        ax.set_ylabel(r"$L_{\rm emp}^{p95} / \sigma^{D+1}$")
        return
    sigmas = sorted(sigma_ratios.keys())
    means = [np.mean(sigma_ratios[s]) for s in sigmas]
    stds = [np.std(sigma_ratios[s]) for s in sigmas]
    for s, ratios in sigma_ratios.items():
        ax.scatter([s] * len(ratios), ratios, color="C1", alpha=0.4, s=20)
    ax.errorbar(sigmas, means, yerr=stds, color="C1", marker="o", linewidth=2)
    ax.axhline(1.0, color="black", linestyle="--", label="bound attained")
    ax.set_xlabel(r"$\sigma$ target")
    ax.set_ylabel(r"$L_{\rm emp}^{\,p95}\,/\,\sigma^{D+1}$")
    ax.set_title("Empirical Lipschitz vs certified bound")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def panel_right(ax, cells):
    """Empirical rollout E_t vs certified envelope (σ=0.5 cells only)."""
    by_t = defaultdict(list)
    env_by_t = defaultdict(list)
    cert_certs = []
    for c in cells:
        rc = c["rollout_certified.json"]
        if rc is None or not rc.get("certified"):
            continue
        if c["model"] != "lemo_pc_nd":
            continue
        E_t = np.array(rc["E_t_empirical"])
        env = np.array(rc["E_t_certified"], dtype=float)
        for t in range(len(E_t)):
            by_t[t].append(E_t[t])
            if not np.isnan(env[t]):
                env_by_t[t].append(env[t])
        cert_certs.append(rc.get("rho"))
    if not by_t:
        ax.text(0.5, 0.5, "no certified cells",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("rollout step t")
        ax.set_ylabel(r"$E_t$ (rel-L$_2$)")
        return
    ts = sorted(by_t.keys())
    emp_mean = [np.mean(by_t[t]) for t in ts]
    emp_std = [np.std(by_t[t]) for t in ts]
    env_mean = [np.mean(env_by_t[t]) for t in ts if t in env_by_t]
    ax.plot(ts, emp_mean, color="C0", linewidth=2,
            label="empirical $E_t$ (σ=0.5, certified)")
    ax.fill_between(ts, np.array(emp_mean) - np.array(emp_std),
                    np.array(emp_mean) + np.array(emp_std),
                    color="C0", alpha=0.2)
    ax.plot(ts, env_mean, color="C2", linewidth=2, linestyle="--",
            label=r"certified envelope $(1-\rho^t)/(1-\rho)\cdot\varepsilon$")
    ax.set_xlabel("rollout step t")
    ax.set_ylabel(r"rel-$L_2$ error $E_t$")
    ax.set_title("Rollout: empirical vs certified envelope")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Sweep roots to crawl, e.g. extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs")
    ap.add_argument("--out_name", default="F_w1_certified_region")
    args = ap.parse_args()

    cells = list(crawl_cells(args.roots))
    print(f"[F_w1] {len(cells)} cells crawled")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panel_left(axes[0], cells)
    panel_mid(axes[1], cells)
    panel_right(axes[2], cells)
    fig.tight_layout()

    pdf_path = FIG_DIR / f"{args.out_name}.pdf"
    png_path = PNG_DIR / f"{args.out_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"[F_w1] saved {pdf_path}")
    print(f"[F_w1] saved {png_path}")


if __name__ == "__main__":
    main()
