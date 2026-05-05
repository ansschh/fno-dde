"""W1-E3: Rollout error vs certified bound.

For each B4 σ-sweep cell that has a `per_frame.json` (from
capture_paper_artifacts.py), compare the empirical per-frame rollout error
E_t to the certified envelope from Cor 5.15b:

    ρ(σ, n, D)   = (√n · σ^{D+1})^{1/n}              (asymptotic closed form)
    E_t_cert(ε)  = (1 - ρ^t) / (1 - ρ) · ε           (geometric error sum)

For uncertified σ values (η = σ^{D+1} ≥ 1/√n) the closed form gives ρ ≥ 1
and the geometric envelope diverges; we record this as `certified=False`
and report the empirical curve only.

Reads:
  - per_frame.json (rel_l2_per_step + naive_copy_per_step)
  - test_results.json (config: n_layers=D, n=lag dim, sigma)

Writes `rollout_certified.json` next to each cell with:
  - rho_certified, eta = sigma^{D+1}
  - E_t_empirical (length T)
  - E_t_certified_envelope (length T) — NaN if uncertified
  - max_envelope_breach (max_t E_t_emp / E_t_cert), NaN if uncertified
  - certified: bool

Aggregator at end produces a single CSV `w1_e3_rollout_summary.csv` for plotting.

Usage:
  python scripts/eval_w1_rollout_certified.py \\
    --layer_root extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs \\
    --summary_csv reports/w1_e3_rollout_summary.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def parse_path(per_frame_path: Path):
    parts = per_frame_path.parts
    try:
        idx = parts.index("raw")
    except ValueError:
        return None
    if idx + 4 >= len(parts):
        return None
    fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
    if not seed_str.startswith("s"):
        return None
    return fam, reg, mdl, int(seed_str[1:])


def get_cell_meta(cell_dir: Path):
    """Read sigma + n_layers from test_results.json or training config."""
    tr = cell_dir / "test_results.json"
    if not tr.exists():
        return None
    try:
        data = json.loads(tr.read_text())
    except Exception:
        return None
    cfg = data.get("config", {})
    n_layers = int(cfg.get("model", {}).get("n_layers", cfg.get("n_layers", 3)))
    sigma = (cfg.get("model", {}).get("sigma")
             if cfg.get("model") else None) or cfg.get("sigma")
    sigma = float(sigma) if sigma is not None else None
    spatial_shape = cfg.get("model", {}).get("spatial_shape", [64, 64])
    n_lag = cfg.get("model", {}).get("length",
            cfg.get("length", 64))
    return {
        "n_layers": n_layers,
        "sigma": sigma,
        "n_lag": int(n_lag),
        "spatial_shape": spatial_shape,
    }


def certified_envelope(sigma: float | None, D: int, n: int, eps: float, T: int):
    """Return (rho, certified, envelope) per Cor 5.15b.

    For η = σ^{D+1} < 1/√n: ρ = (√n · η)^{1/n} < 1; envelope = (1 - ρ^t)/(1 - ρ) · ε.
    Otherwise: certified=False, envelope filled with NaN.
    """
    if sigma is None:
        return None, False, np.full(T, np.nan)
    eta = sigma ** (D + 1)
    threshold = 1.0 / math.sqrt(n)
    if eta >= threshold:
        # Uncertified: closed-form rho >= 1
        return None, False, np.full(T, np.nan)
    rho = (math.sqrt(n) * eta) ** (1.0 / n)
    if rho >= 1.0:
        return rho, False, np.full(T, np.nan)
    t_arr = np.arange(1, T + 1, dtype=np.float64)
    env = (1.0 - rho ** t_arr) / (1.0 - rho) * eps
    return rho, True, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_root", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--eps_per_step", type=float, default=None,
                    help="One-step error ε to use in envelope. Default: per-cell"
                         " mean of rel_l2_per_step[0..3] (early-rollout error).")
    args = ap.parse_args()

    layer_root = Path(args.layer_root)
    if not layer_root.is_absolute():
        layer_root = REPO / layer_root

    pf_paths = sorted(layer_root.glob("**/per_frame.json"))
    print(f"[w1-e3] {len(pf_paths)} per_frame.json files under {layer_root}")

    summary = []
    for pf in pf_paths:
        meta = parse_path(pf)
        if meta is None:
            continue
        fam, reg, mdl, seed = meta
        cell_dir = pf.parent
        cell_meta = get_cell_meta(cell_dir)
        if cell_meta is None:
            continue
        try:
            data = json.loads(pf.read_text())
        except Exception:
            continue
        E_t_key = ("rel_l2_per_step" if "rel_l2_per_step" in data
                   else ("rel_l2_mean_per_step" if "rel_l2_mean_per_step" in data
                         else None))
        if E_t_key is None:
            continue
        E_t = np.array(data[E_t_key], dtype=np.float64)
        T = E_t.size
        if T == 0:
            continue
        # Pick ε as the early-rollout per-step error (proxy for one-step defect).
        eps = (args.eps_per_step
               if args.eps_per_step is not None
               else float(np.mean(E_t[:min(4, T)])))
        sigma = cell_meta["sigma"]
        D = cell_meta["n_layers"]
        n_lag = cell_meta["n_lag"]
        rho, certified, env = certified_envelope(sigma, D, n_lag, eps, T)
        # Compute per-cell metrics
        if certified:
            ratio = E_t / np.where(env > 0, env, np.nan)
            max_breach = float(np.nanmax(ratio))
            within_envelope = bool(np.all(E_t <= env))
        else:
            max_breach = float("nan")
            within_envelope = False
        result = {
            "family": fam, "regime": reg, "model": mdl, "seed": seed,
            "sigma": sigma, "D": D, "n_lag": n_lag,
            "eta": (sigma ** (D + 1)) if sigma else None,
            "threshold_eta": 1.0 / math.sqrt(n_lag),
            "certified": certified,
            "rho": rho,
            "eps": eps,
            "T": T,
            "E_t_empirical": E_t.tolist(),
            "E_t_certified": env.tolist() if certified else [None] * T,
            "max_envelope_breach": max_breach,
            "within_envelope": within_envelope,
        }
        with open(cell_dir / "rollout_certified.json", "w") as fh:
            json.dump(result, fh, indent=2)
        summary.append({
            "family": fam, "regime": reg, "model": mdl, "seed": seed,
            "sigma": sigma, "D": D, "n_lag": n_lag,
            "eta": result["eta"],
            "certified": certified,
            "rho": rho,
            "eps": eps,
            "max_envelope_breach": max_breach,
            "within_envelope": within_envelope,
            "E_t_final": float(E_t[-1]),
            "E_t_max": float(E_t.max()),
        })

    if not summary:
        print("[w1-e3] no cells aggregated")
        return

    out_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[w1-e3] wrote {len(summary)} rows to {out_csv}")
    n_cert = sum(1 for r in summary if r["certified"])
    n_within = sum(1 for r in summary if r["within_envelope"])
    print(f"[w1-e3] {n_cert}/{len(summary)} cells certified; "
          f"{n_within}/{n_cert if n_cert else 1} within envelope")


if __name__ == "__main__":
    main()
