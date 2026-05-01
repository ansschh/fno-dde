"""LDS (Lag-Dependence Statistic) sweep — data-only, no ML ckpts required.

Quantifies how lag-dependent each family's dynamics are.  The expectation
is that LEMO-PC's advantage over Markov baselines correlates with the LDS.

For each family, computes:
  acf_lag_k:    Pearson correlation of u(t) with u(t-k), averaged over
                spatial dims + trajectories, for k = 1..K.
  acf_integral: sum_k |acf_lag_k|        (large = strong lag dependence)
  acf_halflife: smallest k where |acf| < 0.5 (large = long memory)
  R2_markov:    one-step linear-regression goodness on u(t+1) vs u(t)
  R2_full:      one-step linear-regression on u(t+1) vs u(t-K..t)
  LDS:          R2_full - R2_markov   (large = lag knowledge helps)

Outputs:
  paper/stats/lds_per_family.json    aggregate stats
  paper/figures/L01_lds_bar.{pdf,png}     bar chart per family
  paper/figures/L02_acf_curves.{pdf,png}  ACF(k) curves per family
  paper/tables/T07_lds.tex          family x metric table

Usage:
    python3 scripts/lds_sweep.py \\
        --data_dir data_dde_pde \\
        --families dist_exp_rd_2d,dist_gaussian_rd_2d,dist_gamma_rd_2d,dist_uniform_rd_2d,dist_powerlaw_rd_2d \\
        [--K 32]   # max lag offset

If a family's data dir is missing locally, the family is skipped (not an error).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
STATS_DIR = REPO / "paper" / "stats"
FIG_DIR = REPO / "paper" / "figures"
TAB_DIR = REPO / "paper" / "tables"
for d in (STATS_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
                "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]


def load_family_test_shard(data_dir: Path, family: str):
    fam_dir = data_dir / family
    manifest = json.loads((fam_dir / "manifest.json").read_text())
    n_hist = manifest["n_hist"]
    n_out = manifest["n_out"]
    shard = np.load(fam_dir / "test" / "shard_000.npz")
    phi = shard["phi"]
    y = shard["y"]
    traj = np.concatenate([phi, y], axis=1)
    return traj, n_hist, n_out


def compute_acf(traj, K: int = 32):
    T = traj.shape[1]
    K = min(K, T - 1)
    flat = traj.reshape(traj.shape[0], T, -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-12
    z = (flat - mean) / std
    acf = np.zeros(K + 1, dtype=np.float64)
    for k in range(K + 1):
        if k == 0:
            acf[k] = 1.0
            continue
        a = z[:, k:, :]
        b = z[:, :T - k, :]
        acf[k] = float((a * b).mean())
    return acf


def compute_R2_markov_vs_full(traj, K_hist: int = 16):
    N, T = traj.shape[0], traj.shape[1]
    flat = traj.reshape(N, T, -1)
    K_hist = min(K_hist, T - 2)

    X_mk = flat[:, :T - 1]
    Y    = flat[:, 1:T]
    Xm = X_mk.reshape(-1, X_mk.shape[-1])
    Ym = Y.reshape(-1, Y.shape[-1])
    Am, *_ = np.linalg.lstsq(Xm, Ym, rcond=None)
    Ym_hat = Xm @ Am
    ss_res_m = float(((Ym_hat - Ym) ** 2).sum())
    ss_tot = float(((Ym - Ym.mean(axis=0, keepdims=True)) ** 2).sum())
    R2_markov = 1.0 - ss_res_m / max(ss_tot, 1e-12)

    if K_hist + 1 <= T - 1:
        Xs = []
        for j in range(K_hist + 1):
            Xs.append(flat[:, K_hist - j: T - 1 - j])
        Xf = np.concatenate(Xs, axis=-1)
        Yf = flat[:, K_hist + 1: T]
        Xf_2d = Xf.reshape(-1, Xf.shape[-1])
        Yf_2d = Yf.reshape(-1, Yf.shape[-1])
        Af, *_ = np.linalg.lstsq(Xf_2d, Yf_2d, rcond=None)
        Yf_hat = Xf_2d @ Af
        ss_res_f = float(((Yf_hat - Yf_2d) ** 2).sum())
        ss_tot_f = float(((Yf_2d - Yf_2d.mean(axis=0, keepdims=True)) ** 2).sum())
        R2_full = 1.0 - ss_res_f / max(ss_tot_f, 1e-12)
    else:
        R2_full = R2_markov
    return R2_markov, R2_full


def lds_for_family(data_dir: Path, family: str, K: int = 32, K_hist: int = 16,
                   max_traj: int = 64):
    traj, n_hist, n_out = load_family_test_shard(data_dir, family)
    print(f"  {family}: traj shape {traj.shape}")
    if traj.shape[0] > max_traj:
        idx = np.random.RandomState(0).choice(traj.shape[0], max_traj, replace=False)
        traj = traj[idx]
    acf = compute_acf(traj, K=K)
    acf_integral = float(np.abs(acf[1:]).sum())
    half_arr = np.where(np.abs(acf) < 0.5)[0]
    if len(half_arr) == 0:
        half = K
    else:
        half = int(half_arr[0])
    R2_markov, R2_full = compute_R2_markov_vs_full(traj, K_hist=K_hist)
    LDS = R2_full - R2_markov
    return {
        "family": family,
        "n_traj": int(traj.shape[0]),
        "T_total": int(traj.shape[1]),
        "n_hist": n_hist,
        "n_out": n_out,
        "acf": acf.tolist(),
        "acf_integral": acf_integral,
        "acf_halflife_k": half,
        "R2_markov": R2_markov,
        "R2_full": R2_full,
        "LDS": LDS,
    }


def plot_results(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams_present = list(results.keys())
    if not fams_present:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(fams_present))
    lds_values = [results[f]["LDS"] for f in fams_present]
    ax.bar(x, lds_values, color="#1f77b4", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_rd_2d", "").replace("dist_", "") for f in fams_present],
                       rotation=20)
    ax.set_ylabel(r"LDS = $R^2_{full} - R^2_{markov}$")
    ax.set_title("Lag-dependence statistic per family (positive = full history beats Markov)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for xi, v in zip(x, lds_values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "L01_lds_bar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"  -> {out.name}")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for f in fams_present:
        acf = np.array(results[f]["acf"])
        k = np.arange(len(acf))
        ax.plot(k, acf, marker="o", markersize=3, lw=1.0,
                label=f.replace("_rd_2d", "").replace("dist_", ""))
    ax.set_xlabel(r"lag $k$")
    ax.set_ylabel("auto-correlation")
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("Trajectory auto-correlation function per family")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=False)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / "L02_acf_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def write_table(results: dict):
    fams_present = list(results.keys())
    if not fams_present:
        return
    body = [r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Family & ACF integral & ACF half-life $k$ & $R^2_{markov}$ & LDS \\",
            r"\midrule"]
    for f in fams_present:
        d = results[f]
        body.append(f"{f.replace('_rd_2d','').replace('dist_','')} & "
                    f"{d['acf_integral']:.2f} & {d['acf_halflife_k']} & "
                    f"{d['R2_markov']:.3f} & {d['LDS']:.3f} \\\\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = ([r"\begin{table}[h]",
          r"\centering",
          (r"\caption{Lag-dependence statistic (LDS) per family. "
           r"Higher LDS means full-history regression beats Markov-fit "
           r"by a larger margin --- i.e.\ the dynamics are more delay-dependent.}"),
          r"\label{tab:lds}"]
         + body
         + [r"\end{table}"])
    out = TAB_DIR / "T07_lds.tex"
    out.write_text("\n".join(s))
    print(f"  -> {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--families", default=",".join(DEFAULT_FAMS))
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--K_hist", type=int, default=16)
    args = ap.parse_args()

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    data_dir = Path(args.data_dir)
    print(f"[lds] data_dir={data_dir}, K={args.K}, K_hist={args.K_hist}")

    results = {}
    for fam in fams:
        if not (data_dir / fam / "manifest.json").exists():
            print(f"  {fam}: SKIP (missing data dir)")
            continue
        try:
            r = lds_for_family(data_dir, fam, K=args.K, K_hist=args.K_hist)
            results[fam] = r
            print(f"  {fam}: ACF_int={r['acf_integral']:.2f}, half-life={r['acf_halflife_k']}, "
                  f"R2_mk={r['R2_markov']:.3f}, R2_fu={r['R2_full']:.3f}, LDS={r['LDS']:.3f}")
        except Exception as e:
            print(f"  {fam}: FAIL ({type(e).__name__}: {e})")

    out_json = STATS_DIR / "lds_per_family.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  -> {out_json}")

    plot_results(results)
    write_table(results)


if __name__ == "__main__":
    main()
