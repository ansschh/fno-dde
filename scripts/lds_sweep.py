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


def compute_R2_markov_vs_full(traj, K_hist: int = 16, test_frac: float = 0.3,
                                spatial_summary: str = "mean"):
    """R^2 of Markov vs full-history LINEAR regression on the INCREMENT
    Δu(t) = u(t+1) - u(t).  Predicting the value u(t+1) is trivial for
    smooth dynamics (R^2 ≈ 1 even with one feature) — predicting the
    increment isolates the part of the dynamics that depends on
    history beyond the current frame.

    Markov:  Δu(t) = A · u(t)
    Full:    Δu(t) = sum_{j=0..K} A_j · u(t-j)

    Train/test split is over trajectories (70/30) so R^2 reflects
    generalisation, not overfit interpolation.  Spatial summary
    (default: mean) collapses spatial dim to keep regression tractable.
    """
    N, T = traj.shape[0], traj.shape[1]
    K_hist = min(K_hist, T - 2)
    if spatial_summary == "mean":
        flat = traj.reshape(N, T, -1).mean(axis=-1, keepdims=True)
    elif spatial_summary == "raw":
        flat = traj.reshape(N, T, -1)
    else:
        raise ValueError(spatial_summary)
    delta = flat[:, 1:] - flat[:, :-1]     # (N, T-1, D)  the increment
    flat_x = flat[:, :-1]                   # (N, T-1, D)  current value (lined up with delta)

    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    n_test = max(1, int(round(N * test_frac)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if len(train_idx) < 2:
        return 0.0, 0.0

    train_x = flat_x[train_idx]; test_x = flat_x[test_idx]
    train_y = delta[train_idx];  test_y = delta[test_idx]
    Tx = train_x.shape[1]

    # Markov: Δu(t) = A u(t).  Single-frame predictor.
    Xtr_m = train_x.reshape(-1, train_x.shape[-1])
    Ytr_m = train_y.reshape(-1, train_y.shape[-1])
    Xte_m = test_x.reshape(-1, test_x.shape[-1])
    Yte_m = test_y.reshape(-1, test_y.shape[-1])
    Am, *_ = np.linalg.lstsq(Xtr_m, Ytr_m, rcond=None)
    Yte_pred = Xte_m @ Am
    ss_res_m = float(((Yte_pred - Yte_m) ** 2).sum())
    ss_tot_m = float(((Yte_m - Yte_m.mean(axis=0, keepdims=True)) ** 2).sum())
    R2_markov = 1.0 - ss_res_m / max(ss_tot_m, 1e-12)

    # Full: Δu(t) = sum_{j=0..K} A_j u(t-j).  Stack the K_hist past frames.
    if K_hist + 1 <= Tx:
        def stack_history(arr_xt):
            xs = []
            for j in range(K_hist + 1):
                xs.append(arr_xt[:, K_hist - j: Tx - j])
            return np.concatenate(xs, axis=-1)
        Xtr_f_3d = stack_history(train_x)
        Ytr_f_3d = train_y[:, K_hist:]
        Xte_f_3d = stack_history(test_x)
        Yte_f_3d = test_y[:, K_hist:]
        Xtr_f = Xtr_f_3d.reshape(-1, Xtr_f_3d.shape[-1])
        Ytr_f = Ytr_f_3d.reshape(-1, Ytr_f_3d.shape[-1])
        Xte_f = Xte_f_3d.reshape(-1, Xte_f_3d.shape[-1])
        Yte_f = Yte_f_3d.reshape(-1, Yte_f_3d.shape[-1])
        Af, *_ = np.linalg.lstsq(Xtr_f, Ytr_f, rcond=None)
        Yte_f_pred = Xte_f @ Af
        ss_res_f = float(((Yte_f_pred - Yte_f) ** 2).sum())
        ss_tot_f = float(((Yte_f - Yte_f.mean(axis=0, keepdims=True)) ** 2).sum())
        R2_full = 1.0 - ss_res_f / max(ss_tot_f, 1e-12)
    else:
        R2_full = R2_markov
    return float(R2_markov), float(R2_full)


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
    # L01_lds_bar dropped (2026-05-03) — single-bar-per-family figure was
    # information-thin (one number per family, no underlying R^2 values
    # shown, no error bars). The story is now carried by the enhanced
    # T07_lds.tex table which exposes BOTH R^2_Markov (negative across
    # all families) and R^2_full (~1.0) per family, making the gap
    # (LDS) concrete instead of abstract. Figure file kept on disk per
    # the no-delete rule.

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
    # Column layout: Family | R^2_markov | R^2_full | LDS = R^2_full - R^2_markov
    # ACF integral and ACF half-life are essentially constant across families
    # (integral ~22.4, half-life=27 for all) so they were dropped.
    fam_label = {"dist_exp": "Exp", "dist_gaussian": "Gauss", "dist_gamma": "Gamma",
                  "dist_uniform": "Uniform", "dist_powerlaw": "Power"}
    def _label(f):
        key = f.replace("_rd_2d", "")
        return fam_label.get(key, key.replace("dist_", "").capitalize())
    body = [r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Family & $R^2_{\mathrm{Markov}}$ & $R^2_{\mathrm{full}}$ & "
            r"LDS $= R^2_{\mathrm{full}} - R^2_{\mathrm{Markov}}$ \\",
            r"\midrule"]
    for f in fams_present:
        d = results[f]
        body.append(f"{_label(f)} & {d['R2_markov']:.3f} & {d['R2_full']:.3f} & "
                    f"{d['LDS']:.3f} \\\\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = ([r"\begin{table}[h]",
          r"\centering",
          (r"\caption{Lag-dependence statistic (LDS) per family. "
           r"$R^2_{\mathrm{Markov}}$ is the one-step linear-regression $R^2$ "
           r"using only the latest state $u(t)$ as input; $R^2_{\mathrm{full}}$ "
           r"uses the full history window $u(t-K..t)$. "
           r"$R^2_{\mathrm{Markov}} < 0$ across all families means the Markov fit "
           r"is worse than predicting the mean, while $R^2_{\mathrm{full}} \approx 1$ "
           r"shows full history explains the dynamics; their gap (LDS) quantifies "
           r"how much delay knowledge matters.}"),
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
