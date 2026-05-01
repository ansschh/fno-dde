"""
Compute lag-dependence strength (LDS) for each DDE family.

LDS := 1 - R²_now / R²_history

where R²_now is the coefficient of determination for a linear regression
predicting `y_next` (the most recent *target* value) from only the most
recent history value, and R²_history is the same regression using all
history steps. Both numerator and denominator are computed on the same
(train) data.

Interpretation:
  - LDS near 1  => the target depends on non-recent history strongly;
                   lag-equivariance has a lot to exploit (lag-dominant).
  - LDS near 0  => the target is nearly Markovian in the most recent
                   step; lag-equivariance has little to add (weakly
                   lag-dependent).

Emits one line per family with the LDS plus the two R² values. The
cluster-design doc stratifies families at LDS ≥ 0.3 (lag-dominant) vs
LDS < 0.3 (weakly lag-dependent); this script's output is used to
populate the corresponding covariate column in the results table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from datasets.sharded_dataset import ShardedDDEDataset


def _rff(X: np.ndarray, n_features: int = 512, bandwidth: float = 1.0,
         seed: int = 0) -> np.ndarray:
    """Random Fourier Features approximating an RBF kernel.
    X: (n_samples, d). Returns (n_samples, n_features) with cos(w^T x + b).
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.standard_normal((d, n_features)) / bandwidth
    b = rng.uniform(0.0, 2 * np.pi, size=n_features)
    return np.sqrt(2.0 / n_features) * np.cos(X @ W + b).astype(np.float32)


def nonlinear_r2(X: np.ndarray, y: np.ndarray, n_rff: int = 512,
                 bandwidth: Optional[float] = None, alpha: float = 1e-2) -> float:
    """Ridge regression R² on RBF random Fourier features of X."""
    if bandwidth is None:
        # Median-heuristic bandwidth (rough)
        from numpy.linalg import norm
        idx = np.random.default_rng(1).integers(0, X.shape[0], size=min(200, X.shape[0]))
        Xs = X[idx]
        sq = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
        med = np.median(sq[sq > 0]) if (sq > 0).any() else 1.0
        bandwidth = float(np.sqrt(max(med, 1e-8)))

    Z = _rff(X, n_features=n_rff, bandwidth=bandwidth)

    n = Z.shape[0]
    perm = np.random.default_rng(0).permutation(n)
    split = int(0.8 * n)
    tr, te = perm[:split], perm[split:]
    Z_tr, Z_te = Z[tr], Z[te]
    y_tr, y_te = y[tr], y[te]

    my = y_tr.mean()
    yc_tr = y_tr - my

    d = Z_tr.shape[1]
    A = Z_tr.T @ Z_tr + alpha * np.eye(d)
    beta = np.linalg.solve(A, Z_tr.T @ yc_tr)
    y_pred = Z_te @ beta + my

    ss_res = float(((y_te - y_pred) ** 2).sum())
    ss_tot = float(((y_te - y_tr.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return max(-1.0, 1.0 - ss_res / ss_tot)


def compute_lds_for_family(data_dir: str, family: str, max_samples: int = 5000,
                            horizon_steps: int = 10) -> dict:
    """LDS with a short-horizon target: predict y(t + horizon_steps * dt),
    small enough that the DDE dynamics are still somewhat predictable
    but far enough that the trivial 'y ≈ x_now' fit breaks. RBF-RFF
    ridge regression on 80% train / 20% held-out.
    """
    ds = ShardedDDEDataset(data_dir=str(data_dir), family=family, split="train")

    H_full = []
    H_now = []
    y_future = []
    n_samples = min(len(ds), max_samples)
    for i in range(n_samples):
        s = ds[i]
        inp = s['input'].numpy()
        tgt = s['target'].numpy()

        in_mask = inp[:, 1] if inp.shape[-1] > 1 else None
        if in_mask is not None:
            n_hist = int((in_mask > 0.5).sum())
        else:
            n_hist = inp.shape[0] // 4
        if n_hist <= 1 or n_hist >= inp.shape[0]:
            continue

        hist = inp[:n_hist, 0]
        most_recent = hist[-1]

        # System parameters: channels 3 onwards (0=signal, 1=mask, 2=t_norm).
        # These are broadcast-constant across the sequence so we take any row.
        if inp.shape[-1] > 3:
            params = inp[0, 3:].astype(np.float32)
        else:
            params = np.zeros(0, dtype=np.float32)

        # Short-horizon target
        y_idx = min(n_hist + horizon_steps, tgt.shape[0] - 1)
        y_val = tgt[y_idx, 0]

        # Sparse-lag "history" feature: 10 evenly-spaced lag anchors over
        # the full history window plus parameters. "now" feature: just the
        # most recent value plus parameters. Both regressions condition on
        # the parameters identically; LDS measures the incremental value of
        # past history.
        n_anchors = 10
        anchor_idx = np.linspace(0, n_hist - 1, n_anchors).astype(int)
        full_feat = np.concatenate([hist[anchor_idx], params])
        now_feat = np.concatenate([np.array([most_recent]), params])
        H_full.append(full_feat)
        H_now.append(now_feat)
        y_future.append(y_val)

    if not H_full:
        return {"family": family, "error": "no valid samples"}

    L = len(H_full[0])
    H_full_arr = np.stack(H_full).astype(np.float32)
    H_now_arr = np.array(H_now, dtype=np.float32)
    y_arr = np.array(y_future, dtype=np.float32)

    r2_full = nonlinear_r2(H_full_arr, y_arr)
    r2_now = nonlinear_r2(H_now_arr, y_arr)

    if r2_full < 1e-3:
        lds = 0.0
    else:
        lds = max(0.0, 1.0 - max(r2_now, 0.0) / r2_full)

    return {
        "family": family,
        "n_samples": len(y_arr),
        "hist_length": L,
        "r2_now": r2_now,
        "r2_history": r2_full,
        "lds": lds,
        "stratum": "lag-dominant" if lds >= 0.3 else "weakly-lag-dependent",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data_baseline_v2')
    p.add_argument('--families', type=str, nargs='+', default=None,
                   help='family names; default: auto-detect from data_dir subdirectories')
    p.add_argument('--max_samples', type=int, default=5000)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 20, 50],
                   help='forecast horizons (in time-steps) to sweep; report peak LDS')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if args.families is None:
        families = sorted(
            d.name for d in data_dir.iterdir() if d.is_dir()
        )
    else:
        families = args.families

    print(f"LDS for families in {data_dir}: {families}")
    print(f"Horizons swept: {args.horizons}; peak LDS per family reported.")
    print()
    header = (f"{'family':<14} {'n':>6} {'best_h':>7} {'R²_now':>9} {'R²_full':>9} "
              f"{'LDS_peak':>9} {'stratum':>22}")
    print(header)
    print("-" * len(header))

    results = []
    for family in families:
        best = None
        for h in args.horizons:
            try:
                r = compute_lds_for_family(str(data_dir), family, args.max_samples,
                                            horizon_steps=h)
                if 'error' in r:
                    continue
                if best is None or r['lds'] > best['lds']:
                    best = {**r, 'horizon': h}
            except Exception as e:
                print(f"  {family} horizon={h} failed: {e}")
        if best is None:
            print(f"{family:<14} no-valid-horizon")
            continue
        results.append(best)
        print(f"{family:<14} {best['n_samples']:>6} {best['horizon']:>7} "
              f"{best['r2_now']:>9.4f} {best['r2_history']:>9.4f} "
              f"{best['lds']:>9.4f} {best['stratum']:>22}")

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
