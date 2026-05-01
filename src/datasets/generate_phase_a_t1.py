"""
Phase A T1 — continuous-lag transfer toy dataset.

Synthetic operator-learning task where the true target is a fixed
continuous lag convolution composed with a per-sample continuous shift.
This is the cleanest empirical realization of plan §3 T1, the
continuous-lag transfer toy.

Formally:

    Input : history  h : [-τ_max, 0] → ℝ      (random Fourier series)
            delay    τ ∈ [τ_min, τ_max]       (uniform)
    Target: y(t) := ∫_0^{T_K} K(s) · h_ext(t - s - τ) ds,   t ∈ [0, T]

where `K` is a fixed kernel (Gaussian bump centered at T_K/2, unit L¹
norm) and `h_ext` is the periodic extension of `h` with period τ_max.

For each sample we store the history φ, the parameter vector
`params = [τ]`, and the trajectory `y` in the same shard layout as
existing DDE families, so `sweep_phase_b`/`dry_run` can consume it
without a separate loader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from datasets.generate_python import write_shard_npz


def gaussian_kernel(T_K: float, n_K: int = 64, sigma_frac: float = 0.25) -> tuple:
    """Gaussian bump on [0, T_K] centered at T_K/2, normalized to unit L¹."""
    s = np.linspace(0.0, T_K, n_K, dtype=np.float64)
    sigma = sigma_frac * T_K
    K = np.exp(-0.5 * ((s - T_K / 2) / sigma) ** 2)
    # Unit L¹ norm.
    K /= (K.sum() * (T_K / n_K))
    return s, K


def sample_history(rng: np.random.Generator, t_hist: np.ndarray,
                   tau_max: float, n_fourier: int = 5) -> np.ndarray:
    """Random smooth history on `[-tau_max, 0]` via a truncated Fourier series."""
    L = tau_max
    c0 = rng.uniform(-1, 1)
    ak = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
    bk = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
    phi = np.full_like(t_hist, c0, dtype=np.float64)
    for k in range(1, n_fourier + 1):
        omega = 2.0 * np.pi * k / L
        phi += ak[k - 1] * np.cos(omega * t_hist)
        phi += bk[k - 1] * np.sin(omega * t_hist)
    # Rescale to ~unit std.
    phi /= (np.std(phi) + 1e-10)
    return phi.astype(np.float32)


def compute_target(
    phi: np.ndarray, t_hist: np.ndarray, tau: float, t_out: np.ndarray,
    K: np.ndarray, s_K: np.ndarray, tau_max: float,
) -> np.ndarray:
    """Compute y(t) = ∫ K(s) h_ext(t - s - τ) ds on the output grid."""
    # Extend history periodically with period tau_max over R; `h_ext(z)`
    # for any real `z` is the value of phi at index (z mod tau_max).
    # We evaluate at points (t - s - τ) for t ∈ t_out, s ∈ s_K.
    ds = s_K[1] - s_K[0] if len(s_K) > 1 else 1.0
    T = t_out[-1]
    tau_max_f = float(tau_max)

    # For each (t, s), query h_ext at z = t - s - τ.
    # z_grid shape: (n_out, n_K). Vectorize via outer subtraction.
    z = t_out[:, None] - s_K[None, :] - tau                    # (n_out, n_K)
    # Map z ∈ ℝ → index into phi by periodic extension:
    #   z_mod ∈ [-tau_max, 0], then linearly interpolate on t_hist grid.
    z_mod = -((-z) % tau_max_f)                                 # in [-tau_max, 0]
    # Linear interpolation on t_hist (uniform grid in [-tau_max, 0]).
    idx_f = (z_mod + tau_max_f) / tau_max_f * (len(t_hist) - 1)
    i0 = np.clip(np.floor(idx_f).astype(np.int64), 0, len(t_hist) - 1)
    i1 = np.clip(i0 + 1, 0, len(t_hist) - 1)
    w = idx_f - i0
    phi_at_z = (1.0 - w) * phi[i0] + w * phi[i1]               # (n_out, n_K)

    # Integrate ∫ K(s) h(z) ds via midpoint rule.
    y = (K[None, :] * phi_at_z).sum(axis=1) * ds                # (n_out,)
    return y.astype(np.float32)


def generate_sample_t1(
    rng: np.random.Generator, t_hist: np.ndarray, t_out: np.ndarray,
    K: np.ndarray, s_K: np.ndarray, tau_max: float, tau_min: float,
    expose_tau: bool = False,
) -> dict:
    """If `expose_tau=False` (Phase A v2 default), `params` is empty — the
    model sees only `phi`, and the per-sample random τ is hidden inside
    the target. This is the "pure-history toy" design: the model learns
    the τ-marginalized operator, and architectural equivariance shows up
    as lower-variance predictions across the τ randomness rather than as
    explicit transfer."""
    phi = sample_history(rng, t_hist, tau_max)
    tau = float(rng.uniform(tau_min, tau_max))
    y = compute_target(phi, t_hist, tau, t_out, K, s_K, tau_max)
    if expose_tau:
        params = np.array([tau], dtype=np.float32)
        lags   = np.array([tau], dtype=np.float32)
    else:
        params = np.zeros((0,), dtype=np.float32)
        lags   = np.zeros((0,), dtype=np.float32)
    return {
        "phi":    phi.reshape(-1, 1),
        "y":      y.reshape(-1, 1),
        "params": params,
        "lags":   lags,
        "t_hist": t_hist.astype(np.float32),
        "t_out":  t_out.astype(np.float32),
    }


def generate_split(
    out_root: Path, split: str, n: int, seed: int, shard_size: int,
    T: float, tau_max: float, tau_min: float, n_hist: int, n_out: int,
    T_K: float, n_K: int, expose_tau: bool = False, family_name: str = "t1_continuous_lag",
) -> int:
    s_K, K = gaussian_kernel(T_K, n_K=n_K)
    t_hist = np.linspace(-tau_max, 0.0, n_hist, dtype=np.float64)
    t_out  = np.linspace(0.0, T, n_out, dtype=np.float64)

    split_dir = out_root / family_name / split
    split_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (n + shard_size - 1) // shard_size
    total = 0
    for sid in range(n_shards):
        path = split_dir / f"shard_{sid:03d}.npz"
        if path.exists():
            total += shard_size
            continue
        remaining = n - sid * shard_size
        B = min(shard_size, remaining)
        rng = np.random.default_rng(seed + sid * 1000)
        samples: list[dict] = []
        pbar = tqdm(total=B, desc=f"  {family_name} {split} shard {sid}", leave=False)
        for _ in range(B):
            samples.append(generate_sample_t1(
                rng, t_hist, t_out, K, s_K, tau_max, tau_min,
                expose_tau=expose_tau,
            ))
            pbar.update(1)
        pbar.close()
        meta = {
            "family": "t1_continuous_lag",
            "split": split,
            "shard_id": sid,
            "n_samples": len(samples),
            "config": {"T": T, "tau_max": tau_max, "tau_min": tau_min,
                       "n_hist": n_hist, "n_out": n_out,
                       "T_K": T_K, "n_K": n_K},
            "seed": seed + sid * 1000,
            "generator": "phase_a_t1",
        }
        write_shard_npz(str(path), samples, meta)
        total += len(samples)
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=str, default="data_phase_a")
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val",   type=int, default=400)
    p.add_argument("--n_test",  type=int, default=400)
    p.add_argument("--shard_size", type=int, default=256)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--T",       type=float, default=8.0)
    p.add_argument("--tau_max", type=float, default=1.0)
    p.add_argument("--tau_min", type=float, default=0.1)
    p.add_argument("--n_hist",  type=int, default=128)
    p.add_argument("--n_out",   type=int, default=128)
    p.add_argument("--T_K",     type=float, default=1.0)
    p.add_argument("--n_K",     type=int,   default=64)
    p.add_argument("--expose_tau", action="store_true",
                   help="Expose τ as input feature (v1). Default: hide τ (v2).")
    p.add_argument("--family_name", type=str, default=None,
                   help="Override family name. Default: 't1_continuous_lag' "
                        "if expose_tau else 't1_continuous_lag_v2'.")
    args = p.parse_args()

    family_name = args.family_name or (
        "t1_continuous_lag" if args.expose_tau else "t1_continuous_lag_v2"
    )
    out_root = Path(args.output_dir)
    fam_dir = out_root / family_name
    fam_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {family_name} (expose_tau={args.expose_tau}) under {fam_dir}/")
    totals = {}
    for split, n, off in [("train", args.n_train, 0),
                           ("val",   args.n_val,   100_000),
                           ("test",  args.n_test,  200_000)]:
        print(f"  {split}: {n} samples")
        totals[split] = generate_split(
            out_root, split, n, args.seed + off, args.shard_size,
            args.T, args.tau_max, args.tau_min, args.n_hist, args.n_out,
            args.T_K, args.n_K,
            expose_tau=args.expose_tau, family_name=family_name,
        )

    manifest = {
        "family": family_name,
        "description": (
            "Phase A T1 — continuous-lag transfer toy. Target is a fixed-"
            "kernel convolution composed with a per-sample continuous delay "
            "τ. v1 (expose_tau=True): τ is broadcast as an input channel. "
            "v2 (expose_tau=False, default): τ is hidden, the model learns "
            "the τ-marginalized operator from h alone. v2 isolates whether "
            "architectural equivariance helps even without explicit τ access."
        ),
        "config": {
            "T": args.T, "tau_max": args.tau_max, "tau_min": args.tau_min,
            "n_hist": args.n_hist, "n_out": args.n_out,
            "T_K": args.T_K, "n_K": args.n_K,
            "expose_tau": bool(args.expose_tau),
        },
        "param_names": ["tau"] if args.expose_tau else [],
        "param_ranges": ({"tau": [args.tau_min, args.tau_max]}
                         if args.expose_tau else {}),
        "state_dim": 1,
        "input_channels": 1,
        "splits": {s: {"n_samples": totals[s]} for s in totals},
        "seed": args.seed,
        "generator": "phase_a_t1",
    }
    with open(fam_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest: {fam_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
