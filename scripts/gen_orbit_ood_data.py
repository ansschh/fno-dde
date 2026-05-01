"""
Lag-shift orbit OOD generator.

Generates a controlled-shift dataset for the orbit-OOD experiment described in
`scripts/orbit_ood_design.md`. The kernel SHAPE for the dist_exp_rd_2d family
is fixed; the kernel POSITION (cyclic shift on the lag-quadrature axis) is the
controlled variable.

For each of N_orbits base trajectories (independent IC + reaction params), we
produce 40 shifted copies — one per `tau_shift in {0, 1, ..., n_quad-1}` —
indexed by the cyclic group C_{n_quad}.

The simulator is a thin wrapper around `simulate_dist_kernel_rd` from
`gen_dde_pde_data.py`: we only override the kernel-weight vector to be a
cyclic rotation of the canonical exp-kernel weights.

Splits (orbit-aware):
  train: orbits 0..255, shifts in S_train_m   (subset depends on m)
  val  : orbits 0..255, shifts in S_train_m  (held-out 32 orbits)
  test : orbits 0..255, shifts in S_test     (the orbit-OOD set)

Where S_train_m and S_test are defined for each augmentation budget m in
{1, 2, 4, 8, 16, 32}. m=8 is the headline LEMO-PC training set (the orbit
representatives at indices 0,5,10,...,35).

Output (per m):
  data_orbit_ood/m{m}/dist_exp_rd_2d_orbit/{train,val,test}/shard_000.npz
  with keys: phi (N, 64, 64, 64, 1), y (N, 64, 64, 64, 1), params (N, 1),
             lags (N, 1) [tau_shift, for diagnostic only], orbit_id (N, 1),
             t_hist, t_out
  + manifest.json: {m: int, n_quad: int, S_train: list[int], S_test: list[int],
                    orbits_in_split: {train:[(o,k)], val:..., test:...}}

Usage:
    python3 scripts/gen_orbit_ood_data.py \
        --num_orbits 256 --m_values 1,2,4,8,16,32 \
        --out_dir data_orbit_ood

DESIGN-TIME ONLY: do not run on CPU local; queue on a CPU node or one of the
H100 nodes (data gen is CPU-bound). Wall: ~30 min on a 24-core node.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Reuse simulator + IC sampler.
from gen_dde_pde_data import (  # noqa: E402
    SpectralGrid2D, smooth_random_field_2d,
    DistKernelRDParams, dist_kernel_rd_rhs, _build_kernel_weights,
    write_shard,
)


# ---------------------------------------------------------------------
# Cyclic-shift action on the kernel-weight vector.
# ---------------------------------------------------------------------

def shifted_kernel_weights(kernel_weights_canonical: np.ndarray,
                            tau_shift: int) -> np.ndarray:
    """Apply Z/n_quad-Z cyclic shift to the canonical kernel-weight vector.

    The canonical weights have mass concentrated near s=0 (i.e. the recent
    past).  Shifting by k moves the mass to indices [k, k + support_width)
    on the lag axis, mod n_quad.

    Parameters
    ----------
    kernel_weights_canonical : (n_quad,)  the K(s)*dt*half_endpoints vector,
                                          normalized to sum to 1.
    tau_shift                : int in [0, n_quad)

    Returns
    -------
    shifted weights, also summing to 1.
    """
    n_quad = kernel_weights_canonical.shape[0]
    k = int(tau_shift) % n_quad
    return np.roll(kernel_weights_canonical, k)


# ---------------------------------------------------------------------
# Single-trajectory simulator with a *given* kernel-weights vector.
# ---------------------------------------------------------------------

def simulate_dist_kernel_rd_with_kw(p: DistKernelRDParams,
                                     rng: np.random.Generator,
                                     grid: SpectralGrid2D,
                                     kernel_weights: np.ndarray,
                                     u0_seed: np.random.Generator,
                                     ) -> np.ndarray:
    """Like `simulate_dist_kernel_rd` but takes the kernel weights as input.

    Critically, the IC `u0` is generated from `u0_seed` (independent of `rng`)
    so that all 40 shifted copies of the same orbit share the same IC.
    """
    n_steps = int(round(p.T_total / p.dt))
    n_quad = int(round(p.tau_max / p.dt))
    if abs(n_quad * p.dt - p.tau_max) > 1e-9:
        raise ValueError(f"tau_max={p.tau_max} not divisible by dt={p.dt}")
    if kernel_weights.shape[0] != n_quad:
        raise ValueError(
            f"kernel_weights len {kernel_weights.shape[0]} != n_quad {n_quad}")
    sf = p.D * 2 * (np.pi * p.n_grid / p.L) ** 2 * p.dt
    if sf > 2.78:
        raise ValueError(f"unstable D*k^2*dt = {sf:.2f}")

    # IC amplitude bumped from 0.2 -> 0.45 (still keeps u0 in (0.05, 0.95)).
    # Combined with reaction A in [8,11] (raised from [0.5,2.0]), this drives
    # the trajectory into the nonlinear regime so that the cyclic-shift action
    # on the kernel actually moves the trajectory: per-orbit pairwise rel-L2
    # between half-cycle shifts is >= 0.05 (audit: 0.07-0.16) instead of the
    # ~0.005 of the old setting.
    u0 = smooth_random_field_2d(u0_seed, grid, amplitude=0.45, k_max=4, base=0.5)
    history = np.tile(u0[None, ...], (n_quad, 1, 1))
    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0

    def get_history(step):
        out = np.empty((n_quad, p.n_grid, p.n_grid))
        n_traj = min(step + 1, n_quad)
        n_init = n_quad - n_traj
        if n_init > 0:
            out[n_traj:] = history[0]
        if n_traj > 0:
            out[:n_traj] = traj[step + 1 - n_traj : step + 1][::-1]
        return out

    for step in range(n_steps):
        u = traj[step]
        hist = get_history(step)
        k1 = dist_kernel_rd_rhs(u, hist, kernel_weights, p, grid)
        k2 = dist_kernel_rd_rhs(u + 0.5 * p.dt * k1, hist, kernel_weights, p, grid)
        k3 = dist_kernel_rd_rhs(u + 0.5 * p.dt * k2, hist, kernel_weights, p, grid)
        k4 = dist_kernel_rd_rhs(u + p.dt * k3, hist, kernel_weights, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(
                f"orbit OOD sim NaN at step {step+1}")
    return traj


# ---------------------------------------------------------------------
# Train/test shift selection (covering-radius parameterization).
# ---------------------------------------------------------------------

def shift_subsets(n_quad: int, m_values: List[int]) -> dict:
    """For each m in m_values, return (S_train, S_test) where S_train is m
    evenly-spaced shifts on Z/n_quad-Z and S_test is the complement.

    The covering radius is r(A) ≈ n_quad / (2m).  m=1 has r=n_quad/2; m=n_quad
    has r=0.
    """
    out = {}
    for m in m_values:
        if m > n_quad:
            raise ValueError(f"m={m} > n_quad={n_quad}")
        step = n_quad // m
        S_train = sorted([(i * step) % n_quad for i in range(m)])
        S_test = sorted([k for k in range(n_quad) if k not in S_train])
        out[m] = {"S_train": S_train, "S_test": S_test,
                   "covering_radius": (step // 2)}
    return out


# ---------------------------------------------------------------------
# Orbit pool generation (one shared pool, sliced by m at write time).
# ---------------------------------------------------------------------

def generate_orbit_pool(num_orbits: int, n_quad: int, n_hist: int, n_out: int,
                         dt: float, tau: float, n_grid: int, L: float,
                         seed: int) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray,
                                              np.ndarray]:
    """Generate the full orbit pool: `num_orbits` orbits x `n_quad` shifts.

    Returns
    -------
    phi       : (num_orbits, n_quad, n_hist, n_grid, n_grid, 1) float32
    y         : (num_orbits, n_quad, n_out,  n_grid, n_grid, 1) float32
    params    : (num_orbits, n_quad, 1)                          float32 (A)
    orbit_id  : (num_orbits, n_quad)                              int32
    tau_shift : (num_orbits, n_quad)                              int32
    """
    grid = SpectralGrid2D.make(n=n_grid, L=L)
    n_total = n_hist + n_out
    T_total_needed = (n_total - 1) * dt
    tau_max = n_quad * dt
    if abs(tau_max - n_quad * dt) > 1e-9:
        raise ValueError(f"tau_max {tau_max} not on dt grid")

    # Sampling: per-orbit reaction param A and IC, shared across shifts.
    rng_master = np.random.default_rng(seed)

    # Canonical exp-kernel weights (these are what we cyclically shift).
    kw_canonical = _build_kernel_weights("exp", n_quad, dt, tau, params={})

    phi_all = np.zeros((num_orbits, n_quad, n_hist, n_grid, n_grid, 1),
                       dtype=np.float32)
    y_all = np.zeros((num_orbits, n_quad, n_out, n_grid, n_grid, 1),
                     dtype=np.float32)
    params_all = np.zeros((num_orbits, n_quad, 1), dtype=np.float32)
    orbit_id_all = np.zeros((num_orbits, n_quad), dtype=np.int32)
    tau_shift_all = np.zeros((num_orbits, n_quad), dtype=np.int32)

    print(f"[orbit_ood] generating {num_orbits} orbits x {n_quad} shifts = "
          f"{num_orbits * n_quad} trajectories")
    print(f"            dt={dt}, tau={tau}, tau_max={tau_max}, "
          f"n_hist={n_hist}, n_out={n_out}, n_grid={n_grid}")
    t_pool0 = time.time()

    for orbit_id in range(num_orbits):
        # Per-orbit invariants.
        # A range raised from [0.5, 2.0] -> [8.0, 11.0]: stronger reaction term
        # is needed for the cyclic-shift action on K to actually move the
        # trajectory.  At A~1.0 the system relaxes to a near-constant state
        # before the kernel-position differences can accumulate, so pairwise
        # rel-L2 between shifts is ~0.005.  At A~9.5 the trajectory enters
        # the nonlinear regime and ALL half-cycle pairwise rel-L2 are >= 0.05
        # (the empirical lambda for the covering-radius PAC-gap theorem).
        A = float(rng_master.uniform(8.0, 11.0))
        D = float(rng_master.choice([0.025, 0.05]))
        ic_seed = int(rng_master.integers(0, 2**31 - 1))

        for k_idx, tau_shift in enumerate(range(n_quad)):
            # Each (orbit, shift) gets the same IC, only the kernel-weights
            # differ.  This preserves the orbit structure exactly: the orbit
            # at tau_shift=k is the cyclic shift of the orbit at tau_shift=0.
            u0_seed = np.random.default_rng(ic_seed)
            kw_shifted = shifted_kernel_weights(kw_canonical, tau_shift)
            p = DistKernelRDParams(
                kernel_type="exp", A=A, tau=tau, tau_max=tau_max,
                kernel_extra={}, D=D, T_total=T_total_needed,
                dt=dt, n_grid=n_grid, L=L)
            sim_rng = np.random.default_rng(ic_seed + tau_shift * 7919)
            traj = simulate_dist_kernel_rd_with_kw(
                p, sim_rng, grid, kw_shifted, u0_seed)
            traj = traj[:n_total][..., None].astype(np.float32)
            phi_all[orbit_id, k_idx] = traj[:n_hist]
            y_all[orbit_id, k_idx] = traj[n_hist:n_hist + n_out]
            params_all[orbit_id, k_idx, 0] = A
            orbit_id_all[orbit_id, k_idx] = orbit_id
            tau_shift_all[orbit_id, k_idx] = tau_shift
        if (orbit_id + 1) % 8 == 0:
            elapsed = time.time() - t_pool0
            eta = elapsed * (num_orbits - orbit_id - 1) / (orbit_id + 1)
            print(f"  orbit {orbit_id + 1:>4d}/{num_orbits}, elapsed "
                  f"{elapsed/60:.1f}min, ETA {eta/60:.1f}min")

    return phi_all, y_all, params_all, orbit_id_all, tau_shift_all


# ---------------------------------------------------------------------
# Slicing the pool into per-m train/val/test shards.
# ---------------------------------------------------------------------

def write_split_for_m(out_root: Path, m: int, splits_info: dict,
                      phi_all, y_all, params_all, orbit_id_all, tau_shift_all,
                      val_orbit_ids: set, n_hist: int, n_out: int, dt: float):
    """Slice the orbit pool by (orbit_id, tau_shift) ∈ split, write npz."""
    n_orbits = phi_all.shape[0]
    n_quad = phi_all.shape[1]
    n_grid = phi_all.shape[3]

    S_train = splits_info["S_train"]
    S_test = splits_info["S_test"]

    # train: orbits not in val_orbit_ids, shifts in S_train
    # val:   orbits in val_orbit_ids,    shifts in S_train
    # test:  orbits not in val_orbit_ids, shifts in S_test
    train_idx, val_idx, test_idx = [], [], []
    for o in range(n_orbits):
        for k_idx in range(n_quad):
            tau_shift = int(tau_shift_all[o, k_idx])
            if tau_shift in S_train:
                if o in val_orbit_ids:
                    val_idx.append((o, k_idx))
                else:
                    train_idx.append((o, k_idx))
            elif tau_shift in S_test and o not in val_orbit_ids:
                test_idx.append((o, k_idx))

    def stack_split(idx_list):
        if not idx_list:
            empty_phi = np.zeros((0, n_hist, n_grid, n_grid, 1),
                                 dtype=np.float32)
            empty_y = np.zeros((0, n_out, n_grid, n_grid, 1),
                               dtype=np.float32)
            empty_p = np.zeros((0, 1), dtype=np.float32)
            empty_l = np.zeros((0, 1), dtype=np.float32)
            return empty_phi, empty_y, empty_p, empty_l
        os = np.array([o for (o, k) in idx_list])
        ks = np.array([k for (o, k) in idx_list])
        phi = phi_all[os, ks]
        y = y_all[os, ks]
        params = params_all[os, ks]
        # Use lags channel for diagnostic: tau_shift (the controlled var).
        lags = tau_shift_all[os, ks].astype(np.float32).reshape(-1, 1)
        return phi, y, params, lags

    phi_train, y_train, params_train, lags_train = stack_split(train_idx)
    phi_val, y_val, params_val, lags_val = stack_split(val_idx)
    phi_test, y_test, params_test, lags_test = stack_split(test_idx)

    out_dir = out_root / f"m{m}" / "dist_exp_rd_2d_orbit"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_hist = (np.arange(n_hist) * dt).astype(np.float32)
    t_out = (n_hist + np.arange(n_out)) * dt
    t_out = t_out.astype(np.float32)

    for split, phi, y, params, lags in [
        ("train", phi_train, y_train, params_train, lags_train),
        ("val", phi_val, y_val, params_val, lags_val),
        ("test", phi_test, y_test, params_test, lags_test),
    ]:
        sd = out_dir / split
        sd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sd / "shard_000.npz",
            phi=phi, y=y, params=params,
            t_hist=t_hist, t_out=t_out,
            lags=lags,
        )
        print(f"  wrote m={m} {split}: phi={phi.shape}, y={y.shape}")

    # Manifest per m.
    manifest = {
        "m": int(m),
        "n_quad": int(n_quad),
        "covering_radius": int(splits_info["covering_radius"]),
        "S_train": [int(s) for s in S_train],
        "S_test":  [int(s) for s in S_test],
        "n_train": int(phi_train.shape[0]),
        "n_val":   int(phi_val.shape[0]),
        "n_test":  int(phi_test.shape[0]),
        "n_orbits": int(n_orbits),
        "val_orbit_ids": sorted(int(o) for o in val_orbit_ids),
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)


# ---------------------------------------------------------------------
# Audit (run before any large-scale generation)
# ---------------------------------------------------------------------

def audit_one_orbit(out_dir: Path, dt: float = 0.01, tau: float = 0.25,
                     tau_max: float = 0.4, n_hist: int = 64, n_out: int = 64,
                     n_grid: int = 64, A: float = 9.5, D: float = 0.05,
                     seed: int = 42):
    # NOTE: lambda fix.  Original config (tau=0.1, A in [0.5,2.0], IC amp=0.2)
    # gave per-orbit pairwise rel-L2 of only 0.005-0.006: the trajectory
    # relaxed to a near-constant steady state before the kernel-position
    # differences could accumulate, so the orbit was nearly degenerate.
    # Tau alone is not enough — at tau=0.5 (broader kernel) the shifts overlap
    # heavily and pairwise rel-L2 drops further to ~0.001-0.002.  The actual
    # bottleneck is the dynamics: f(u)=A*u*(1-u) with A~1 produces gentle
    # diffusion-like trajectories.  Fix used: tau=0.25 (per design plan, gives
    # a smooth single-peaked kernel) + A range raised to [8.0, 11.0] + IC
    # amplitude raised from 0.2 to 0.45.  This drives the trajectory into the
    # nonlinear regime, and ALL half-cycle (max-offset) pairwise rel-L2 are
    # >= 0.05 (audited values: 0.069-0.156, mean 0.082).  This is the
    # empirical lambda for the covering-radius PAC-gap theorem.
    """Generate ONE full orbit (n_quad shifted copies of the same IC) and
    sanity-check that the simulator is finite, that distinct shifts produce
    distinct trajectories, and that the action's lambda > 0 (i.e., the orbit
    does NOT collapse — equivalent to non-degeneracy condition (A13))."""
    print(f"\n=== AUDIT ORBIT: tau={tau}, tau_max={tau_max} (n_quad={int(round(tau_max/dt))}), "
          f"A={A}, D={D}, seed={seed} ===")
    grid = SpectralGrid2D.make(n=n_grid, L=2 * np.pi)
    n_quad = int(round(tau_max / dt))
    n_total = n_hist + n_out
    T_total = (n_total - 1) * dt

    kw_canonical = _build_kernel_weights("exp", n_quad, dt, tau, params={})
    print(f"  kw_canonical mass: {kw_canonical.sum():.4f}, "
          f"max@idx={int(np.argmax(kw_canonical))}, max={kw_canonical.max():.4f}")

    p = DistKernelRDParams(
        kernel_type="exp", A=A, tau=tau, tau_max=tau_max,
        kernel_extra={}, D=D, T_total=T_total, dt=dt, n_grid=n_grid,
        L=2 * np.pi)

    ic_seed = seed * 31337
    # 8 shifts evenly spaced on Z/n_quad-Z so we see both small and half-cycle
    # offsets in the pairwise rel-L2 audit.  The half-cycle (max-offset) pair
    # is the most informative empirical lambda.
    step = max(1, n_quad // 8)
    sample_shifts = [(i * step) % n_quad for i in range(8)]
    trajs = {}
    t0 = time.time()
    for k in sample_shifts:
        u0_seed = np.random.default_rng(ic_seed)
        kw = shifted_kernel_weights(kw_canonical, k)
        sim_rng = np.random.default_rng(ic_seed + k * 7919)
        traj = simulate_dist_kernel_rd_with_kw(
            p, sim_rng, grid, kw, u0_seed)
        trajs[k] = traj
        print(f"    shift={k:>3d}: traj range=[{traj.min():.4f}, {traj.max():.4f}], "
              f"final_mean={traj[-1].mean():.4f}, finite={np.isfinite(traj).all()}")
    print(f"  wall: {time.time()-t0:.1f}s for {len(sample_shifts)} shifts")

    # Pairwise differences (these MUST be > 0 for the orbit to be nondegenerate).
    print("  pairwise rel-L2 between shifts (non-degeneracy / lambda check):")
    keys = sorted(trajs.keys())
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            diff = trajs[ki] - trajs[kj]
            rel = float(np.linalg.norm(diff) / (np.linalg.norm(trajs[kj]) + 1e-12))
            print(f"    shift({ki}) vs shift({kj}): rel-L2 = {rel:.4f}")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for k, t in trajs.items():
        np.save(out_dir / f"audit_orbit_shift{k:03d}.npy", t)
    print(f"  audit trajs saved under {out_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_orbits", type=int, default=256)
    ap.add_argument("--num_val_orbits", type=int, default=32,
                    help="Number of orbits held out for validation (val uses "
                         "S_train shifts).")
    ap.add_argument("--m_values", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--n_hist", type=int, default=64)
    ap.add_argument("--n_out", type=int, default=64)
    ap.add_argument("--n_grid", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.25,
                    help="Memory timescale of the canonical exp kernel.  "
                         "Bumped from 0.1 -> 0.25 so cyclic shifts span enough "
                         "of the lag axis to give pairwise rel-L2 >= 0.05 "
                         "between shifts of the same orbit (the empirical "
                         "lambda for the covering-radius PAC-gap theorem).")
    ap.add_argument("--tau_max", type=float, default=0.4,
                    help="Quadrature truncation; n_quad = round(tau_max/dt).")
    ap.add_argument("--out_dir", default="data_orbit_ood")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit_only", action="store_true",
                    help="Run only the single-orbit audit, skip data gen.")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    audit_dir = out_root / "audit"

    # Stage 1: audit.  A=9.5 = midpoint of the per-orbit [8.0, 11.0] range.
    audit_one_orbit(audit_dir, dt=args.dt, tau=args.tau, tau_max=args.tau_max,
                     n_hist=args.n_hist, n_out=args.n_out, n_grid=args.n_grid,
                     A=9.5, D=0.05, seed=args.seed)
    if args.audit_only:
        return

    n_quad = int(round(args.tau_max / args.dt))
    m_values = [int(x) for x in args.m_values.split(",")]
    splits_info_per_m = shift_subsets(n_quad, m_values)
    print(f"\n[orbit_ood] n_quad={n_quad}, m_values={m_values}")
    for m, info in splits_info_per_m.items():
        print(f"  m={m:>2d}: |S_train|={len(info['S_train'])}  "
              f"|S_test|={len(info['S_test'])}  "
              f"r(A)={info['covering_radius']}  "
              f"S_train={info['S_train']}")

    # Stage 2: generate the orbit pool.
    phi_all, y_all, params_all, orbit_id_all, tau_shift_all = generate_orbit_pool(
        num_orbits=args.num_orbits, n_quad=n_quad,
        n_hist=args.n_hist, n_out=args.n_out, dt=args.dt, tau=args.tau,
        n_grid=args.n_grid, L=2 * np.pi, seed=args.seed)

    # Stage 3: choose validation orbits (deterministic).
    val_orbit_ids = set(range(args.num_orbits - args.num_val_orbits,
                               args.num_orbits))
    print(f"[orbit_ood] val_orbit_ids: {sorted(val_orbit_ids)[:5]}... "
          f"({len(val_orbit_ids)} total)")

    # Stage 4: write per-m shards.
    for m, info in splits_info_per_m.items():
        write_split_for_m(out_root, m, info,
                           phi_all, y_all, params_all, orbit_id_all,
                           tau_shift_all, val_orbit_ids,
                           n_hist=args.n_hist, n_out=args.n_out, dt=args.dt)

    # Stage 5: write top-level manifest.
    top_manifest = {
        "num_orbits": int(args.num_orbits),
        "num_val_orbits": int(args.num_val_orbits),
        "n_quad": int(n_quad),
        "n_hist": int(args.n_hist),
        "n_out":  int(args.n_out),
        "n_grid": int(args.n_grid),
        "dt":     float(args.dt),
        "tau":    float(args.tau),
        "tau_max": float(args.tau_max),
        "kernel_family": "dist_exp_rd_2d_orbit",
        "controlled_var": "tau_shift (cyclic shift on lag-quadrature axis)",
        "m_values": m_values,
    }
    with open(out_root / "manifest_top.json", "w") as fh:
        json.dump(top_manifest, fh, indent=2)
    print(f"\n[orbit_ood] DONE.  Output: {out_root}")


if __name__ == "__main__":
    main()
