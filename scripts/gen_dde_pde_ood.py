"""Generate OOD test sets — same simulators, but tau values OUTSIDE training range.

Training tau ranges:
    mackey_glass:    {1.5, 2.0, 2.5}      OOD: {3.0, 3.5}
    wright:          {1.0, 1.5, 2.0}      OOD: {0.5, 2.5}
    hutchinson:      {1.0, 1.5, 2.0}      OOD: {0.5, 2.5}
    dist_delay_rd:   {0.25, 0.5, 1.0}     OOD: {0.125, 1.5}
    delay_burgers:   {0.25, 0.5, 1.0}     OOD: {0.125, 1.5}
    kuramoto:        {0.25, 0.5, 1.0}     OOD: {0.125, 1.5}

Output: data_dde_pde_ood/{family}/test/shard_000.npz (test split only — for eval)
        plus matching train/val with same OOD distribution (so dataset
        normalization stats can be computed; train/val also OOD).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Reuse simulators + IC sampler from main gen file.
from gen_dde_pde_data import (
    SpectralGrid2D, smooth_random_field_2d,
    MackeyGlassParams, simulate_mackey_glass,
    WrightParams, simulate_wright,
    HutchinsonParams, simulate_hutchinson,
    DistDelayRDParams, simulate_dist_delay_rd,
    DelayBurgersParams, simulate_delay_burgers,
    KuramotoParams, simulate_kuramoto,
    DistKernelRDParams, simulate_dist_kernel_rd,
    sample_dist_kernel_rd_param_set,
    write_shard,
)


OOD_TAUS = {
    "mackey_glass_2d":  [3.0, 3.5],
    "wright_2d":        [0.5, 2.5],
    "hutchinson_2d":    [0.5, 2.5],
    "dist_delay_rd_2d": [0.1, 1.5],   # 0.1 = 10*dt (under train min 0.25), 1.5 above
    "delay_burgers_2d": [0.1, 1.5],
    "kuramoto_2d":      [0.1, 1.5],
    "dist_exp_rd_2d":      [0.1, 1.5],
    "dist_gaussian_rd_2d": [0.1, 1.5],
    "dist_gamma_rd_2d":    [0.1, 1.5],
    "dist_uniform_rd_2d":  [0.1, 1.5],
    "dist_powerlaw_rd_2d": [0.1, 1.5],
}


def make_ood_params(family: str, rng: np.random.Generator,
                     T_total: float, dt: float, n_grid: int, L: float):
    tau = float(rng.choice(OOD_TAUS[family]))
    if family == "mackey_glass_2d":
        return MackeyGlassParams(
            beta=float(rng.uniform(1.5, 2.5)), gamma=1.0, n=10.0,
            tau=tau, D=float(rng.choice([0.025, 0.05, 0.1])),
            T_total=T_total, dt=dt, n_grid=n_grid, L=L)
    if family == "wright_2d":
        return WrightParams(
            alpha=float(rng.uniform(0.5, 1.75)), tau=tau,
            D=float(rng.choice([0.025, 0.05])),
            T_total=T_total, dt=dt, n_grid=n_grid, L=L)
    if family == "hutchinson_2d":
        return HutchinsonParams(
            r=float(rng.uniform(0.5, 1.5)), K=1.0, tau=tau,
            D=float(rng.choice([0.025, 0.05])),
            T_total=T_total, dt=dt, n_grid=n_grid, L=L)
    if family == "dist_delay_rd_2d":
        return DistDelayRDParams(
            A=float(rng.uniform(0.5, 2.0)), tau=tau, tau_max=4.0 * tau,
            D=float(rng.choice([0.025, 0.05])),
            T_total=T_total, dt=dt, n_grid=n_grid, L=L)
    if family == "delay_burgers_2d":
        return DelayBurgersParams(
            nu=float(rng.choice([0.025, 0.05])),
            alpha=float(rng.uniform(0.5, 2.0)), tau=tau,
            u_target_amp=float(rng.uniform(-0.3, 0.3)),
            T_total=T_total, dt=dt, n_grid=n_grid, L=L)
    if family == "kuramoto_2d":
        return KuramotoParams(
            K=float(rng.uniform(0.5, 2.0)),
            sigma=float(rng.choice([0.3, 0.5, 0.8])), tau=tau,
            omega_std=float(rng.choice([0.05, 0.1, 0.2])),
            T_total=T_total, dt=dt, n_grid=n_grid, L=L)
    if family.startswith("dist_") and family.endswith("_rd_2d") and family != "dist_delay_rd_2d":
        kernel_type = family.replace("dist_", "").replace("_rd_2d", "")
        p_train = sample_dist_kernel_rd_param_set(rng, kernel_type)
        new_tau_max = max(4.0 * tau, 0.4)
        new_tau_max = round(new_tau_max / dt) * dt
        return DistKernelRDParams(
            kernel_type=kernel_type, A=p_train.A, tau=tau, tau_max=new_tau_max,
            kernel_extra=p_train.kernel_extra, D=p_train.D,
            T_total=T_total, dt=dt, n_grid=n_grid, L=L)
    raise ValueError(family)


def simulate(family: str, p, rng, grid):
    if family == "mackey_glass_2d":   return simulate_mackey_glass(p, rng, grid)
    if family == "wright_2d":         return simulate_wright(p, rng, grid)
    if family == "hutchinson_2d":     return simulate_hutchinson(p, rng, grid)
    if family == "dist_delay_rd_2d":  return simulate_dist_delay_rd(p, rng, grid)
    if family == "delay_burgers_2d":  return simulate_delay_burgers(p, rng, grid)
    if family == "kuramoto_2d":       return simulate_kuramoto(p, rng, grid)
    if family.startswith("dist_") and family.endswith("_rd_2d") and family != "dist_delay_rd_2d":
        return simulate_dist_kernel_rd(p, rng, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(OOD_TAUS.keys()))
    ap.add_argument("--num_train", type=int, default=128)
    ap.add_argument("--num_val", type=int, default=64)
    ap.add_argument("--num_test", type=int, default=200)
    ap.add_argument("--num_points", type=int, default=64)
    ap.add_argument("--T_total", type=float, default=1.28)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--n_hist", type=int, default=64)
    ap.add_argument("--n_out", type=int, default=64)
    ap.add_argument("--out_dir", default="data_dde_pde_ood")
    args = ap.parse_args()

    L = 2 * np.pi
    grid = SpectralGrid2D.make(n=args.num_points, L=L)
    out_root = Path(args.out_dir) / args.family
    out_root.mkdir(parents=True, exist_ok=True)

    n_total = args.n_hist + args.n_out

    for split, num, seed in [("train", args.num_train, 0),
                               ("val",   args.num_val, 100_000),
                               ("test",  args.num_test, 200_000)]:
        rng = np.random.default_rng(seed)
        phi_list, y_list, params_list = [], [], []
        print(f"\n--- {args.family} OOD {split} (n={num}) ---")
        for i in range(num):
            p = make_ood_params(args.family, rng, args.T_total, args.dt,
                                  args.num_points, L)
            traj = simulate(args.family, p, rng, grid)[:n_total]
            traj = traj[..., None].astype(np.float32)
            phi_list.append(traj[:args.n_hist])
            y_list.append(traj[args.n_hist:n_total])
            # Variable-length params per family — use a small marker.
            params_list.append([p.tau, 0.0, 0.0])
            if (i + 1) % 32 == 0:
                print(f"  [{i+1}/{num}]")
        phi = np.stack(phi_list)
        y = np.stack(y_list)
        params = np.array(params_list, dtype=np.float32)
        t_hist = np.arange(args.n_hist) * args.dt
        t_out = (args.n_hist + np.arange(args.n_out)) * args.dt
        write_shard(out_root, split, phi, y, params, t_hist, t_out)

    manifest = {
        "family": args.family,
        "spatial_dims": 2, "num_channels": 1,
        "spatial_shape": [args.num_points, args.num_points],
        "n_hist": args.n_hist, "n_out": args.n_out,
        "n_samples": {"train": args.num_train, "val": args.num_val,
                       "test": args.num_test},
        "params_dim": 3,
        "source": "gen_dde_pde_ood.py",
        "ood_tau_values": OOD_TAUS[args.family],
        "delay_relevance": "OOD lag-shift test (tau outside training range)",
        "dt": args.dt,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {out_root}/manifest.json")


if __name__ == "__main__":
    main()
