"""
APEBench data generation script.

Generates standard APEBench scenarios in the format used by our pipeline:
  shards of NPZ files with keys:
    phi   : (n, n_hist, *spatial, c)         — history window
    y     : (n, n_out, *spatial, c)          — future window
    params: (n, params_dim)                  — per-trajectory params
    t_hist: (n_hist,)                        — history time axis
    t_out : (n_out,)                         — future time axis
    lags  : (n, ...)                         — placeholder for delay info

For each trajectory of total length T+1, we use the first n_hist as input
history and the next n_out as targets.  Train/val/test splits are by
trajectory (no leakage of time within a trajectory).

Currently supports KolmogorovFlow 2D, Burgers 3D, GrayScott 3D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import apebench
import jax


def gen_scenario(scenario_cls, *, num_spatial_dims, num_points, num_train,
                  num_test, T, num_channels=1, train_seed=0, test_seed=773,
                  num_warmup=0):
    """Construct an APEBench scenario and generate train+test trajectories."""
    scenario = scenario_cls(
        num_spatial_dims=num_spatial_dims,
        num_points=num_points,
        num_channels=num_channels,
        num_train_samples=num_train,
        train_temporal_horizon=T,
        num_test_samples=num_test,
        test_temporal_horizon=T,
        train_seed=train_seed,
        test_seed=test_seed,
        num_warmup_steps=num_warmup,
    )
    train_data = np.asarray(scenario.get_train_data())          # (N, T+1, C, *spatial)
    test_data = np.asarray(scenario.get_test_data())            # (N, T+1, C, *spatial)
    return train_data, test_data


def to_history_format(traj_block: np.ndarray, n_hist: int, n_out: int):
    """Cut a (N, T+1, C, *spatial) trajectory block into (history, future).

    Returns dict with keys phi (history) and y (future), shape
    `(N, n_*, *spatial, C)` with the channel axis moved to the last
    dim to match our pipeline's `(B, lag, *spatial, C)` convention.
    """
    # Channel-last permutation: (N, T+1, C, *spatial) -> (N, T+1, *spatial, C)
    n_dims = traj_block.ndim
    perm = [0, 1] + list(range(3, n_dims)) + [2]
    traj = np.transpose(traj_block, perm)
    phi = traj[:, :n_hist]                                       # (N, n_hist, *spatial, C)
    y   = traj[:, n_hist:n_hist + n_out]                          # (N, n_out, *spatial, C)
    return phi, y


def write_shard(out_dir: Path, split: str, phi: np.ndarray, y: np.ndarray,
                params: np.ndarray, t_hist: np.ndarray, t_out: np.ndarray):
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "shard_000.npz",
        phi=phi.astype(np.float32),
        y=y.astype(np.float32),
        params=params.astype(np.float32),
        t_hist=t_hist.astype(np.float32),
        t_out=t_out.astype(np.float32),
        lags=np.zeros((phi.shape[0], 1), dtype=np.float32),
    )
    print(f"  wrote {out_dir}/shard_000.npz  phi={phi.shape}  y={y.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=str, required=True,
                    choices=["kolmogorov_2d", "decaying_turbulence_2d",
                              "burgers_3d", "gray_scott_3d",
                              "burgers_2d", "burgers_1d", "diffusion_2d"])
    ap.add_argument("--num_train", type=int, default=200)
    ap.add_argument("--num_val", type=int, default=40)
    ap.add_argument("--num_test", type=int, default=40)
    ap.add_argument("--num_points", type=int, default=64,
                    help="Grid resolution per spatial axis.")
    ap.add_argument("--T", type=int, default=64,
                    help="Trajectory length in timesteps.")
    ap.add_argument("--n_hist", type=int, default=32)
    ap.add_argument("--n_out", type=int, default=32)
    ap.add_argument("--out_dir", default="data_apebench")
    args = ap.parse_args()

    print(f"=== Generating {args.family} ===")
    print(f"  num_train={args.num_train}, num_test={args.num_test}, num_val={args.num_val}")
    print(f"  num_points={args.num_points}, T={args.T}, n_hist={args.n_hist}, n_out={args.n_out}")
    print(f"  jax backend: {jax.default_backend()}")

    if args.family == "kolmogorov_2d":
        scenario_cls = apebench.scenarios.physical.KolmogorovFlow
        num_spatial_dims, num_channels = 2, 1
    elif args.family == "decaying_turbulence_2d":
        scenario_cls = apebench.scenarios.physical.DecayingTurbulence
        num_spatial_dims, num_channels = 2, 1
    elif args.family == "burgers_2d":
        scenario_cls = apebench.scenarios.physical.Burgers
        num_spatial_dims, num_channels = 2, 2
    elif args.family == "burgers_3d":
        scenario_cls = apebench.scenarios.physical.Burgers
        num_spatial_dims, num_channels = 3, 3
    elif args.family == "gray_scott_3d":
        scenario_cls = apebench.scenarios.physical.GrayScott
        num_spatial_dims, num_channels = 3, 2
    elif args.family == "burgers_1d":
        scenario_cls = apebench.scenarios.physical.BurgersSingleChannel
        num_spatial_dims, num_channels = 1, 1
    elif args.family == "diffusion_2d":
        scenario_cls = apebench.scenarios.physical.Diffusion
        num_spatial_dims, num_channels = 2, 1
    else:
        raise ValueError(args.family)

    # Generate train + test together so APEBench's seed conventions apply.
    train_traj, test_traj = gen_scenario(
        scenario_cls,
        num_spatial_dims=num_spatial_dims,
        num_points=args.num_points,
        num_train=args.num_train + args.num_val,           # we'll split off val later
        num_test=args.num_test,
        num_channels=num_channels,
        T=args.T,
    )
    print(f"  train_traj shape: {train_traj.shape}")
    print(f"  test_traj  shape: {test_traj.shape}")

    # Split off val from train.
    val_traj = train_traj[:args.num_val]
    train_traj = train_traj[args.num_val:]

    # Cut into history/future windows.
    train_phi, train_y = to_history_format(train_traj, args.n_hist, args.n_out)
    val_phi, val_y     = to_history_format(val_traj,   args.n_hist, args.n_out)
    test_phi, test_y   = to_history_format(test_traj,  args.n_hist, args.n_out)

    # APEBench scenarios run with simulator-internal "params" that we don't
    # easily extract; for now we use a placeholder zero-vector per sample.
    # Future: extract IC/forcing parameters as conditioning vector.
    params_dim = 1
    train_params = np.zeros((train_phi.shape[0], params_dim), dtype=np.float32)
    val_params   = np.zeros((val_phi.shape[0],   params_dim), dtype=np.float32)
    test_params  = np.zeros((test_phi.shape[0],  params_dim), dtype=np.float32)

    # Time axes (unitless step index).
    t_hist = np.arange(args.n_hist, dtype=np.float32)
    t_out  = np.arange(args.n_hist, args.n_hist + args.n_out, dtype=np.float32)

    out_root = Path(args.out_dir) / args.family
    out_root.mkdir(parents=True, exist_ok=True)
    write_shard(out_root, "train", train_phi, train_y, train_params, t_hist, t_out)
    write_shard(out_root, "val",   val_phi,   val_y,   val_params,   t_hist, t_out)
    write_shard(out_root, "test",  test_phi,  test_y,  test_params,  t_hist, t_out)

    manifest = {
        "family":       args.family,
        "spatial_dims": num_spatial_dims,
        "num_channels": num_channels,
        "spatial_shape": list(train_phi.shape[2:-1]),
        "n_hist":       args.n_hist,
        "n_out":        args.n_out,
        "n_samples":    {"train": train_phi.shape[0],
                          "val":   val_phi.shape[0],
                          "test":  test_phi.shape[0]},
        "params_dim":   params_dim,
        "source":       "apebench.scenarios.physical",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {out_root}/manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
