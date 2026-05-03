"""Orbit OOD cells (B3) for parallel execution on H100 pod.

Includes:
  - LEMO-PC at m in {8, 16, 32} x 3 seeds = 9 cells (m1/m2/m4 already secured
    on orbit pod, restored locally).
  - per_lag_mlp_nd at m in {1, 2, 4, 8, 16, 32} x 3 seeds = 18 cells (the
    non-equivariant baseline arm — needed for B3 1/m trend).

Total: 27 cells. Run with the parallel dispatcher in `_launch_orbit_h100.sh`
which assigns cells round-robin across 8 H100 GPUs. m32 cells form the
critical-path floor (~4.8h each on H100); other cells fit in parallel.
"""
from __future__ import annotations
import json
import sys

LEMO_M_VALUES = (8, 16, 32)            # m1, m2, m4 already done elsewhere
PER_LAG_M_VALUES = (1, 2, 4, 8, 16, 32)  # baseline arm needs all m
SEEDS = (42, 123, 456)


def _orbit_args(model, m, seed,
                epochs=100, batch_size=4, width=64, n_layers=3,
                lag_modes=24, spatial_modes=12):
    """Orbit OOD cell args. epochs=100 to match offload sweep cadence
    (was 200 in original launch_orbit_ood.sh; convergence checked OK at 100)."""
    return [
        "--family", "dist_exp_rd_2d_orbit",
        "--model", model,
        "--regime", "clean",
        "--noise_std", "0.0",
        "--downsample_factor", "1",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--width", str(width),
        "--n_layers", str(n_layers),
        "--lag_modes", str(lag_modes),
        "--spatial_modes", str(spatial_modes),
        "--seed", str(seed),
        "--data_dir", f"data_orbit_ood/m{m}",
        "--output_dir", f"outputs/orbit_ood_h100/lemo_pc_nd_m{m}/raw"
            if model == "lemo_pc_nd" else
            f"outputs/orbit_ood_h100/per_lag_mlp_nd_m{m}/raw",
        "--residual_anchor",
    ]


def cells_lemo():
    """LEMO-PC cells at m=8,16,32 × 3 seeds = 9 cells."""
    cells = []
    for m in LEMO_M_VALUES:
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_lemo_pc",
                "fam": "dist_exp_rd_2d_orbit", "reg": "clean", "seed": seed,
                "model": "lemo_pc_nd", "m": m,
                "args": _orbit_args("lemo_pc_nd", m, seed),
            })
    return cells


def cells_per_lag_mlp():
    """per_lag_mlp_nd cells at m=1,2,4,8,16,32 × 3 seeds = 18 cells."""
    cells = []
    for m in PER_LAG_M_VALUES:
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_per_lag_mlp",
                "fam": "dist_exp_rd_2d_orbit", "reg": "clean", "seed": seed,
                "model": "per_lag_mlp_nd", "m": m,
                "args": _orbit_args("per_lag_mlp_nd", m, seed),
            })
    return cells


def all_cells():
    """All 27 orbit cells. ORDERED: longest cells first so they get scheduled
    on GPUs early. m32 cells (slowest, 4.8h each) come first to start them
    on GPUs 0-5, leaving GPUs 6-7 free for shorter cells."""
    cells = []
    # m32 first (longest)
    for m in (32,):
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_lemo_pc", "fam": "dist_exp_rd_2d_orbit",
                "reg": "clean", "seed": seed, "model": "lemo_pc_nd", "m": m,
                "args": _orbit_args("lemo_pc_nd", m, seed),
            })
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_per_lag_mlp", "fam": "dist_exp_rd_2d_orbit",
                "reg": "clean", "seed": seed, "model": "per_lag_mlp_nd", "m": m,
                "args": _orbit_args("per_lag_mlp_nd", m, seed),
            })
    # m16 next
    for m in (16,):
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_lemo_pc", "fam": "dist_exp_rd_2d_orbit",
                "reg": "clean", "seed": seed, "model": "lemo_pc_nd", "m": m,
                "args": _orbit_args("lemo_pc_nd", m, seed),
            })
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_per_lag_mlp", "fam": "dist_exp_rd_2d_orbit",
                "reg": "clean", "seed": seed, "model": "per_lag_mlp_nd", "m": m,
                "args": _orbit_args("per_lag_mlp_nd", m, seed),
            })
    # m8 next
    for m in (8,):
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_lemo_pc", "fam": "dist_exp_rd_2d_orbit",
                "reg": "clean", "seed": seed, "model": "lemo_pc_nd", "m": m,
                "args": _orbit_args("lemo_pc_nd", m, seed),
            })
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_per_lag_mlp", "fam": "dist_exp_rd_2d_orbit",
                "reg": "clean", "seed": seed, "model": "per_lag_mlp_nd", "m": m,
                "args": _orbit_args("per_lag_mlp_nd", m, seed),
            })
    # m4, m2, m1 (per_lag_mlp only, fast)
    for m in (4, 2, 1):
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_per_lag_mlp", "fam": "dist_exp_rd_2d_orbit",
                "reg": "clean", "seed": seed, "model": "per_lag_mlp_nd", "m": m,
                "args": _orbit_args("per_lag_mlp_nd", m, seed),
            })
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            print(f"{i:3d}  {c['sweep']:18s}  {c['model']:18s}  m={c['m']:3d}  s{c['seed']}")
        print(f"---\nTotal: {len(cells)} cells")
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells")


if __name__ == "__main__":
    main()
