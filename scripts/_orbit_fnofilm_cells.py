"""Orbit OOD with FNO+FiLM only — focused B3 baseline test.

The most dangerous baseline (B1 reviewer panel cited it as "potentially
already cyclically lag-equivariant by construction"). If LEMO-PC dominates
FNO+FiLM on orbit OOD, the equivariance case is strong because FNO+FiLM
has FiLM conditioning + FNO architecture but lacks explicit lag-equivariance.

18 cells: 6 m-values × 3 seeds × 1 model.

m32 at 50 epochs (matches Pod 2 setup); m1-m16 at 100 epochs.
"""
from __future__ import annotations
import json
import sys

M_VALUES = (1, 2, 4, 8, 16, 32)
SEEDS = (42, 123, 456)


def _orbit_args(m, seed,
                 epochs=100, batch_size=4, width=64, n_layers=3,
                 lag_modes=24, spatial_modes=12):
    return [
        "--family", "dist_exp_rd_2d_orbit",
        "--model", "fno_film_nd",
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
        "--output_dir", f"outputs/orbit_fnofilm_h100/m{m}/raw",
        "--residual_anchor",
    ]


def all_cells():
    """18 cells, m32 first (longest, critical path)."""
    cells = []
    for m in (32, 16, 8, 4, 2, 1):
        for seed in SEEDS:
            cells.append({
                "sweep": "orbit_fno_film",
                "fam": "dist_exp_rd_2d_orbit", "reg": "clean",
                "seed": seed, "model": "fno_film_nd", "m": m,
                "args": _orbit_args(m, seed, epochs=50 if m == 32 else 100),
            })
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            print(f"{i:3d}  fno_film_nd  m={c['m']:3d}  s{c['seed']:4d}  ep={c['args'][c['args'].index('--epochs')+1]}")
        print(f"---\nTotal: {len(cells)} cells")
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells (FNO+FiLM × 6 m × 3 seeds)")


if __name__ == "__main__":
    main()
