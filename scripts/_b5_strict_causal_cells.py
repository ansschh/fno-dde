"""B5 strict-causal Toeplitz pilot training cells.

Per CHANGE_1F: the existing B5 sweep used CausalSmoother on top of cyclic
LEMO-PC (smoother stack). To validate the cyclic-vs-causal story cleanly,
also train the strictly-causal Toeplitz variant via FiLMLagSpectralND(causal=True).

15 cells: 5 fams × 3 seeds × clean × LEMO-PC with causal=True.

Output: outputs/b5_strict_causal_runpod/raw/<fam>/clean/lemo_pc_nd/s<seed>/
"""
from __future__ import annotations
import json
import sys

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
SEEDS = (42, 123, 456)


def _args(fam, seed,
          epochs=100, batch_size=4, width=64, n_layers=3, lag_modes=24, spatial_modes=12):
    return [
        "--family", fam,
        "--model", "lemo_pc_nd",
        "--regime", "clean",
        "--noise_std", "0.05",
        "--downsample_factor", "2",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--width", str(width),
        "--n_layers", str(n_layers),
        "--lag_modes", str(lag_modes),
        "--spatial_modes", str(spatial_modes),
        "--seed", str(seed),
        "--data_dir", "data_dde_pde",
        "--output_dir", "outputs/b5_strict_causal_runpod/raw",
        "--causal",   # strict causal Toeplitz via FiLMLagSpectralND
        "--residual_anchor",
    ]


def all_cells():
    cells = []
    for fam in FAMS:
        for seed in SEEDS:
            cells.append({
                "sweep": "b5_strict_causal",
                "fam": fam, "reg": "clean", "seed": seed, "model": "lemo_pc_nd",
                "args": _args(fam, seed, epochs=100),
            })
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            print(f"{i:3d}  lemo_pc_nd[causal=True]  {c['fam']:25s}  s{c['seed']:4d}")
        print(f"---\nTotal: {len(cells)} cells")
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells (LEMO-PC strict-causal × 5 fams × 3 seeds, clean)")


if __name__ == "__main__":
    main()
