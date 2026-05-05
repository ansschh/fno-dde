"""W11 — compute-matched FNO sweep cells.

Trains vanilla `fno_nd` (the model in T04 compute table) at 400 epochs to
match LEMO-PC@200ep wallclock (~5951s vs FNO@200ep ~2835s ≈ 2.1× ratio).

5 families × 3 seeds × clean regime × FNO@400ep = 15 cells.

The hyperparameters mirror the original T01 FNO sweep: width=64, n_layers=3,
spatial_modes=12 (no lag_modes, since FNO is delay-blind).

Output dir: outputs/w11_compute_matched_runpod/raw/<fam>/clean/fno_nd/s<seed>/
"""
from __future__ import annotations
import json
import sys

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
SEEDS = (42, 123, 456)


def _args(fam, seed,
          epochs=400, batch_size=4, width=64, n_layers=3, spatial_modes=12,
          data_dir="data_dde_pde"):
    return [
        "--family", fam,
        "--model", "fno_nd",
        "--regime", "clean",
        "--noise_std", "0.05",
        "--downsample_factor", "2",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--width", str(width),
        "--n_layers", str(n_layers),
        "--spatial_modes", str(spatial_modes),
        "--seed", str(seed),
        "--data_dir", data_dir,
        "--output_dir", "outputs/w11_compute_matched_runpod/raw",
    ]


def all_cells():
    """5 fams × 3 seeds = 15 cells of FNO@400ep."""
    cells = []
    for fam in FAMS:
        for seed in SEEDS:
            cells.append({
                "sweep": "w11_compute_matched",
                "fam": fam, "reg": "clean", "seed": seed, "model": "fno_nd",
                "args": _args(fam, seed, epochs=400),
            })
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            ep = c["args"][c["args"].index("--epochs") + 1]
            print(f"{i:3d}  fno_nd  {c['fam']:25s}  s{c['seed']:4d}  ep={ep}")
        print(f"---\nTotal: {len(cells)} cells")
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells (FNO@400ep × 5 fams × 3 seeds, clean only)")


if __name__ == "__main__":
    main()
