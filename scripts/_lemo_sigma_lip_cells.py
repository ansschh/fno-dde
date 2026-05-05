"""LEMO_σ-Lip pilot training cells.

3 cells: dist_exp_rd_2d × clean × σ=0.7 × 3 seeds.

Trains the new LEMO_σ-Lip variant with σ-projection on lag conv AND spatial
FNO + spectral-norm on B/heads + ReLU. Backwards compatible interface; uses
existing train_apebench_smoke.py.

Output: outputs/lemo_sigma_lip_pilot_runpod/raw/dist_exp_rd_2d/clean/lemo_sigma_lip_nd/s<seed>/

Usage:
  python scripts/_lemo_sigma_lip_cells.py --print
  python scripts/_lemo_sigma_lip_cells.py --json | python scripts/_lemo_sigma_lip_runner.py
"""
from __future__ import annotations
import json
import sys

FAMS = ["dist_exp_rd_2d"]
SEEDS = (42, 123, 456)
SIGMA = 0.7   # within the certified band at n=128, D=3 (threshold ≈ 0.545)


def _args(fam, seed,
          epochs=100, batch_size=4, width=64, n_layers=3, lag_modes=24, spatial_modes=12):
    return [
        "--family", fam,
        "--model", "lemo_sigma_lip_nd",
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
        "--output_dir", "outputs/lemo_sigma_lip_pilot_runpod/raw",
        "--sigma", str(SIGMA),
        "--residual_anchor",
    ]


def all_cells():
    cells = []
    for fam in FAMS:
        for seed in SEEDS:
            cells.append({
                "sweep": "lemo_sigma_lip_pilot",
                "fam": fam, "reg": "clean", "seed": seed, "model": "lemo_sigma_lip_nd",
                "args": _args(fam, seed, epochs=100),
            })
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            print(f"{i:3d}  lemo_sigma_lip_nd  {c['fam']:25s}  s{c['seed']:4d}  σ={SIGMA}")
        print(f"---\nTotal: {len(cells)} cells")
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells (LEMO_σ-Lip × dist_exp_rd_2d × σ=0.7 × 3 seeds)")


if __name__ == "__main__":
    main()
