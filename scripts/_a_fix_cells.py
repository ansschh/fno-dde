"""A-fix LEMO-PC retrain cell list: 5 fams x 3 regimes x 3 seeds = 45 cells.

LEMO-PC architecture now has zero_beta_above_dc=True baked in (commit
that adds the parameter to FiLMLagSpectralND -> LEMOPCNDBlock -> LEMOPCND
factory). These retrain cells produce checkpoints that natively satisfy
T1 cyclic-shift equivariance at FP32 floor (~3e-7) without post-hoc
projection. Output dir is `outputs/a_fix_runpod/raw/...`.
"""
from __future__ import annotations

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ("clean", "lowres", "noisy")
SEEDS = (42, 123, 456)


def _base_args(fam, reg, seed):
    return [
        "--family", fam,
        "--model", "lemo_pc_nd",
        "--regime", reg,
        "--noise_std", "0.05",
        "--downsample_factor", "2",
        "--epochs", "100",
        "--batch_size", "4",
        "--width", "64",
        "--n_layers", "3",
        "--lag_modes", "24",
        "--spatial_modes", "12",
        "--seed", str(seed),
        "--data_dir", "data_dde_pde",
        "--output_dir", "outputs/a_fix_runpod/raw",
        "--residual_anchor",
    ]


def all_cells():
    cells = []
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                cells.append({
                    "sweep": "a_fix",
                    "fam": fam, "reg": reg, "seed": seed,
                    "model": "lemo_pc_nd",
                    "args": _base_args(fam, reg, seed),
                })
    return cells


if __name__ == "__main__":
    cs = all_cells()
    print(f"a_fix: {len(cs)} cells")
    for i, c in enumerate(cs[:3]):
        print(f"  {i}: {c['fam']} {c['reg']} s{c['seed']}")
