"""Fill-table-gaps sweep cell list (84 cells).

Targets the missing cells in Tables 4, 5, 6, 13 of the NeurIPS paper:
  - fno_film_nd: Power x clean, Gamma x all 3, Gauss x noisy (15 cells)
  - nide_nd: (Exp, Gauss) x clean (6 cells)
  - memno_nd: all 5 fams x clean (15 cells)
  - ffno_nd: all 5 fams x clean (15 cells)
  - noneq_film_nd: Exp x clean (3 cells)
  - causal_smooth_lemo_pc_nd: all 5 fams x {lowres, noisy} (30 cells)

Total = 84 cells. Output dir: outputs/fill_gaps_runpod/raw/.
"""
from __future__ import annotations

FAMS_ALL = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
            "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
SEEDS = (42, 123, 456)


def _base_args(model, fam, reg, seed,
               residual_anchor=True, epochs=100, batch_size=4,
               width=64, n_layers=3, lag_modes=24, spatial_modes=12):
    args = [
        "--family", fam,
        "--model", model,
        "--regime", reg,
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
        "--output_dir", "outputs/fill_gaps_runpod/raw",
    ]
    if residual_anchor:
        args.append("--residual_anchor")
    return args


def _make(sweep, model, fam, reg, seed, **kwargs):
    return {
        "sweep": sweep, "model": model, "fam": fam, "reg": reg, "seed": seed,
        "args": _base_args(model, fam, reg, seed, **kwargs),
    }


def cells_fno_film():
    """fno_film_nd missing: Power-clean, Gamma-all, Gauss-noisy."""
    cells = []
    targets = [
        ("dist_powerlaw_rd_2d", "clean"),
        ("dist_gamma_rd_2d", "clean"),
        ("dist_gamma_rd_2d", "lowres"),
        ("dist_gamma_rd_2d", "noisy"),
        ("dist_gaussian_rd_2d", "noisy"),
    ]
    for fam, reg in targets:
        for seed in SEEDS:
            cells.append(_make("fill_fno_film", "fno_film_nd", fam, reg, seed))
    return cells


def cells_nide():
    """nide_nd missing: Exp-clean, Gauss-clean."""
    cells = []
    for fam in ("dist_exp_rd_2d", "dist_gaussian_rd_2d"):
        for seed in SEEDS:
            # NIDE: residual_anchor=True per Table 4 conventions.
            cells.append(_make("fill_nide", "nide_nd", fam, "clean", seed))
    return cells


def cells_memno():
    """memno_nd missing: all 5 fams x clean."""
    cells = []
    for fam in FAMS_ALL:
        for seed in SEEDS:
            # MemNO: NO residual_anchor (per memno_ffno_sweep convention).
            cells.append(_make("fill_memno", "memno_nd", fam, "clean", seed,
                                residual_anchor=False))
    return cells


def cells_ffno():
    """ffno_nd missing: all 5 fams x clean."""
    cells = []
    for fam in FAMS_ALL:
        for seed in SEEDS:
            # F-FNO: NO residual_anchor (per memno_ffno_sweep convention).
            cells.append(_make("fill_ffno", "ffno_nd", fam, "clean", seed,
                                residual_anchor=False))
    return cells


def cells_noneq_film():
    """noneq_film_nd missing: Exp-clean."""
    cells = []
    for seed in SEEDS:
        cells.append(_make("fill_noneq_film", "noneq_film_nd",
                             "dist_exp_rd_2d", "clean", seed))
    return cells


def cells_causal_smooth():
    """causal_smooth_lemo_pc_nd missing: all 5 fams x lowres + noisy."""
    cells = []
    for fam in FAMS_ALL:
        for reg in ("lowres", "noisy"):
            for seed in SEEDS:
                cells.append(_make("fill_causal_smooth",
                                     "causal_smooth_lemo_pc_nd",
                                     fam, reg, seed))
    return cells


def all_cells():
    cells = []
    cells.extend(cells_fno_film())     # 15
    cells.extend(cells_nide())         # 6
    cells.extend(cells_memno())        # 15
    cells.extend(cells_ffno())         # 15
    cells.extend(cells_noneq_film())   # 3
    cells.extend(cells_causal_smooth()) # 30
    return cells  # 84


if __name__ == "__main__":
    cs = all_cells()
    print(f"fill_gaps: {len(cs)} cells")
    by_sweep = {}
    for c in cs:
        by_sweep.setdefault(c["sweep"], 0)
        by_sweep[c["sweep"]] += 1
    for k, v in by_sweep.items():
        print(f"  {k}: {v}")
