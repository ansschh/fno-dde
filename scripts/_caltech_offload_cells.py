"""Master cell list for the 6-pod RunPod offload of pending Caltech sweeps.

Order matters — pods take a contiguous [START, START+COUNT) slice of this
list, so cells must NEVER be reordered between pods. Each cell carries
ALL its training arguments (data-driven launcher invokes `train_apebench_smoke.py`
with the cell's `args` list directly — no hardcoded values).

Sweeps included:
  film_ablation       (135 cells: 3 models × 5 fams × 3 regs × 3 seeds, residual_anchor)
  sigma_sweep         (60  cells: 1 model × 5 fams × 4 sigmas × 3 seeds, residual_anchor + --sigma)
  memno_ffno          (60  cells: 2 models × 5 fams × 2 regimes × 3 seeds, NO residual_anchor)
  memory_aware        (117 cells: 3 models × 5 fams × 3 regs × 3 seeds, MINUS first 18
                                  = 135 - 18 cells; first 18 stay on Caltech)

Total: 372 cells. Skips orbit_ood (needs pre-generated data_orbit_ood/m{1,2,4,8,16,32}/
which lives only on Caltech; Caltech keeps that one).

Output dir convention: every sweep writes to `outputs/{layer}_runpod/raw/{fam}/{reg}/{model}/s{seed}/`.
The `/raw/` suffix is REQUIRED so the figure loaders (`viz_for`, `residuals_for`,
`_viz_for_model_any_layer`) discover the checkpoints when the data is rsynced
back to the local repo.
"""
from __future__ import annotations
import json
import sys

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ("clean", "lowres", "noisy")
SEEDS = (42, 123, 456)


# Common training args shared across all sweeps.  Per-sweep overrides
# (residual_anchor, sigma, output_dir, noise_std, downsample_factor) get
# composed on top of these.
def _base_args(model, fam, reg, seed, output_root,
               noise_std=0.05, downsample_factor=2,
               residual_anchor=False, sigma=None,
               data_dir="data_dde_pde",
               epochs=100, batch_size=4,
               width=64, n_layers=3, lag_modes=24, spatial_modes=12):
    args = [
        "--family", fam,
        "--model", model,
        "--regime", reg,
        "--noise_std", str(noise_std),
        "--downsample_factor", str(downsample_factor),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--width", str(width),
        "--n_layers", str(n_layers),
        "--lag_modes", str(lag_modes),
        "--spatial_modes", str(spatial_modes),
        "--seed", str(seed),
        "--data_dir", data_dir,
        "--output_dir", f"{output_root}/raw",
    ]
    if residual_anchor:
        args.append("--residual_anchor")
    if sigma is not None:
        args.extend(["--sigma", str(sigma)])
    return args


def cells_film_ablation():
    cells = []
    for model in ("fno_film_nd", "noneq_film_nd", "lemo_bcorrect_nd"):
        for fam in FAMS:
            for reg in REGIMES:
                for seed in SEEDS:
                    cells.append({
                        "sweep": "film_ablation",
                        "fam": fam, "reg": reg, "seed": seed, "model": model,
                        "args": _base_args(model, fam, reg, seed,
                                            output_root="outputs/film_ablation_runpod",
                                            residual_anchor=True),
                    })
    return cells


def cells_sigma_sweep():
    """60 cells: 4 sigma × 5 fams × 3 seeds, lemo_pc_nd, residual_anchor.
    Output dir is sigma-specific (mirrors Caltech sigma_sweep.sbatch behavior).
    """
    cells = []
    for sigma in (0.5, 0.7, 0.9, 0.99):
        for fam in FAMS:
            for seed in SEEDS:
                cells.append({
                    "sweep": "sigma_sweep",
                    "fam": fam, "reg": "clean", "seed": seed,
                    "model": "lemo_pc_nd", "sigma": sigma,
                    "args": _base_args("lemo_pc_nd", fam, "clean", seed,
                                        output_root=f"outputs/sigma_{sigma}_runpod",
                                        residual_anchor=True, sigma=sigma),
                })
    return cells


def cells_memno_ffno():
    """60 cells: 2 models × 5 fams × 2 regs (lowres + noisy, since clean
    coverage for memno/ffno was already done on Caltech) × 3 seeds.
    NO residual_anchor (matches memno_ffno_sweep.sbatch).
    """
    cells = []
    for model in ("memno_nd", "ffno_nd"):
        for fam in FAMS:
            for reg in ("lowres", "noisy"):
                for seed in SEEDS:
                    cells.append({
                        "sweep": "memno_ffno",
                        "fam": fam, "reg": reg, "seed": seed, "model": model,
                        "args": _base_args(model, fam, reg, seed,
                                            output_root="outputs/memno_ffno_runpod",
                                            residual_anchor=True),
                    })
    return cells


def cells_memory_aware(skip_first: int = 18):
    """135 cells, skip first N (Caltech is doing those).  3 models × 5 fams × 3 regs × 3 seeds.
    NO residual_anchor (matches memory_aware_sweep.sbatch).
    """
    cells = []
    for model in ("nide_nd", "ndde_nd", "s4_nd"):
        for fam in FAMS:
            for reg in REGIMES:
                for seed in SEEDS:
                    cells.append({
                        "sweep": "memory_aware",
                        "fam": fam, "reg": reg, "seed": seed, "model": model,
                        "args": _base_args(model, fam, reg, seed,
                                            output_root="outputs/memory_aware_runpod",
                                            residual_anchor=True),
                    })
    return cells[skip_first:]


def all_cells():
    cells = []
    cells.extend(cells_film_ablation())
    cells.extend(cells_sigma_sweep())
    cells.extend(cells_memno_ffno())
    cells.extend(cells_memory_aware(skip_first=18))
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            print(f"{i:4d}  {c['sweep']:14s}  {c['model']:18s}  "
                  f"{c['fam']:24s}  {c['reg']:7s}  s{c['seed']}")
        print(f"---\nTotal: {len(cells)} cells")
    elif len(sys.argv) > 1 and sys.argv[1] == "--slice":
        start = int(sys.argv[2]); count = int(sys.argv[3])
        for c in cells[start:start + count]:
            print(json.dumps(c))
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells")
        from collections import Counter
        sweeps = Counter(c["sweep"] for c in cells)
        for s, n in sweeps.items():
            print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
