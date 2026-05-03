"""Master cell list for the 6-pod RunPod offload of pending Caltech sweeps.

Order matters — pods take a contiguous [START, START+COUNT) slice of this
list, so cells must NEVER be reordered between pods. All 5 sweeps
concatenated in this fixed order:

  0-134   film_ablation       (135 cells: 3 models × 5 fams × 3 regs × 3 seeds)
  135-194 sigma_sweep         (60  cells: 1 model × 5 fams × 4 sigmas × 3 seeds)
  195-230 orbit_ood           (36  cells: 2 models × 1 fam × 2 OOD regs × 3 seeds × 3 quants)
  231-290 memno_ffno          (60  cells: 2 models × 5 fams × 2 regimes × 3 seeds)
  291-407 memory_aware        (117 cells: 3 models × 5 fams × 3 regs × 3 seeds, MINUS first 18
                                 = 135 - 18 cells; first 18 stay on Caltech)

Total: 408 cells. Assignment by throughput-weighted share across 6 pods.
"""
from __future__ import annotations
import json
import sys

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ("clean", "lowres", "noisy")
SEEDS = (42, 123, 456)


def cells_film_ablation():
    cells = []
    for model in ("fno_film_nd", "noneq_film_nd", "lemo_bcorrect_nd"):
        for fam in FAMS:
            for reg in REGIMES:
                for seed in SEEDS:
                    cells.append({
                        "sweep": "film_ablation",
                        "fam": fam, "reg": reg, "seed": seed, "model": model,
                        "out_dir": "outputs/film_ablation_runpod",
                        "extra_flags": ["--residual_anchor"],
                    })
    return cells


def cells_sigma_sweep():
    cells = []
    for fam in FAMS:
        for sigma in (0.5, 1.0, 2.0, 4.0):
            for seed in SEEDS:
                cells.append({
                    "sweep": "sigma_sweep",
                    "fam": fam, "reg": "clean", "seed": seed,
                    "model": "lemo_pc_nd",
                    "out_dir": "outputs/sigma_sweep_runpod",
                    "extra_flags": ["--residual_anchor",
                                    "--sigma_lipschitz", str(sigma)],
                })
    return cells


def cells_orbit_ood():
    """36 cells: 2 models × 1 family (dist_exp_rd_2d_orbit) × 2 OOD-strength
    regimes × 3 seeds × 3 OOD-quant settings.  This is the cleanest read of
    slurm/orbit_ood.sbatch's array indexing — see that file for ground truth.
    """
    cells = []
    for model in ("lemo_pc_nd", "per_lag_mlp_nd"):
        for ood_strength in ("ood_low", "ood_high"):
            for seed in SEEDS:
                for ood_quant in (0, 1, 2):
                    cells.append({
                        "sweep": "orbit_ood",
                        "fam": "dist_exp_rd_2d_orbit",
                        "reg": "clean", "seed": seed, "model": model,
                        "out_dir": "outputs/orbit_ood_runpod",
                        "extra_flags": ["--residual_anchor",
                                        "--ood_regime", ood_strength,
                                        "--ood_quant", str(ood_quant)],
                    })
    return cells[:36]  # ensure exact count


def cells_memno_ffno():
    cells = []
    for model in ("memno_nd", "ffno_nd"):
        for fam in FAMS:
            for reg in ("lowres", "noisy"):
                for seed in SEEDS:
                    cells.append({
                        "sweep": "memno_ffno",
                        "fam": fam, "reg": reg, "seed": seed, "model": model,
                        "out_dir": "outputs/memno_ffno_runpod",
                        "extra_flags": [],
                    })
    return cells


def cells_memory_aware(skip_first: int = 18):
    cells = []
    for model in ("nide_nd", "ndde_nd", "s4_nd"):
        for fam in FAMS:
            for reg in REGIMES:
                for seed in SEEDS:
                    cells.append({
                        "sweep": "memory_aware",
                        "fam": fam, "reg": reg, "seed": seed, "model": model,
                        "out_dir": "outputs/memory_aware_runpod",
                        "extra_flags": [],
                    })
    return cells[skip_first:]  # leave first N to Caltech


def all_cells():
    cells = []
    cells.extend(cells_film_ablation())
    cells.extend(cells_sigma_sweep())
    cells.extend(cells_orbit_ood())
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
