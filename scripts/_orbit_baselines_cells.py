"""Orbit OOD baselines comprehensive sweep (Phase 2 — for 16x H100 pod).

5 baselines × 6 m-values × 3 seeds = 90 cells. Tests B3 paper claim against
the architectural classes the reviewer panel asked for (B1 list):
  - ndde_nd        (memory-aware NN-DE)
  - memno_nd       (convolutional memory operator)
  - nide_nd        (neural integral / delay equation = ANIE/NIDE)
  - s4_nd          (state-space sequence model)
  - fno_film_nd    (FNO + FiLM conditioning, no equivariance)

Used together with Pod 2's LEMO-PC + per_lag_mlp orbit cells to demonstrate
that lag-equivariance specifically is the mechanism — none of these
non-equivariant baselines should match LEMO-PC's orbit-constant test error.

m32 cells at 50 epochs (consistent with Pod 2's orbit setup: cuts m32 wall
from 10h -> 5h). All other m at 100 epochs.

Output: outputs/orbit_baselines_h100/<model>_m<m>/raw/dist_exp_rd_2d_orbit/clean/<model>/s<seed>/

Cells ordered with m32 first so they get scheduled on GPUs early
(critical-path bottleneck).
"""
from __future__ import annotations
import json
import sys

BASELINE_MODELS = ("ndde_nd", "memno_nd", "nide_nd", "s4_nd", "fno_film_nd")
M_VALUES = (1, 2, 4, 8, 16, 32)
SEEDS = (42, 123, 456)


def _orbit_baseline_args(model, m, seed,
                         epochs=100, batch_size=4, width=64, n_layers=3,
                         lag_modes=24, spatial_modes=12):
    """Orbit OOD baseline training args. Same hyperparams as LEMO-PC orbit
    cells for fair comparison."""
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
        "--output_dir", f"outputs/orbit_baselines_h100/{model}_m{m}/raw",
        "--residual_anchor",
    ]


def all_cells():
    """All 90 baseline orbit cells. ORDERED: m32 first (longest, critical
    path), then m16, m8, ..., m1. Within each m, models alternate seeds
    so different baselines start in parallel on GPUs 0-15."""
    cells = []
    # Reverse-m order: m32 cells get GPU slots first (5h each, dominant)
    for m in (32, 16, 8, 4, 2, 1):
        # Interleave models so first 5 cells per m hit different baselines
        for seed in SEEDS:
            for model in BASELINE_MODELS:
                cells.append({
                    "sweep": f"orbit_{model}",
                    "fam": "dist_exp_rd_2d_orbit", "reg": "clean",
                    "seed": seed, "model": model, "m": m,
                    "args": _orbit_baseline_args(
                        model, m, seed,
                        epochs=50 if m == 32 else 100,
                    ),
                })
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            print(f"{i:3d}  {c['model']:18s}  m={c['m']:3d}  s{c['seed']:4d}  ep={c['args'][c['args'].index('--epochs')+1]}")
        print(f"---\nTotal: {len(cells)} cells")
        from collections import Counter
        models = Counter(c["model"] for c in cells)
        for mdl, n in models.items():
            print(f"  {mdl}: {n}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells across {len(BASELINE_MODELS)} models, {len(M_VALUES)} m-values, {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
