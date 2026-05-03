"""B5 (causal-smooth LEMO) + P3 (sensitivity sweeps) cell lists.

B5: tests the strictly-causal LEMO-PC variant (CausalSmoother on output of
cyclic-equivariant body). Reviewer asks (B5) for empirical quantification of
the cyclic-vs-causal boundary trade-off.

P3: sensitivity sweep over key hyperparameters (lag_modes, n_layers, width,
spatial_modes), one family + one seed per setting (sufficient for ablation_hawk
who only asked for "modest" sweeps).

Both produce results into outputs/{b5_causal_smooth,p3_sensitivity}_runpod/raw/...
matching the figure-loader convention used by the rest of the pipeline.
"""
from __future__ import annotations
import json
import sys

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
SEEDS_3 = (42, 123, 456)


def _b5_args(fam, seed,
             epochs=100, batch_size=4, width=64, n_layers=3,
             lag_modes=24, spatial_modes=12):
    """B5 causal-smooth LEMO-PC training args. residual_anchor required so
    cyclic boundary stays continuous in the body; smoother applies to output."""
    return [
        "--family", fam,
        "--model", "causal_smooth_lemo_pc_nd",
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
        "--output_dir", "outputs/b5_causal_smooth_runpod/raw",
        "--residual_anchor",
    ]


def _p3_args(model, fam, seed, *,
             epochs=100, batch_size=4, width=64, n_layers=3,
             lag_modes=24, spatial_modes=12, output_subdir="default"):
    """P3 sensitivity training args. Pass overrides for the parameter being swept."""
    return [
        "--family", fam,
        "--model", model,
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
        "--output_dir", f"outputs/p3_sensitivity_runpod/{output_subdir}/raw",
        "--residual_anchor",
    ]


def cells_b5():
    """B5: causal_smooth_lemo_pc_nd × 5 fams × 3 seeds = 15 cells.
    Compares directly with cyclic LEMO-PC (already trained at lemo_pc_nd
    cells in film_ablation sweep). Same fam+seed → boundary contamination
    measurable as (causal_smooth result) - (cyclic LEMO-PC result)."""
    cells = []
    for fam in FAMS:
        for seed in SEEDS_3:
            cells.append({
                "sweep": "b5_causal_smooth",
                "fam": fam, "reg": "clean", "seed": seed,
                "model": "causal_smooth_lemo_pc_nd",
                "args": _b5_args(fam, seed),
            })
    return cells


def cells_p3():
    """P3 sensitivity sweep on dist_exp_rd_2d, seed=42, lemo_pc_nd.

    Defaults: width=64, n_layers=3, lag_modes=24, spatial_modes=12.
    Vary one parameter per row, hold others at default.
    """
    cells = []
    fam = "dist_exp_rd_2d"
    seed = 42
    # 1. lag_modes sweep: 4 values (default 24)
    for lm in (8, 16, 32, 48):
        cells.append({
            "sweep": "p3_lag_modes",
            "fam": fam, "reg": "clean", "seed": seed, "model": "lemo_pc_nd",
            "param": "lag_modes", "value": lm,
            "args": _p3_args("lemo_pc_nd", fam, seed,
                              lag_modes=lm,
                              output_subdir=f"lag_modes_{lm}"),
        })
    # 2. n_layers sweep: 4 values (default 3)
    for nl in (2, 3, 4, 5):
        if nl == 3:
            continue  # default, already covered by main sweep
        cells.append({
            "sweep": "p3_n_layers",
            "fam": fam, "reg": "clean", "seed": seed, "model": "lemo_pc_nd",
            "param": "n_layers", "value": nl,
            "args": _p3_args("lemo_pc_nd", fam, seed,
                              n_layers=nl,
                              output_subdir=f"n_layers_{nl}"),
        })
    # 3. width sweep: 3 values (default 64)
    for w in (32, 96, 128):
        cells.append({
            "sweep": "p3_width",
            "fam": fam, "reg": "clean", "seed": seed, "model": "lemo_pc_nd",
            "param": "width", "value": w,
            "args": _p3_args("lemo_pc_nd", fam, seed,
                              width=w,
                              output_subdir=f"width_{w}"),
        })
    # 4. spatial_modes sweep: 3 values (default 12)
    for sm in (6, 16, 24):
        cells.append({
            "sweep": "p3_spatial_modes",
            "fam": fam, "reg": "clean", "seed": seed, "model": "lemo_pc_nd",
            "param": "spatial_modes", "value": sm,
            "args": _p3_args("lemo_pc_nd", fam, seed,
                              spatial_modes=sm,
                              output_subdir=f"spatial_modes_{sm}"),
        })
    # 5. lag_modes × 3 seeds for stability of headline (12 → 24 transition)
    for lm in (12, 24):
        for s in (123, 456):
            cells.append({
                "sweep": "p3_lag_modes_seeds",
                "fam": fam, "reg": "clean", "seed": s, "model": "lemo_pc_nd",
                "param": "lag_modes", "value": lm,
                "args": _p3_args("lemo_pc_nd", fam, s,
                                  lag_modes=lm,
                                  output_subdir=f"lag_modes_{lm}_seeds"),
            })
    return cells


def all_cells():
    cells = []
    cells.extend(cells_b5())
    cells.extend(cells_p3())
    return cells


def main():
    cells = all_cells()
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        for i, c in enumerate(cells):
            print(f"{i:3d}  {c['sweep']:24s}  {c['model']:30s}  "
                  f"{c['fam']:24s}  s{c['seed']}  "
                  f"{c.get('param', ''):16s}={c.get('value', '')}")
        print(f"---\nTotal: {len(cells)} cells")
        from collections import Counter
        sweeps = Counter(c["sweep"] for c in cells)
        for s, n in sweeps.items():
            print(f"  {s}: {n}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(cells))
    else:
        print(f"Total: {len(cells)} cells")


if __name__ == "__main__":
    main()
