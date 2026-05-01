"""
D3 (residual structure) + D4 (phase vs amplitude) analysis.

Reads the residuals.npz files produced by extract_residuals.py for a
list of (cell, model) pairs and prints:

  D3 — per-sample correlation matrix of relL2 across models in a cell.
       If correlation high, all models fail on the same samples → task
       ceiling. If low, different models own different regions → ensemble
       gain available.

  D4 — fraction of total energy in amplitude-error vs phase-error per
       cell, per model. If phase >> amplitude, relative-L2 metrics
       penalize phase mismatch heavily (oscillator cells).

Reads from existing residuals.npz; expects extract_residuals.py to have
been run first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_res(run_dir: Path):
    """Find the most recent residuals.npz in run_dir or its subdirs."""
    files = list(run_dir.rglob("residuals.npz"))
    if not files:
        return None
    return np.load(sorted(files)[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+",
                   default=["linear2/id", "linear2/lag_shift",
                             "predator_prey/id", "vdp/id", "dist_exp/id"])
    p.add_argument("--out", default="d3_d4_summary.json")
    args = p.parse_args()

    # For each cell, which (model, run_dir) pairs to load:
    cell_models = {
        "linear2/id": [
            ("tcn",                   "outputs/phase_b_core_dde_v1/linear2/id/tcn_s42"),
            ("lemo_sigma_09",         "outputs/phase_b_core_dde_v1/linear2/id/lemo_sigma_09_s42"),
            ("v3_long",               "outputs/phase_b_lemo_pc_v3_long/linear2/id/lemo_pc_v3_s42"),
            ("v3_sigma_long",         "outputs/phase_b_lemo_pc_v3_long/linear2/id/lemo_pc_v3_sigma_09_s42"),
        ],
        "linear2/lag_shift": [
            ("tcn",                   "outputs/phase_b_core_dde_v1/linear2/lag_shift/tcn_s42"),
            ("v3_long",               "outputs/phase_b_lemo_pc_v3_long/linear2/lag_shift/lemo_pc_v3_s42"),
            ("v3_sigma_long",         "outputs/phase_b_lemo_pc_v3_long/linear2/lag_shift/lemo_pc_v3_sigma_09_s42"),
        ],
        "predator_prey/id": [
            ("plainmlp",              "outputs/phase_b_core_dde_v1/predator_prey/id/plainmlp_s42"),
            ("v3_long",               "outputs/phase_b_lemo_pc_v3_long/predator_prey/id/lemo_pc_v3_s42"),
            ("v3_sigma_long",         "outputs/phase_b_lemo_pc_v3_long/predator_prey/id/lemo_pc_v3_sigma_09_s42"),
        ],
        "vdp/id": [
            ("fno1d",                 "outputs/phase_b_core_dde_v1/vdp/id/fno1d_s42"),
            ("v3_long",               "outputs/phase_b_lemo_pc_v3_long/vdp/id/lemo_pc_v3_s42"),
            ("v3_sigma_long",         "outputs/phase_b_lemo_pc_v3_long/vdp/id/lemo_pc_v3_sigma_09_s42"),
        ],
        "dist_exp/id": [
            ("tcn",                   "outputs/phase_b_core_dde_v1/dist_exp/id/tcn_s42"),
            ("v3_long",               "outputs/phase_b_lemo_pc_v3_long/dist_exp/id/lemo_pc_v3_s42"),
            ("v3_sigma_long",         "outputs/phase_b_lemo_pc_v3_long/dist_exp/id/lemo_pc_v3_sigma_09_s42"),
        ],
    }

    summary = {}

    for cell in args.cells:
        if cell not in cell_models:
            continue
        print(f"\n=== {cell} ===")
        loaded = {}
        for name, run_dir in cell_models[cell]:
            res = load_res(Path(run_dir))
            if res is None:
                print(f"  {name}: no residuals found at {run_dir}")
                continue
            loaded[name] = res

        if not loaded:
            continue

        # D3: per-sample relL2 correlation matrix
        print("\n  D3: per-sample relL2 correlation across models")
        names = list(loaded.keys())
        rls = {n: loaded[n]["rel_l2"] for n in names}
        # Align: take the min length (in case test sets differ)
        n_min = min(len(v) for v in rls.values())
        rls = {n: v[:n_min] for n, v in rls.items()}
        # Print correlation matrix
        header = "    " + " ".join(f"{n:>14}" for n in names)
        print(header)
        for ni in names:
            row = []
            for nj in names:
                c = np.corrcoef(rls[ni], rls[nj])[0, 1]
                row.append(f"{c:>14.3f}")
            print(f"    {ni:<14} {''.join(row[:1])} {' '.join(row)}".rstrip())

        # D3b: top-decile overlap. For each pair, what fraction of the
        # top-10% hardest samples are shared?
        print("\n  D3b: hardest-decile overlap (Jaccard)")
        for i, ni in enumerate(names):
            top_i = set(np.argsort(rls[ni])[-n_min // 10:].tolist())
            for j, nj in enumerate(names):
                if i == j:
                    continue
                top_j = set(np.argsort(rls[nj])[-n_min // 10:].tolist())
                jac = len(top_i & top_j) / len(top_i | top_j)
                print(f"    {ni} vs {nj}:  jaccard={jac:.3f}")

        # D4: amplitude vs phase error
        print("\n  D4: amplitude vs phase error (energy-weighted)")
        d4_rows = []
        for n in names:
            res = loaded[n]
            amp = res["amplitude_err"]    # (N, n_modes)
            phase = res["phase_err"]      # (N, n_modes)
            energy = res["energy_target"] # (N, n_modes)
            # Weight by mode energy so high-energy modes dominate.
            w = energy / (energy.sum(axis=1, keepdims=True) + 1e-10)
            amp_w = (amp * w).sum(axis=1).mean()
            phase_w = (phase * w).sum(axis=1).mean()  # phase in radians; ~0 = perfect
            d4_rows.append((n, amp_w, phase_w))
            print(f"    {n:<20s}  amp_err_w={amp_w:.4f}  phase_err_w={phase_w:.4f} rad")

        summary[cell] = {
            "models": names,
            "n_samples": n_min,
            "correlation": {ni: {nj: float(np.corrcoef(rls[ni], rls[nj])[0, 1])
                                  for nj in names} for ni in names},
            "amp_phase": [{"model": n, "amp_w": float(a), "phase_w": float(p)}
                           for n, a, p in d4_rows],
        }

    out_path = PROJECT_ROOT / args.out
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
