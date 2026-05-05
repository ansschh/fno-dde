"""Generate T_lag_modes_ablation.tex from P3 sensitivity sweep data.

Aggregates lag_modes ∈ {12, 16, 24, 32, 48} × dist_exp_rd_2d × clean × seeds.

Reads from: extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/p3_sensitivity_runpod/lag_modes_*/raw/

Outputs:
  - paper/tables/T_lag_modes_ablation.tex (LaTeX appendix table)

Usage:
  python scripts/make_T_lag_modes_ablation.py
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
TABLE_OUT = NEURIPS / "tables" / "T_lag_modes_ablation.tex"
TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)

ROOT = REPO / "extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/p3_sensitivity_runpod"


def collect():
    """Return dict[lag_modes][seed] -> {test_rel_l2, params, wall_seconds}."""
    out = defaultdict(dict)
    for sweep_dir in sorted(ROOT.glob("lag_modes_*")):
        # Some dirs are named lag_modes_N or lag_modes_N_seeds; extract N.
        name = sweep_dir.name
        try:
            lm = int(name.split("_")[2])
        except (IndexError, ValueError):
            continue
        for tr in sweep_dir.glob("**/test_results.json"):
            parts = tr.parts
            if "raw" not in parts:
                continue
            idx = parts.index("raw")
            if idx + 4 >= len(parts):
                continue
            fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
            if fam != "dist_exp_rd_2d" or reg != "clean":
                continue
            if not seed_str.startswith("s"):
                continue
            seed = int(seed_str[1:])
            try:
                data = json.loads(tr.read_text())
            except Exception:
                continue
            test = data.get("test_rel_l2_mean") or data.get("test_rel_l2")
            params = data.get("params") or data.get("n_params")
            wall = data.get("wall_seconds") or data.get("wall_clock_s")
            if test is None:
                continue
            out[lm][seed] = {
                "test_rel_l2": float(test),
                "params": int(params) if params else 0,
                "wall_s": float(wall) if wall else 0.0,
            }
    return out


def emit_table(data):
    if not data:
        print("[T_lag_modes] no data found")
        return
    lms = sorted(data.keys())
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Lag-mode ablation (LEMO-PC on dist\_exp\_rd\_2d, clean regime). "
        r"Test rel-$L_2$, parameters, and wall-clock time as a function of the "
        r"number of retained lag-spectrum modes. Mean $\pm$ std across "
        r"$n=3$ seeds. Lag-window length is 64.}",
        r"\label{tab:lag-modes-ablation}",
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"Lag modes & Test rel-$L_2$ & Params (M) & Wall-clock (s) & $n$ \\",
        r"\midrule",
    ]
    for lm in lms:
        seeds = data[lm]
        if not seeds:
            continue
        rels = [s["test_rel_l2"] for s in seeds.values()]
        params = [s["params"] for s in seeds.values()]
        walls = [s["wall_s"] for s in seeds.values() if s["wall_s"] > 0]
        n = len(rels)
        rel_mean = np.mean(rels)
        rel_std = np.std(rels)
        params_m = np.mean(params) / 1e6
        wall_mean = np.mean(walls) if walls else 0.0
        wall_str = f"{wall_mean:.0f}" if wall_mean else "--"
        lines.append(
            rf"{lm} & {rel_mean:.4f} $\pm$ {rel_std:.4f} & {params_m:.2f} & "
            rf"{wall_str} & {n} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    body = "\n".join(lines) + "\n"
    TABLE_OUT.write_text(body)
    print(f"[T_lag_modes] wrote {len(lms)} lag-mode rows -> {TABLE_OUT}")
    for lm in lms:
        seeds = data[lm]
        rels = [s["test_rel_l2"] for s in seeds.values()]
        if rels:
            print(f"  lag_modes={lm}: rel_l2={np.mean(rels):.4f} ± {np.std(rels):.4f} (n={len(rels)})")


def main():
    data = collect()
    emit_table(data)


if __name__ == "__main__":
    main()
