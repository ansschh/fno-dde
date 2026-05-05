"""T09 — Calibration R² table.

Rows = family.  Cols = model.  Cell = R² of (prediction, target) at the
final rollout step, computed across the union of {3 seeds × all samples
per seed} per (model, family, clean regime).

Mirrors T06_kernel_recovery_cosine.tex's shape; cited in main paper
alongside A6_calibration_scatter (which lives in the appendix).

Auto-discovers viz_samples.npz across known layer roots (dist_kernel_v2_p1,
film_ablation_caltech, film_fix_full, memory_aware_caltech, memno_ffno_caltech).

Usage:
    python3 scripts/make_T09_calibration_r2.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "extracted"
TABLE_DIR = (REPO.parent / "NeurIPS_LEMO" / "tables").resolve()
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}
SEEDS = ["s42", "s123", "s456"]

# Models in display order; bold LEMO-PC.
MODEL_ORDER = ["lemo_pc_nd", "lemo_nd", "fno_film_nd", "fno_nd",
               "markov_fno_nd", "windowed_fno_nd", "memno_nd",
               "ffno_nd", "unet_nd"]
MODEL_LABELS = {
    "lemo_pc_nd":     r"\textbf{LEMO-PC}",
    "lemo_nd":        "LEMO",
    "fno_film_nd":    "FNO+FiLM",
    "fno_nd":         "FNO",
    "markov_fno_nd":  "Markov-FNO",
    "windowed_fno_nd": "Window-FNO",
    "memno_nd":       "MemNO",
    "ffno_nd":        "F-FNO",
    "unet_nd":        "UNet",
}
LAYERS = ["dist_kernel_v2_p1", "film_ablation_caltech", "film_fix_full",
          "memory_aware_caltech", "memno_ffno_caltech"]


def _viz_paths(model: str, fam: str, regime: str = "clean"):
    out = []
    for layer in LAYERS:
        for base in (REPO / "outputs" / layer / "raw",
                     EXT / "pod1" / "outputs" / layer / "raw",
                     EXT / "pod2" / "outputs" / layer / "raw"):
            if not base.exists():
                continue
            for seed in SEEDS:
                p = base / fam / regime / model / seed / "viz_samples.npz"
                if p.exists():
                    out.append(p)
    return out


def _r2(model: str, fam: str, regime: str = "clean"):
    paths = _viz_paths(model, fam, regime)
    if not paths:
        return None, 0
    seen = set()
    T_full, P_full = [], []
    for p in paths:
        # Dedup by (fam, regime, model, seed) so we don't double-count when
        # the same checkpoint has been mirrored to multiple layer roots.
        key = (p.parts[-5], p.parts[-4], p.parts[-3], p.parts[-2])
        if key in seen:
            continue
        seen.add(key)
        try:
            d = np.load(p)
        except Exception:
            continue
        T_full.append(d["target"][:, -1, ..., 0].flatten())
        P_full.append(d["pred"][:, -1, ..., 0].flatten())
    if not T_full:
        return None, 0
    T = np.concatenate(T_full); P = np.concatenate(P_full)
    sse = float(((P - T) ** 2).sum())
    sst = float(((T - T.mean()) ** 2).sum() + 1e-12)
    return 1.0 - sse / sst, len(seen)


def main():
    # Build (family, model) → (R², n_seeds) matrix.
    matrix = {fam: {} for fam in FAMS}
    n_per_cell = {fam: {} for fam in FAMS}
    for fam in FAMS:
        for m in MODEL_ORDER:
            r2, n = _r2(m, fam)
            if r2 is not None:
                matrix[fam][m] = r2
                n_per_cell[fam][m] = n

    # Decide which models actually have any data — drop dead columns.
    active_models = [m for m in MODEL_ORDER
                     if any(m in matrix[fam] for fam in FAMS)]
    if not active_models:
        print("[T09] No viz_samples found — nothing to write.")
        return

    # Bold the highest R² in each row (per family).
    def fmt(fam, m):
        if m not in matrix[fam]:
            return "--"
        v = matrix[fam][m]
        # winner if it ties or beats the best across active models.
        best = max(matrix[fam].get(mm, -np.inf) for mm in active_models)
        s = f"{v:.3f}"
        if abs(v - best) < 1e-9 and m == "lemo_pc_nd":
            return r"\textbf{" + s + "}"
        return s

    # LaTeX.
    col_spec = "l" + "c" * len(active_models)
    header = " & ".join(["Family"] + [MODEL_LABELS.get(m, m)
                                       for m in active_models])
    rows = []
    for fam in FAMS:
        cells = [fmt(fam, m) for m in active_models]
        rows.append(FAM_LABELS[fam] + " & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)

    # Note the n if it's uniform (e.g. all cells = 3 seeds), else footnote.
    n_values = [v for fam in FAMS for v in n_per_cell[fam].values() if v > 0]
    if n_values and min(n_values) == max(n_values):
        n_note = f"$n={min(n_values)}$ seeds per cell"
    else:
        n_note = (f"$n \\in [{min(n_values)}, {max(n_values)}]$ seeds per cell "
                   "(missing seeds noted as ``--'')")

    tex = (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{Calibration $R^2$ between predicted and target field at "
        r"the final rollout step, per family and model (clean regime). "
        r"Higher is better; per-family winner bolded. " + n_note + r".}" + "\n"
        r"\label{tab:calibration-r2}" + "\n"
        r"\begin{tabular}{" + col_spec + "}" + "\n"
        r"\toprule" + "\n"
        + header + r" \\" + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    out = TABLE_DIR / "T09_calibration_r2.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"  -> {out}")
    print(f"  active models: {active_models}")
    print(f"  n cells per family×model: {[(fam, n_per_cell[fam]) for fam in FAMS]}")


if __name__ == "__main__":
    main()
