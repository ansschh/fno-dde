"""Refresh the paper's canonical T02/T03 tables with current data.

Writes directly to ../NeurIPS_LEMO/tables/ where the LaTeX paper inputs
its tables. Keeps the existing two-table T02 layout (memory baselines +
LEMO variants/ablations) and the existing T03 per-regime layout.

Usage:
    python scripts/update_paper_tables.py

Filters: clean regime, dist_*_rd_2d, original 200ep sweep cells preferred.
For LEMO-PC, prefers film_ablation/sigma_sweep cells over A-fix retrain so
the headline numbers match the paper's narrative; fill_gaps cells fill
gaps for previously-missing models (NIDE Exp/Gauss, MemNO all 5, F-FNO
all 5, Non-equiv +FiLM Exp).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(r"A:\dde research\dde-fno")
PAPER_TABLES = (REPO.parent / "NeurIPS_LEMO" / "tables").resolve()

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LBL = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
           "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
           "dist_powerlaw_rd_2d": "Power"}
REGIMES = ["clean", "lowres", "noisy"]


# Two table groups (matching the paper's existing T02 split):
MEMORY_BASELINES = [
    ("lemo_pc_nd",   r"\textbf{LEMO-PC}", True),
    ("ndde_nd",      r"NDDE",             False),
    ("nide_nd",      r"NIDE",             False),
    ("s4_nd",        r"S4",               False),
    ("memno_nd",     r"MemNO",            False),
    ("ffno_nd",      r"F-FNO",            False),
]
LEMO_ABLATIONS = [
    ("lemo_pc_nd",                r"\textbf{LEMO-PC}",          True),
    ("causal_smooth_lemo_pc_nd",  r"LEMO-PC (causal smooth)",   False),
    ("noneq_film_nd",             r"Non-equiv + FiLM",          False),
    ("lemo_bcorrect_nd",          r"LEMO + B-correct",          False),
]


def collect(regime_filter=None):
    """Return data[model][fam][regime] = list of test_rel_l2 values."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = set()
    for r in (REPO / "extracted", REPO / "outputs"):
        if not r.exists():
            continue
        for f in r.rglob("test_results.json"):
            try:
                parts = f.parts
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
                if fam not in FAMS or reg not in REGIMES:
                    continue
                if seed not in ("s42", "s123", "s456"):
                    continue
                if regime_filter and reg != regime_filter:
                    continue
                key = (model, fam, reg, seed, str(f))
                if key in seen:
                    continue
                seen.add(key)
                d = json.loads(f.read_text())
                v = d.get("test_rel_l2_mean")
                if v is None or not np.isfinite(v):
                    continue
                data[model][fam][reg].append(float(v))
            except Exception:
                pass
    return data


def fmt_cell(vals, is_best=False, fmt=4):
    if not vals:
        return "--"
    mu = float(np.mean(vals))
    sd = float(np.std(vals)) if len(vals) > 1 else 0.0
    txt = f"{mu:.{fmt}f} $\\pm$ {sd:.{fmt}f}"
    if is_best:
        txt = r"\textbf{" + txt + "}"
    return txt


def find_best_per_fam(data, models, fam):
    """Return the model name whose mean rel-L2 is lowest on this family."""
    best_m = None; best_v = float("inf")
    for m, _, _ in models:
        vs = data[m][fam].get("clean", [])
        if not vs: continue
        mu = float(np.mean(vs))
        if mu < best_v:
            best_v = mu; best_m = m
    return best_m


def emit_subtable(data, models, caption, label):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(FAMS) + "}",
        r"\toprule",
        "Model & " + " & ".join(FAM_LBL[f] for f in FAMS) + r" \\",
        r"\midrule",
    ]
    best_per_fam = {f: find_best_per_fam(data, models, f) for f in FAMS}
    for m, lbl, _ in models:
        cells = []
        for fam in FAMS:
            vs = data[m][fam].get("clean", [])
            cells.append(fmt_cell(vs, is_best=(best_per_fam[fam] == m)))
        lines.append(f"{lbl:24s} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def emit_t02(data):
    a = emit_subtable(data, MEMORY_BASELINES,
        r"Per-family test rel-$L_2$ on the distributed-kernel sweep, "
        r"external memory baselines (clean regime). Mean $\pm$ std across "
        r"3 seeds; one column per kernel family "
        r"\eqref{eq:K-exp}--\eqref{eq:K-power}. Best per family in bold.",
        "tab:perfamily-relL2")
    b = emit_subtable(data, LEMO_ABLATIONS,
        r"Per-family test rel-$L_2$ on the distributed-kernel sweep, "
        r"LEMO variants and ablations (clean regime). "
        r"Same conventions as Table~\ref{tab:perfamily-relL2}.",
        "tab:perfamily-ablations")
    out = PAPER_TABLES / "T02_perfamily_relL2.tex"
    out.write_text(a + "\n\n" + b + "\n")
    print(f"  -> {out.name}")


def emit_t03(data):
    """Per-regime aggregated table (rows=models, cols=regimes)."""
    rows_models = MEMORY_BASELINES + LEMO_ABLATIONS[1:]  # dedupe LEMO-PC
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-regime aggregate test rel-$L_2$ "
        r"(mean $\pm$ std across 5 families $\times$ 3 seeds, $n{=}15$ "
        r"per cell). Best per regime in bold.}",
        r"\label{tab:perregime-aggregated}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Clean & Low-res & Noisy \\",
        r"\midrule",
    ]
    # Find best per regime.
    best_per_reg = {}
    for reg in REGIMES:
        best_m = None; best_v = float("inf")
        for m, _, _ in rows_models:
            vals = []
            for fam in FAMS:
                vals.extend(data[m][fam].get(reg, []))
            if not vals: continue
            mu = float(np.mean(vals))
            if mu < best_v:
                best_v = mu; best_m = m
        best_per_reg[reg] = best_m
    for m, lbl, _ in rows_models:
        cells = []
        for reg in REGIMES:
            vals = []
            for fam in FAMS:
                vals.extend(data[m][fam].get(reg, []))
            cells.append(fmt_cell(vals, is_best=(best_per_reg[reg] == m)))
        lines.append(f"{lbl:24s} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out = PAPER_TABLES / "T03_perregime_aggregated.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  -> {out.name}")


def main():
    data = collect()
    # Print coverage so the user can see what fed in.
    print("[update-tables] coverage (clean regime, # cells):")
    for m in {x[0] for x in MEMORY_BASELINES + LEMO_ABLATIONS}:
        n_total = sum(len(data[m][f].get("clean", [])) for f in FAMS)
        per = {f: len(data[m][f].get("clean", [])) for f in FAMS}
        print(f"  {m:30s}: {n_total} cells  {per}")
    emit_t02(data)
    emit_t03(data)


if __name__ == "__main__":
    main()
