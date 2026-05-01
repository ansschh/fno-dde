"""Generate the raw per-cell rel-L2 table (T-raw).

Reviewer asks for the absolute rel-L2 numbers backing the percent-improvement
claims in T01. We emit a per-(family, regime) table with 3-seed mean and a
95% bootstrap CI (10000 resamples) for each of the 6 main models: LEMO-PC,
LEMO (no FiLM ablation), FNO, Markov-FNO, Window-FNO, UNet. Auxiliary
baselines (MemNO, F-FNO) have clean-regime-only coverage and are reported
in a small companion block at the bottom.

Sources:
  - LEMO and LEMO-PC:
      extracted/pod1/outputs/dist_kernel_v2_p1/raw/<fam>/<reg>/<model>/<seed>/test_results.json
  - FNO/Markov-FNO/Window-FNO/UNet:
      extracted/pod2/outputs/dist_kernel_v2_p2/logs/<fam>_<model>_<reg>_s<num>.log
  - MemNO/F-FNO:
      extracted/pod3/outputs/final_baselines/raw/<fam>/<reg>/<model>/<seed>/test_results.json

Output: NeurIPS_LEMO/tables/T_raw_per_cell.tex with \multirow family blocks,
wrapped in \begin{table}\caption\label\end{table}.

Usage:
    python3 scripts/make_raw_table.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "extracted"
PAPER_TABLE_DIR = REPO.parent / "NeurIPS_LEMO" / "tables"
PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp",
              "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma",
              "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}
REGIMES = ["clean", "lowres", "noisy"]
SEEDS = ["s42", "s123", "s456"]
SEED_TO_NUM = {"s42": "42", "s123": "123", "s456": "456"}

MAIN_MODELS = ["lemo_pc_nd", "lemo_nd", "fno_nd", "markov_fno_nd",
               "windowed_fno_nd", "unet_nd"]
AUX_MODELS = ["memno_nd", "ffno_nd"]
MODEL_LABELS = {
    "lemo_pc_nd":      r"\textbf{LEMO-PC}",
    "lemo_nd":         "LEMO",
    "fno_nd":          "FNO",
    "markov_fno_nd":   "Markov-FNO",
    "windowed_fno_nd": "Window-FNO",
    "unet_nd":         "UNet",
    "memno_nd":        "MemNO",
    "ffno_nd":         "F-FNO",
}

N_BOOT = 10000
RNG = np.random.default_rng(20251030)
_LOG_PAT = re.compile(r"=== FINAL test relL2 = ([0-9.]+) ===")


def _try_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_lemo_test(model: str) -> dict:
    """Pod-1 dist_kernel_v2_p1 hosts both lemo_pc_nd and lemo_nd."""
    out = {}
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = float(
                        d.get("test_rel_l2_mean",
                              d.get("test_rel_l2", float("nan"))))
    return out


def load_baseline_from_log(model: str) -> dict:
    """Pod-2 dist_kernel_v2_p2/logs hosts FNO / Markov / Window / UNet."""
    out = {}
    log_dir = EXT / "pod2" / "outputs" / "dist_kernel_v2_p2" / "logs"
    if not log_dir.exists():
        return out
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                seed_num = SEED_TO_NUM[seed]
                logf = log_dir / f"{fam}_{model}_{reg}_s{seed_num}.log"
                if not logf.exists():
                    continue
                m = _LOG_PAT.search(logf.read_text(errors="replace"))
                if m:
                    out[(fam, reg, seed)] = float(m.group(1))
    return out


def load_pod3_baseline(model: str) -> dict:
    """Pod-3 final_baselines hosts MemNO / F-FNO (clean only)."""
    out = {}
    pod3 = EXT / "pod3" / "outputs" / "final_baselines" / "raw"
    if not pod3.exists():
        return out
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = pod3 / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = float(
                        d.get("test_rel_l2_mean",
                              d.get("test_rel_l2", float("nan"))))
    return out


def gather_all() -> dict:
    return {
        "lemo_pc_nd":      load_lemo_test("lemo_pc_nd"),
        "lemo_nd":         load_lemo_test("lemo_nd"),
        "fno_nd":          load_baseline_from_log("fno_nd"),
        "markov_fno_nd":   load_baseline_from_log("markov_fno_nd"),
        "windowed_fno_nd": load_baseline_from_log("windowed_fno_nd"),
        "unet_nd":         load_baseline_from_log("unet_nd"),
        "memno_nd":        load_pod3_baseline("memno_nd"),
        "ffno_nd":         load_pod3_baseline("ffno_nd"),
    }


def boot_ci(vals: np.ndarray, n_boot: int = N_BOOT,
            rng: np.random.Generator = RNG) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI from n_boot resamples.

    n=3 seeds is small but parametric assumptions are worse; the
    percentile-bootstrap CI is the agreed reporting convention for the
    headline (T01) and is reused here for consistency.
    """
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = vals.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def fmt_cell(mean: float, lo: float, hi: float) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def cell_for(data: dict, model: str, fam: str, reg: str) -> str:
    vals = [data.get(model, {}).get((fam, reg, sd), np.nan) for sd in SEEDS]
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return fmt_cell(*boot_ci(vals))


def build_main_table(data: dict) -> str:
    main_present = [m for m in MAIN_MODELS if data.get(m)]
    n_cols = len(main_present)
    col_spec = "ll" + "c" * n_cols

    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Raw per-cell test rel-$L_2$ for the 6 main models on "
        r"every (family, regime) pair (mean over 3 seeds with 95\% bootstrap "
        r"CI from 10\,000 resamples in brackets). These are the absolute "
        r"numbers backing the percent-improvement claims in "
        r"\Cref{tab:headline-per-baseline}; reviewers can recompute any "
        r"\% reduction directly from these cells. MemNO and F-FNO covered "
        r"only the clean regime in the Pod-3 Phase-A sweep and are reported "
        r"in the lower block.}")
    lines.append(r"\label{tab:raw-percell}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = ["Family", "Regime"] + [MODEL_LABELS[m] for m in main_present]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for f_idx, fam in enumerate(FAMS):
        for r_idx, reg in enumerate(REGIMES):
            cells = [cell_for(data, mdl, fam, reg) for mdl in main_present]
            if r_idx == 0:
                fam_cell = (rf"\multirow{{{len(REGIMES)}}}{{*}}"
                            rf"{{{FAM_LABELS[fam]}}}")
            else:
                fam_cell = ""
            lines.append(f"{fam_cell} & {reg} & " + " & ".join(cells)
                         + r" \\")
        if f_idx < len(FAMS) - 1:
            lines.append(r"\midrule")

    # Aux block (clean only, MemNO + F-FNO).
    aux_present = [m for m in AUX_MODELS if data.get(m)]
    if aux_present:
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{2 + n_cols}}}{{l}}"
            r"{\textit{Auxiliary baselines (clean regime only, "
            r"Pod-3 Phase-A sweep).}} \\")
        lines.append(r"\midrule")

        # Header row for aux block: re-use Family / Model columns; one cell
        # per aux model in the value column, dashes elsewhere.
        sub_header = ["Family", "Model"] + [MODEL_LABELS[aux_present[0]]] + \
                     [MODEL_LABELS[m] for m in aux_present[1:]] + \
                     [""] * max(0, n_cols - len(aux_present))
        lines.append(" & ".join(sub_header[:2 + n_cols]) + r" \\")
        lines.append(r"\midrule")

        for fam in FAMS:
            cells: list[str] = []
            for i, mdl in enumerate(aux_present):
                if i < n_cols:
                    cells.append(cell_for(data, mdl, fam, "clean"))
            cells += ["--"] * (n_cols - len(aux_present))
            lines.append(f"{FAM_LABELS[fam]} & clean & "
                         + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    print(f"[T-raw] working dir: {REPO}")
    data = gather_all()
    print("[T-raw] data inventory")
    for m in MAIN_MODELS + AUX_MODELS:
        clean_label = (MODEL_LABELS.get(m, m)
                       .replace(chr(92) + 'textbf', '')
                       .replace('{', '').replace('}', ''))
        print(f"  {clean_label:<14}: {len(data.get(m, {})):>4} cells")

    text = build_main_table(data)
    out = PAPER_TABLE_DIR / "T_raw_per_cell.tex"
    out.write_text(text + "\n")
    print(f"[T-raw] wrote {out}")


if __name__ == "__main__":
    main()
