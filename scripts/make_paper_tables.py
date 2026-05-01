"""Paper table generation — LaTeX tables backing the 2D dist-kernel claims.

Reads same data sources as make_paper_figures.py.

Produces (in paper/tables/):
  - T01_headline_per_baseline.tex        impr% / Hedges g / p-value per comparison
  - T02_perfamily_relL2.tex              5 fams x N models test relL2 (clean)
  - T03_perregime_aggregated.tex         3 regimes x N models, mean +/- std
  - T04_compute_costs.tex                params, train wallclock, peak mem per model
  - T05_per_baseline_per_regime_breakdown.tex   per-regime impr% per baseline

Usage:
    python3 scripts/make_paper_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "extracted"
TABLE_DIR = REPO / "paper" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = REPO / "paper" / "stats" / "paired_permutation.json"

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}
REGIMES = ["clean", "lowres", "noisy"]
SEEDS = ["s42", "s123", "s456"]
MODEL_LABELS = {
    "lemo_pc_nd":     r"\textbf{LEMO-PC}",
    "lemo_nd":        "LEMO",
    "fno_nd":         "FNO",
    "markov_fno_nd":  "Markov-FNO",
    "windowed_fno_nd": "Window-FNO",
    "memno_nd":       "MemNO",
    "ffno_nd":        "F-FNO",
}
MODEL_ORDER = ["lemo_pc_nd", "lemo_nd", "fno_nd", "markov_fno_nd",
               "windowed_fno_nd", "memno_nd", "ffno_nd"]


def _try_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_lemo_test(model: str = "lemo_pc_nd") -> dict:
    out = {}
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = float(d.get("test_rel_l2_mean",
                                                        d.get("test_rel_l2", float("nan"))))
    return out


_LOG_PAT = re.compile(r"=== FINAL test relL2 = ([0-9.]+) ===")


def load_baseline_from_log(model: str) -> dict:
    out = {}
    log_dir = EXT / "pod2" / "outputs" / "dist_kernel_v2_p2" / "logs"
    if not log_dir.exists():
        return out
    seed_to_num = {"s42": "42", "s123": "123", "s456": "456"}
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                seed_num = seed_to_num[seed]
                logf = log_dir / f"{fam}_{model}_{reg}_s{seed_num}.log"
                if not logf.exists():
                    continue
                m = _LOG_PAT.search(logf.read_text(errors="replace"))
                if m:
                    out[(fam, reg, seed)] = float(m.group(1))
    return out


def load_pod3_baseline(model: str) -> dict:
    out = {}
    pod3 = EXT / "pod3" / "outputs" / "final_baselines" / "raw"
    if not pod3.exists():
        for alt in (EXT / "pod3", EXT / "pod3_phaseA"):
            cand = alt / "outputs" / "final_baselines" / "raw"
            if cand.exists():
                pod3 = cand
                break
    if not pod3.exists():
        return out
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = pod3 / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d is not None:
                    out[(fam, reg, seed)] = float(d.get("test_rel_l2_mean",
                                                        d.get("test_rel_l2", float("nan"))))
    return out


def gather_all_models() -> dict:
    data = {
        "lemo_pc_nd": load_lemo_test("lemo_pc_nd"),
        "lemo_nd":    load_lemo_test("lemo_nd"),
        "fno_nd":         load_baseline_from_log("fno_nd"),
        "markov_fno_nd":  load_baseline_from_log("markov_fno_nd"),
        "windowed_fno_nd": load_baseline_from_log("windowed_fno_nd"),
        "memno_nd":       load_pod3_baseline("memno_nd"),
        "ffno_nd":        load_pod3_baseline("ffno_nd"),
    }
    return {m: d for m, d in data.items() if len(d) >= 1}


def get_params(model: str) -> int:
    """Try to read params from a representative test_results.json."""
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / model / seed / "test_results.json"
                d = _try_json(p)
                if d and "params" in d:
                    return int(d["params"])
    pod3 = EXT / "pod3" / "outputs" / "final_baselines" / "raw"
    if pod3.exists():
        for fam in FAMS:
            for reg in REGIMES:
                for seed in SEEDS:
                    p = pod3 / fam / reg / model / seed / "test_results.json"
                    d = _try_json(p)
                    if d and "params" in d:
                        return int(d["params"])
    return 0


# -------------------- tables --------------------

def _wrap_table(body_lines, caption, label, position="h"):
    return ([f"\\begin{{table}}[{position}]",
             r"\centering",
             f"\\caption{{{caption}}}",
             f"\\label{{{label}}}",
            ] + body_lines +
            [r"\end{table}"])


def t01_headline(stats: dict):
    """Headline: % impr / Hedges g / p-value / CI per baseline comparison."""
    if not stats:
        return None
    keys = [k for k in ("FNO", "MarkovFNO", "WindFNO", "UNet", "LEMO_ND_ablation")
            if k in stats]
    if not keys:
        return None
    label_map = {"FNO": "FNO", "MarkovFNO": "Markov-FNO",
                 "WindFNO": "Window-FNO", "UNet": "UNet",
                 "LEMO_ND_ablation": "LEMO (no FiLM)"}
    rows = []
    for k in keys:
        a = stats[k]["aggregate"]
        impr = a["improvement_ratio_mean_pct"]
        ci = a["improvement_95ci_pct"]
        g = a.get("hedges_g", 0.0)
        p = a["paired_permutation_p"]
        p_str = "$<10^{-4}$" if p < 1e-4 else f"${p:.1e}$"
        rows.append((label_map[k], impr, ci, g, p_str, a["n_paired_cells"]))
    body = ["\\begin{tabular}{lccccr}",
            r"\toprule",
            r"Comparison & \% improvement & 95\% CI & Hedges $g$ & $p$-value & $n$ \\",
            r"\midrule"]
    for label, impr, ci, g, p, n in rows:
        body.append(f"vs.\\ {label} & {impr:.1f} & [{ci[0]:.1f}, {ci[1]:.1f}] & "
                    f"{g:.2f} & {p} & {n} \\\\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = _wrap_table(body,
                    caption=("LEMO-PC vs.\\ each baseline: percent improvement "
                             "in test rel-$L_2$, paired-permutation $p$-value, "
                             "Hedges $g$, and bootstrap 95\\% CI across "
                             "$n=45$ paired cells."),
                    label="tab:headline-per-baseline")
    out = TABLE_DIR / "T01_headline_per_baseline.tex"
    out.write_text("\n".join(s))
    return out


def t02_perfamily_clean(data: dict):
    """5 fams x N models test relL2 (clean regime, mean ± std over 3 seeds)."""
    models = [m for m in MODEL_ORDER if m in data]
    if not models:
        return None
    body = ["\\begin{tabular}{l" + "c" * len(models) + "}",
            r"\toprule",
            "Family & " + " & ".join(MODEL_LABELS[m] for m in models) + r" \\",
            r"\midrule"]
    for fam in FAMS:
        cells = []
        for mdl in models:
            vals = [data[mdl].get((fam, "clean", sd), np.nan) for sd in SEEDS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                m, sd = np.mean(vals), np.std(vals)
                cells.append(f"{m:.4f} $\\pm$ {sd:.4f}")
            else:
                cells.append("--")
        body.append(f"{FAM_LABELS[fam]} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = _wrap_table(body,
                    caption=("Per-family test rel-$L_2$ on the clean regime "
                             "(mean $\\pm$ std over 3 seeds)."),
                    label="tab:perfamily")
    out = TABLE_DIR / "T02_perfamily_relL2.tex"
    out.write_text("\n".join(s))
    return out


def t03_perregime(data: dict):
    """3 regimes x N models, mean ± std aggregated across 5 fams x 3 seeds."""
    models = [m for m in MODEL_ORDER if m in data]
    if not models:
        return None
    body = ["\\begin{tabular}{l" + "c" * len(models) + "}",
            r"\toprule",
            "Regime & " + " & ".join(MODEL_LABELS[m] for m in models) + r" \\",
            r"\midrule"]
    for reg in REGIMES:
        cells = []
        for mdl in models:
            vals = [data[mdl].get((fam, reg, sd), np.nan)
                    for fam in FAMS for sd in SEEDS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                m, sd = np.mean(vals), np.std(vals)
                cells.append(f"{m:.4f} $\\pm$ {sd:.4f}")
            else:
                cells.append("--")
        body.append(f"{reg} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = _wrap_table(body,
                    caption=("Per-regime aggregate test rel-$L_2$ "
                             "(mean $\\pm$ std across 5 families $\\times$ 3 seeds)."),
                    label="tab:perregime")
    out = TABLE_DIR / "T03_perregime_aggregated.tex"
    out.write_text("\n".join(s))
    return out


def t04_compute_costs(data: dict):
    """params, train wallclock per model (best-of-each config)."""
    models = [m for m in MODEL_ORDER if m in data]
    if not models:
        return None
    base = EXT / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw"
    pod3 = EXT / "pod3" / "outputs" / "final_baselines" / "raw"
    body = [r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Model & Params & Train wallclock (s) \\",
            r"\midrule"]
    for mdl in models:
        params = get_params(mdl)
        wall_secs = []
        for src in (base, pod3):
            if not src.exists():
                continue
            for fam in FAMS:
                for reg in REGIMES:
                    for seed in SEEDS:
                        p = src / fam / reg / mdl / seed / "test_results.json"
                        d = _try_json(p)
                        if d and "wall_seconds" in d:
                            wall_secs.append(d["wall_seconds"])
        wall_str = f"{np.mean(wall_secs):.0f} $\\pm$ {np.std(wall_secs):.0f}" if wall_secs else "--"
        param_str = f"{params/1e6:.2f}M" if params > 0 else "--"
        body.append(f"{MODEL_LABELS[mdl]} & {param_str} & {wall_str} \\\\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = _wrap_table(body,
                    caption=("Compute costs: trainable parameter counts and "
                             "per-cell wall-clock training time "
                             "(mean $\\pm$ std across cells)."),
                    label="tab:compute-costs")
    out = TABLE_DIR / "T04_compute_costs.tex"
    out.write_text("\n".join(s))
    return out


def t05_perregime_perbaseline(stats: dict):
    """For each baseline, % impr in each regime (clean / lowres / noisy)."""
    if not stats:
        return None
    keys = [k for k in ("FNO", "MarkovFNO", "WindFNO", "UNet", "LEMO_ND_ablation")
            if k in stats and "per_regime" in stats[k]]
    if not keys:
        return None
    label_map = {"FNO": "FNO", "MarkovFNO": "Markov-FNO",
                 "WindFNO": "Window-FNO", "UNet": "UNet",
                 "LEMO_ND_ablation": "LEMO (no FiLM)"}
    body = [r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Baseline & clean & lowres & noisy \\",
            r"\midrule"]
    for k in keys:
        per_reg = stats[k]["per_regime"]
        cells = []
        for reg in REGIMES:
            r = per_reg.get(reg, {})
            if "improvement_pct_mean" in r:
                m = r["improvement_pct_mean"]
                ci = r.get("improvement_95ci_pct", [m, m])
                cells.append(f"{m:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]")
            else:
                cells.append("--")
        body.append(f"vs.\\ {label_map[k]} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = _wrap_table(body,
                    caption=("Percent improvement of LEMO-PC over each "
                             "baseline, broken down by regime "
                             "(mean with bootstrap 95\\% CI in brackets, "
                             "$n=15$ pairs per cell)."),
                    label="tab:perbaselineperregime")
    out = TABLE_DIR / "T05_per_baseline_per_regime_breakdown.tex"
    out.write_text("\n".join(s))
    return out


def t08_single_delay():
    """Single-delay 2D leaderboard: 3 families x 3 regimes x 5 models
    (UNet excluded from headline; LEMO ablation kept).  Best entry per row
    is bolded.  Replaces the C30 heatmap."""
    base_p1 = EXT / "pod1" / "outputs" / "layer5_final_sweep_p1" / "raw"
    base_p2 = EXT / "pod2" / "outputs" / "layer5_final_sweep_p2" / "raw"
    sd_fams = ["mackey_glass_2d", "wright_2d", "hutchinson_2d"]
    sd_labels = {"mackey_glass_2d": "Mackey-Glass", "wright_2d": "Wright",
                  "hutchinson_2d": "Hutchinson"}
    models = ["lemo_pc_nd", "lemo_nd", "fno_nd", "markov_fno_nd",
              "windowed_fno_nd"]
    if not (base_p1.exists() or base_p2.exists()):
        return None
    rows_data = []
    for fam in sd_fams:
        for reg in REGIMES:
            row_vals = []
            for mdl in models:
                vals = []
                for base in (base_p1, base_p2):
                    if not base.exists():
                        continue
                    for seed in SEEDS:
                        p = base / fam / reg / mdl / seed / "test_results.json"
                        d = _try_json(p)
                        if d is not None:
                            r = d.get("test_rel_l2_mean", d.get("test_rel_l2"))
                            if r is not None:
                                vals.append(float(r))
                row_vals.append(float(np.mean(vals)) if vals else None)
            rows_data.append((fam, reg, row_vals))
    if not rows_data:
        return None
    body = ["\\begin{tabular}{ll" + "c" * len(models) + "}",
            r"\toprule",
            r"Family & Regime & " + " & ".join(MODEL_LABELS[m].replace(r"\textbf{", "").replace("}", "") for m in models) + r" \\",
            r"\midrule"]
    for fam, reg, vals in rows_data:
        finite = [v for v in vals if v is not None]
        if not finite:
            continue
        best = min(finite)
        cells = []
        for v in vals:
            if v is None:
                cells.append("--")
            elif abs(v - best) < 1e-9:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        body.append(f"{sd_labels[fam]} & {reg} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule",
             r"\end{tabular}"]
    s = _wrap_table(body,
                    caption=("Single-delay 2D test rel-$L_2$ across "
                             "Mackey-Glass, Wright, Hutchinson families "
                             "(mean over 3 seeds).  Best per row in bold.  "
                             "UNet excluded (not delay-aware)."),
                    label="tab:single-delay")
    out = TABLE_DIR / "T08_single_delay.tex"
    out.write_text("\n".join(s))
    return out


def main():
    print(f"[paper-tables] working dir: {REPO}")
    stats = _try_json(STATS_PATH) or {}
    data = gather_all_models()
    print("\n[paper-tables] data inventory")
    print(f"  paired_permutation.json: {'OK' if stats else 'MISSING'}")
    for m in MODEL_ORDER:
        print(f"  {MODEL_LABELS.get(m, m).replace(chr(92)+'textbf','').replace('{','').replace('}',''):<14}: {len(data.get(m, {})):>4} cells")

    out_files = []
    for name, fn, args in [
        ("T01 headline",       t01_headline,           (stats,)),
        ("T02 per-family",     t02_perfamily_clean,    (data,)),
        ("T03 per-regime",     t03_perregime,          (data,)),
        ("T04 compute costs",  t04_compute_costs,      (data,)),
        ("T05 per-regime impr", t05_perregime_perbaseline, (stats,)),
        ("T08 single-delay",   t08_single_delay,       ()),
    ]:
        try:
            out = fn(*args)
        except Exception as e:
            print(f"  {name:<22}: FAIL ({type(e).__name__}: {e})")
            continue
        if out is None:
            print(f"  {name:<22}: skip (data missing)")
        else:
            out_files.append(out)
            print(f"  {name:<22}: -> {out.name}")
    print(f"\n[paper-tables] generated {len(out_files)} tables in {TABLE_DIR}")
    return out_files


if __name__ == "__main__":
    main()
