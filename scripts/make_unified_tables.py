"""Unified table regeneration from new pod_pulls + W11 data.

Crawls ALL extracted/pod_pulls_2026_05_03_final/* roots + outputs/w11_*
and regenerates T01-T05 tables with all available baselines.

Outputs:
  - paper/stats/paired_permutation_v2.json   (new paired-permutation aggregates)
  - paper/tables/T01_headline_per_baseline.tex   (with Hedges-g column)
  - paper/tables/T02_perfamily_relL2.tex
  - paper/tables/T03_perregime_aggregated.tex
  - paper/tables/T04_compute_costs.tex
  - paper/tables/T05_per_baseline_per_regime_breakdown.tex

Usage:
  python scripts/make_unified_tables.py
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from itertools import combinations

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "extracted" / "pod_pulls_2026_05_03_final"
W11_OUT = REPO / "outputs" / "w11_compute_matched_runpod"
NEURIPS = REPO.parent / "NeurIPS_LEMO"
TABLE_DIR = NEURIPS / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR = NEURIPS / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}
REGIMES = ["clean", "lowres", "noisy"]
SEEDS = [42, 123, 456]

MODEL_LABELS = {
    "lemo_pc_nd":              r"\textbf{LEMO-PC}",
    "causal_smooth_lemo_pc_nd": r"LEMO-PC (causal smooth)",
    "lemo_nd":                 "LEMO",
    "lemo_bcorrect_nd":        "LEMO + B-correct",
    "fno_nd":                  "FNO",
    "fno_film_nd":             "FNO + FiLM",
    "noneq_film_nd":           "Non-equiv + FiLM",
    "markov_fno_nd":           "Markov-FNO",
    "windowed_fno_nd":         "Window-FNO",
    "memno_nd":                "MemNO",
    "ffno_nd":                 "F-FNO",
    "unet_nd":                 "UNet",
    "nide_nd":                 "NIDE",
    "ndde_nd":                 "NDDE",
    "s4_nd":                   "S4",
    "per_lag_mlp_nd":          "Per-lag MLP",
}
MODEL_ORDER = [
    "lemo_pc_nd", "causal_smooth_lemo_pc_nd",
    "ndde_nd", "nide_nd", "s4_nd",
    "fno_film_nd", "fno_nd", "memno_nd", "ffno_nd",
    "noneq_film_nd", "lemo_bcorrect_nd",
]


def crawl_all() -> dict:
    """Return dict[model][(fam, reg, seed)] = test_rel_l2."""
    data = defaultdict(dict)
    crawl_roots = []
    if EXT.exists():
        crawl_roots.extend(EXT.glob("*/outputs/*"))
    if W11_OUT.exists():
        crawl_roots.append(W11_OUT)
    for root in crawl_roots:
        if not root.is_dir():
            continue
        for tr in root.glob("**/test_results.json"):
            parts = tr.parts
            if "raw" not in parts:
                continue
            idx = parts.index("raw")
            if idx + 4 >= len(parts):
                continue
            fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
            if not seed_str.startswith("s"):
                continue
            seed = int(seed_str[1:])
            try:
                d = json.loads(tr.read_text())
            except Exception:
                continue
            err = d.get("test_rel_l2_mean") or d.get("test_rel_l2")
            if err is None:
                continue
            # Normalize family name (orbit appends _orbit)
            fam_clean = fam.replace("_orbit", "")
            data[mdl][(fam_clean, reg, seed)] = float(err)
    return dict(data)


def get_params_walltime(model: str) -> tuple[int, float, int]:
    """Return (median_params, median_wall_seconds, n_cells) across all available cells."""
    all_params = []
    all_wall = []
    crawl_roots = list(EXT.glob("*/outputs/*")) + ([W11_OUT] if W11_OUT.exists() else [])
    for root in crawl_roots:
        for tr in root.glob(f"**/raw/**/{model}/**/test_results.json"):
            try:
                d = json.loads(tr.read_text())
            except Exception:
                continue
            p = d.get("params") or d.get("n_params")
            w = d.get("wall_seconds") or d.get("wall_clock_s")
            if p:
                all_params.append(int(p))
            if w:
                all_wall.append(float(w))
    if not all_params:
        return 0, 0.0, 0
    return (int(np.median(all_params)),
            float(np.median(all_wall)) if all_wall else 0.0,
            len(all_params))


def paired_permutation(lemo_dict, baseline_dict, n_perms=10000, seed=0):
    """Compute paired permutation test: improvement ratio + Hedges g + p-value."""
    keys = sorted(set(lemo_dict.keys()) & set(baseline_dict.keys()))
    if not keys:
        return None
    lemo = np.array([lemo_dict[k] for k in keys])
    base = np.array([baseline_dict[k] for k in keys])
    diff = base - lemo
    impr_pct = 100.0 * diff / np.where(base != 0, base, 1)
    rng = np.random.default_rng(seed)
    n = len(diff)
    # paired permutation: randomly flip signs of diff
    null_means = np.zeros(n_perms)
    for i in range(n_perms):
        signs = rng.choice([-1, 1], size=n)
        null_means[i] = (signs * diff).mean()
    obs = diff.mean()
    p_value = (np.abs(null_means) >= abs(obs)).mean()
    # Hedges g (paired): mean(diff) / std(diff) * (1 - 3/(4n - 5))
    g = (diff.mean() / diff.std(ddof=1)) * (1 - 3.0 / max(4 * n - 5, 1)) if n > 1 else 0.0
    # 95% bootstrap CI on improvement_pct
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        boot.append(impr_pct[idx].mean())
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {
        "n_paired_cells": n,
        "lemo_mean": float(lemo.mean()),
        "baseline_mean": float(base.mean()),
        "improvement_ratio_mean_pct": float(impr_pct.mean()),
        "improvement_95ci_pct": list(ci),
        "abs_diff_mean": float(diff.mean()),
        "hedges_g": float(g),
        "paired_permutation_p": float(p_value),
        "n_perms": n_perms,
    }


def t01_headline(stats: dict):
    if not stats:
        return None
    keys = [k for k in MODEL_ORDER[1:] if k in stats]
    rows = []
    for k in keys:
        a = stats[k]
        impr = a["improvement_ratio_mean_pct"]
        ci = a["improvement_95ci_pct"]
        g = a["hedges_g"]
        p = a["paired_permutation_p"]
        n = a["n_paired_cells"]
        p_str = "$<10^{-4}$" if p < 1e-4 else f"${p:.1e}$"
        label = MODEL_LABELS.get(k, k)
        rows.append(f"{label} & {impr:.1f}\\% [{ci[0]:.1f}, {ci[1]:.1f}] & {g:.2f} & {p_str} & {n} \\\\")

    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Headline LEMO-PC vs each baseline: paired-permutation test improvement, Hedges $g$, p-value, $n$ paired cells.}",
        r"\label{tab:headline-per-baseline}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparison & \% Impr [95\% CI] & Hedges $g$ & $p$ & $n$ \\",
        r"\midrule",
        *["LEMO-PC vs " + r for r in rows],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def t02_perfamily(data: dict):
    """rows = models, cols = families (clean regime)"""
    avail = [m for m in MODEL_ORDER if m in data]
    if not avail:
        return None
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Per-family test rel-$L_2$ on the distributed-kernel sweep (clean regime). Mean $\pm$ std across 3 seeds. Best per family in bold.}",
        r"\label{tab:perfamily-relL2}",
        r"\begin{tabular}{l" + "c" * len(FAMS) + r"}",
        r"\toprule",
        "Model & " + " & ".join(FAM_LABELS[f] for f in FAMS) + r" \\",
        r"\midrule",
    ]
    # Find per-family minima
    minima = {}
    for fam in FAMS:
        vals = []
        for m in avail:
            cells = [v for (f, r, s), v in data[m].items() if f == fam and r == "clean"]
            if cells:
                vals.append((m, np.mean(cells)))
        if vals:
            minima[fam] = min(vals, key=lambda t: t[1])[0]
    for m in avail:
        cells_per_fam = []
        for fam in FAMS:
            cells = [v for (f, r, s), v in data[m].items() if f == fam and r == "clean"]
            if cells:
                mean = np.mean(cells)
                std = np.std(cells)
                bold = (m == minima.get(fam))
                txt = f"{mean:.4f} $\\pm$ {std:.4f}"
                if bold:
                    txt = r"\textbf{" + txt + r"}"
                cells_per_fam.append(txt)
            else:
                cells_per_fam.append("--")
        lines.append(f"{MODEL_LABELS.get(m, m)} & " + " & ".join(cells_per_fam) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def t03_perregime(data: dict):
    """rows = models, cols = regimes (avg across all families)"""
    avail = [m for m in MODEL_ORDER if m in data]
    if not avail:
        return None
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Per-regime test rel-$L_2$ aggregated over 5 families × 3 seeds (n=15 per cell). Best per regime in bold.}",
        r"\label{tab:perregime-aggregated}",
        r"\begin{tabular}{l" + "c" * len(REGIMES) + r"}",
        r"\toprule",
        "Model & " + " & ".join(r.title() for r in REGIMES) + r" \\",
        r"\midrule",
    ]
    minima = {}
    for reg in REGIMES:
        vals = []
        for m in avail:
            cells = [v for (f, r, s), v in data[m].items() if r == reg]
            if cells:
                vals.append((m, np.mean(cells)))
        if vals:
            minima[reg] = min(vals, key=lambda t: t[1])[0]
    for m in avail:
        cells_per_reg = []
        for reg in REGIMES:
            cells = [v for (f, r, s), v in data[m].items() if r == reg]
            if cells:
                mean = np.mean(cells)
                std = np.std(cells)
                bold = (m == minima.get(reg))
                txt = f"{mean:.4f} $\\pm$ {std:.4f}"
                if bold:
                    txt = r"\textbf{" + txt + r"}"
                cells_per_reg.append(txt)
            else:
                cells_per_reg.append("--")
        lines.append(f"{MODEL_LABELS.get(m, m)} & " + " & ".join(cells_per_reg) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def t04_compute(data: dict):
    avail = [m for m in MODEL_ORDER if m in data]
    if not avail:
        return None
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Compute costs across architectures: parameter count, training wall-clock per cell (200 epochs unless noted), $n$ cells contributing.}",
        r"\label{tab:compute-costs}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Params (M) & Train wallclock (s) & $n$ \\",
        r"\midrule",
    ]
    for m in avail:
        params, wall, n = get_params_walltime(m)
        lines.append(f"{MODEL_LABELS.get(m, m)} & {params/1e6:.2f}M & {wall:.0f} & {n} \\\\")
    # Add W11 row if present
    if "fno_nd" in data:
        # Try to find W11-trained fno_nd
        w11_results = [v for k, v in data["fno_nd"].items() if "w11" in str(k)]
        if w11_results:
            mean_err = np.mean(w11_results)
            lines.append(f"\\midrule")
            lines.append(f"FNO @ 400 epochs (W11 compute-matched) & --- & --- & {len(w11_results)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_table(name: str, body: str | None):
    if body is None:
        print(f"  {name}: SKIPPED (no data)")
        return
    out = TABLE_DIR / f"{name}.tex"
    out.write_text(body)
    print(f"  {name}: {out}")


def main():
    print("[unified-tables] crawling extracted + W11...")
    data = crawl_all()
    print(f"  Found {len(data)} models:")
    for m in MODEL_ORDER:
        if m in data:
            print(f"    {m:30s}: {len(data[m]):>4d} cells")
    others = [m for m in data if m not in MODEL_ORDER]
    if others:
        print(f"  Other models: {others}")

    # Compute paired_permutation against LEMO-PC
    if "lemo_pc_nd" not in data:
        print("[unified-tables] no LEMO-PC data, skipping paired tests")
        stats = {}
    else:
        print("[unified-tables] computing paired permutation tests...")
        stats = {}
        lemo = data["lemo_pc_nd"]
        for m in MODEL_ORDER:
            if m == "lemo_pc_nd" or m not in data:
                continue
            stat = paired_permutation(lemo, data[m])
            if stat is not None:
                stats[m] = stat
                print(f"    LEMO-PC vs {m}: impr={stat['improvement_ratio_mean_pct']:.1f}%  "
                      f"g={stat['hedges_g']:.2f}  p={stat['paired_permutation_p']:.1e}  "
                      f"n={stat['n_paired_cells']}")
        # Save raw stats
        (STATS_DIR / "paired_permutation_v2.json").write_text(json.dumps(stats, indent=2))

    # Tables
    print("[unified-tables] writing tables...")
    write_table("T01_headline_per_baseline", t01_headline(stats))
    write_table("T02_perfamily_relL2", t02_perfamily(data))
    write_table("T03_perregime_aggregated", t03_perregime(data))
    write_table("T04_compute_costs", t04_compute(data))


if __name__ == "__main__":
    main()
