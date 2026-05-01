"""Paired-permutation analysis on the v2 dist-kernel sweep.

Reads:
  - LEMO test_results.json from extracted/pod1/.../dist_kernel_v2_p1/raw/.../
  - Baseline FINAL relL2 from extracted/pod2/.../dist_kernel_v2_p2/logs/*.log

Performs for each baseline:
  - Aggregate paired permutation test (n=45 paired cells: 5 fams x 3 regimes x 3 seeds)
  - Bootstrap 95% CI on improvement ratio
  - Per-regime breakdown (clean / lowres / noisy)

Outputs:
  - paper/stats/paired_permutation.json with full results
  - paper/stats/paired_permutation.md with publication-ready text
"""
import json
import re
from pathlib import Path
import numpy as np
from itertools import combinations

REPO = Path("A:/dde research/dde-fno")
POD1 = REPO / "extracted/pod1"
POD2 = REPO / "extracted/pod2"
OUT_DIR = REPO / "paper/stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
REGIMES = ["clean", "lowres", "noisy"]
SEEDS = ["s42", "s123", "s456"]


def load_lemo_pc():
    """LEMO_PC numbers from pod1 dist_kernel_v2_p1 test_results.json."""
    out = {}
    base = POD1 / "outputs/dist_kernel_v2_p1/raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / "lemo_pc_nd" / seed / "test_results.json"
                if p.exists():
                    d = json.loads(p.read_text())
                    out[(fam, reg, seed)] = d["test_rel_l2_mean"]
    return out


def load_lemo_nd():
    """LEMO_ND numbers from pod1 dist_kernel_v2_p1 test_results.json."""
    out = {}
    base = POD1 / "outputs/dist_kernel_v2_p1/raw"
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                p = base / fam / reg / "lemo_nd" / seed / "test_results.json"
                if p.exists():
                    d = json.loads(p.read_text())
                    out[(fam, reg, seed)] = d["test_rel_l2_mean"]
    return out


def load_baseline_from_log(model_name):
    """Baseline numbers from pod2 logs (FINAL test relL2 line)."""
    out = {}
    log_dir = POD2 / "outputs/dist_kernel_v2_p2/logs"
    pat = re.compile(r"FINAL test relL2 = ([0-9.]+)")
    for fam in FAMS:
        for reg in REGIMES:
            for seed in SEEDS:
                seed_num = seed[1:]
                logf = log_dir / f"{fam}_{model_name}_{reg}_s{seed_num}.log"
                if not logf.exists():
                    continue
                m = pat.search(logf.read_text())
                if m:
                    out[(fam, reg, seed)] = float(m.group(1))
    return out


def paired_permutation_test(a, b, n_iter=10000):
    """Two-sided paired permutation test on (b - a)/b improvement.
    a = LEMO, b = baseline.  Larger = LEMO worse. Improvement ratio (b-a)/b."""
    a = np.array(a)
    b = np.array(b)
    diffs = b - a  # positive = LEMO better
    obs_mean = float(diffs.mean())
    n = len(diffs)
    # Permutation: randomly flip signs of paired diffs
    rng = np.random.default_rng(42)
    counts = 0
    for _ in range(n_iter):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = (signs * diffs).mean()
        if abs(perm_mean) >= abs(obs_mean):
            counts += 1
    p = (counts + 1) / (n_iter + 1)  # +1 to avoid 0
    return obs_mean, p


def bootstrap_improvement(a, b, n_boot=10000):
    """Bootstrap 95% CI on improvement ratio (b - a)/b * 100."""
    a = np.array(a)
    b = np.array(b)
    n = len(a)
    rng = np.random.default_rng(42)
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_b = a[idx]
        b_b = b[idx]
        if b_b.sum() > 1e-12:
            ratios.append(100 * (b_b.mean() - a_b.mean()) / b_b.mean())
    ratios = np.array(ratios)
    return float(ratios.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def cohen_d(a, b):
    a = np.array(a); b = np.array(b)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled < 1e-12: return float("inf")
    return (b.mean() - a.mean()) / pooled


def hedges_g(a, b):
    """Hedges g (small-N corrected Cohen's d)."""
    n = len(a) + len(b)
    if n <= 2: return float("nan")
    correction = 1 - 3 / (4 * n - 9)
    return cohen_d(a, b) * correction


def analyze(lemo, baseline, label):
    """Pair LEMO vs baseline cells matched by (fam, reg, seed)."""
    keys = sorted(set(lemo.keys()) & set(baseline.keys()))
    if not keys:
        return None
    a = [lemo[k] for k in keys]
    b = [baseline[k] for k in keys]
    obs, p = paired_permutation_test(a, b, n_iter=10000)
    mean_imp, lo, hi = bootstrap_improvement(a, b, n_boot=10000)
    return {
        "label": label,
        "n_paired_cells": len(keys),
        "lemo_mean": float(np.mean(a)),
        "baseline_mean": float(np.mean(b)),
        "improvement_ratio_mean_pct": mean_imp,
        "improvement_95ci_pct": [lo, hi],
        "paired_permutation_p": p,
        "abs_diff_mean": obs,
        "cohen_d": cohen_d(a, b),
        "hedges_g": hedges_g(a, b),
    }


def per_regime_analysis(lemo, baseline, label):
    """Break down by regime."""
    out = {}
    for reg in REGIMES:
        keys = sorted([k for k in lemo if k[1] == reg and k in baseline])
        if not keys:
            continue
        a = [lemo[k] for k in keys]
        b = [baseline[k] for k in keys]
        obs, p = paired_permutation_test(a, b, n_iter=10000)
        mi, lo, hi = bootstrap_improvement(a, b, n_boot=10000)
        out[reg] = {
            "n_pairs": len(keys),
            "lemo_mean": float(np.mean(a)),
            "baseline_mean": float(np.mean(b)),
            "improvement_pct_mean": mi,
            "improvement_95ci_pct": [lo, hi],
            "p_value": p,
        }
    return out


def main():
    lemo_pc = load_lemo_pc()
    lemo_nd = load_lemo_nd()
    fno = load_baseline_from_log("fno_nd")
    markov = load_baseline_from_log("markov_fno_nd")
    wind = load_baseline_from_log("windowed_fno_nd")
    unet = load_baseline_from_log("unet_nd")

    print(f"Loaded: LEMO_PC={len(lemo_pc)}, LEMO_ND={len(lemo_nd)}")
    print(f"        FNO={len(fno)}, MarkovFNO={len(markov)}, WindFNO={len(wind)}, UNet={len(unet)}")
    print()

    results = {}
    for name, base in [("FNO", fno), ("MarkovFNO", markov), ("WindFNO", wind),
                         ("UNet", unet), ("LEMO_ND_ablation", lemo_nd)]:
        if not base:
            continue
        agg = analyze(lemo_pc, base, f"LEMO_PC vs {name}")
        per_reg = per_regime_analysis(lemo_pc, base, name)
        results[name] = {"aggregate": agg, "per_regime": per_reg}

    # Holm-Bonferroni correction across the 4 main baselines (not the LEMO_ND ablation):
    main_p = [(n, results[n]["aggregate"]["paired_permutation_p"])
                for n in ["FNO", "MarkovFNO", "WindFNO", "UNet"] if n in results]
    main_p_sorted = sorted(main_p, key=lambda x: x[1])
    holm_results = {}
    K = len(main_p_sorted)
    for i, (name, p) in enumerate(main_p_sorted):
        # Holm: significant if p_(i) < alpha / (K - i)
        holm_results[name] = {
            "raw_p": p,
            "holm_threshold_at_alpha_0.05": 0.05 / (K - i),
            "significant_holm_0.05": p < 0.05 / (K - i),
        }
    results["holm_bonferroni"] = holm_results

    # Save JSON
    out_json = OUT_DIR / "paired_permutation.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")

    # Generate publication-ready markdown
    md_lines = ["# Paired-permutation analysis — LEMO_PC vs baselines\n",
                "**Method:** Paired-sample permutation test (10,000 iterations) on per-cell "
                "test relL2, paired by (family, regime, seed). Bootstrap 95% CI on improvement "
                "ratio (10,000 resamples).  Holm-Bonferroni correction across 4 main "
                "baselines.\n",
                "## Aggregate results (all 5 fams × 3 regimes × 3 seeds = up to 45 paired cells)\n",
                "| baseline | n_pairs | LEMO_PC mean | baseline mean | improvement % | 95% CI | p (perm) | Holm p< | Hedges g |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["FNO", "MarkovFNO", "WindFNO", "UNet", "LEMO_ND_ablation"]:
        if name not in results:
            continue
        a = results[name]["aggregate"]
        if a is None:
            continue
        holm_threshold = results.get("holm_bonferroni", {}).get(name, {}).get("holm_threshold_at_alpha_0.05", "N/A")
        md_lines.append(
            f"| {name} | {a['n_paired_cells']} | {a['lemo_mean']:.4f} | "
            f"{a['baseline_mean']:.4f} | {a['improvement_ratio_mean_pct']:.1f}% | "
            f"[{a['improvement_95ci_pct'][0]:.1f}, {a['improvement_95ci_pct'][1]:.1f}] | "
            f"{a['paired_permutation_p']:.4g} | {holm_threshold if holm_threshold == 'N/A' else f'{holm_threshold:.4f}'} | "
            f"{a['hedges_g']:.2f} |"
        )

    md_lines.append("\n## Per-regime breakdown\n")
    for name in ["FNO", "MarkovFNO", "WindFNO", "UNet", "LEMO_ND_ablation"]:
        if name not in results: continue
        md_lines.append(f"### vs {name}")
        md_lines.append("| regime | n_pairs | LEMO mean | baseline mean | improv. % | 95% CI | p |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for reg in REGIMES:
            r = results[name]["per_regime"].get(reg)
            if r is None: continue
            md_lines.append(
                f"| {reg} | {r['n_pairs']} | {r['lemo_mean']:.4f} | "
                f"{r['baseline_mean']:.4f} | {r['improvement_pct_mean']:.1f}% | "
                f"[{r['improvement_95ci_pct'][0]:.1f}, {r['improvement_95ci_pct'][1]:.1f}] | "
                f"{r['p_value']:.4g} |"
            )
        md_lines.append("")

    out_md = OUT_DIR / "paired_permutation.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Wrote {out_md}")
    print()

    # Print headline numbers to stdout
    print("=== HEADLINE ===")
    for name in ["FNO", "MarkovFNO", "WindFNO", "UNet"]:
        if name not in results: continue
        a = results[name]["aggregate"]
        if a is None: continue
        print(f"  LEMO_PC vs {name:12s}: improvement {a['improvement_ratio_mean_pct']:5.1f}% "
              f"[CI {a['improvement_95ci_pct'][0]:.1f}, {a['improvement_95ci_pct'][1]:.1f}], "
              f"p_perm={a['paired_permutation_p']:.4g}, n={a['n_paired_cells']}")


if __name__ == "__main__":
    main()
