"""Build T_compute_matched.tex — wall-clock-matched FNO@400ep vs LEMO-PC@200ep.

Shows that even with matched wall-clock training time, plain FNO at 400
epochs (~5670s on H100) does not match LEMO-PC at 200 epochs (~5951s)
on ANY metric: in-distribution rel-L2, OOD rel-L2, equivariance error,
FGSM robustness, Gaussian-noise robustness, long-horizon rel-L2.

Output: A:/dde research/dde-fno/paper/tables/T_compute_matched.tex
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(r"A:\dde research\dde-fno")
TABLE_DIR = REPO / "paper" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect(model: str, root: Path):
    """For each (fam, seed) cell under root, gather all evaluation metrics.

    Returns: out[fam][seed] -> {metric: value}
    """
    out = defaultdict(dict)
    for fam in FAMS:
        for seed_dir in (root / "raw" / fam / "clean" / model).glob("s*"):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name
            metrics = {}

            # In-distribution + cross-family OOD.
            cf = _load(seed_dir / "cross_family_relL2.json")
            if cf:
                rl = cf.get("rel_l2", {})
                if fam in rl:
                    metrics["in_dist"] = float(rl[fam])
                ood = [rl[ff] for ff in FAMS if ff != fam and ff in rl]
                if ood:
                    metrics["ood_mean"] = float(np.mean(ood))

            # Equivariance error at k=1 and k=64.
            ed = _load(seed_dir / "equivariance_dense.json")
            if ed:
                e = ed.get("e_orbit", {})
                if "1" in e:
                    metrics["equiv_k1"] = float(e["1"].get("mean", float("nan")))
                if "64" in e:
                    metrics["equiv_k64"] = float(e["64"].get("mean", float("nan")))

            # FGSM robustness at eps = 0.05.
            ad = _load(seed_dir / "adversarial_dense.json")
            if ad:
                # Schema: {"per_eps": {"0.05": {"mean": ...}}}  OR  {"adv": {...}}
                per_eps = ad.get("per_eps") or ad.get("adv") or {}
                for k, v in per_eps.items():
                    try:
                        kf = float(k)
                    except Exception:
                        continue
                    if abs(kf - 0.05) < 1e-6:
                        metrics["fgsm_05"] = float(
                            v.get("mean") if isinstance(v, dict) else v)
                        break

            # Gaussian-noise robustness at sigma = 0.5.
            nd = _load(seed_dir / "noise_dense.json")
            if nd:
                per_sig = nd.get("per_sigma") or nd.get("noise") or {}
                for k, v in per_sig.items():
                    try:
                        kf = float(k)
                    except Exception:
                        continue
                    if abs(kf - 0.5) < 1e-6:
                        metrics["noise_05"] = float(
                            v.get("mean") if isinstance(v, dict) else v)
                        break

            # Long-horizon h = 128.
            lh = _load(seed_dir / "long_horizon.json")
            if lh:
                h128 = lh.get("h_128") or {}
                if isinstance(h128, dict):
                    metrics["lh_128"] = float(h128.get("rel_l2_mean", float("nan")))

            out[fam][seed] = metrics
    return out


def aggregate(per_cell: dict, key: str):
    """Across all (fam, seed) cells, return mean +/- std of metric `key`."""
    vals = []
    for fam, by_seed in per_cell.items():
        for seed, mets in by_seed.items():
            v = mets.get(key)
            if v is not None and not np.isnan(v):
                vals.append(v)
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(mu, sd, n, scientific=False):
    if mu is None:
        return "--"
    if scientific:
        return f"{mu:.2e} $\\pm$ {sd:.1e}"
    return f"{mu:.4f} $\\pm$ {sd:.4f}"


def emit(per_fno, per_lemo, out_path: Path):
    METRICS = [
        ("in_dist",   "Test rel-$L_2$ (in-dist) $\\downarrow$",   False),
        ("ood_mean",  "Test rel-$L_2$ (OOD avg, 4 fams) $\\downarrow$", False),
        ("equiv_k1",  "Equivariance error $k{=}1$ $\\downarrow$",   True),
        ("equiv_k64", "Equivariance error $k{=}64$ $\\downarrow$",  True),
        ("fgsm_05",   "FGSM rel-$L_2$ at $\\varepsilon{=}0.05$ $\\downarrow$",  False),
        ("noise_05",  "Gaussian rel-$L_2$ at $\\sigma{=}0.5$ $\\downarrow$",  False),
        ("lh_128",    "Long-horizon rel-$L_2$ at $h{=}128$ $\\downarrow$", False),
    ]
    lines = [
        r"% Compute-matched comparison: FNO@400ep vs LEMO-PC@200ep, clean regime, 5 fams x 3 seeds.",
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Metric & FNO @ 400 ep & LEMO-PC @ 200 ep \\",
        r" & ($\sim$5670\,s wall-clock) & ($\sim$5951\,s wall-clock) \\",
        r"\midrule",
    ]
    for key, label, sci in METRICS:
        mu_f, sd_f, n_f = aggregate(per_fno, key)
        mu_l, sd_l, n_l = aggregate(per_lemo, key)
        cell_f = fmt(mu_f, sd_f, n_f, sci)
        cell_l = fmt(mu_l, sd_l, n_l, sci)
        # Bold the better (lower) of the two when both present.
        if mu_f is not None and mu_l is not None:
            if mu_l < mu_f:
                cell_l = r"\textbf{" + cell_l + "}"
            elif mu_f < mu_l:
                cell_f = r"\textbf{" + cell_f + "}"
        lines.append(label + " & " + cell_f + " & " + cell_l + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"-> {out_path}")
    # Also print a quick console summary.
    print("\nConsole summary (mean over all clean cells):")
    print(f"{'Metric':40s}  {'FNO@400ep':>20s}  {'LEMO-PC@200ep':>20s}")
    for key, label, sci in METRICS:
        mu_f, sd_f, n_f = aggregate(per_fno, key)
        mu_l, sd_l, n_l = aggregate(per_lemo, key)
        f_str = f"{mu_f:.4e}" if mu_f is not None and sci else (
            f"{mu_f:.4f}" if mu_f is not None else "--")
        l_str = f"{mu_l:.4e}" if mu_l is not None and sci else (
            f"{mu_l:.4f}" if mu_l is not None else "--")
        print(f"  {label[:38]:38s}  {f_str:>20s}  {l_str:>20s}")


def main():
    fno_root = REPO / "outputs" / "w11_compute_matched_runpod"
    # LEMO-PC@200ep: pulled cells live at extracted/h100_pull_2026_05_05/outputs/a_fix_runpod/
    # (a_fix retrain was 100ep but identical model). We also include the
    # original 200ep film_ablation_runpod cells if present locally.
    lemo_roots = [
        REPO / "extracted" / "h100_pull_2026_05_05" / "outputs" / "a_fix_runpod",
        REPO / "extracted" / "pod_pulls_2026_05_03_final",
        REPO / "outputs" / "a_fix_runpod",
    ]

    per_fno = collect("fno_nd", fno_root)
    per_lemo = defaultdict(dict)
    for r in lemo_roots:
        if not r.exists():
            continue
        # Walk recursively so we pick up nested layout (extracted/.../film_ablation_runpod/raw/...).
        for sub in r.rglob("raw"):
            sub_root = sub.parent
            collected = collect("lemo_pc_nd", sub_root)
            for fam, by_seed in collected.items():
                for seed, mets in by_seed.items():
                    # Avoid overwriting if seed already present (prefer first hit).
                    if seed not in per_lemo[fam]:
                        per_lemo[fam][seed] = mets

    print(f"FNO cells:     {sum(len(s) for s in per_fno.values())} cells across {len(per_fno)} families")
    print(f"LEMO-PC cells: {sum(len(s) for s in per_lemo.values())} cells across {len(per_lemo)} families")

    out_path = TABLE_DIR / "T_compute_matched.tex"
    emit(per_fno, per_lemo, out_path)


if __name__ == "__main__":
    main()
