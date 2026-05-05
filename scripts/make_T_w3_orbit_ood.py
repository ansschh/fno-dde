"""W3 orbit-OOD table — clean replacement for F_w3 figure.

Reports per-m test rel-L2 for LEMO-PC (exactly equivariant) vs per-lag MLP
(non-equivariant) on the orbit OOD experiment, with the ratio column making
the equivariance gap quantitatively obvious.

Output: paper/tables/T_w3_orbit_ood.tex
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
TABLE_PATH = NEURIPS / "tables" / "T_w3_orbit_ood.tex"
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)


def crawl(roots):
    """Return dict[model][m] = list of test_rel_l2."""
    out = defaultdict(lambda: defaultdict(list))
    m_pattern = re.compile(r"(?:^|_)m(\d+)$")
    for root in roots:
        rp = Path(root)
        if not rp.is_absolute():
            rp = REPO / rp
        if not rp.exists():
            continue
        for tr in rp.glob("**/test_results.json"):
            parts = tr.parts
            if "raw" not in parts:
                continue
            m_seg = None
            for seg in parts:
                m = m_pattern.search(seg)
                if m:
                    m_seg = int(m.group(1))
                    break
            if m_seg is None:
                continue
            mdl = None
            for known in ("lemo_pc_nd", "per_lag_mlp_nd"):
                if known in parts:
                    mdl = known
                    break
            if mdl is None:
                continue
            try:
                d = json.loads(tr.read_text())
            except Exception:
                continue
            err = d.get("test_rel_l2_mean") or d.get("test_rel_l2")
            if err is None:
                continue
            out[mdl][m_seg].append(float(err))
    return out


def emit_table(data):
    if not data:
        print("[T_w3] no data")
        return
    all_ms = sorted({m for d in data.values() for m in d.keys()})
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Orbit-OOD test rel-$L_2$: cyclic-shift orbits of "
        r"\texttt{dist\_exp\_rd\_2d} on $C_n$ (one orbit per kernel-shift "
        r"position). Models train on $m$ representative shifts and test on "
        r"held-out shifts of the same orbit. \textbf{LEMO-PC} (exactly "
        r"$C_n$-equivariant) generalises across shifts by construction; "
        r"\textbf{per-lag MLP} (non-equivariant) cannot extrapolate at any "
        r"augmentation budget. Mean across seeds; $n$ in parentheses.}",
        r"\label{tab:w3-orbit-ood}",
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"$m$ & $r(A) \approx L/(2m)$ & LEMO-PC (eq.) & per-lag MLP (non-eq.) & Ratio \\",
        r"\midrule",
    ]
    L = 2 * np.pi   # cyclic shift orbit length proxy
    for m in all_ms:
        r_A = L / (2 * m)
        lemo_vals = data.get("lemo_pc_nd", {}).get(m, [])
        mlp_vals = data.get("per_lag_mlp_nd", {}).get(m, [])
        if lemo_vals:
            lemo_mean = np.mean(lemo_vals)
            lemo_std = np.std(lemo_vals)
            lemo_str = rf"{lemo_mean:.4f} $\pm$ {lemo_std:.4f} ($n=${len(lemo_vals)})"
        else:
            lemo_mean = None
            lemo_str = "--"
        if mlp_vals:
            mlp_mean = np.mean(mlp_vals)
            mlp_std = np.std(mlp_vals)
            mlp_str = rf"{mlp_mean:.4f} $\pm$ {mlp_std:.4f} ($n=${len(mlp_vals)})"
        else:
            mlp_mean = None
            mlp_str = "--"
        if lemo_mean is not None and mlp_mean is not None and lemo_mean > 0:
            ratio = mlp_mean / lemo_mean
            ratio_str = rf"{ratio:.1f}$\times$"
        else:
            ratio_str = "--"
        lines.append(rf"{m} & {r_A:.3f} & {lemo_str} & {mlp_str} & {ratio_str} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    body = "\n".join(lines) + "\n"
    TABLE_PATH.write_text(body)
    print(f"[T_w3] wrote {TABLE_PATH}")
    for mdl, m_to_errs in sorted(data.items()):
        for m in sorted(m_to_errs.keys()):
            errs = m_to_errs[m]
            print(f"  {mdl:18s} m={m:3d}: n={len(errs):>2d}  mean={np.mean(errs):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    args = ap.parse_args()
    data = crawl(args.roots)
    emit_table(data)


if __name__ == "__main__":
    main()
