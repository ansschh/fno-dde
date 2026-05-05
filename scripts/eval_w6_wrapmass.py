"""W6: post-hoc WrapMass + τ_R diagnostic from kernel_snapshot.npz.

Computes per-checkpoint:
  - WrapMass(R) = sum_{r > R} ‖K_r‖_F / sum_{r=0..n-1} ‖K_r‖_F
                  fraction of kernel Frobenius mass beyond lag R
  - τ_R(R)      = sup_{r > R} σ_max(K_r)
                  operator-norm tail (max singular value of any tail block)
  - lag_n        = ‖K_r‖_F profile vs r (full curve)

Aggregates per family + computes summary stats.

Output: paper/tables/T_w6_wrapmass.tex (5-row table) + per-cell wrapmass.json.

Usage:
  python scripts/eval_w6_wrapmass.py --roots extracted/pod_pulls_2026_05_03_final ...
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NEURIPS = REPO.parent / "NeurIPS_LEMO"
TABLE_PATH = NEURIPS / "tables" / "T_w6_wrapmass.tex"
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

R_VALUES = [16, 24, 32]


def parse_path(snap_path):
    parts = snap_path.parts
    try:
        idx = parts.index("raw")
    except ValueError:
        return None
    if idx + 4 >= len(parts):
        return None
    fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
    if not seed_str.startswith("s"):
        return None
    return fam, reg, mdl, int(seed_str[1:])


def kernel_block_norms(K_complex, n):
    """K_complex shape (in, out, n_modes). Return per-time-r block norms.

    Time-domain reconstruction: K_t[r] = ifft(K_complex along last axis).
    Per-r block norm: ‖K_t[:, :, r]‖_F (Frobenius) and σ_max(K_t[:, :, r]) (op norm).
    """
    # Pad to length n along last axis
    n_modes = K_complex.shape[-1]
    if n_modes < (n // 2 + 1):
        pad_width = [(0, 0)] * (K_complex.ndim - 1) + [(0, (n // 2 + 1) - n_modes)]
        K_padded = np.pad(K_complex, pad_width, mode='constant')
    else:
        K_padded = K_complex
    # Inverse FFT to time domain
    K_time = np.fft.irfft(K_padded, n=n, axis=-1)   # (in, out, n)
    frob_per_r = np.linalg.norm(K_time, axis=(0, 1))   # (n,)
    op_per_r = np.array([
        float(np.linalg.svd(K_time[:, :, r], compute_uv=False).max())
        for r in range(n)
    ])
    return frob_per_r, op_per_r


def wrapmass_for_cell(snap_path: Path, n_lag: int = 64):
    """Return per-cell WrapMass + tau_R diagnostics."""
    try:
        npz = np.load(snap_path)
    except Exception as e:
        return None
    keys = list(npz.keys())
    # Find lag-conv kernel keys: "blocks.{i}.A_lag.weights__re" / "_im"
    block_indices = sorted({
        int(k.split(".")[1]) for k in keys
        if k.startswith("blocks.") and ".A_lag.weights__" in k
    })
    if not block_indices:
        return None
    per_layer = []
    for ell in block_indices:
        re_key = f"blocks.{ell}.A_lag.weights__re"
        im_key = f"blocks.{ell}.A_lag.weights__im"
        if re_key not in npz or im_key not in npz:
            continue
        K = npz[re_key] + 1j * npz[im_key]   # (in, out, n_modes)
        frob_per_r, op_per_r = kernel_block_norms(K, n_lag)
        layer = {
            "frob_per_r": frob_per_r.tolist(),
            "op_per_r": op_per_r.tolist(),
        }
        for R in R_VALUES:
            wrapmass = float(frob_per_r[R+1:].sum() / max(frob_per_r.sum(), 1e-12))
            tau_R = float(op_per_r[R+1:].max() if R+1 < len(op_per_r) else 0.0)
            layer[f"wrapmass_{R}"] = wrapmass
            layer[f"tau_{R}"] = tau_R
        per_layer.append(layer)
    if not per_layer:
        return None
    # Aggregate over layers (max wrapmass / max tau)
    summary = {"per_layer": per_layer}
    for R in R_VALUES:
        summary[f"wrapmass_{R}_max"] = max(L[f"wrapmass_{R}"] for L in per_layer)
        summary[f"tau_{R}_max"] = max(L[f"tau_{R}"] for L in per_layer)
    return summary


def emit_table(by_family):
    if not by_family:
        print("[w6-wrapmass] no data")
        return
    fams = sorted(by_family.keys())
    fam_labels = {
        "dist_exp_rd_2d": "Exp",
        "dist_gaussian_rd_2d": "Gauss",
        "dist_gamma_rd_2d": "Gamma",
        "dist_uniform_rd_2d": "Uniform",
        "dist_powerlaw_rd_2d": "Power",
    }
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{W6 wrap-mass diagnostic from learned cyclic-FFT lag kernels. "
        r"$\mathrm{WrapMass}(R) = \sum_{r>R} \|K_r\|_F / \sum_r \|K_r\|_F$ is the "
        r"fraction of kernel Frobenius mass beyond truncation $R$. "
        r"$\tau_R = \sup_{r>R} \sigma_{\max}(K_r)$ is the operator-norm tail "
        r"(the quantity that actually controls Prop 5.10b's boundary discrepancy). "
        r"All values are max-over-layers, mean-over-(seed, regime). "
        r"Lag window $n=128$.}",
        r"\label{tab:w6-wrapmass}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Family & WrapMass($R$=24) & WrapMass($R$=32) & $\tau_{24}$ & $\tau_{32}$ \\",
        r"\midrule",
    ]
    for fam in fams:
        if fam not in fam_labels:
            continue
        cells = by_family[fam]
        if not cells:
            continue
        wm24 = np.mean([c["wrapmass_24_max"] for c in cells])
        wm32 = np.mean([c["wrapmass_32_max"] for c in cells])
        t24 = np.mean([c["tau_24_max"] for c in cells])
        t32 = np.mean([c["tau_32_max"] for c in cells])
        n = len(cells)
        lines.append(
            rf"{fam_labels[fam]} & {wm24:.3f} & {wm32:.3f} & {t24:.3f} & {t32:.3f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    body = "\n".join(lines) + "\n"
    TABLE_PATH.write_text(body)
    print(f"[w6-wrapmass] wrote {TABLE_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--n_lag", type=int, default=128)
    args = ap.parse_args()

    by_family = defaultdict(list)
    n_total = 0
    for r in args.roots:
        rp = Path(r)
        if not rp.is_absolute():
            rp = REPO / rp
        for snap in rp.glob("**/kernel_snapshot.npz"):
            meta = parse_path(snap)
            if meta is None:
                continue
            fam, reg, mdl, seed = meta
            # Only LEMO-PC / LEMO_σ / variants (cyclic-FFT lag kernel)
            if mdl not in ("lemo_pc_nd", "causal_smooth_lemo_pc_nd",
                            "lemo_bcorrect_nd", "lemo_nd"):
                continue
            res = wrapmass_for_cell(snap, n_lag=args.n_lag)
            if res is None:
                continue
            res["family"] = fam
            res["regime"] = reg
            res["model"] = mdl
            res["seed"] = seed
            (snap.parent / "wrapmass.json").write_text(json.dumps(res, indent=2))
            by_family[fam].append(res)
            n_total += 1

    print(f"[w6-wrapmass] processed {n_total} cells")
    for fam in sorted(by_family.keys()):
        cells = by_family[fam]
        if not cells:
            continue
        wm24 = np.mean([c["wrapmass_24_max"] for c in cells])
        t24 = np.mean([c["tau_24_max"] for c in cells])
        print(f"  {fam:25s}: n={len(cells):>3d}  WrapMass(24)={wm24:.3f}  tau_24={t24:.3f}")

    emit_table(by_family)


if __name__ == "__main__":
    main()
