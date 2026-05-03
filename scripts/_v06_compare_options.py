"""V06 redesign — generate three variants for comparison.

Outputs (under NeurIPS_LEMO/figures/):
  V06_optionA_cumulative.{pdf,png}   2-panel: per-mode + cumulative %
  V06_optionB_overlay.{pdf,png}      single panel, residual energy + LEMO op-norm overlay
  V06_optionC_baselines.{pdf,png}    per-mode residual energy across LEMO-PC vs FNO+FiLM
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"A:\dde research\dde-fno")
FIG = REPO.parent / "NeurIPS_LEMO" / "figures"
sys.path.insert(0, str(REPO / "scripts"))
from make_visual_figures import (FAMS, FAM_LABELS, SEEDS, load_viz, load_viz_fno,
                                  load_kernel)


def collect_residual_energy(model: str = "lemo_pc_nd"):
    """Returns {fam: [energy_array, ...]} where each entry is (T_mode,) FFT energy
    averaged across batch+space+channels for one seed."""
    series = defaultdict(list)
    for fam in FAMS:
        for seed in SEEDS:
            if model == "lemo_pc_nd":
                d = load_viz(fam, "clean", seed)
            elif model == "fno_film_nd":
                d = load_viz_fno(fam, "clean", seed)
            else:
                d = None
            if d is None:
                continue
            target = d["target"]
            pred = d["pred"]
            r = pred - target
            perm = [0] + list(range(2, r.ndim)) + [1]
            r_p = np.transpose(r, axes=perm)
            R = np.fft.rfft(r_p, axis=-1)
            energy = np.mean(np.abs(R) ** 2, axis=tuple(range(R.ndim - 1)))
            series[fam].append(energy)
    return series


def collect_kernel_op_norm():
    """Per-mode operator norm σ_max(K̂(ω_m)) for LEMO-PC, averaged across families
    and seeds. Returns (modes, mean, std)."""
    all_curves = []
    for fam in FAMS:
        for seed in SEEDS:
            d = load_kernel(fam, "clean", seed)
            if d is None:
                continue
            keys = list(d.keys())
            re_keys = [k for k in keys if k.endswith("__re") and "weights" in k
                        and "film" not in k and "A_lag" in k]
            if not re_keys:
                continue
            re = d[re_keys[0]]
            im_key = re_keys[0].replace("__re", "__im")
            if im_key not in d:
                continue
            K = re + 1j * d[im_key]
            in_ch, out_ch, M = K.shape
            sigmas = []
            for m in range(M):
                Km = K[:, :, m]
                # SVD top singular value
                s = np.linalg.svd(Km, compute_uv=False)
                sigmas.append(float(s[0]))
            all_curves.append(np.array(sigmas))
    if not all_curves:
        return None, None, None
    L = min(len(c) for c in all_curves)
    arr = np.stack([c[:L] for c in all_curves], axis=0)
    return np.arange(L), arr.mean(axis=0), arr.std(axis=0)


# ----- Option A: 2-panel (per-mode + cumulative %) -----

def make_option_A(series_lemo):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                    gridspec_kw={"wspace": 0.30})
    for fam, es in series_lemo.items():
        e = np.array(es)
        m = e.mean(axis=0)
        s = e.std(axis=0)
        modes = np.arange(len(m))
        axL.plot(modes, m, lw=1.5, label=FAM_LABELS[fam])
        axL.fill_between(modes, m - s, m + s, alpha=0.18)
        # Cumulative fraction of energy
        cumulative = np.cumsum(m)
        cumulative /= cumulative[-1]
        axR.plot(modes, cumulative * 100, lw=1.5, label=FAM_LABELS[fam])
    axL.set_yscale("log")
    axL.set_xlabel("spectral lag mode $m$")
    axL.set_ylabel(r"$\mathbb{E}\,|\hat{r}_m|^2$")
    axL.set_title("Per-mode residual energy")
    axL.grid(linestyle=":", alpha=0.5, which="both")
    for sp in ("top", "right"): axL.spines[sp].set_visible(False)
    axR.axvline(24, color="dimgrey", linestyle="--", lw=1.2)
    # Compute cumulative at m=24 as annotation
    avg_cumulative = np.mean(
        [np.cumsum(np.array(series_lemo[f]).mean(axis=0))
         / np.cumsum(np.array(series_lemo[f]).mean(axis=0))[-1]
         for f in series_lemo], axis=0)
    if len(avg_cumulative) > 24:
        axR.annotate(f"M=24:\n{avg_cumulative[24]*100:.1f}% energy",
                      xy=(24, avg_cumulative[24] * 100),
                      xytext=(34, 60),
                      fontsize=10, color="dimgrey",
                      arrowprops=dict(arrowstyle="->", color="dimgrey", lw=0.8))
    axR.set_xlabel("modes retained $M$")
    axR.set_ylabel("% of total residual energy")
    axR.set_title("Cumulative energy capture")
    axR.set_ylim(0, 105)
    axR.grid(linestyle=":", alpha=0.5)
    for sp in ("top", "right"): axR.spines[sp].set_visible(False)
    axL.legend(loc="upper right", fontsize=9, frameon=False, ncol=1)
    fig.suptitle("Residual spectrum and truncation justification", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG / "V06_optionA_cumulative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ----- Option B: residual energy + kernel op-norm overlay (single panel) -----

def make_option_B(series_lemo, kernel_modes, kernel_mean, kernel_std):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    # Mean residual energy across families
    avg = np.mean([np.array(series_lemo[f]).mean(axis=0) for f in series_lemo], axis=0)
    avg_std = np.mean([np.array(series_lemo[f]).std(axis=0) for f in series_lemo], axis=0)
    modes = np.arange(len(avg))
    color_e = "#7e2dac"
    color_k = "#d62728"
    line_e = ax.plot(modes, avg, color=color_e, lw=1.8,
                      label="residual energy $\\mathbb{E}\\,|\\hat{r}_m|^2$")[0]
    ax.fill_between(modes, np.maximum(avg - avg_std, 1e-3), avg + avg_std,
                     color=color_e, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("spectral lag mode $m$")
    ax.set_ylabel(r"residual energy  $\mathbb{E}\,|\hat{r}_m|^2$",
                   color=color_e)
    ax.tick_params(axis="y", labelcolor=color_e)
    # Right axis: kernel op-norm
    ax2 = ax.twinx()
    line_k = ax2.plot(kernel_modes, kernel_mean, color=color_k, lw=1.8,
                       label="LEMO-PC per-mode op-norm $\\sigma_{\\max}$")[0]
    ax2.fill_between(kernel_modes,
                      np.maximum(kernel_mean - kernel_std, 0),
                      kernel_mean + kernel_std,
                      color=color_k, alpha=0.18, lw=0)
    ax2.set_ylabel(r"LEMO-PC kernel  $\sigma_{\max}(\widehat{K}(\omega_m))$",
                    color=color_k)
    ax2.tick_params(axis="y", labelcolor=color_k)
    ax.axvline(24, color="dimgrey", linestyle="--", lw=1.0, alpha=0.6)
    ax.text(24.5, ax.get_ylim()[1] * 0.5, "lag_modes = 24",
             fontsize=9, color="dimgrey", rotation=0)
    ax.grid(linestyle=":", alpha=0.4)
    for sp in ("top",): ax.spines[sp].set_visible(False)
    for sp in ("top",): ax2.spines[sp].set_visible(False)
    fig.legend(handles=[line_e, line_k], loc="upper right",
                bbox_to_anchor=(0.85, 0.95), fontsize=9, frameon=False)
    fig.suptitle("Residual spectrum and learned kernel mode usage",
                  fontsize=12, y=1.0)
    fig.tight_layout()
    out = FIG / "V06_optionB_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ----- Option C: multi-baseline residual energy comparison -----

def make_option_C(series_lemo, series_fno):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if series_lemo:
        avg = np.mean([np.array(series_lemo[f]).mean(axis=0)
                        for f in series_lemo], axis=0)
        std = np.mean([np.array(series_lemo[f]).std(axis=0)
                        for f in series_lemo], axis=0)
        modes = np.arange(len(avg))
        ax.plot(modes, avg, color="#d62728", lw=1.8, label="LEMO-PC")
        ax.fill_between(modes, np.maximum(avg - std, 1e-3), avg + std,
                         color="#d62728", alpha=0.15, lw=0)
    if series_fno:
        avg_f = np.mean([np.array(series_fno[f]).mean(axis=0)
                          for f in series_fno], axis=0)
        std_f = np.mean([np.array(series_fno[f]).std(axis=0)
                          for f in series_fno], axis=0)
        modes_f = np.arange(len(avg_f))
        ax.plot(modes_f, avg_f, color="#17becf", lw=1.8, label="FNO+FiLM")
        ax.fill_between(modes_f, np.maximum(avg_f - std_f, 1e-3), avg_f + std_f,
                         color="#17becf", alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("spectral lag mode $m$")
    ax.set_ylabel(r"$\mathbb{E}\,|\hat{r}_m|^2$")
    ax.axvline(24, color="dimgrey", linestyle="--", lw=1.0, alpha=0.6)
    ax.text(24.5, ax.get_ylim()[1] * 0.5, "lag_modes = 24",
             fontsize=9, color="dimgrey")
    ax.grid(linestyle=":", alpha=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    fig.suptitle("Residual energy: LEMO-PC vs FNO+FiLM (averaged over families)",
                  fontsize=12, y=1.0)
    fig.tight_layout()
    out = FIG / "V06_optionC_baselines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("Collecting LEMO-PC residual energy...")
    series_lemo = collect_residual_energy("lemo_pc_nd")
    print(f"  found {sum(len(v) for v in series_lemo.values())} cells across {len(series_lemo)} families")

    print("Collecting FNO+FiLM residual energy...")
    series_fno = collect_residual_energy("fno_film_nd")
    print(f"  found {sum(len(v) for v in series_fno.values())} cells across {len(series_fno)} families")

    print("Collecting LEMO-PC kernel per-mode op-norm...")
    k_modes, k_mean, k_std = collect_kernel_op_norm()
    if k_mean is None:
        print("  [warn] no kernel snapshots found; option B will skip")

    if series_lemo:
        out = make_option_A(series_lemo)
        print(f"-> {out.name}")
    if series_lemo and k_mean is not None:
        out = make_option_B(series_lemo, k_modes, k_mean, k_std)
        print(f"-> {out.name}")
    if series_lemo:
        out = make_option_C(series_lemo, series_fno)
        print(f"-> {out.name}")


if __name__ == "__main__":
    main()
