"""M4 â€” Cyclic-shift equivariance demo (T1 in pictures).

For ONE input trajectory `x` from one family's test set, render side-by-side:

  Top row:    rho_k x      for k in {0, 4, 16, 32}
  Bottom row: LEMO-PC(rho_k x)  for the same k's

The bottom row should look visually identical to a cyclic shift of
LEMO-PC(rho_0 x) â€” that's Theorem T1 in pictures.

Annotations: per-shift rel_L2 error
  e_k = ||LEMO(rho_k x) - rho_k LEMO(x)||_2 / ||LEMO(x)||_2
overlaid on each bottom-row panel.  T1 holds iff e_k < 1e-3.

Output: paper/figures/M4_equivariance_demo.{pdf,png}

Source: one ckpt + one test sample.  Picks the first ckpt under a chosen
family's clean/s42/.  The state visualization shows ONE state channel
at the LAST history frame for both rho_k x (input) and LEMO-PC's output
(also at the last frame).
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401  (sets Times New Roman globally)
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
FIG = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG.mkdir(parents=True, exist_ok=True)
# Checkpoint roots: prefer the most recent (8-channel inputs from data_dde_pde),
# fall back to legacy 6-channel snapshot for compatibility.
_BASE_CANDIDATES = [
    REPO / "extracted" / "244_FULL_pull" / "workspace" / "dde-fno" / "extracted"
        / "pod1" / "outputs" / "dist_kernel_v2_p1" / "raw",
    REPO / "outputs" / "dist_kernel_v2_p1" / "raw",
    REPO / "extracted_lemo_pc" / "outputs" / "dist_kernel_v2_p1" / "raw",
]
BASE = next((p for p in _BASE_CANDIDATES if p.exists()), _BASE_CANDIDATES[-1])

# Data directory: prefer local data_dde_pde, fall back to extracted bundles.
_LOCAL_DATA = REPO / "data_dde_pde"
_EXT_DATA_CANDIDATES = [
    REPO / "extracted" / "244_FULL_pull" / "workspace" / "dde-fno" / "data_dde_pde",
    REPO / "extracted" / "full_pulls" / "227" / "workspace" / "dde-fno" / "data_dde_pde",
    REPO / "extracted" / "full_pulls" / "154" / "workspace" / "dde-fno" / "data_dde_pde",
]
DATA_DIR = str(_LOCAL_DATA) if (_LOCAL_DATA / "dist_exp_rd_2d").exists() else next(
    (str(p) for p in _EXT_DATA_CANDIDATES if (p / "dist_exp_rd_2d").exists()),
    "data_dde_pde")

SHIFTS = (0, 4, 16, 32)
FAM = "dist_exp_rd_2d"
SEED = "s42"


def main():
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt_path = BASE / FAM / "clean" / "lemo_pc_nd" / SEED / "best_model.pt"
    if not ckpt_path.exists():
        print(f"[M4] ckpt not found: {ckpt_path}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ra = bool(cfg.get("residual_anchor", False))
    regime = cfg.get("regime", "clean")
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    _, _, test_loader = create_apebench_dataloaders(
        DATA_DIR, FAM, batch_size=1,
        regime=regime, noise_std=noise_std, downsample_factor=downsample_factor,
        residual_anchor=ra, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    x = sample["input"].to(device).float()
    C_state = sample["target"].shape[-1]
    with torch.no_grad():
        y_unshifted = model(x)
    # T1 in pictures: visualize LEMO(roll_k x) (TOP row) vs roll_k(LEMO(x))
    # (BOTTOM row) at a fixed output lag.  The two rows should be visually
    # identical up to float32 numerical precision (~5e-3 rel-err).
    t_show = -1     # output frame to render (last future frame)
    chan = 0
    left_panels, right_panels, errs = [], [], []
    with torch.no_grad():
        for k in SHIFTS:
            x_shifted = torch.roll(x, shifts=int(k), dims=1)
            y_shift = model(x_shifted)              # LEMO(roll_k x)
            y_roll  = torch.roll(y_unshifted, shifts=int(k), dims=1)  # roll_k(LEMO x)
            num = (y_shift - y_roll).flatten(1).norm(dim=1).item()
            den = max(y_roll.flatten(1).norm(dim=1).item(), 1e-12)
            errs.append(num / den)
            left_panels.append(y_shift[0, t_show, ..., chan].cpu().numpy())
            right_panels.append(y_roll[0, t_show, ..., chan].cpu().numpy())
    # Per-column SIGNED difference: LEMO(Ï_k x) - Ï_k LEMO(x). T1 -> near zero
    # (FP32 floor). Sign carries information (over- vs under-shoot of T1) so we
    # render with a diverging RdBu_r colormap matching the V01 error-difference
    # palette: red = positive deviation, blue = negative deviation, white = 0.
    diff_panels = [(t - b) for t, b in zip(left_panels, right_panels)]
    diff_vmax = float(max(max(np.abs(d).max() for d in diff_panels), 1e-9))
    # 32x cubic upsample to match V01 aesthetic.
    from scipy.ndimage import zoom as _zoom
    UPSAMPLE = 32

    fig, axes = plt.subplots(1, len(SHIFTS), figsize=(3.5 * len(SHIFTS), 4.0),
                              gridspec_kw={"wspace": 0.05})
    if len(SHIFTS) == 1:
        axes = [axes]
    for j, (k, diff, err) in enumerate(zip(SHIFTS, diff_panels, errs)):
        ax = axes[j]
        d_hi = _zoom(diff, UPSAMPLE, order=3)
        ax.imshow(d_hi, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax,
                  interpolation="bilinear",
                  extent=[-0.5, diff.shape[1] - 0.5,
                          diff.shape[0] - 0.5, -0.5])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        # Per-panel k label inside the panel (no axis title)
        ax.text(0.5, 0.97, f"$k = {k}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=11, color="black")
        ax.text(0.02, 0.98, f"$e_k = {err:.1e}$",
                  transform=ax.transAxes, fontsize=10,
                  va="top", ha="left", color="black",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.7, edgecolor="none"))
    axes[0].set_ylabel(r"$\mathrm{LEMO}(\rho_k x) - \rho_k\,\mathrm{LEMO}(x)$",
                         rotation=90, ha="center", va="center", fontsize=11,
                         labelpad=10)
    out = FIG / "M4_equivariance_demo.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}  errs = {['%.2e' % e for e in errs]}")


if __name__ == "__main__":
    main()
