"""M4 — Cyclic-shift equivariance demo (T1 in pictures).

For ONE input trajectory `x` from one family's test set, render side-by-side:

  Top row:    rho_k x      for k in {0, 4, 16, 32}
  Bottom row: LEMO-PC(rho_k x)  for the same k's

The bottom row should look visually identical to a cyclic shift of
LEMO-PC(rho_0 x) — that's Theorem T1 in pictures.

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
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
FIG = REPO / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BASE = REPO / "extracted_lemo_pc" / "outputs" / "dist_kernel_v2_p1" / "raw"

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
    ra = bool(cfg.get("residual_anchor", True))
    regime = cfg.get("regime", "clean")
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    _, _, test_loader = create_apebench_dataloaders(
        "data_dde_pde", FAM, batch_size=1,
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
    rows_input, rows_pred, errs = [], [], []
    with torch.no_grad():
        for k in SHIFTS:
            x_state = x[..., :C_state]
            x_aux = x[..., C_state:]
            x_state_shifted = torch.roll(x_state, shifts=int(k), dims=1)
            x_shifted = torch.cat([x_state_shifted, x_aux], dim=-1)
            y_shift = model(x_shifted)
            y_roll  = torch.roll(y_unshifted, shifts=int(k), dims=1)
            num = (y_shift - y_roll).flatten(1).norm(dim=1).item()
            den = max(y_roll.flatten(1).norm(dim=1).item(), 1e-12)
            rel = num / den
            errs.append(rel)
            # Pick the LAST history frame of the input (residual-anchor
            # makes signal[n_hist-1]=0 so use frame 0 instead).
            rows_input.append(x_state_shifted[0, 0, ..., 0].cpu().numpy())
            # Pred at the LAST output frame.
            rows_pred.append(y_shift[0, -1, ..., 0].cpu().numpy())
    fig, axes = plt.subplots(2, len(SHIFTS), figsize=(2.0 * len(SHIFTS), 4.4),
                              gridspec_kw={"wspace": 0.06, "hspace": 0.18})
    if len(SHIFTS) == 1:
        axes = axes.reshape(2, 1)
    inp_max = max(np.abs(arr).max() for arr in rows_input)
    pred_max = max(np.abs(arr).max() for arr in rows_pred)
    for j, (k, inp, pred, err) in enumerate(zip(SHIFTS, rows_input, rows_pred, errs)):
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, j].imshow(inp, cmap="RdBu_r", vmin=-inp_max, vmax=inp_max)
        axes[1, j].imshow(pred, cmap="RdBu_r", vmin=-pred_max, vmax=pred_max)
        axes[0, j].set_title(f"$k = {k}$", fontsize=10)
        # Annotate equivariance error on bottom row.
        col = "green" if err < 1e-3 else "red"
        axes[1, j].text(0.02, 0.98, f"$e_k = {err:.1e}$",
                          transform=axes[1, j].transAxes, fontsize=8,
                          va="top", ha="left", color=col,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                      alpha=0.85, edgecolor="none"))
    axes[0, 0].set_ylabel(r"input $\rho_k x$",
                            rotation=0, ha="right", va="center", fontsize=10)
    axes[1, 0].set_ylabel(r"pred $\mathrm{LEMO}(\rho_k x)$",
                            rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(("Cyclic-shift equivariance at deployment "
                  + r"($e_k = \|\mathrm{LEMO}(\rho_k x) - \rho_k \mathrm{LEMO}(x)\|_2/\|\mathrm{LEMO}(x)\|_2$)"),
                 fontsize=11)
    out = FIG / "M4_equivariance_demo.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}  errs = {['%.2e' % e for e in errs]}")


if __name__ == "__main__":
    main()
