"""M8 — Receptive-field saliency. **DROPPED 2026-05-03** — saliency vs GT
kernel correlations (Pearson ρ ∈ [-0.16, 0.46]) currently make LEMO-PC
look weaker than it is, due to FiLM nullification preventing per-family
kernel adaptation. Even after the FiLM-fix retrain improves the numbers,
the figure competes with V05 (which we also dropped for the same reason)
and the headline kernel-recovery story is already carried by T06 (cosine
similarity table). Script kept on disk; not invoked by any pipeline.

Receptive-field saliency: which past frames does LEMO-PC's
prediction depend on?

For each family's first-seed clean ckpt:
  - load model + one test sample
  - require_grad on input state-channels
  - forward → take the predicted state at output frame T (last)
  - sum |predicted state at T| over spatial+channel → scalar L
  - dL / d(input_state(t_in)) for each t_in in [0, n_total-1]
  - aggregate |gradient| over (sample, spatial, channel) → curve over t_in

Overlay the curve with the analytic ground-truth distributed-delay kernel
shape per family (normalized).  If LEMO-PC genuinely learned the
distributed-delay structure the saliency peak should align with the
GT kernel mass.

5 panels, one per family, in one figure.

Source: best_model.pt + one test sample per family.

Output: paper/figures/M8_saliency_map.{pdf,png}
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
BASE = REPO / "extracted_lemo_pc" / "outputs" / "dist_kernel_v2_p1" / "raw"

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
FAM_LABELS = {"dist_exp_rd_2d": "Exp", "dist_gaussian_rd_2d": "Gauss",
              "dist_gamma_rd_2d": "Gamma", "dist_uniform_rd_2d": "Uniform",
              "dist_powerlaw_rd_2d": "Power"}
SEED = "s42"


def gt_kernel_shape(family: str, n: int):
    """Same hand-coded kernel shapes used in post_hoc_analyses.py."""
    t = np.arange(n) / max(n - 1, 1)
    if family.startswith("dist_exp"):       gt = np.exp(-3 * t)
    elif family.startswith("dist_gaussian"):gt = np.exp(-((t - 0.3) ** 2) / 0.05)
    elif family.startswith("dist_gamma"):   gt = (t ** 1.5) * np.exp(-3 * t)
    elif family.startswith("dist_uniform"): gt = (t < 0.5).astype(np.float32)
    elif family.startswith("dist_powerlaw"):gt = (t + 0.05) ** (-1.2)
    else:                                    gt = np.zeros_like(t)
    if gt.sum() > 0:
        gt = gt / gt.max()
    return gt


def saliency_for_family(fam: str, device: str):
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt_path = BASE / fam / "clean" / "lemo_pc_nd" / SEED / "best_model.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ra = bool(cfg.get("residual_anchor", False))
    regime = cfg.get("regime", "clean")
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    _, _, test_loader = create_apebench_dataloaders(
        "data_dde_pde", fam, batch_size=2,
        regime=regime, noise_std=noise_std, downsample_factor=downsample_factor,
        residual_anchor=ra, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    n_state_ch = sample["target"].shape[-1]
    x = sample["input"].to(device).float()
    x_state = x[..., :n_state_ch].detach().clone().requires_grad_(True)
    x_aux = x[..., n_state_ch:].detach()
    x_full = torch.cat([x_state, x_aux], dim=-1)
    y = model(x_full)
    # Loss = sum of |predicted state at last output frame|.
    L = y[:, -1].abs().sum()
    grad = torch.autograd.grad(L, x_state)[0]
    # Aggregate |grad| over (batch, spatial, channel) → curve over t_in.
    sal = grad.abs().mean(dim=tuple([0] + list(range(2, grad.dim()))))
    sal = sal.detach().cpu().numpy()
    return sal, n_total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(1, len(FAMS), figsize=(2.8 * len(FAMS), 3.2),
                              sharey=True,
                              gridspec_kw={"wspace": 0.10})
    if len(FAMS) == 1:
        axes = [axes]
    n_done = 0
    for ax, fam in zip(axes, FAMS):
        res = saliency_for_family(fam, device)
        if res is None:
            ax.set_visible(False); continue
        sal, n_total = res
        sal_norm = sal / (sal.max() + 1e-12)
        gt = gt_kernel_shape(fam, n_total)
        # Reverse time so that x-axis represents "lag from output frame"
        # = t_out - t_in (so 0 is most recent past, large is far past).
        t_lag = np.arange(n_total)[::-1]
        ax.plot(t_lag, sal_norm, color="#d62728", lw=1.6, label="saliency")
        ax.plot(t_lag, gt, color="black", lw=1.0, linestyle="--", label="GT kernel")
        # Pearson correlation as a quantitative metric per family.
        a = sal_norm - sal_norm.mean()
        b = gt - gt.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        rho = float((a * b).sum() / denom)
        ax.set_xlabel(r"lag $t_{out} - t_{in}$", fontsize=10)
        ax.set_title(f"{FAM_LABELS[fam]}  (ρ={rho:.2f})", fontsize=11)
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_xlim(0, n_total - 1)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        n_done += 1
    if n_done == 0:
        plt.close(fig); return
    axes[0].set_ylabel("normalized")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Receptive-field saliency vs GT kernel",
                  fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = FIG / "M8_saliency_map.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
